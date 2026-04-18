# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os, logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask
from cosyvoice.utils.onnx import SpeechTokenExtractor, online_feature, onnx_path


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        # NOTE this is unnecessary, feat/h already same shape
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat speech token and prompt speech token
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v2.batch.onnx'))

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if 'speech_token' not in batch:
            token, token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            token = batch['speech_token'].to(device)
            token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h, h_lengths = self.encoder(token, token_len, streaming=streaming)
        else:
            token, context = token[:, :-self.pre_lookahead_len], token[:, -self.pre_lookahead_len:]
            h, h_lengths = self.encoder(token, token_len, context=context, streaming=streaming)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 pre_lookahead_layer: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 # Plan B: 让 Flow 直接接收 instruct 条件,绕过 LLM 的 token 瓶颈
                 instruct_cond: bool = False,
                 instruct_vocab_size: int = 152064,  # Qwen tokenizer vocab
                 instruct_embed_dim: int = 128):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        # Instruct FiLM + 独立 100× 学习率(见 train_utils._split_param_groups)
        # scale=1(不放大),init std=0.01(初始扰动极小,保证语音可懂)
        # 让 100× lr 慢慢把 FiLM 参数推到应有的方向,不破坏已学的 Flow 主体
        self.instruct_cond = instruct_cond
        self.film_scale = 5.0
        if instruct_cond:
            self.instruct_embedding = nn.Embedding(instruct_vocab_size, instruct_embed_dim)
            self.instruct_proj_gamma = nn.Linear(instruct_embed_dim, output_size)
            self.instruct_proj_beta = nn.Linear(instruct_embed_dim, output_size)
            nn.init.zeros_(self.instruct_proj_gamma.bias)
            nn.init.zeros_(self.instruct_proj_beta.bias)
            nn.init.normal_(self.instruct_proj_gamma.weight, std=0.01)
            nn.init.normal_(self.instruct_proj_beta.weight, std=0.01)
            logging.info(f"Plan B FiLM: instruct_cond enabled (vocab={instruct_vocab_size}, embed={instruct_embed_dim}, scale={self.film_scale})")
        if online_feature is True:
            self.speech_token_extractor = SpeechTokenExtractor(model_path=os.path.join(onnx_path, 'speech_tokenizer_v3.batch.onnx'))

    def _compute_instruct_film(self, instruct_token, instruct_token_len, device, dtype):
        """instruct token → mean-pooled embed → gamma/beta 用于 FiLM 调制。
        返回 (gamma, beta),两者形状 [B, output_size],推荐用法:h = h * (1 + gamma) + beta"""
        if not self.instruct_cond or instruct_token is None:
            return None, None
        instruct_token = instruct_token.to(device)
        instruct_token_len = instruct_token_len.to(device)
        emb = self.instruct_embedding(torch.clamp(instruct_token, min=0))  # [B, L, D]
        L = emb.shape[1]
        mask = (torch.arange(L, device=device).unsqueeze(0) < instruct_token_len.unsqueeze(1)).float()
        emb = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, D]
        gamma = self.instruct_proj_gamma(emb).to(dtype)  # [B, output_size]
        beta = self.instruct_proj_beta(emb).to(dtype)
        return gamma, beta

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if 'speech_token' not in batch:
            token, token_len = self.speech_token_extractor.inference(batch['whisper_feat'], batch['whisper_feat_len'], device)
        else:
            token = batch['speech_token'].to(device)
            token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h = self.pre_lookahead_layer(token)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mask = mask.repeat_interleave(self.token_mel_ratio, dim=1).squeeze(dim=-1)

        # Plan B FiLM: h = h * (1 + gamma) + beta
        gamma, beta = self._compute_instruct_film(
            batch.get('instruct_token'), batch.get('instruct_token_len'), device, h.dtype)
        if gamma is not None:
            # film_scale 放大 FiLM 影响,弥补小 lr 下的慢学习
            g = gamma.unsqueeze(1) * self.film_scale
            b = beta.unsqueeze(1) * self.film_scale
            h = h * (1.0 + g) + b

        # get conditions
        # Plan B: 训练时强制 conds=0,断掉"reverb mel 直接复制到 conds"的泄漏
        # (原代码 50% 概率把 reverb mel 前 30% 当 conds,Flow 不学 instruct 也能复用)
        conds = torch.zeros(feat.shape, device=token.device).transpose(1, 2)

        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  streaming,
                  finalize,
                  instruct_token=None,
                  instruct_token_len=None):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        # Plan B FiLM: h = h * (1 + gamma) + beta
        gamma, beta = self._compute_instruct_film(
            instruct_token, instruct_token_len, token.device, h.dtype)
        if gamma is not None:
            # film_scale 放大 FiLM 影响,弥补小 lr 下的慢学习
            g = gamma.unsqueeze(1) * self.film_scale
            b = beta.unsqueeze(1) * self.film_scale
            h = h * (1.0 + g) + b

        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from hyperpyyaml import load_hyperpyyaml
    with open('./pretrained_models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'hift': None})
    model = configs['flow']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    max_len = 10 * model.decoder.estimator.static_chunk_size
    chunk_size = model.decoder.estimator.static_chunk_size
    context_size = model.pre_lookahead_layer.pre_lookahead_len
    token = torch.randint(0, 6561, size=(1, max_len)).to(device)
    token_len = torch.tensor([max_len]).to(device)
    prompt_token = torch.randint(0, 6561, size=(1, chunk_size)).to(device)
    prompt_token_len = torch.tensor([chunk_size]).to(device)
    prompt_feat = torch.rand(1, chunk_size * 2, 80).to(device)
    prompt_feat_len = torch.tensor([chunk_size * 2]).to(device)
    prompt_embedding = torch.rand(1, 192).to(device)
    pred_gt, _ = model.inference(token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, prompt_embedding, streaming=True, finalize=True)
    for i in range(0, max_len, chunk_size):
        finalize = True if i + chunk_size + context_size >= max_len else False
        pred_chunk, _ = model.inference(token[:, :i + chunk_size + context_size], torch.tensor([token[:, :i + chunk_size + context_size].shape[1]]).to(device),
                                        prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, prompt_embedding, streaming=True, finalize=finalize)
        pred_chunk = pred_chunk[:, :, i * model.token_mel_ratio:]
        print((pred_gt[:, :, i * model.token_mel_ratio: i * model.token_mel_ratio + pred_chunk.shape[2]] - pred_chunk).abs().max().item())
