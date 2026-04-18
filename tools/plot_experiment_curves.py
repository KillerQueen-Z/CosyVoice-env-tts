"""从训练日志里提取 CV loss 并画成曲线,保存为 PNG 供 markdown 使用。"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

LOGS_DIR = Path('/home/exouser/CosyVoice-env-tts/logs')
OUT_DIR = Path('/home/exouser/CosyVoice-env-tts/docs/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_cv(log_path, max_epoch=None):
    """提取 CV info 的 (epoch, loss, acc),acc 可能缺失"""
    epochs, losses, accs = [], [], []
    if not log_path.exists():
        return epochs, losses, accs
    for line in log_path.read_text(errors='ignore').splitlines():
        m = re.search(r'Epoch\s+(\d+).*CV info.*loss\s+([\d.]+)(?:.*acc\s+([\d.]+))?', line)
        if m:
            e = int(m.group(1))
            if max_epoch is not None and e > max_epoch:
                continue
            epochs.append(e)
            losses.append(float(m.group(2)))
            accs.append(float(m.group(3)) if m.group(3) else None)
    return epochs, losses, accs


def parse_neural_reverb(log_path):
    epochs, train, dev = [], [], []
    for line in log_path.read_text().splitlines():
        m = re.search(r'E\s*(\d+)\s+train=([\d.]+)\s+dev=([\d.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            train.append(float(m.group(2)))
            dev.append(float(m.group(3)))
    return epochs, train, dev


def save(fig, name):
    path = OUT_DIR / f'{name}.png'
    fig.savefig(str(path), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {path}')


def main():
    # ---- Figure 1: LLM SFT 三次尝试的 CV loss 对比 ----
    e1, l1, _ = parse_cv(LOGS_DIR / 'train_both_20260416_003538.log', max_epoch=14)
    e2, l2, _ = parse_cv(LOGS_DIR / 'train_both_canonical_20260417_005725.log', max_epoch=9)
    e3, l3, _ = parse_cv(LOGS_DIR / 'train_llm_strong_20260418_005710.log', max_epoch=24)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(e1, l1, 'r-o', label='v1: lr=1e-5 (no canonical) — collapses', linewidth=2)
    ax.plot(e2, l2, 'g-s', label='v2: lr=1e-6 (canonical) — healthy but weak', linewidth=2)
    ax.plot(e3, l3, 'b-^', label='v3: lr=5e-6 — rebounds at E5', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('CV loss')
    ax.set_title('LLM SFT: Three attempts (lower = better)')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, 'llm_sft_comparison')

    # ---- Figure 2: LLM SFT v2 细节(合适的 zoom)----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(e2, l2, 'g-o', linewidth=2, label='CV loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('CV loss')
    ax.set_title('LLM SFT v2 (canonical, lr=1e-6): healthy but weak convergence\n'
                 'CV loss drops 3.92 → 3.83 (Δ=0.09), no audible reverb difference')
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, 'llm_sft_v2_detail')

    # ---- Figure 3: LLM SFT v3(lr=5e-6)—— 呈现 U 形反弹 ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(e3, l3, 'b-o', linewidth=2)
    best = min(range(len(l3)), key=lambda i: l3[i])
    ax.axvline(e3[best], color='red', linestyle='--', alpha=0.5,
               label=f'best at E{e3[best]} (loss={l3[best]:.3f})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('CV loss')
    ax.set_title('LLM SFT v3 (lr=5e-6): U-shape — best at E4 then overfits')
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, 'llm_sft_v3_rebound')

    # ---- Figure 4: Plan B Flow 训练曲线 ----
    e4, l4, _ = parse_cv(LOGS_DIR / 'train_flow_planb_v3_safe_20260417_110735.log')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(e4, l4, 'm-o', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('CV loss')
    ax.set_title('Flow Plan B (FiLM conditioning, v3-safe):\nCV loss 0.64 -> 0.58 (15 epochs), but FiLM params barely activated')
    ax.grid(alpha=0.3)
    save(fig, 'flow_planb_v3_safe')

    # ---- Figure 5: Neural Reverb 训练曲线 ----
    ne, nt, nd = parse_neural_reverb(LOGS_DIR / 'train_neural_reverb_20260418_024551.log')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ne, nt, 'b-o', label='Train loss', linewidth=2)
    ax.plot(ne, nd, 'r-s', label='Dev loss', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (L1 + multi-STFT)')
    ax.set_title('Neural Reverb (4-class learnable RIR, 96k params):\nhealthy convergence, 30 epochs from 5.67 -> 3.14')
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, 'neural_reverb_training')

    # ---- Figure 6: Classifier confusion matrix ----
    import numpy as np
    cm = np.array([[250, 0, 0, 0], [1, 246, 28, 7], [1, 10, 192, 15], [0, 2, 16, 232]])
    classes = ['clean', 'small', 'medium', 'large']
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Reverb Classifier Confusion Matrix (92% dev acc)')
    for i in range(4):
        for j in range(4):
            color = 'white' if cm[i,j] > cm.max() / 2 else 'black'
            ax.text(j, i, cm[i,j], ha='center', va='center', color=color, fontweight='bold')
    plt.colorbar(im)
    save(fig, 'classifier_confusion')

    # ---- Figure 7: 诊断 tail/whole ratio 对比 ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(4); width = 0.35
    orig = [0.150, 0.065, 0.158, 0.379]
    recon = [0.152, 0.060, 0.150, 0.272]
    ax.bar(x - width/2, orig, width, label='Original reverb audio', color='steelblue')
    ax.bar(x + width/2, recon, width, label='Reconstructed (token→Flow→HifiGAN)', color='coral')
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel('tail/whole RMS ratio')
    ax.set_title('Diagnostic: tokens preserve ~70% of reverb signal\n'
                 'Large: original 0.379 → reconstructed 0.272 (~70%)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    save(fig, 'diagnostic_tail_ratio')

    # ---- Figure 8: epoch → wet ratio sigmoid 曲线(演示逻辑)----
    import math
    epochs = list(range(0, 26))
    wets = [1.0 / (1.0 + math.exp(-(e - 7) / 2.5)) for e in epochs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, wets, 'purple', linewidth=2.5, marker='o')
    for e in [0, 5, 7, 10, 14, 20]:
        w = 1.0 / (1.0 + math.exp(-(e - 7) / 2.5))
        ax.annotate(f'{w:.2f}', (e, w), textcoords='offset points', xytext=(5, -15))
    ax.set_xlabel('LLM checkpoint epoch')
    ax.set_ylabel('Neural Reverb wet/dry ratio')
    ax.set_title('Demo Dashboard: epoch-based wet mix curve\n'
                 '(sigmoid centered at E7, slope=2.5)')
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    save(fig, 'epoch_wet_curve')

    # ---- Figure 9: 实验最终指标对比 ----
    fig, ax = plt.subplots(figsize=(9, 4.5))
    phases = ['base\n(no training)', 'LLM SFT\n(best)', 'Flow Plan B\n(best)', 'Neural Reverb\n(isolated)', 'Full Pipeline\n(modular)']
    accs = [25, 25, 25, 100, 86]
    colors = ['gray', 'lightcoral', 'plum', 'mediumseagreen', 'forestgreen']
    bars = ax.bar(phases, accs, color=colors)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v}%',
                ha='center', fontweight='bold')
    ax.set_ylabel('Classifier-benchmark accuracy (%)')
    ax.set_title('Final Comparison: reverb-class accuracy across approaches')
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3, axis='y')
    save(fig, 'final_comparison')

    print('\n[done] 9 figures saved to', OUT_DIR)


if __name__ == '__main__':
    main()
