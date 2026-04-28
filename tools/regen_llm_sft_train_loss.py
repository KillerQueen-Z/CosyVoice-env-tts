"""Plot per-epoch TRAIN loss for LLM SFT v1/v2/v3.

Each TRAIN Batch line: `TRAIN Batch <epoch>/<step> loss <X> acc <Y> lr <Z> grad_norm <W>`.
We require the `acc` field so Flow training (no acc) is filtered out.
For each epoch, take the mean of all batch losses within that epoch.
"""
import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

LOGS = Path('/home/exouser/CosyVoice-env-tts/logs')
OUT = Path('/home/exouser/CosyVoice-env-tts/docs/figures')
OUT.mkdir(parents=True, exist_ok=True)

PAT = re.compile(
    r'TRAIN Batch\s+(\d+)/\d+\s+loss\s+([\d.eE+\-]+)\s+acc\s+([\d.eE+\-]+)'
)


def parse_train(path, max_epoch=None):
    """Return sorted [(epoch, mean_loss)]; mean over batches within epoch."""
    by_epoch = defaultdict(list)
    for line in Path(path).read_text(errors='ignore').splitlines():
        m = PAT.search(line)
        if not m:
            continue
        e, loss = int(m.group(1)), float(m.group(2))
        if max_epoch is not None and e > max_epoch:
            continue
        by_epoch[e].append(loss)
    items = sorted(by_epoch.items())
    return [(e, sum(ls) / len(ls)) for e, ls in items]


def unzip(pts):
    if not pts:
        return [], []
    es, ls = zip(*pts)
    return list(es), list(ls)


def save(fig, name):
    p = OUT / f'{name}.png'
    fig.savefig(str(p), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('saved', p)


v1 = parse_train(LOGS / 'train_both_20260416_003538.log', max_epoch=14)
v2 = parse_train(LOGS / 'train_both_canonical_20260417_005725.log', max_epoch=9)
v3 = parse_train(LOGS / 'train_llm_strong_20260418_005710.log', max_epoch=24)

print(f'v1: {len(v1)} epochs, train loss [{min(t[1] for t in v1):.3f}, {max(t[1] for t in v1):.3f}]')
print(f'v2: {len(v2)} epochs, train loss [{min(t[1] for t in v2):.3f}, {max(t[1] for t in v2):.3f}]')
print(f'v3: {len(v3)} epochs, train loss [{min(t[1] for t in v3):.3f}, {max(t[1] for t in v3):.3f}]')

e1, l1 = unzip(v1)
e2, l2 = unzip(v2)
e3, l3 = unzip(v3)

# Three-attempt train-loss comparison
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(e1, l1, 'r-o', label='v1: lr=1e-5 (no canonical)', linewidth=2, markersize=5)
ax.plot(e2, l2, 'g-s', label='v2: lr=1e-6 (canonical)', linewidth=2, markersize=5)
ax.plot(e3, l3, 'b-^', label='v3: lr=5e-6', linewidth=2, markersize=5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Train loss (per-epoch mean)')
ax.set_title('LLM SFT: Train loss across three attempts (lower = better)')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
save(fig, 'llm_sft_train_loss_comparison')

# Log-scale variant (helps see v1's catastrophic memorization)
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(e1, l1, 'r-o', label='v1: lr=1e-5 (no canonical)', linewidth=2, markersize=5)
ax.semilogy(e2, l2, 'g-s', label='v2: lr=1e-6 (canonical)', linewidth=2, markersize=5)
ax.semilogy(e3, l3, 'b-^', label='v3: lr=5e-6', linewidth=2, markersize=5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Train loss (log scale)')
ax.set_title('LLM SFT train loss — log scale (v1 collapses train loss to ~1e-3)')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3, which='both')
save(fig, 'llm_sft_train_loss_logscale')

print('\n[done] 2 train-loss figures saved.')
