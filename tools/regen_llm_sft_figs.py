"""Regenerate clean LLM SFT figures using full per-epoch data from logs.
Filters: only LLM CV lines (which contain `acc`), drops Flow lines (no acc).
Deduplicates: keeps the first occurrence per epoch (some logs contain
multiple training runs concatenated).
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt

LOGS = Path('/home/exouser/CosyVoice-env-tts/logs')
OUT = Path('/home/exouser/CosyVoice-env-tts/docs/figures')
OUT.mkdir(parents=True, exist_ok=True)

# Match LLM CV lines: must contain `acc` (filters out Flow which only logs loss).
PAT = re.compile(
    r'Epoch\s+(\d+)\s+Step\s+\d+\s+CV info.*?loss\s+([\d.eE+\-]+)\s+acc\s+([\d.eE+\-]+)'
)


def parse(path, max_epoch=None):
    """Return sorted [(epoch, loss, acc)], deduping by epoch (first wins)."""
    seen = {}
    for line in Path(path).read_text(errors='ignore').splitlines():
        m = PAT.search(line)
        if not m:
            continue
        e, loss, acc = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if max_epoch is not None and e > max_epoch:
            continue
        if e not in seen:
            seen[e] = (loss, acc)
    return [(e, *seen[e]) for e in sorted(seen)]


def unzip3(triples):
    if not triples:
        return [], [], []
    es, ls, accs = zip(*triples)
    return list(es), list(ls), list(accs)


def save(fig, name):
    p = OUT / f'{name}.png'
    fig.savefig(str(p), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('saved', p)


# Parse all three runs (caps from the journey md)
v1 = parse(LOGS / 'train_both_20260416_003538.log', max_epoch=14)
v2 = parse(LOGS / 'train_both_canonical_20260417_005725.log', max_epoch=9)
v3 = parse(LOGS / 'train_llm_strong_20260418_005710.log', max_epoch=24)

print(f'v1: {len(v1)} epochs, range loss [{min(t[1] for t in v1):.3f}, {max(t[1] for t in v1):.3f}]')
print(f'v2: {len(v2)} epochs, range loss [{min(t[1] for t in v2):.3f}, {max(t[1] for t in v2):.3f}]')
print(f'v3: {len(v3)} epochs, range loss [{min(t[1] for t in v3):.3f}, {max(t[1] for t in v3):.3f}]')

e1, l1, _ = unzip3(v1)
e2, l2, _ = unzip3(v2)
e3, l3, _ = unzip3(v3)

# Fig 1: three-attempt comparison
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(e1, l1, 'r-o', label='v1: lr=1e-5 (no canonical) — collapses', linewidth=2, markersize=5)
ax.plot(e2, l2, 'g-s', label='v2: lr=1e-6 (canonical) — healthy but weak', linewidth=2, markersize=5)
ax.plot(e3, l3, 'b-^', label='v3: lr=5e-6 — rebounds at E5', linewidth=2, markersize=5)
ax.set_xlabel('Epoch')
ax.set_ylabel('CV loss')
ax.set_title('LLM SFT: Three attempts (lower = better)')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
save(fig, 'llm_sft_comparison')

# Fig 2: v2 detail (zoomed y)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(e2, l2, 'g-o', linewidth=2, markersize=7, label='CV loss')
best_i = min(range(len(l2)), key=lambda i: l2[i])
ax.axvline(e2[best_i], color='red', linestyle='--', alpha=0.6,
           label=f'best at E{e2[best_i]} (loss={l2[best_i]:.3f})')
ax.set_xlabel('Epoch')
ax.set_ylabel('CV loss')
ax.set_title('LLM SFT v2 (canonical, lr=1e-6): healthy but weak convergence\n'
             f'CV loss {l2[0]:.3f} → {min(l2):.3f} (Δ={l2[0]-min(l2):.3f}), no audible reverb')
ax.grid(alpha=0.3)
ax.legend()
save(fig, 'llm_sft_v2_detail')

# Fig 3: v3 U-shape rebound (with inset showing the shallow dip)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(e3, l3, 'b-o', linewidth=2, markersize=5)
best_i = min(range(len(l3)), key=lambda i: l3[i])
ax.axvline(e3[best_i], color='red', linestyle='--', alpha=0.6,
           label=f'best at E{e3[best_i]} (loss={l3[best_i]:.3f})')
ax.set_xlabel('Epoch')
ax.set_ylabel('CV loss')
ax.set_title('LLM SFT v3 (lr=5e-6): U-shape — best at E4 then overfits')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

# Inset zoom on the shallow U (E0-E10)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
zoom_n = 11  # E0..E10
axin = inset_axes(ax, width='40%', height='40%', loc='center right',
                  bbox_to_anchor=(-0.05, 0.15, 1, 1), bbox_transform=ax.transAxes)
axin.plot(e3[:zoom_n], l3[:zoom_n], 'b-o', linewidth=2, markersize=5)
axin.axvline(e3[best_i], color='red', linestyle='--', alpha=0.6)
axin.set_title('Zoom: E0–E10 (the U)', fontsize=9)
axin.grid(alpha=0.3)
axin.set_ylim(min(l3[:zoom_n]) - 0.02, max(l3[:zoom_n]) + 0.02)
axin.tick_params(labelsize=8)
save(fig, 'llm_sft_v3_rebound')

print('\n[done] 3 LLM SFT figures regenerated with per-epoch data.')
