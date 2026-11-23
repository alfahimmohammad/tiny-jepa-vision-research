"""Figure 2: two sampled masks per strategy on the 8x8 patch grid, with every target block outlined.

Uses the repository samplers (models/masks.py, configs/masks/*.yaml) on a batch of 64 and shows image 0, so batch-level
rules (upstream's batch-minimum context, the overlap cap's batch minimum) apply as in training. The target blocks are
recorded while sampling, without changing the samplers: the fixed-count samplers pass each image's blocks to
masks._targets, and upstream builds its blocks with _block(grid, top, left, h, w), image 0 first.
"""
import os, random, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from models import masks, modified_ijepa  # noqa: E402

G, B = 8, 64
BLUE, ORANGE, LIGHT, GREY = (np.array([42, 120, 214]) / 255, np.array([235, 104, 52]) / 255,
                             np.array([248, 205, 182]) / 255, np.array([0.90] * 3))
EDGE = "#1f1f1f"
rec, tcalls = [], []                                        # _block calls / per-image block lists of the current draw


def _bits_rect(m):
    idx = masks._bits_to_idx(m); r = [i // G for i in idx]; c = [i % G for i in idx]
    return min(r), min(c), max(r) - min(r) + 1, max(c) - min(c) + 1


_targets = masks._targets
def targets_rec(rng, union, blocks, *a, **k):              # fixed-count samplers: every block of the image
    tcalls.append([_bits_rect(t) for t in blocks])
    return _targets(rng, union, blocks, *a, **k)
masks._targets = targets_rec

_block = modified_ijepa._block
def block_rec(g, top, left, h, w):                          # upstream: target blocks are built with _block
    rec.append((g, top, left, h, w)); return _block(g, top, left, h, w)
modified_ijepa._block = block_rec; masks._block = block_rec


def draw(name, seed):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "masks", f"{name}.yaml")))
    args = {k: tuple(map(tuple, v)) if k == "block_sizes" else tuple(v) if isinstance(v, list) else v
            for k, v in cfg.items() if k != "strategy"}
    rec.clear(); tcalls.clear()
    out = masks.STRATEGIES[cfg["strategy"]](B, G, random.Random(seed), **args)
    ctx, tgts = out[0], out[1]
    masks.check_masks(ctx, tgts, G)
    st = cfg["strategy"]
    if st == "upstream":                                    # the first n_targets _block calls are the target blocks
        blocks = [r[1:] for r in rec if r[0] == G][:cfg.get("n_targets", 4)]
    else:                                                   # the last B calls are the returned batch, in image order
        blocks = tcalls[-B]
    return ctx[0], [t[0] for t in tgts], blocks, st


def panel(ax, name, seed, title):
    ctx, sets, blocks, st = draw(name, seed)
    img = np.tile(GREY, (G * G, 1))
    for t, l, h, w in blocks:                               # block patches that are not drawn as targets
        for r in range(t, t + h):
            for c in range(l, l + w):
                img[r * G + c] = LIGHT
    for s in sets:
        for p in s:
            img[p] = ORANGE
    for p in ctx:
        img[p] = BLUE
    ax.imshow(img.reshape(G, G, 3)); ax.set_xticks([]); ax.set_yticks([])
    for k in range(G + 1):
        ax.axhline(k - .5, color="white", lw=.6); ax.axvline(k - .5, color="white", lw=.6)
    for j, (t, l, h, w) in enumerate(blocks):               # outline each target block (inset so overlaps stay visible)
        d = 0.08 + 0.07 * (j % 3)
        ax.add_patch(Rectangle((l - .5 + d, t - .5 + d), w - 2 * d, h - 2 * d, fill=False, ec=EDGE, lw=1.0))
    ax.set_title(title, fontsize=7)
    ax.set_xlabel(f"{len(blocks)} block{'s' * (len(blocks) > 1)} | ctx {len(ctx)} | tgt {sum(len(s) for s in sets)}", fontsize=6)


ROWS = [("upstream", "baseline (upstream)"), ("1_large", "1_large (proposed)"), ("scattered", "scattered (ablation)"),
        ("2_large_overlap", "2_large_overlap"), ("2_large_nonoverlap", "2_large_nonoverlap"), ("overlap_frac", "overlap cap <25%")]
fig, axs = plt.subplots(2, len(ROWS), figsize=(7.2, 3.05))
for j, (name, title) in enumerate(ROWS):
    for i, seed in enumerate((3, 11)):
        panel(axs[i, j], name, seed, title if i == 0 else "")
plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=1.0)
OUT = os.path.join(ROOT, "report", "figures")
plt.savefig(f"{OUT}/fig_masks.pdf"); plt.savefig(f"{OUT}/fig_masks.png", dpi=150)
print("ok")
