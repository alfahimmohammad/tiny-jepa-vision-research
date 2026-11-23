"""Mask strategies for Tiny I-JEPA (H1: masking geometry).

Every strategy has the signature  fn(B, grid, rng, **args) -> (ctx_list, tgt_lists)  like the upstream
`sample_ijepa_masks`:
  ctx_list[b]     sorted visible patch indices of image b (same length for every image in the batch)
  tgt_lists[m][b] sorted target patch indices of target set m for image b (same length within a set)
`args` come from a YAML file in configs/masks/ (one per strategy variant). `mask_sampler(cfg)` turns such a
config into the `mask_fn(B, grid, rng)` that `models.modified_ijepa.train` calls once per step.

Strategies
  upstream     the original sampler: 4 blocks, context = context block minus targets, trimmed to the batch minimum
  blocks       baseline-style blocks with FIXED counts (ctx_size visible, tgt_size target tokens per image);
               1_large (proposed), 2_large_overlap, 2_large_nonoverlap
  scattered    (ablation) many small non-overlapping blocks until tgt_size patches are covered (same fixed counts);
               scattered_var5: exactly 5 blocks, no token budgets, context trimmed to the batch minimum
  overlap_frac 4 upstream-size blocks with |A n B| / |A| < max_overlap_frac for every pair (exact enumeration)
Every strategy predicts one target set per block (per_block: true, the default, as upstream I-JEPA); the loss is the
mean over the sets. per_block: false (one set from the union of the blocks) exists only to reproduce old runs.
"""
import functools
import itertools

import numpy as np

from models.modified_ijepa import _block, _bsize, sample_ijepa_masks


# ---------- helpers (patch sets as 64-bit masks: bit i = patch i) ----------

def _take(rng, idx, k):
    """Unbiased random subset of k patch indices, sorted."""
    return sorted(rng.sample(idx, k))


@functools.lru_cache(maxsize=None)
def _block_bits(grid, th, tw):
    """Bit masks of every th x tw block position, row-major over (top, left)."""
    return [sum(1 << i for i in _block(grid, r, c, th, tw))
            for r in range(grid - th + 1) for c in range(grid - tw + 1)]


def _bits_to_idx(m):
    return [i for i in range(m.bit_length()) if m >> i & 1]


def _popcount64(a):
    return np.bitwise_count(a).astype(np.int64)


@functools.lru_cache(maxsize=None)
def _combo_unions(grid, th, tw, n):
    """Every set of n distinct block positions and the bit mask of its union (C(36, 4) = 58,905 at most)."""
    allb = np.array(_block_bits(grid, th, tw), dtype=np.uint64)
    combos = np.array(list(itertools.combinations(range(len(allb)), n)), dtype=np.int16)
    u = np.zeros(len(combos), dtype=np.uint64)
    for j in range(n):
        u |= allb[combos[:, j]]
    return combos, u


# ---------- upstream ----------

def upstream(B, grid, rng, n_targets=4, min_ctx=4):
    return sample_ijepa_masks(B, grid, n_targets=n_targets, min_ctx=min_ctx, rng=rng)


# ---------- fixed-count blocks (1_large, 2_large_*) and scattered ----------

def _place_blocks(grid, rng, th, tw, n, overlap, need):
    """Place n th x tw blocks (n=0: keep adding blocks until the union has >= need patches). With
    overlap=False each block is placed uniformly among the positions that do not touch earlier blocks.
    Returns (union bit mask, list of block bit masks), or None if no free position is left."""
    allb, union, k, placed = _block_bits(grid, th, tw), 0, 0, []
    while (k < n) if n else (union.bit_count() < need):
        if overlap:
            t = allb[rng.randint(0, grid - th) * (grid - tw + 1) + rng.randint(0, grid - tw)]
        else:
            free = [t for t in allb if not (t & union)]
            if not free:
                return None
            t = rng.choice(free)
        union |= t; k += 1; placed.append(t)
    return union, placed


def _targets(rng, union, blocks, tgt_size, per_block, whole=False):
    """Target sets of one image. per_block=False: one set, tgt_size patches drawn from the union of the blocks.
    per_block=True (as upstream I-JEPA): one set per block, tgt_size // n_blocks patches drawn from that block,
    so a patch covered by two blocks can be a target twice. whole=True (scattered: disjoint small blocks, as
    many as needed to cover tgt_size patches) makes each set the whole block."""
    if not per_block:
        return [_take(rng, _bits_to_idx(union), tgt_size)]
    if whole:
        return [_bits_to_idx(t) for t in blocks]
    k = tgt_size // len(blocks)
    return [_take(rng, _bits_to_idx(t), k) for t in blocks]


def blocks(B, grid, rng, ctx_size=20, tgt_size=32, n_targets=4, tgt_scale=(0.15, 0.20),
           tgt_aspect=(0.75, 1.5), ctx_scale=(0.85, 1.0), overlap=True, per_block=True, max_draw_tries=200,
           max_geom_tries=50):
    """Upstream block geometry (one target-block size and one context-block size per batch, random positions
    per image) with FIXED token counts: a draw that cannot supply ctx_size context and tgt_size target patches is
    redrawn, then ctx_size context indices are drawn uniformly. Targets (see _targets): per_block=True gives one
    set per block (n predictor calls with tgt_size // n mask tokens each; scattered: one set per small block, all
    its patches), per_block=False one set of tgt_size patches from the union (one predictor call)."""
    if ctx_size + tgt_size > grid * grid:
        raise ValueError(f"ctx_size + tgt_size = {ctx_size + tgt_size} > {grid * grid} patches")
    n = n_targets
    for _ in range(max_geom_tries):
        th, tw = _bsize(grid, rng.uniform(*tgt_scale), rng.uniform(*tgt_aspect))
        ch, cw = _bsize(grid, rng.uniform(*ctx_scale), 1.0)
        if (n and n * th * tw < tgt_size) or ch * cw < ctx_size:     # these sizes can never supply the counts
            continue
        ctx_blocks = _block_bits(grid, ch, cw)
        ctx_list, tgt_sets = [], []
        for b in range(B):
            for _ in range(max_draw_tries):
                placed = _place_blocks(grid, rng, th, tw, n, overlap, tgt_size)
                if placed is None or placed[0].bit_count() < tgt_size:
                    continue
                union, blks = placed
                c = rng.choice(ctx_blocks) & ~union
                if c.bit_count() < ctx_size:
                    continue
                ctx_list.append(_take(rng, _bits_to_idx(c), ctx_size))
                tgt_sets.append(_targets(rng, union, blks, tgt_size, per_block, whole=not n))
                break
            else:
                break                                                 # redraw the batch's block sizes
        if len(ctx_list) == B:
            return ctx_list, [list(t) for t in zip(*tgt_sets)]
    raise RuntimeError(f"could not draw ctx_size={ctx_size}, tgt_size={tgt_size} with tgt_scale={tgt_scale}")


def scattered(B, grid, rng, tgt_scale=(0.04, 0.07), n_targets=0, ctx_size=20, tgt_size=32, **args):
    """H1 ablation: many SMALL non-overlapping blocks (2-6 patches) until tgt_size patches are covered.
    n_targets > 0 places exactly that many blocks instead. ctx_size = tgt_size = None: no token budgets
    (see _scattered_variable)."""
    if ctx_size is None and tgt_size is None:
        return _scattered_variable(B, grid, rng, n_targets, tgt_scale, **args)
    return blocks(B, grid, rng, ctx_size=ctx_size, tgt_size=tgt_size, n_targets=n_targets, overlap=False,
                  tgt_scale=tgt_scale, **args)


def _scattered_variable(B, grid, rng, n_targets, tgt_scale, tgt_aspect=(0.75, 1.5), ctx_scale=(0.85, 1.0),
                        min_ctx=4, per_block=True, max_draw_tries=200, max_geom_tries=50):
    """Scattered blocks WITHOUT token budgets, upstream-style: one small-block size and one context-block size per
    batch; per image exactly n_targets non-overlapping blocks at random positions, each predicted whole as its
    own target set (so the target count n_targets * block area varies from batch to batch); the context is the
    context block minus all target patches (redrawn if < min_ctx), then randomly trimmed to the batch minimum as in
    sample_ijepa_masks."""
    if not n_targets:
        raise ValueError("scattered without token budgets needs n_targets > 0")
    if not per_block:
        raise ValueError("scattered without token budgets predicts one target set per block (per_block: true)")
    for _ in range(max_geom_tries):
        th, tw = _bsize(grid, rng.uniform(*tgt_scale), rng.uniform(*tgt_aspect))
        ctx_blocks = _block_bits(grid, *_bsize(grid, rng.uniform(*ctx_scale), 1.0))
        ctx_list, tgt_sets = [], []
        for b in range(B):
            for _ in range(max_draw_tries):
                placed = _place_blocks(grid, rng, th, tw, n_targets, False, 0)
                if placed is None:
                    continue
                union, blks = placed
                c = rng.choice(ctx_blocks) & ~union
                if c.bit_count() < min_ctx:
                    continue
                ctx_list.append(_bits_to_idx(c)); tgt_sets.append([_bits_to_idx(t) for t in blks])
                break
            else:
                break                                                 # redraw the batch's block sizes
        if len(ctx_list) == B:
            L = min(len(c) for c in ctx_list)                         # batch-minimum trim (upstream)
            return [_take(rng, c, L) for c in ctx_list], [list(t) for t in zip(*tgt_sets)]
    raise RuntimeError(f"could not place {n_targets} blocks with tgt_scale={tgt_scale}")


# ---------- overlap_frac ----------

@functools.lru_cache(maxsize=None)
def _ovfrac_valid(grid, th, tw, n, tgt_size, max_frac, min_ctx):
    """Block sets whose every pair has |A n B| / |A| < max_frac, whose union has >= tgt_size patches and that
    leave >= min_ctx patches on the full grid. None if no set qualifies."""
    combos, u = _combo_unions(grid, th, tw, n)
    allb = np.array(_block_bits(grid, th, tw), dtype=np.uint64)
    inter = _popcount64(allb[:, None] & allb[None, :])
    frac = np.zeros(len(combos))
    for a, b in itertools.combinations(range(n), 2):
        frac = np.maximum(frac, inter[combos[:, a], combos[:, b]] / (th * tw))
    size = _popcount64(u)
    ok = (frac < max_frac) & (size >= tgt_size) & (grid * grid - size >= min_ctx)
    return np.flatnonzero(ok) if ok.any() else None


def overlap_frac(B, grid, rng, ctx_size=20, tgt_size=32, n_targets=4, tgt_scale=(0.15, 0.20),
                 tgt_aspect=(0.75, 1.5), ctx_scale=(0.85, 1.0), max_overlap_frac=0.25, full_ctx_below=16,
                 min_ctx=12, per_block=True, max_geom_tries=50):
    """4 upstream-size blocks (fixed target count) with a pairwise overlap cap. Per image: a uniform context-block position and a uniform
    valid block set. Context = context block minus the targets (the full grid if fewer than full_ctx_below patches
    are left); each image offers up to ctx_size patches and the batch uses the batch minimum."""
    n, full = n_targets, (1 << grid * grid) - 1
    for _ in range(max_geom_tries):
        th, tw = _bsize(grid, rng.uniform(*tgt_scale), rng.uniform(*tgt_aspect))
        ch, cw = _bsize(grid, rng.uniform(*ctx_scale), 1.0)
        valid = _ovfrac_valid(grid, th, tw, n, tgt_size, max_overlap_frac, min_ctx)
        if valid is None:
            continue
        combos, unions = _combo_unions(grid, th, tw, n)
        allb = _block_bits(grid, th, tw)
        ctx_blocks = _block_bits(grid, ch, cw)
        avail, tgt_sets = [], []
        for b in range(B):
            cb = rng.choice(ctx_blocks)
            j = valid[rng.randrange(len(valid))]
            union = int(unions[j])
            if (cb & ~union).bit_count() < full_ctx_below:
                cb = full
            avail.append(_bits_to_idx(cb & ~union))
            tgt_sets.append(_targets(rng, union, [allb[k] for k in combos[j]], tgt_size, per_block))
        L = min(ctx_size, *map(len, avail))                          # batch minimum
        return [_take(rng, a, L) for a in avail], [list(t) for t in zip(*tgt_sets)]
    raise RuntimeError(f"no {n}-block layout with pairwise overlap < {max_overlap_frac}")


STRATEGIES = {"upstream": upstream, "blocks": blocks, "scattered": scattered, "overlap_frac": overlap_frac}


# ---------- checks and the sampler used by training ----------

class MaskLeakError(RuntimeError):
    pass


def check_masks(ctx_list, tgt_lists, grid):
    """Raise if an index is invalid, lengths are ragged, or a context patch is also a target patch."""
    n, L = grid * grid, len(ctx_list[0])
    for t in tgt_lists:
        if len({len(x) for x in t}) != 1:
            raise ValueError("target sets must have equal length within the batch")
    for b, c in enumerate(ctx_list):
        if len(c) != L or len(set(c)) != L or not all(0 <= i < n for i in c):
            raise ValueError(f"bad context indices for item {b}")
        union = set()
        for t in tgt_lists:
            if not t[b] or not all(0 <= i < n for i in t[b]):
                raise ValueError(f"bad target indices for item {b}")
            union.update(t[b])
        if union & set(c):
            raise MaskLeakError(f"context/target overlap for item {b}")


def sample(B, grid, rng, strategy="upstream", max_tries=20, **args):
    """Draw masks with the given strategy; a leaking draw (possible only on tiny grids) is redrawn."""
    for _ in range(max_tries):
        ctx, tgts = STRATEGIES[strategy](B, grid, rng, **args)
        try:
            check_masks(ctx, tgts, grid)
            return ctx, tgts
        except MaskLeakError:
            continue
    raise MaskLeakError("could not sample leak-free masks")


def mask_sampler(cfg):
    """cfg = a configs/masks/*.yaml dict -> mask_fn(B, grid, rng) for models.modified_ijepa.train."""
    args = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}
    return lambda B, grid, rng: sample(B, grid, rng, **args)
