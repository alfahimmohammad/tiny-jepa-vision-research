"""Every mask strategy: valid indices, context/target disjoint, declared counts, determinism, strategy rules."""
import glob
import os
import random

import pytest
import yaml

from models import masks
from models.modified_ijepa import sample_ijepa_masks

GRID = 8
MASK_FILES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "masks", "*.yaml")))
NAMES = [os.path.basename(f)[:-5] for f in MASK_FILES]


def _cfg(name):
    return yaml.safe_load(open(MASK_FILES[NAMES.index(name)]))


@pytest.mark.parametrize("name", NAMES)
def test_valid_disjoint_and_counts(name):
    cfg, fn = _cfg(name), masks.mask_sampler(_cfg(name))
    rng = random.Random(0)
    for B in (1, 16, 64):
        for _ in range(5):
            ctx, tgts = fn(B, GRID, rng)
            masks.check_masks(ctx, tgts, GRID)                     # raises on invalid index or leak
            assert all(c == sorted(set(c)) for c in ctx)
            if cfg.get("tgt_size"):                                # fixed counts per image, independent of B
                sizes = [len(t[0]) for t in tgts]
                if not cfg.get("per_block"):
                    assert sizes == [cfg["tgt_size"]]              # one set drawn from the union
                elif cfg["strategy"] == "scattered":               # one set per whole small block
                    assert len(set(sizes)) == 1 and cfg["tgt_size"] <= sum(sizes) < cfg["tgt_size"] + sizes[0]
                else:                                              # one set per block, tgt_size // n each
                    n = cfg.get("n_targets", 4)
                    assert sizes == [cfg["tgt_size"] // n] * n
                if cfg["strategy"] == "overlap_frac":
                    assert 1 <= len(ctx[0]) <= cfg["ctx_size"]     # batch minimum of up to ctx_size
                else:
                    assert len(ctx[0]) == cfg["ctx_size"]


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(name):
    fn = masks.mask_sampler(_cfg(name))
    assert fn(32, GRID, random.Random(7)) == fn(32, GRID, random.Random(7))


def test_upstream_strategy_is_the_original_sampler():
    fn = masks.mask_sampler(_cfg("upstream"))
    for s in range(5):
        ctx, tgts = fn(64, GRID, random.Random(s))
        assert (ctx, tgts) == sample_ijepa_masks(64, GRID, rng=random.Random(s))


def test_overlap_frac_cap():
    for th, tw in ((3, 3), (3, 4), (4, 3)):
        valid = masks._ovfrac_valid(GRID, th, tw, 4, 32, 0.25, 12)
        combos, _ = masks._combo_unions(GRID, th, tw, 4)
        allb = masks._block_bits(GRID, th, tw)
        for j in valid[:: max(1, len(valid) // 200)]:
            bl = [allb[k] for k in combos[j]]
            assert all((a & b).bit_count() / (th * tw) < 0.25 for i, a in enumerate(bl) for b in bl[i + 1:])


def test_scattered_blocks_do_not_overlap():
    for _ in range(50):
        placed = masks._place_blocks(GRID, random.Random(_), 2, 2, 0, False, 32)
        if placed is not None:
            u, blocks = placed
            assert u.bit_count() >= 32 and sum(b.bit_count() for b in blocks) == u.bit_count()   # disjoint


@pytest.mark.parametrize("name", ["1_large", "2_large_overlap", "2_large_nonoverlap", "overlap_frac", "scattered"])
def test_per_block_sets_lie_in_one_block(name):
    """per_block=True: every target set of an image is inside one block of the target-block geometry, i.e. a
    contiguous th x tw rectangle (scattered: the whole block)."""
    cfg = _cfg(name)
    max_area = int(1.25 * cfg["tgt_scale"][1] * GRID * GRID) + 2      # largest block _bsize can return, with slack
    ctx, tgts = masks.mask_sampler(cfg)(32, GRID, random.Random(1))
    for b in range(32):
        for t in tgts:
            rows, cols = [p // GRID for p in t[b]], [p % GRID for p in t[b]]
            assert (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1) <= max_area


def test_per_block_false_is_one_union_set():
    for name in ("1_large", "2_large_overlap", "overlap_frac", "scattered"):
        cfg = {**_cfg(name), "per_block": False}
        ctx, tgts = masks.mask_sampler(cfg)(16, GRID, random.Random(0))
        assert len(tgts) == 1 and len(tgts[0][0]) == cfg["tgt_size"]


def test_scattered_without_budgets():
    """scattered_var5: exactly 5 disjoint whole blocks per image (one set each, same block size within the batch),
    context = context block minus targets trimmed to the batch minimum, both counts varying between batches."""
    cfg = _cfg("scattered_var5")
    fn, rng, ctx_lens, tgt_lens = masks.mask_sampler(cfg), random.Random(0), set(), set()
    for _ in range(40):
        ctx, tgts = fn(64, GRID, rng)
        masks.check_masks(ctx, tgts, GRID)
        assert len(tgts) == cfg["n_targets"]
        assert len({len(t[0]) for t in tgts}) == 1                           # one block size per batch
        for b in range(64):
            blocks = [set(t[b]) for t in tgts]
            assert sum(map(len, blocks)) == len(set().union(*blocks))      # disjoint
            for t in blocks:                                               # each set is a whole rectangle
                r, c = [p // GRID for p in t], [p % GRID for p in t]
                assert (max(r) - min(r) + 1) * (max(c) - min(c) + 1) == len(t)
        assert len(ctx[0]) >= cfg["min_ctx"]
        ctx_lens.add(len(ctx[0])); tgt_lens.add(sum(len(t[0]) for t in tgts))
    assert len(ctx_lens) > 1 and len(tgt_lens) > 1                         # counts are not fixed
    with pytest.raises(ValueError):
        masks.mask_sampler({**cfg, "per_block": False})(4, GRID, random.Random(0))


@pytest.mark.parametrize("name", NAMES)
def test_every_strategy_predicts_one_set_per_block(name):
    """Default for every experiment: the loss is computed per target block (per_block true or absent)."""
    assert _cfg(name).get("per_block", True) is True


def test_leak_is_detected():
    with pytest.raises(masks.MaskLeakError):
        masks.check_masks([[0, 1]], [[[1, 2]]], GRID)
