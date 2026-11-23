"""Shapes, finite values, context isolation, context noise, and a CPU smoke test of every experiment config."""
import glob
import os
import random

import pytest
import torch

from models.masks import mask_sampler
from models.modified_ijepa import Encoder, Predictor, add_context_noise, sample_ijepa_masks, train
from train import load_config

CONFIGS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "*.yaml")))


def test_shapes_and_finite():
    torch.manual_seed(0)
    enc, pred = Encoder(), Predictor(grid=8)
    imgs = torch.randn(8, 3, 32, 32)
    ctx, tgts = sample_ijepa_masks(8, 8, rng=random.Random(0))
    ci = torch.tensor(ctx)
    assert enc(imgs).shape == (8, 64, 128)
    ce = enc(imgs, ci)
    assert ce.shape == (8, ci.size(1), 128)
    for t in tgts:
        out = pred(ce, ci, torch.tensor(t))
        assert out.shape == (8, len(t[0]), 128) and torch.isfinite(out).all()


def test_context_encoder_never_sees_target_tokens():
    """Changing pixels inside target patches must not change the context encoding."""
    torch.manual_seed(0)
    enc = Encoder()
    imgs = torch.randn(4, 3, 32, 32); imgs2 = imgs.clone()
    ctx, tgts = sample_ijepa_masks(4, 8, rng=random.Random(0))
    for b in range(4):
        for p in set().union(*(t[b] for t in tgts)):
            r, c = divmod(p, 8)
            imgs2[b, :, r * 4:(r + 1) * 4, c * 4:(c + 1) * 4] = 99.0
    ci = torch.tensor(ctx)
    assert torch.allclose(enc(imgs, ci), enc(imgs2, ci))


def test_context_noise():
    x = torch.randn(256, 3, 32, 32)
    g = lambda: torch.Generator().manual_seed(1)
    y = add_context_noise(x, g(), p=0.5, sigma_max=0.2)
    assert torch.equal(y, add_context_noise(x, g(), p=0.5, sigma_max=0.2))      # seeded
    same = (y == x).flatten(1).all(1)
    assert 0.35 < same.float().mean() < 0.65                                     # ~half the images unchanged
    assert torch.equal(add_context_noise(x, g(), p=0.0), x)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.basename(p))
def test_config_smoke_cpu(path):
    """Every experiment config: 2 optimizer steps on CPU, finite loss, declared token counts."""
    cfg = load_config(path)
    ds = [torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(i)) for i in range(16)]
    out = train(ds, steps=2, batch_size=8, mask_fn=mask_sampler(cfg["mask"]), ctx_noise=cfg["ctx_noise"],
                num_workers=0, device="cpu")
    assert len(out["log"]) == 2 and all(torch.isfinite(torch.tensor(r["loss"])) for r in out["log"])
    if cfg["mask"].get("ctx_size") and cfg["mask"]["strategy"] != "overlap_frac":   # null: no budget
        assert out["log"][0]["ctx_tokens"] == cfg["mask"]["ctx_size"]
        assert out["log"][0]["tgt_tokens"] == cfg["mask"]["tgt_size"]
