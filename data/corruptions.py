"""Deterministic stress tests on the held-out test images, applied in [0, 1] pixel space BEFORE
CIFAR-10 normalisation. Every corruption depends only on its own fixed seed and the number of
images, never on the model, so all model variants see byte-identical inputs (paired comparison).

Occlusion: one axis-aligned SQUARE per image, side = round(sqrt(frac * 32 * 32)) pixels
           (10% -> 10 px = 9.8% area, 25% -> 16 px = 25.0%, 40% -> 20 px = 39.1%).
           Top-left corner uniform over all positions that keep the square fully inside the image.
           Fill value: the CIFAR-10 per-channel mean (0.4914, 0.4822, 0.4465), which becomes exactly
           0 after normalisation, i.e. "no signal" rather than an artificial black/white edge.
Noise:     x + N(0, sigma^2) independently per pixel and channel, sigma in {0.05, 0.15},
           then clipped to [0, 1].
"""
import hashlib
import math

import numpy as np
import torch

CORRUPTION_SEED = 1000
OCCLUSION_FRACS = (0.10, 0.25, 0.40)
NOISE_STDS = (0.05, 0.15)
IMG = 32


def occlusion_boxes(n, frac, seed, size=IMG):
    side = int(round(math.sqrt(frac * size * size)))
    g = np.random.default_rng(seed)
    tops = g.integers(0, size - side + 1, n)
    lefts = g.integers(0, size - side + 1, n)
    return np.stack([tops, lefts, np.full(n, side), np.full(n, side)], 1)


def apply_occlusion(x01, boxes, fill):
    out = x01.clone()
    f = torch.tensor(fill, dtype=out.dtype).view(3, 1, 1)
    for i, (t, l, h, w) in enumerate(boxes.tolist()):
        out[i, :, t:t + h, l:l + w] = f
    return out


def apply_noise(x01, std, seed):
    g = torch.Generator().manual_seed(seed)
    return (x01 + torch.randn(x01.shape, generator=g) * std).clamp(0.0, 1.0)


def build_conditions(x01, fill, seed_offset=0):
    """Return ({name: tensor in [0,1]}, {name: metadata}) for clean + 3 occlusion + 2 noise levels.
    seed_offset=0 is the held-out TEST protocol; a different offset gives independent corruptions
    (used only for the diagnostic oracle probe on probe-train images)."""
    conds, meta = {"clean": x01}, {"clean": {}}
    for k, frac in enumerate(OCCLUSION_FRACS):
        seed = CORRUPTION_SEED + seed_offset + k
        boxes = occlusion_boxes(len(x01), frac, seed)
        name = f"occ{int(frac * 100)}"
        conds[name] = apply_occlusion(x01, boxes, fill)
        side = int(boxes[0, 2])
        meta[name] = {"seed": seed, "side_px": side, "area_frac": side * side / IMG ** 2,
                      "fill": list(fill)}
    for k, std in enumerate(NOISE_STDS):
        seed = CORRUPTION_SEED + seed_offset + 10 + k
        name = f"noise{std:.2f}"
        conds[name] = apply_noise(x01, std, seed)
        meta[name] = {"seed": seed, "std": std}
    for name, x in conds.items():
        meta[name]["sha256"] = tensor_hash(x)
    return conds, meta


def tensor_hash(x):
    return hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
