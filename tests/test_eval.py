"""Splits, corruptions and the frozen-encoder probe."""
import numpy as np
import torch

from data.corruptions import apply_noise, build_conditions, occlusion_boxes
from data.splits import SIZES, make_splits
from evaluate import extract_features, param_hash, select_probe, standardizer
from models.modified_ijepa import MEAN, Encoder


def _fake_labels(n, seed):
    return np.random.default_rng(seed).integers(0, 10, n)


def test_splits_disjoint_sized_deterministic():
    ytr, yte = _fake_labels(50_000, 0), _fake_labels(10_000, 1)
    s1, s2 = make_splits(ytr, yte), make_splits(ytr, yte)
    assert s1["fingerprint"] == s2["fingerprint"] and s1["indices"] == s2["indices"]
    ind = s1["indices"]
    for k, n in SIZES.items():
        assert len(ind[k]) == n == len(set(ind[k]))
    tr = [set(ind[k]) for k in ("pretrain", "probe_train", "val")]
    assert not (tr[0] & tr[1]) and not (tr[0] & tr[2]) and not (tr[1] & tr[2])
    for k in ("probe_train", "val"):
        assert np.all(np.bincount(ytr[ind[k]], minlength=10) == SIZES[k] // 10)
    assert np.all(np.bincount(yte[ind["test"]], minlength=10) == 200)


def test_corruptions_are_paired_and_valid():
    x = torch.rand(50, 3, 32, 32)
    c1, m1 = build_conditions(x, fill=MEAN)
    c2, m2 = build_conditions(x.clone(), fill=MEAN)
    assert m1 == m2                                            # identical bytes on every call
    for k in c1:
        assert torch.equal(c1[k], c2[k]) and c1[k].min() >= 0 and c1[k].max() <= 1
    for name, target in (("occ10", 0.10), ("occ25", 0.25), ("occ40", 0.40)):
        assert abs(m1[name]["area_frac"] - target) < 0.02
        diff = (c1[name] != x).any(1).float().mean((1, 2))
        assert (diff <= m1[name]["area_frac"] + 1e-6).all()
    boxes = occlusion_boxes(1000, 0.4, 0)
    assert (boxes[:, 0] + boxes[:, 2] <= 32).all() and (boxes[:, 1] + boxes[:, 3] <= 32).all()
    assert torch.equal(apply_noise(x, 0.15, 3), apply_noise(x, 0.15, 3))


def test_probe_keeps_encoder_frozen(monkeypatch):
    import evaluate
    monkeypatch.setattr(evaluate, "PROBE_EPOCHS", 3)
    torch.manual_seed(0)
    enc = Encoder()
    h0 = param_hash(enc)
    x = torch.rand(120, 3, 32, 32); y = torch.arange(120) % 10
    f = extract_features(enc, x, "cpu")
    assert f.shape == (120, 128)
    z = standardizer(f[:100])
    select_probe(z(f[:100]), y[:100], (z(f[100:]), y[100:]), "cpu")
    assert param_hash(enc) == h0
