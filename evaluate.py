"""Frozen-encoder evaluation (the role of upstream ijepa_extras.py, with the assignment's protocol).

Protocol: the frozen context encoder sees all 64 patches, tokens are mean-pooled (128-d), standardised with
probe-train statistics, and a linear probe is trained on the 2k probe-train images. (lr, wd) is selected on the
1k validation images only; the clean-trained probe is then applied to the clean and corrupted test images.
Also: random-init control, oracle probe (fit on corrupted probe-train images; diagnostic only), held-out latent
loss on fixed images and masks, collapse metrics, and a probe trained on a mix of clean and noisy images (diagnostic only).

Usage:  python evaluate.py --run_dir runs/proposed_s0        # writes eval.json, appends to results/metrics.csv
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.corruptions import build_conditions
from data.splits import load_cifar, load_or_create_splits, normalize, to_unit_tensor
from models.modified_ijepa import MEAN, STD, Encoder, Predictor, sample_ijepa_masks

CONDS = ["clean", "occ10", "occ25", "occ40", "noise0.05", "noise0.15"]
PROBE_EPOCHS, PROBE_BATCH, PROBE_SEED = 100, 256, 0
PROBE_GRID = {"lr": [1e-3, 3e-3, 1e-2], "wd": [0.0, 1e-4, 1e-2]}
FIXED_PROBE = dict(lr=3e-3, wd=1e-4)           # single setting used for the curves during training
EVAL_MASK_SEED, EVAL_MASK_BATCH = 12345, 64     # held-out loss: fixed images (val) and fixed upstream masks
ORACLE_SEED_OFFSET = 100                        # independent corruptions of the probe-train images (oracle only)
NOISY_PROBE_SIGMA, NP_TRAIN_SEED, NP_VAL_SEED, NP_MIX_SEED = 0.15, 20_000, 30_000, 40_000


class EvalData:
    """Fixed evaluation tensors in [0, 1] (corruptions are applied before normalisation)."""

    def __init__(self, data_root, splits_path, device):
        xtr, ytr, xte, yte = load_cifar(data_root)
        self.splits = load_or_create_splits(splits_path, ytr, yte)
        ind = self.splits["indices"]
        self.device = device
        self.x_probe, self.y_probe = to_unit_tensor(xtr, ind["probe_train"]), torch.from_numpy(ytr[ind["probe_train"]])
        self.x_val, self.y_val = to_unit_tensor(xtr, ind["val"]), torch.from_numpy(ytr[ind["val"]])
        self.y_test = torch.from_numpy(yte[ind["test"]])
        self.test, self.cond_meta = build_conditions(to_unit_tensor(xte, ind["test"]), MEAN)
        self.probe_conds, _ = build_conditions(self.x_probe, MEAN, seed_offset=ORACLE_SEED_OFFSET)


# ---------- features, probe, metrics ----------

def param_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def frozen(encoder):
    enc = copy.deepcopy(encoder).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def extract_features(encoder, x01, device, bs=500):
    """All patches visible (idx=None), mean over tokens."""
    was_training = encoder.training
    encoder.eval()
    out = [encoder(normalize(x01[i:i + bs].to(device), MEAN, STD)).mean(1).float().cpu() for i in range(0, len(x01), bs)]
    encoder.train(was_training)
    return torch.cat(out)


def standardizer(f_train):
    mu, sd = f_train.mean(0, keepdim=True), f_train.std(0, keepdim=True).clamp_min(1e-6)
    return lambda f: (f - mu) / sd


@torch.no_grad()
def probe_acc(clf, f, y, device):
    return (clf(f.to(device)).argmax(-1).cpu() == y.cpu()).float().mean().item()


def fit_probe(train_feats, y, lr, wd, device, val=None):
    """Linear probe on standardised features. train_feats: one tensor, or a list with one tensor per epoch (fresh
    noise each epoch). val = (features, labels), or a list of them: the (mean) validation accuracy after the last
    epoch is returned."""
    with torch.random.fork_rng(devices=[]):          # never disturb the RNG of a training run
        torch.manual_seed(PROBE_SEED)
        clf = nn.Linear(train_feats[0].size(-1) if isinstance(train_feats, list) else train_feats.size(1), 10).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
        g = torch.Generator().manual_seed(PROBE_SEED)
        y = y.to(device)
        for ep in range(PROBE_EPOCHS):
            f = (train_feats[ep] if isinstance(train_feats, list) else train_feats).to(device)
            perm = torch.randperm(len(f), generator=g).to(device)
            for i in range(0, len(f), PROBE_BATCH):
                idx = perm[i:i + PROBE_BATCH]
                loss = F.cross_entropy(clf(f[idx]), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
    vals = val if isinstance(val, list) else [val]
    return clf, (sum(probe_acc(clf, *v, device) for v in vals) / len(vals) if val is not None else None)


def select_probe(train_feats, y, val, device):
    """Grid over (lr, wd), selected on validation accuracy; ties -> smaller wd, then smaller lr."""
    best = None
    for lr, wd in itertools.product(PROBE_GRID["lr"], PROBE_GRID["wd"]):
        clf, acc = fit_probe(train_feats, y, lr, wd, device, val)
        if best is None or (acc, -wd, -lr) > best[0]:
            best = ((acc, -wd, -lr), clf, lr, wd, acc)
    return best[1:]


def collapse_stats(f):
    s = torch.linalg.svdvals(f - f.mean(0, keepdim=True))
    p = s / s.sum()
    fn = F.normalize(f, dim=1); n = len(f)
    return {"feat_std_mean": f.std(0).mean().item(),
            "eff_rank": torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item(),
            "top1_sv_frac": (s[0] ** 2 / (s ** 2).sum()).item(),
            "mean_pairwise_cos": (((fn @ fn.T).sum() - n) / (n * (n - 1))).item()}


@torch.no_grad()
def heldout_loss(ctx_enc, tgt_enc, pred, x01, device):
    """Latent loss on fixed val images and fixed upstream masks, with the real context, with another image's
    context (batch rolled by one), and for a content-free per-position mean target."""
    rng, D = random.Random(EVAL_MASK_SEED), ctx_enc.dim
    g = lambda t, ti: t.gather(1, ti.unsqueeze(-1).expand(-1, -1, D))
    out = {"heldout_loss": [], "heldout_loss_other_ctx": [], "heldout_loss_mean_target": []}
    for i in range(0, len(x01) - EVAL_MASK_BATCH + 1, EVAL_MASK_BATCH):
        x = normalize(x01[i:i + EVAL_MASK_BATCH].to(device), MEAN, STD)
        cl, tls = sample_ijepa_masks(len(x), ctx_enc.grid, rng=rng)
        ci = torch.tensor(cl, device=device); tis = [torch.tensor(t, device=device) for t in tls]
        full = F.layer_norm(tgt_enc(x), (D,))
        ce, ce_other = ctx_enc(x, ci), ctx_enc(x.roll(1, dims=0), ci)
        mean_full = full.mean(0, keepdim=True).expand_as(full)
        loss = lambda p: (sum(F.smooth_l1_loss(p(ti), g(full, ti)) for ti in tis) / len(tis)).item()
        out["heldout_loss"].append(loss(lambda ti: pred(ce, ci, ti)))
        out["heldout_loss_other_ctx"].append(loss(lambda ti: pred(ce_other, ci, ti)))
        out["heldout_loss_mean_target"].append(loss(lambda ti: g(mean_full, ti)))
    return {k: float(np.mean(v)) for k, v in out.items()}


# ---------- during training: cheap metrics for the curves ----------

def quick_eval(data, ctx_enc, tgt_enc, pred):
    """Fixed probe setting (no selection), test accuracy per condition, collapse stats, held-out loss."""
    dev = data.device
    with torch.random.fork_rng(devices=[torch.device(dev)] if torch.device(dev).type == "cuda" else []):
        f_probe = extract_features(ctx_enc, data.x_probe, dev); z = standardizer(f_probe)
        clf, val_acc = fit_probe(z(f_probe), data.y_probe, FIXED_PROBE["lr"], FIXED_PROBE["wd"], dev,
                                 (z(extract_features(ctx_enc, data.x_val, dev)), data.y_val))
        feats = {c: extract_features(ctx_enc, data.test[c], dev) for c in CONDS}
        row = {"val_acc": val_acc, **{f"acc_{c}": probe_acc(clf, z(feats[c]), data.y_test, dev) for c in CONDS}}
        row.update({f"cos_{c}": F.cosine_similarity(feats[c], feats["clean"], dim=1).mean().item() for c in CONDS[1:]})
        row.update(collapse_stats(feats["clean"]))
        row.update(heldout_loss(ctx_enc, tgt_enc, pred, data.x_val, dev))
    return row


# ---------- final protocol ----------

def full_evaluation(data, ctx_enc, tgt_enc, pred, oracle=True):
    """(lr, wd) selected on val; the clean-trained probe applied to every test condition; oracle probe; collapse
    stats; held-out loss. The encoder is a frozen copy whose parameter hash is checked before and after."""
    dev, enc = data.device, frozen(ctx_enc)
    h0 = param_hash(enc)
    f_probe = extract_features(enc, data.x_probe, dev); z = standardizer(f_probe)
    f_test = {c: extract_features(enc, data.test[c], dev) for c in CONDS}
    clf, lr, wd, val_acc = select_probe(z(f_probe), data.y_probe, (z(extract_features(enc, data.x_val, dev)), data.y_val), dev)
    row = {"probe_lr": lr, "probe_wd": wd, "val_acc": val_acc}
    row.update({f"acc_{c}": probe_acc(clf, z(f_test[c]), data.y_test, dev) for c in CONDS})
    row.update({f"cos_{c}": F.cosine_similarity(f_test[c], f_test["clean"], dim=1).mean().item() for c in CONDS[1:]})
    if oracle:                                       # diagnostic only: probe fit on corrupted probe-train images
        for c in CONDS[1:]:
            fc = extract_features(enc, data.probe_conds[c], dev); zc = standardizer(fc)
            clf_c, _ = fit_probe(zc(fc), data.y_probe, lr, wd, dev)
            row[f"oracle_acc_{c}"] = probe_acc(clf_c, zc(f_test[c]), data.y_test, dev)
    row.update(collapse_stats(f_test["clean"]))
    row.update(heldout_loss(enc, tgt_enc, pred, data.x_val, dev))
    assert param_hash(enc) == h0, "encoder changed during probing"
    return row


def mixed_probe(data, ctx_enc):
    """Diagnostic: does the frozen encoder keep the class information under noise? The probe is trained on a mix
    of clean and noisy images: every epoch each probe-train image is clean or, with probability 1/2, a noisy copy
    (a fresh sigma=0.15 draw every epoch). Standardisation statistics come from the clean and all noisy features;
    (lr, wd) is selected on the mean of clean and noisy validation accuracy. Test accuracy for every condition."""
    dev, enc = data.device, frozen(ctx_enc)
    noisy = lambda x, seed: (x + torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))
                             * NOISY_PROBE_SIGMA).clamp(0, 1)
    f_clean = extract_features(enc, data.x_probe, dev)
    f_noisy = [extract_features(enc, noisy(data.x_probe, NP_TRAIN_SEED + e), dev) for e in range(PROBE_EPOCHS)]
    use_noisy = [(torch.rand(len(f_clean), generator=torch.Generator().manual_seed(NP_MIX_SEED + e)) < 0.5)
                 .to(f_clean.device)[:, None] for e in range(PROBE_EPOCHS)]
    z = standardizer(torch.cat([f_clean, *f_noisy]))
    f_train = [z(torch.where(use_noisy[e], f_noisy[e], f_clean)) for e in range(PROBE_EPOCHS)]
    val = [(z(extract_features(enc, data.x_val, dev)), data.y_val),
           (z(extract_features(enc, noisy(data.x_val, NP_VAL_SEED), dev)), data.y_val)]
    clf, lr, wd, val_acc = select_probe(f_train, data.y_probe, val, dev)
    return {"mixed_probe_lr": lr, "mixed_probe_wd": wd, "mixed_probe_val_mean": val_acc,
            **{f"mixed_probe_acc_{c}": probe_acc(clf, z(extract_features(enc, data.test[c], dev)), data.y_test, dev)
               for c in CONDS}}


def load_models(ckpt, device, init=False):
    """Models of a finished run (init=True: its step-0 random initialisation, the random-encoder control)."""
    ctx_enc = Encoder().to(device); pred = Predictor(grid=ctx_enc.grid).to(device)
    ctx_enc.load_state_dict(ckpt["init"]["ctx_enc"] if init else ckpt["ctx_enc"])
    pred.load_state_dict(ckpt["init"]["predictor"] if init else ckpt["predictor"])
    tgt_enc = copy.deepcopy(ctx_enc)
    if not init:
        tgt_enc.load_state_dict(ckpt["tgt_enc"])
    return ctx_enc, tgt_enc, pred


def evaluate_run(run_dir, data, metrics_csv="results/metrics.csv", mixed=True):
    ckpt = torch.load(os.path.join(run_dir, "ckpt_final.pt"), map_location=data.device, weights_only=False)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    rows = []
    for enc_name, init in (("random_init", True), ("pretrained", False)):
        models = load_models(ckpt, data.device, init)
        row = {"run": summary["run"], "config": summary["config_path"], "seed": summary["seed"],
               "steps": summary["steps"], "encoder": enc_name, **full_evaluation(data, *models)}
        if mixed:
            row.update(mixed_probe(data, models[0]))
        row.update({k: summary[k] for k in ("params_trainable", "train_time_s", "peak_mem_mb", "hardware")})
        rows.append(row)
        print(enc_name, {k: round(100 * v, 1) for k, v in row.items() if k.startswith("acc_")}, flush=True)
    json.dump({"rows": rows, "corruptions": data.cond_meta, "splits_fingerprint": data.splits["fingerprint"]},
              open(os.path.join(run_dir, "eval.json"), "w"), indent=1)
    if metrics_csv:
        new = not os.path.exists(metrics_csv)
        os.makedirs(os.path.dirname(metrics_csv) or ".", exist_ok=True)
        with open(metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_root", default="./cifar_data")
    ap.add_argument("--splits_path", default="results/splits.json")
    ap.add_argument("--metrics_csv", default="results/metrics.csv")
    ap.add_argument("--no_mixed_probe", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_run(a.run_dir, EvalData(a.data_root, a.splits_path, dev), a.metrics_csv, mixed=not a.no_mixed_probe)
