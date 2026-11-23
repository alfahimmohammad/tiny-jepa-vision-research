"""Shared-offset decomposition of the pooled clean test features (Sec. Part C): f = m + v, |m| vs RMS|v|.
Needs the final checkpoints (not stored in the repo):
  python report/scripts/offset.py --ckpts DIR [--data_root ./cifar_data]
DIR holds one sub-directory per run with ckpt_final.pt (notebook runs: key "ctx_enc"; train.py runs: same key)."""
import argparse, json, os, sys
import torch
from torchvision import datasets
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, REPO)
from models.modified_ijepa import Encoder, MEAN, STD   # byte-identical to upstream Encoder

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", required=True)
ap.add_argument("--data_root", default="./cifar_data")
ap.add_argument("--runs", nargs="+", default=["h1_masking__baseline", "h1_masking__scattered", "h1_masking__1_large",
                                              "noise_aug__ctx_noise", "ablation_perblock_s0"])
a = ap.parse_args()
sp = json.load(open(os.path.join(REPO, "results", "splits.json")))
idx = sp["indices"]["test"]
ds = datasets.CIFAR10(a.data_root, train=False, download=True)
x = torch.tensor(ds.data[idx]).permute(0, 3, 1, 2).float() / 255
xn = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
for run in a.runs:
    enc = Encoder(); enc.load_state_dict(torch.load(f"{a.ckpts}/{run}/ckpt_final.pt", map_location="cpu", weights_only=False)["ctx_enc"]); enc.eval()
    with torch.no_grad(): f = torch.cat([enc(xn[i:i + 500]).mean(1) for i in range(0, len(xn), 500)])
    m = f.mean(0); v = f - m
    print(f"{run:36s} |m| {m.norm():.2f}  rms|v| {v.norm(dim=1).pow(2).mean().sqrt():.2f}  share |m|^2/E|f|^2 {m.norm()**2/f.norm(dim=1).pow(2).mean():.3f}  std {f.std(0).mean():.3f}")
