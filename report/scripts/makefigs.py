import json, os, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = os.path.join(REPO, "results", "report_runs")   # notebook runs: result.json, train_log.csv, eval_curve.csv
A = os.path.join(REPO, "report", "figures")
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "lines.linewidth": 1.6,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#b5b4ae", "legend.frameon": False,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
ev = lambda r: pd.read_csv(f"{R}/{r}/eval_curve.csv"); tl = lambda r: pd.read_csv(f"{R}/{r}/train_log.csv")
b, t = ev("h1_masking__baseline"), tl("h1_masking__baseline")
# Figure 2: baseline dynamics
fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.35))
a = ax[0]
a.plot(t.step, t.loss.rolling(200, min_periods=20).mean(), color="#b5b4ae", lw=1.2, label="train loss (200-step mean)")
a.plot(b.step, b.heldout_loss, color=S[0], label="held-out: real context")
a.plot(b.step, b.heldout_loss_shuffled_ctx, color=S[1], ls="--", label="held-out: other image's context")
a.plot(b.step, b.heldout_loss_mean_target, color=S[2], ls=":", lw=1.9, label="position-only (mean target)")
a.set_xscale("symlog", linthresh=500); a.set_xlim(0, 15000); a.set_ylim(0, 0.66)
a.axvline(500, color="#52514e", lw=0.7, ls="-."); a.text(540, 0.015, "500 steps", fontsize=6.5, color="#52514e")
a.set_title("(a) latent loss (smooth-L1)"); a.set_xlabel("pretraining step"); a.legend(loc="upper right", fontsize=6.2)
a = ax[1]
a.plot(b.step, b.eff_rank, color=S[6]); a.set_title("(b) effective rank of features (/128)"); a.set_xlabel("pretraining step")
a.set_xscale("symlog", linthresh=500); a.set_xlim(0, 15000)
a = ax[2]
for c, col, lab in (("acc_clean", S[0], "clean"), ("acc_occ40", S[3], "occlusion 40%"), ("acc_noise0.15", S[5], "noise $\\sigma$=.15")):
    a.plot(b.step, 100 * b[c], color=col, label=lab)
    a.axhline(100 * b[c].iloc[0], color=col, lw=0.7, ls=":")
a.set_title("(c) test accuracy, clean-trained probe (%)"); a.set_xlabel("pretraining step"); a.legend(loc="upper left", fontsize=6.5)
a.text(15000, 36.3, "dotted: step-0 (random) encoder", fontsize=6, ha="right", color="#52514e")
a = ax[3]
a.plot(b.step, b["cos_noise0.15"], color=S[5], label="noise $\\sigma$=.15"); a.plot(b.step, b["cos_occ40"], color=S[3], label="occlusion 40%")
a.set_title("(d) cosine(clean, corrupted) features"); a.set_xlabel("pretraining step"); a.legend(loc="upper right", fontsize=6.5)
plt.tight_layout(pad=0.4, w_pad=0.8); plt.savefig(f"{A}/fig_baseline.pdf"); plt.savefig(f"{A}/fig_baseline.png", dpi=130); plt.close()
# Figure 4: noise-augmented pretraining vs baseline
n = ev("noise_aug__ctx_noise")
fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.3))
for a, c, ttl in ((ax[0], "acc_noise0.15", "(a) noise $\\sigma$=.15"), (ax[1], "acc_noise0.05", "(b) noise $\\sigma$=.05"), (ax[2], "acc_clean", "(c) clean")):
    a.plot(b.step, 100 * b[c], color=S[1], label="baseline")
    a.plot(n.step, 100 * n[c], color=S[0], label="context noise (p=0.5, $\\sigma\\sim$U(0,0.2))")
    a.axhline(100 * b[c].iloc[0], color="#52514e", lw=0.7, ls=":", label="random init")
    a.set_title(ttl + " test accuracy (%)"); a.set_xlabel("pretraining step")
ax[0].legend(loc="center right", fontsize=6.5)
plt.tight_layout(pad=0.4, w_pad=0.8); plt.savefig(f"{A}/fig_noise.pdf"); plt.savefig(f"{A}/fig_noise.png", dpi=130); plt.close()
print("figs ok")
