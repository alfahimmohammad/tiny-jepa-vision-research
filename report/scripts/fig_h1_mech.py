"""Figure: why 1_large is the most noise-robust geometry and scattered the least (seed 0, eval every 50 steps).
baseline / 1_large: notebook runs; scattered: per-block repo run (results/h1_perblock/runs/scattered)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "report_runs")
PB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "h1_perblock", "runs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "lines.linewidth": 1.5,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#b5b4ae", "legend.frameon": False,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})


def curve(path):
    d = pd.read_csv(path)
    if "heldout_loss_shuffled_ctx" in d:
        d = d.rename(columns={"heldout_loss_shuffled_ctx": "heldout_loss_other_ctx"})
    return d


RUNS = [("baseline (upstream)", curve(f"{R}/h1_masking__baseline/eval_curve.csv"), "#8a8986", "--"),
        ("1_large (proposed)", curve(f"{R}/h1_masking__1_large/eval_curve.csv"), "#2a78d6", "-"),
        ("scattered (ablation)", curve(f"{PB}/scattered/eval_curve.csv"), "#eb6834", "-")]
sm = lambda s: s.rolling(5, min_periods=1, center=True).mean()       # 250-step window: the per-50-step evals are noisy
PANELS = [("(a) test acc., noise $\\sigma$=.15 (%)", lambda d: 100 * d["acc_noise0.15"]),
          ("(b) mean per-feature std", lambda d: d["feat_std_mean"]),
          ("(c) mean pairwise cosine", lambda d: d["mean_pairwise_cos"]),
          ("(d) loss gap: other-image ctx.", lambda d: d["heldout_loss_other_ctx"] - d["heldout_loss"]),
          ("(e) loss gap: content-free", lambda d: d["heldout_loss_mean_target"] - d["heldout_loss"])]
fig, ax = plt.subplots(1, 5, figsize=(10.5, 2.25))
for a, (title, f) in zip(ax, PANELS):
    for name, d, col, ls in RUNS:
        a.plot(d["step"], sm(f(d)), color=col, ls=ls, label=name)
    a.set_title(title); a.set_xlabel("pretraining step"); a.set_xlim(0, 15000)
    a.set_xticks([0, 5000, 10000, 15000]); a.set_xticklabels(["0", "5k", "10k", "15k"])
ax[3].axhline(0, color="#b5b4ae", lw=0.7); ax[4].axhline(0, color="#b5b4ae", lw=0.7); ax[4].set_ylim(-0.05, 0.17)
ax[0].legend(loc="upper right", fontsize=6.3)
plt.tight_layout(pad=0.4, w_pad=0.7)
plt.savefig(f"{OUT}/fig_h1_mech.pdf"); plt.savefig(f"{OUT}/fig_h1_mech.png", dpi=130); plt.close()
for name, d, _, _ in RUNS:
    e = d.iloc[-1]
    print(f"{name:22s} n15 {100 * e['acc_noise0.15']:.1f} std {e['feat_std_mean']:.3f} pcos {e['mean_pairwise_cos']:.2f} "
          f"gap_other {e['heldout_loss_other_ctx'] - e['heldout_loss']:.3f} gap_pos {e['heldout_loss_mean_target'] - e['heldout_loss']:.3f}")
