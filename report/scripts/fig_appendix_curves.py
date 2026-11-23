"""Appendix figures: the notebooks' training graphs (plot_training / plot_metrics / plot_robustness_curves) for every
15k-step seed-0 run of the report, one curve per run in every panel.

  fig_app_training.pdf    per-step training signals (train_log.csv)
  fig_app_metrics.pdf     fixed-probe val accuracy, the three held-out losses (one panel each), collapse statistics
  fig_app_robustness.pdf  test accuracy per condition (one panel each)

Sources: baseline, 1_large, ctx_noise = notebook runs; scattered, 2_large_*, overlap cap = per-block repo runs
(results/h1_perblock/runs). The per-block runs were trained before grad_norm / pooled std / unique-target logging
existed, so their train_log comes from a same-seed, same-config retraining (train_log_relog.csv, python train.py --config
configs/<cfg>.yaml); its loss is checked against the original log (max |diff| < 1e-5).
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "report_runs")
PB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "h1_perblock", "runs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7, "legend.fontsize": 7.2,
                     "lines.linewidth": 1.2, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
                     "legend.frameon": False, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.titlepad": 3})

# label, train_log, eval_curve, colour, line style (colours follow the other figures of the report)
RUNS = [("baseline (upstream)", f"{R}/h1_masking__baseline/train_log.csv", f"{R}/h1_masking__baseline/eval_curve.csv", "#8a8986", (0, (4, 1.5))),
        ("1_large (proposed)", f"{R}/h1_masking__1_large/train_log.csv", f"{R}/h1_masking__1_large/eval_curve.csv", "#2a78d6", "-"),
        ("scattered (ablation)", f"{PB}/scattered/train_log_relog.csv", f"{PB}/scattered/eval_curve.csv", "#eb6834", "-"),
        ("2_large_overlap", f"{PB}/2_large_overlap/train_log_relog.csv", f"{PB}/2_large_overlap/eval_curve.csv", "#1baf7a", "-"),
        ("2_large_nonoverlap", f"{PB}/2_large_nonoverlap/train_log_relog.csv", f"{PB}/2_large_nonoverlap/eval_curve.csv", "#eda100", "-"),
        ("overlap cap <25%", f"{PB}/overlap_frac/train_log_relog.csv", f"{PB}/overlap_frac/eval_curve.csv", "#6250d6", "-"),
        ("context noise (H3)", f"{R}/noise_aug__ctx_noise/train_log.csv", f"{R}/noise_aug__ctx_noise/eval_curve.csv", "#e87ba4", (0, (5, 1.5, 1, 1.5)))]
ORIG_TRAIN = {"scattered (ablation)": f"{PB}/scattered/train_log.csv", "2_large_overlap": f"{PB}/2_large_overlap/train_log.csv",
              "2_large_nonoverlap": f"{PB}/2_large_nonoverlap/train_log.csv", "overlap cap <25%": f"{PB}/overlap_frac/train_log.csv"}


def load():
    out = []
    for name, tl, ec, col, ls in RUNS:
        L = pd.read_csv(tl).rename(columns={"ctx_tokens": "ctx_len"})
        E = pd.read_csv(ec).rename(columns={"heldout_loss_shuffled_ctx": "heldout_loss_other_ctx"})
        if name in ORIG_TRAIN:                                   # relog run must reproduce the reported run's loss
            o = pd.read_csv(ORIG_TRAIN[name])["loss"].to_numpy()
            d = np.abs(L["loss"].to_numpy() - o)
            print(f"{name:22s} relog vs original loss: max |diff| {d.max():.2e}, mean {d.mean():.2e}")
        out.append((name, L, E, col, ls))
    return out


smt = lambda s: s.rolling(300, min_periods=1, center=True).mean()     # per-step logs: 300-step window
sme = lambda s: s.rolling(5, min_periods=1, center=True).mean()       # per-50-step evals (accuracies): 250-step window


def axis(a, title):
    a.set_title(title); a.set_xlim(0, 15000)
    a.set_xticks([0, 5000, 10000, 15000]); a.set_xticklabels(["0", "5k", "10k", "15k"])


def legend(fig, runs, y):
    h = [plt.Line2D([], [], color=c, ls=ls, lw=1.6) for _, _, _, c, ls in runs]
    fig.legend(h, [r[0] for r in runs], loc="upper center", ncol=4, bbox_to_anchor=(0.5, y),
               handlelength=2.8, columnspacing=1.6, labelspacing=0.3)


def save(fig, name, top):
    fig.supxlabel("pretraining step", fontsize=7, y=0.005)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.7, rect=(0, 0.01, 1, top))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200)
    plt.close(fig)


def training(runs):
    P = [("(a) training loss (log scale)", "loss"), ("(b) gradient norm", "grad_norm"),
         ("(c) pooled std, target enc.", "tgt_pooled_std"), ("(d) pooled std, context enc.", "ctx_pooled_std"),
         ("(e) visible context tokens", "ctx_len"), ("(f) unique target tokens", "tgt_union"),
         ("(g) learning rate", "lr"), ("(h) EMA momentum", "ema")]
    fig, ax = plt.subplots(2, 4, figsize=(7.1, 3.5))
    for a, (t, k) in zip(ax.flat, P):
        if k in ("lr", "ema"):                                  # same schedule for every run: one line
            L = runs[0][1]; a.plot(L.step, L[k], color="#0b0b0b", lw=1.2)
        else:
            for _, L, _, col, ls in runs:
                a.plot(L.step, smt(L[k]), color=col, ls=ls)
        axis(a, t)
    ax[0, 0].set_yscale("log")
    legend(fig, runs, 1.0); save(fig, "fig_app_training", 0.875)


def metrics(runs):
    P = [("(a) val acc. (%), fixed probe", lambda E: sme(100 * E.val_acc)),
         ("(b) held-out: own context", lambda E: E.heldout_loss),
         ("(c) held-out: other context", lambda E: E.heldout_loss_other_ctx),
         ("(d) held-out: content-free", lambda E: E.heldout_loss_mean_target),
         ("(e) effective rank (/128)", lambda E: E.eff_rank),
         ("(f) top-1 singular-value share", lambda E: E.top1_sv_frac),
         ("(g) mean pairwise cosine", lambda E: E.mean_pairwise_cos),
         ("(h) mean per-feature std", lambda E: E.feat_std_mean)]
    fig, ax = plt.subplots(2, 4, figsize=(7.1, 3.5))
    for a, (t, f) in zip(ax.flat, P):
        for _, _, E, col, ls in runs:
            a.plot(E.step, f(E), color=col, ls=ls)
        axis(a, t)
    lo = min(min(E[k].iloc[2:].min() for _, _, E, _, _ in runs)
             for k in ("heldout_loss", "heldout_loss_other_ctx", "heldout_loss_mean_target"))
    hi = max(max(E[k].iloc[2:].max() for _, _, E, _, _ in runs)
             for k in ("heldout_loss", "heldout_loss_other_ctx", "heldout_loss_mean_target"))
    for a in ax[0, 1:]:                                           # one y-range for the 3 held-out losses (comparable)
        a.set_ylim(lo - 0.01, hi + 0.01)
    legend(fig, runs, 1.0); save(fig, "fig_app_metrics", 0.875)


def robustness(runs):
    C = [("clean", "clean"), ("occ10", "occlusion 10%"), ("occ25", "occlusion 25%"), ("occ40", "occlusion 40%"),
         ("noise0.05", "Gaussian noise $\\sigma$=0.05"), ("noise0.15", "Gaussian noise $\\sigma$=0.15")]
    fig, ax = plt.subplots(2, 3, figsize=(7.1, 3.45))
    for i, (a, (c, t)) in enumerate(zip(ax.flat, C)):
        for _, _, E, col, ls in runs:
            a.plot(E.step, sme(100 * E[f"acc_{c}"]), color=col, ls=ls)
        axis(a, f"({'abcdef'[i]}) {t}")
        if i % 3 == 0:
            a.set_ylabel("test accuracy (%)")
    legend(fig, runs, 1.0); save(fig, "fig_app_robustness", 0.875)


if __name__ == "__main__":
    runs = load()
    training(runs); metrics(runs); robustness(runs)
