"""Training graphs of finished runs, as printed by the notebooks' 11c cell:
    plot_training([RUN_NAME]); plot_metrics([RUN_NAME]); plot_robustness_curves([RUN_NAME])

Usage:  python scripts/plot_run.py runs/<name>_s<seed> [more run dirs] [--out DIR]
Reads train_log.csv / eval_curve.csv written by train.py and saves training.png, metrics.png, robustness.png
(to --out, default: the first run dir). The three functions are the notebook's (tiny_jepa_h1_masking.ipynb,
section 9), with plt.show() replaced by savefig.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONDS = ["clean", "occ10", "occ25", "occ40", "noise0.05", "noise0.15"]
COND_LABEL = {"clean": "clean", "occ10": "occ 10%", "occ25": "occ 25%", "occ40": "occ 40%",
              "noise0.05": "noise σ=.05", "noise0.15": "noise σ=.15"}
plt.rcParams.update({
    "figure.dpi": 110, "axes.prop_cycle": plt.cycler(color=SERIES), "lines.linewidth": 2,
    "lines.markersize": 5, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
    "axes.edgecolor": "#b5b4ae", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "text.color": "#0b0b0b",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
})
RUNS, OUT = {}, "."


def load_run(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    E = pd.read_csv(os.path.join(run_dir, "eval_curve.csv")).rename(
        columns={"heldout_loss_other_ctx": "heldout_loss_shuffled_ctx"})
    RUNS[name] = {"name": name, "log": pd.read_csv(os.path.join(run_dir, "train_log.csv")), "evals": E}
    return name


def _runs(names):
    names = names or list(RUNS)
    return [RUNS[n] for n in names]


def _smooth(s, frac=0.02):
    return s.rolling(max(1, int(len(s) * frac)), min_periods=1).mean()


def _save(fname):
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()


def plot_training(names=None):
    """Per-step training signals. Raw loss is drawn faintly behind a moving average."""
    runs = _runs(names)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    panels = [("loss", "training loss (smooth-L1 on LN targets)"), ("lr", "learning rate"),
              ("ema", "EMA momentum"), ("grad_norm", "gradient norm"),
              ("tgt_pooled_std", "pooled feature std (solid target, dashed context)"),
              ("ctx_len", "visible context tokens (solid) / unique target tokens (dashed)")]
    for k, r in enumerate(runs):
        col, L = SERIES[k % len(SERIES)], r["log"]
        for a, (key, _) in zip(ax.flat, panels):
            if key in ("loss", "grad_norm"):
                a.plot(L.step, L[key], color=col, alpha=0.18, lw=1)
                a.plot(L.step, _smooth(L[key]), color=col, label=r["name"])
            elif key == "tgt_pooled_std":
                a.plot(L.step, _smooth(L.tgt_pooled_std), color=col, label=r["name"])
                a.plot(L.step, _smooth(L.ctx_pooled_std), color=col, ls="--")
            elif key == "ctx_len":
                a.plot(L.step, _smooth(L.ctx_len), color=col, label=r["name"])
                a.plot(L.step, _smooth(L.tgt_union), color=col, ls="--")
            else:
                a.plot(L.step, L[key], color=col, label=r["name"])
    for a, (key, title) in zip(ax.flat, panels):
        a.set_title(title); a.set_xlabel("step")
    ax[0, 0].set_yscale("log")
    if len(runs) > 1:
        ax[0, 0].legend()
    _save("training.png")


def plot_metrics(names=None):
    """Metrics curves from the periodic evaluation (step 0 = random init)."""
    runs = _runs(names)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for k, r in enumerate(runs):
        col, E = SERIES[k % len(SERIES)], r["evals"]
        ax[0, 0].plot(E.step, 100 * E.val_acc, "o-", color=col, label=r["name"])
        ax[0, 1].plot(E.step, E.heldout_loss, "o-", color=col, label=f"{r['name']}: real context")
        ax[0, 1].plot(E.step, E.heldout_loss_shuffled_ctx, "--", color=col, label=f"{r['name']}: other image's context")
        ax[0, 1].plot(E.step, E.heldout_loss_mean_target, ":", color=col, label=f"{r['name']}: content-free")
        ax[0, 2].plot(E.step, E.eff_rank, "o-", color=col, label=r["name"])
        ax[1, 0].plot(E.step, E.top1_sv_frac, "o-", color=col, label=r["name"])
        ax[1, 1].plot(E.step, E.mean_pairwise_cos, "o-", color=col, label=r["name"])
        ax[1, 2].plot(E.step, E.feat_std_mean, "o-", color=col, label=r["name"])
    titles = ["probe val accuracy (%), fixed probe setting", "held-out latent loss (val, fixed masks)",
              "effective rank of pooled features (/128)", "top-1 singular-value share (collapse ↑)",
              "mean pairwise cosine (collapse ↑)", "mean per-feature std (collapse ↓)"]
    for a, t in zip(ax.flat, titles):
        a.set_title(t); a.set_xlabel("pretraining step")
    ax[0, 1].legend(fontsize=7)
    if len(runs) > 1:
        ax[0, 0].legend()
    _save("metrics.png")


def plot_robustness_curves(names=None):
    """Test accuracy under each stress test vs pretraining step (descriptive only, never for selection)."""
    runs = [r for r in _runs(names) if "acc_clean" in r["evals"]]
    if not runs:
        print("no test tracking"); return
    fig, ax = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 4), squeeze=False)
    for a, r in zip(ax[0], runs):
        E = r["evals"]
        for k, c in enumerate(CONDS):
            a.plot(E.step, 100 * E[f"acc_{c}"], "o-", color=SERIES[k], label=COND_LABEL[c])
        a.axhline(10, color="#b5b4ae", ls=":", lw=1)
        a.set_title(f"{r['name']}: test accuracy (%) by condition"); a.set_xlabel("pretraining step")
        a.legend(ncol=2)
    _save("robustness.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    names = [load_run(d) for d in a.run_dirs]
    OUT = a.out or a.run_dirs[0]; os.makedirs(OUT, exist_ok=True)
    plot_training(names); plot_metrics(names); plot_robustness_curves(names)
    print("saved", [os.path.join(OUT, f) for f in ("training.png", "metrics.png", "robustness.png")])
