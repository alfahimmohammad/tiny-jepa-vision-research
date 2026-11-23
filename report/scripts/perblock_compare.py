"""H1 fixed-count samplers: one target set from the union (report runs) vs one set per block (per_block: true).

Reads the report's notebook runs (union) and the per-block reruns, writes to results/h1_perblock/:
  curves_<config>.png   training graphs, union vs per-block (upstream baseline in grey for reference)
  curves_all.png        all configs, the four headline curves
  final_compare.png     final protocol accuracies, union vs per-block
  comparison.csv        final metrics side by side
  runs/<config>/        logs + eval.json of the per-block runs (checkpoints stay on scratch)
"""
import csv, json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "h1_perblock")
OLD = os.path.join(OUT, "..", "report_runs")                 # notebook runs (union target set)
NEW = os.environ.get("PERBLOCK_RUNS")                          # dir of fresh train.py runs; default: the copies in OUT/runs
# config -> (old notebook run, new run dir, label)
RUNS = {"2_large_overlap": ("h1_masking__2_large_overlap", "2_large_overlap_perblock_s0", "2_large_overlap (2 blocks)"),
        "2_large_nonoverlap": ("h1_masking__2_large_nonoverlap", "2_large_nonoverlap_perblock_s0", "2_large_nonoverlap (2 blocks)"),
        "scattered": ("h1_masking__scattered", "ablation_perblock_s0", "scattered (ablation, 6-16 small blocks)"),
        "overlap_frac": ("h1_masking__baseline_ovfrac", "overlap_frac_perblock_s0", "overlap_frac (4 blocks)")}
BASE = "h1_masking__baseline"
C = ["clean", "occ10", "occ25", "occ40", "noise0.05", "noise0.15"]
COL = {"union": "#2a6fdb", "per-block": "#d9480f", "baseline": "#8a8a8a"}
PANELS = [("loss", "training loss (rolling mean, 200 steps)", "train"), ("heldout_loss", "held-out loss", "eval"),
          ("val_acc", "val acc (fixed probe)", "eval"), ("acc_clean", "test clean", "eval"),
          ("acc_occ25", "test occ 25%", "eval"), ("acc_occ40", "test occ 40%", "eval"),
          ("acc_noise0.15", "test noise 0.15", "eval"), ("eff_rank", "effective rank", "eval")]


def newdir(cfg, nd):
    return f"{NEW}/{nd}" if NEW else f"{OUT}/runs/{cfg}"


def curves(old_dir, new_dir):
    o = {"train": pd.read_csv(f"{OLD}/{old_dir}/train_log.csv"), "eval": pd.read_csv(f"{OLD}/{old_dir}/eval_curve.csv")}
    n = {"train": pd.read_csv(f"{new_dir}/train_log.csv"), "eval": pd.read_csv(f"{new_dir}/eval_curve.csv")}
    return o, n


def panel(ax, key, title, src, series):
    for name, d, ls, lw in series:
        df = d[src]
        y = df[key].rolling(200, min_periods=1).mean() if src == "train" else df[key] * (100 if key.startswith(("acc", "val")) else 1)
        ax.plot(df["step"], y, color=COL[name], ls=ls, lw=lw, label=name)
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    base = {"train": pd.read_csv(f"{OLD}/{BASE}/train_log.csv"), "eval": pd.read_csv(f"{OLD}/{BASE}/eval_curve.csv")}
    rows, allc = [], {}
    for cfg, (od, nd, label) in RUNS.items():
        nd = newdir(cfg, nd)
        if not os.path.exists(f"{nd}/eval.json"):
            print("missing", nd); continue
        o, n = curves(od, nd); allc[cfg] = (o, n, label)
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
        for ax, (k, t, src) in zip(axes.flat, PANELS):
            panel(ax, k, t, src, [("baseline", base, "--", 0.9), ("union", o, "-", 1.4), ("per-block", n, "-", 1.4)])
        axes.flat[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"{label}: one target set from the union vs one set per block (seed 0, 15k steps)", fontsize=11)
        fig.supxlabel("step", fontsize=9); fig.tight_layout(); fig.savefig(f"{OUT}/curves_{cfg}.png", dpi=130); plt.close(fig)
        # final protocol metrics
        po = json.load(open(f"{OLD}/{od}/result.json"))["protocol"]["pretrained"]
        en = json.load(open(f"{nd}/eval.json")); pn = [r for r in en["rows"] if r["encoder"] == "pretrained"][0]
        sn = json.load(open(f"{nd}/summary.json"))
        r = {"config": cfg}
        for c in C:
            r[f"union_{c}"], r[f"perblock_{c}"] = round(100 * po[f"acc_{c}"], 2), round(100 * pn[f"acc_{c}"], 2)
            r[f"delta_{c}"] = round(r[f"perblock_{c}"] - r[f"union_{c}"], 2)
        for k in ("val_acc", "heldout_loss", "feat_std_mean", "eff_rank", "mean_pairwise_cos"):
            s = 100 if k == "val_acc" else 1
            r[f"union_{k}"], r[f"perblock_{k}"] = round(s * po[k], 4), round(s * pn[k], 4)
        r["perblock_train_time_s"] = round(sn["train_time_s"]); r["perblock_tgt_tokens"] = int(n["train"]["tgt_tokens"].median())
        rows.append(r)
        dst = f"{OUT}/runs/{cfg}"; os.makedirs(dst, exist_ok=True)
        if os.path.abspath(nd) != os.path.abspath(dst):
            for f in ("train_log.csv", "eval_curve.csv", "eval.json", "summary.json"):
                shutil.copy(f"{nd}/{f}", dst)
    with open(f"{OUT}/comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # all configs, headline curves
    keys = [PANELS[0], PANELS[1], PANELS[5], PANELS[6]]
    fig, axes = plt.subplots(len(allc), 4, figsize=(14, 2.3 * len(allc)), sharex=True)
    for i, (cfg, (o, n, label)) in enumerate(allc.items()):
        for j, (k, t, src) in enumerate(keys):
            ax = axes[i, j]
            panel(ax, k, t if i == 0 else "", src, [("baseline", base, "--", 0.8), ("union", o, "-", 1.2), ("per-block", n, "-", 1.2)])
        axes[i, 0].set_ylabel(cfg, fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False); fig.supxlabel("step", fontsize=9)
    fig.suptitle("H1 fixed-count samplers: union (report) vs per-block target sets", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/curves_all.png", dpi=120); plt.close(fig)
    # final accuracies
    fig, axes = plt.subplots(1, len(C), figsize=(16, 3.6), sharey=True)
    names = [r["config"] for r in rows]; y = range(len(rows))
    for ax, c in zip(axes, C):
        for yi, r in zip(y, rows):
            ax.plot([r[f"union_{c}"], r[f"perblock_{c}"]], [yi, yi], color="#c8c8c8", lw=1.5, zorder=1)
        ax.scatter([r[f"union_{c}"] for r in rows], y, color=COL["union"], s=36, label="union", zorder=2)
        ax.scatter([r[f"perblock_{c}"] for r in rows], y, color=COL["per-block"], s=36, label="per-block", zorder=2)
        ax.set_title(f"{c} (%)", fontsize=9); ax.grid(axis="x", alpha=0.25, lw=0.5); ax.tick_params(labelsize=7)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].set_yticks(list(y)); axes[0].set_yticklabels(names, fontsize=8); axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Final test accuracy, validation-selected probe (seed 0, 15k steps)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/final_compare.png", dpi=130); plt.close(fig)
    print(pd.DataFrame(rows)[["config"] + [f"{p}_{c}" for c in C for p in ("union", "perblock", "delta")]].to_string(index=False))


if __name__ == "__main__":
    main()
