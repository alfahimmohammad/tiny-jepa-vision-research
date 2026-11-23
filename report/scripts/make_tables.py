"""Generate the LaTeX tables of the report from the saved result.json files (no hand-copied numbers)."""
import json, os, re, statistics as st
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = os.path.join(REPO, "results", "report_runs")   # notebook runs: result.json, train_log.csv, eval_curve.csv
SEEDS = os.path.join(R, "seeds_1_large")                # 1_large, pretraining seeds 0-4
T = os.path.join(REPO, "report", "tables"); os.makedirs(T, exist_ok=True)
C = ["clean", "occ10", "occ25", "occ40", "noise0.05", "noise0.15"]
load = lambda p: json.load(open(p))
def agg(rows, key, scale=100.0):
    v = [scale * r[key] for r in rows]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else None)
def fmt(m, s=None, d=1, bold=False):
    x = f"{m:.{d}f}" + (f"{{\\scriptsize$\\pm${s:.{d}f}}}" if s is not None else "")
    return f"\\textbf{{{x}}}" if bold else x
prot = lambda r, which="pretrained": r["protocol"][which]
# ---------------- baseline table (Part A summary + 15k) ----------------
md = open(os.path.join(REPO, "results", "partA", "summary_table.md")).read().splitlines()
hdr = [h.strip() for h in md[0].strip("|").split("|")]; pa = {}
for ln in md[2:]:
    c = [x.strip() for x in ln.strip("|").split("|")]
    if len(c) == len(hdr): pa[(c[0], c[1], c[2])] = dict(zip(hdr, c))
def parow(label, key, loss=True):
    d = pa[key]; cells = [d[c].replace(" ± ", "{\\scriptsize$\\pm$") + "}" for c in C]
    hl = d["heldout_loss"].split(" ± ")[0] if loss else "--"; sd = d["feat_std_mean"].split(" ± ")[0]; er = f"{float(d['eff_rank'].split(' ± ')[0]):.1f}"
    return f"{label} & 3 & " + " & ".join(cells) + f" & {hl} & {sd} & {er} \\\\"
base = [load(f"{R}/h1_masking__baseline/result.json")] + [load(f"{R}/h1_masking__baseline__s{k}/result.json") for k in range(1, 5)]
pb = [prot(r) for r in base]; pr = [prot(r, "random init") for r in base]
def row15(label, rows, n):
    cells = [fmt(*agg(rows, "acc_" + c)) for c in C]
    return (f"{label} & {n} & " + " & ".join(cells) + f" & {st.mean(r['heldout_loss'] for r in rows):.3f} & "
            f"{st.mean(r['feat_std_mean'] for r in rows):.3f} & {st.mean(r['eff_rank'] for r in rows):.1f} \\\\")
lines = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Baseline: frozen context encoder, clean-trained linear probe, test accuracy (\%, mean $\pm$ std over $n$ pretraining seeds). "
         r"Held-out latent loss on fixed validation images and masks; feature std = mean per-feature std of pooled features; rank = effective rank (/128).}",
         r"\label{tab:baseline}",
         r"\begin{tabular}{lcccccccccc}\toprule",
         r"Encoder & $n$ & Clean & Occ 10\% & Occ 25\% & Occ 40\% & Noise .05 & Noise .15 & Loss & Feat.\ std & Rank \\\midrule",
         parow("I-JEPA, 500 steps", ("baseline_500", "pretrained", "ctx")),
         parow("I-JEPA, 5k steps", ("baseline_5000", "pretrained", "ctx")),
         row15("I-JEPA, 15k steps", pb, len(pb)),
         row15("\\quad its random init", pr, len(pr)),
         r"\midrule",
         parow("\\textit{Supervised ref., 5k}", ("supervised_5000", "pretrained", "ctx"), loss=False).replace(" & 3 & ", " & 3 & "),
         r"\bottomrule\end{tabular}\end{table}"]
open(f"{T}/tab_baseline.tex", "w").write("\n".join(lines) + "\n")
# ---------------- H1 table ----------------
# fixed-count samplers: per-block target sets (repo runs, results/h1_perblock/runs/<cfg>); the others: notebook runs
PB = os.path.join(REPO, "results", "h1_perblock", "runs")
def run_metrics(d):
    """(protocol metrics, training time) of a notebook run or of a per-block repo run ("pb:<cfg>")."""
    if d.startswith("pb:"):
        rows = load(f"{PB}/{d[3:]}/eval.json")["rows"]; p = [r for r in rows if r["encoder"] == "pretrained"][0]
        return {**p, "heldout_loss_shuffled_ctx": p["heldout_loss_other_ctx"]}, load(f"{PB}/{d[3:]}/summary.json")["train_time_s"]
    r = load(f"{R}/{d}/result.json"); return prot(r), r["train_time_s"]
V = [("Baseline (upstream)", "h1_masking__baseline", "20$^\\dagger$/45$^\\ddagger$"),
     ("\\textbf{1\\_large} (proposed)", "h1_masking__1_large", "20/32"),
     ("\\textbf{scattered} (ablation)", "pb:scattered", "20/32--36$^\\S$"),
     ("2\\_large\\_overlap", "pb:2_large_overlap", "20/2$\\times$16"),
     ("2\\_large\\_nonoverlap", "pb:2_large_nonoverlap", "20/2$\\times$16"),
     ("overlap cap $<$25\\%", "pb:overlap_frac", "16.8/4$\\times$8")]
bm = {c: st.mean(100 * r["acc_" + c] for r in pb) for c in C}
bs = {c: st.stdev(100 * r["acc_" + c] for r in pb) for c in C}
lines = [r"\begin{table*}[t]\centering\setlength{\tabcolsep}{3pt}",
         r"\caption{H1 masking strategies, 15k steps, pretraining seed 0 (validation-selected probe; test accuracy in \%). "
         r"Small numbers: difference to the baseline \emph{mean over 5 seeds} (first row), coloured when it exceeds twice the baseline's seed std for that condition. "
         r"Loss columns: held-out latent loss with the real context / another image's context / content-free per-position mean target. "
         r"Tokens = visible context / target tokens per image; $n{\times}k$: $n$ target blocks predicted as separate sets of $k$ tokens, as upstream "
         r"($^\dagger$batch-minimum trimmed, 4--31; $^\ddagger$4 blocks, $\approx$32 distinct; $^\S$one set per 2--6-patch block). "
         r"Time = training time on one L40S.}",
         r"\label{tab:h1}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccccccc}\toprule",
         r"Strategy & Tokens & Clean & Occ 10\% & Occ 25\% & Occ 40\% & Noise .05 & Noise .15 & Loss (real/other/pos.) & Feat.\ std & Rank & Pair cos & Time (s)\\\midrule"]
def dcol(x, ref, thr):
    d = x - ref
    s = f"{d:+.1f}"
    return f"\\up{{{s}}}" if d >= thr else (f"\\dn{{{s}}}" if d <= -thr else f"{{\\color{{gray}}{s}}}")
# baseline row: mean +- std over 5 seeds
cells = [fmt(*agg(pb, "acc_" + c)) for c in C]
lines.append(f"Baseline (upstream), 5 seeds & 20$^\\dagger$/45$^\\ddagger$ & " + " & ".join(cells) +
             f" & {st.mean(r['heldout_loss'] for r in pb):.3f}/{st.mean(r['heldout_loss_shuffled_ctx'] for r in pb):.3f}/{st.mean(r['heldout_loss_mean_target'] for r in pb):.3f}"
             f" & {st.mean(r['feat_std_mean'] for r in pb):.3f} & {st.mean(r['eff_rank'] for r in pb):.1f} & {st.mean(r['mean_pairwise_cos'] for r in pb):.2f} & "
             f"{st.mean(r['train_time_s'] for r in base):.0f} \\\\")
# 1_large over 5 pretraining seeds (same splits / eval seeds)
L5 = [load(f"{SEEDS}/1_large_pair0/result.json")] + [load(f"{SEEDS}/1_large_pair0_train{k}/result.json") for k in range(1, 5)]
p5 = [prot(r) for r in L5]
cells = [fmt(*agg(p5, "acc_" + c)) for c in C]
lines.append(r"\textbf{1\_large}, 5 seeds & 20/32 & " + " & ".join(cells) +
             f" & {st.mean(r['heldout_loss'] for r in p5):.3f}/{st.mean(r['heldout_loss_shuffled_ctx'] for r in p5):.3f}/{st.mean(r['heldout_loss_mean_target'] for r in p5):.3f}"
             f" & {st.mean(r['feat_std_mean'] for r in p5):.3f} & {st.mean(r['eff_rank'] for r in p5):.1f} & {st.mean(r['mean_pairwise_cos'] for r in p5):.2f} & "
             f"{st.mean(r['train_time_s'] for r in L5):.0f} \\\\\\midrule")
for label, d, tok in V:
    p, ttime = run_metrics(d)
    cells = [f"{100 * p['acc_' + c]:.1f}\\,{{\\scriptsize {dcol(100 * p['acc_' + c], bm[c], 2 * bs[c])}}}" for c in C]
    lines.append(f"{label} & {tok} & " + " & ".join(cells) +
                 f" & {p['heldout_loss']:.3f}/{p['heldout_loss_shuffled_ctx']:.3f}/{p['heldout_loss_mean_target']:.3f} & {p['feat_std_mean']:.3f} & "
                 f"{p['eff_rank']:.1f} & {p['mean_pairwise_cos']:.2f} & {ttime:.0f} \\\\")
    if label.startswith("\\textbf{scattered}"): lines.append(r"\midrule")
lines += [r"\bottomrule\end{tabular}}\end{table*}"]
open(f"{T}/tab_h1.tex", "w").write("\n".join(lines) + "\n")
# ---------------- noise table ----------------
b0 = base[0]; npb = b0["noisy_probe"]
n0 = load(f"{R}/noise_aug__ctx_noise/result.json") if os.path.exists(f"{R}/noise_aug__ctx_noise/result.json") else None
lines = [r"\begin{table}[t]\centering\small\setlength{\tabcolsep}{3.2pt}",
         r"\caption{The noise failure and two remedies (test accuracy, \%, 15k steps). Rows 3--4: the \emph{frozen} encoder (seed 0) with a \emph{mixed-trained} probe: "
         r"every epoch each probe-train image is clean or, with probability $\tfrac12$, a fresh $\sigma{=}0.15$ noisy copy; $(\mathrm{lr},\mathrm{wd})$ chosen on the mean of "
         r"clean and noisy validation accuracy. Row 5: pretraining with noise on the "
         r"context-encoder input only ($p{=}0.5$, $\sigma\sim U(0,0.2)$), evaluated with the standard clean-trained probe; its ablation ($p{=}0$) is row 1. "
         r"cos$_{.15}$: mean cosine between the clean and $\sigma{=}0.15$ features of the same test image.}",
         r"\label{tab:noise}",
         r"\begin{tabular}{lccccccc}\toprule",
         r"Encoder / probe & Clean & Occ 10\% & Occ 25\% & Occ 40\% & Noise .05 & Noise .15 & cos$_{.15}$ \\\midrule"]
cells = [fmt(*agg(pb, "acc_" + c)) for c in C]
lines.append("Baseline, clean probe (5 seeds) & " + " & ".join(cells) + f" & {st.mean(r['cos_noise0.15'] for r in pb):.2f} \\\\")
cells = [fmt(*agg(pr, "acc_" + c)) for c in C]
lines.append("Random init, clean probe (5 seeds) & " + " & ".join(cells) + f" & {st.mean(r['cos_noise0.15'] for r in pr):.2f} \\\\\\midrule")
for lab, k in (("Baseline (s0), \\emph{mixed-trained probe}", "pretrained | mixed"), ("Random init (s0), \\emph{mixed-trained probe}", "random init | mixed")):
    v = npb[k]; lines.append(f"{lab} & " + " & ".join(f"{100 * v['acc_' + c]:.1f}" for c in C) + " & -- \\\\")
lines.append(r"\midrule")
if n0:
    p = prot(n0); lines.append("\\textbf{Context-noise pretraining} (s0), clean probe & " + " & ".join(f"{100 * p['acc_' + c]:.1f}" for c in C) + f" & {p['cos_noise0.15']:.2f} \\\\")
lines += [r"\bottomrule\end{tabular}\end{table}"]
open(f"{T}/tab_noise.tex", "w").write("\n".join(lines) + "\n")
# numbers used in the prose
out = {"baseline5": {c: agg(pb, "acc_" + c) for c in C}, "random5": {c: agg(pr, "acc_" + c) for c in C},
       "1large5": {c: agg(p5, "acc_" + c) for c in C}, "val5": agg(pb, "val_acc"),
       "noisy_probe_base": {k: {c: 100 * v["acc_" + c] for c in C} for k, v in npb.items()},
       "ctx_noise": ({c: 100 * prot(n0)["acc_" + c] for c in C} | {"val": 100 * prot(n0)["val_acc"], "hl": prot(n0)["heldout_loss"],
                     "std": prot(n0)["feat_std_mean"], "rank": prot(n0)["eff_rank"], "t": n0["train_time_s"], "mem": n0["peak_mem_mb"],
                     "np": {k: {c: 100 * v["acc_" + c] for c in C} for k, v in n0["noisy_probe"].items()}}) if n0 else None,
       "base_time": [r["train_time_s"] for r in base], "base_mem": [r["peak_mem_mb"] for r in base]}
json.dump(out, open(f"{T}/numbers.json", "w"), indent=1)
print(json.dumps(out, indent=1))
