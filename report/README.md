# Technical report (LaTeX)

Overleaf-ready: upload this folder and compile `main.tex` with pdfLaTeX + BibTeX (`refs.bib`, style `unsrtnat`).

* `main.tex` includes `text_*.tex` (analysis sections, appendix) and `tables/*.tex`.
* `tables/*.tex` and `figures/*` are **generated** from the run results stored in the repository; do not edit them by hand.
  Run from the repository root:
  * tables: `python report/scripts/make_tables.py` (also writes `tables/numbers.json`);
  * figures: `makefigs.py` (baseline / noise curves), `maskfig.py` (mask examples), `fig_h1_mech.py` (H1 mechanism),
    `fig_appendix_curves.py` (appendix training curves), `perblock_compare.py` (`results/h1_perblock/`);
  * diagnostics quoted in the text: `traj.py` (loss/rank trajectory, Spearman), `offset.py` (shared-offset analysis;
    needs the run checkpoints: `--ckpts DIR`).
* Inputs: `results/report_runs/` (result.json, train_log.csv, eval_curve.csv of each run; `seeds_1_large/` for the
  1_large seed study) and `results/h1_perblock/runs/` (the per-block runs written by `train.py` / `evaluate.py`).
* Any run can be repeated with the repository code: `bash scripts/train.sh configs/<experiment>.yaml [seed]`, then
  `bash scripts/evaluate.sh runs/<name>_s<seed>`.
