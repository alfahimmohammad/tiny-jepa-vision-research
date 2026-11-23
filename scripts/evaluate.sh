#!/usr/bin/env bash
# Evaluate a finished run with the fixed protocol (frozen context encoder -> mean pooling -> linear probe; (lr, wd)
# selected on the validation split; clean / occlusion 10-25-40% / noise 0.05-0.15 test accuracy; random-init control;
# held-out latent loss; collapse statistics). Default: the baseline run written by scripts/train.sh.
#
#   bash scripts/evaluate.sh                               # runs/baseline_s0
#   bash scripts/evaluate.sh runs/baseline_s0_500steps     # any run dir
#   bash scripts/evaluate.sh runs/baseline_s0 --no_mixed_probe   # extra flags are passed to evaluate.py
#
# Environment (optional): DATA_ROOT (default ./cifar_data), METRICS_CSV (rows appended, default results/metrics.csv),
# PYTHON (default python).
# Output: <run dir>/eval.json, one row per encoder (random_init, pretrained) appended to $METRICS_CSV,
# and the training graphs <run dir>/{training,metrics,robustness}.png.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=${1:-runs/baseline_s0}
DATA_ROOT=${DATA_ROOT:-./cifar_data}
METRICS_CSV=${METRICS_CSV:-results/metrics.csv}
PYTHON=${PYTHON:-python}

"$PYTHON" evaluate.py --run_dir "$RUN_DIR" --data_root "$DATA_ROOT" --metrics_csv "$METRICS_CSV" "${@:2}"
"$PYTHON" scripts/plot_run.py "$RUN_DIR"
