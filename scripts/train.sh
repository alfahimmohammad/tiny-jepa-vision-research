#!/usr/bin/env bash
# Pretrain Tiny I-JEPA from a YAML config (default: the unmodified baseline). Runs on any machine with
# Python + PyTorch (GPU used automatically if available, CPU otherwise).
#
#   bash scripts/train.sh                                  # baseline, seed 0, 15,000 steps (report budget)
#   bash scripts/train.sh configs/baseline.yaml 0 500      # baseline, seed 0, 500 steps (assignment budget)
#   bash scripts/train.sh configs/proposed.yaml 1          # any experiment config / seed
#
# Environment (optional): DATA_ROOT (CIFAR-10 download dir, default ./cifar_data), RUNS (output root, default ./runs),
# PYTHON (interpreter, default python).
# Output: $RUNS/<config name>_s<seed>[_<steps>steps]/ with ckpt_final.pt, train_log.csv, eval_curve.csv, summary.json
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/baseline.yaml}
SEED=${2:-0}
STEPS=${3:-}
DATA_ROOT=${DATA_ROOT:-./cifar_data}
RUNS=${RUNS:-./runs}
PYTHON=${PYTHON:-python}

NAME=$("$PYTHON" -c "import sys; from train import load_config; print(load_config(sys.argv[1])['name'])" "$CONFIG")
OUT="$RUNS/${NAME}_s${SEED}${STEPS:+_${STEPS}steps}"

"$PYTHON" train.py --config "$CONFIG" --seed "$SEED" ${STEPS:+--steps "$STEPS"} --data_root "$DATA_ROOT" --out_dir "$OUT"
echo "run dir: $OUT"
