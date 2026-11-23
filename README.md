# Tiny I-JEPA vision research: masking geometry and noise robustness on CIFAR-10

This is a controlled study of Tiny I-JEPA ([keon/jepa](https://github.com/keon/jepa) `ijepa.py` @ `d55bab7b`, MIT) on
CIFAR-10. We pretrain on 10k unlabelled images, then probe the frozen context encoder linearly. We evaluate it on
clean images, under occlusion (10/25/40%) and under Gaussian noise (σ = 0.05/0.15).

The report is in [`report/main.pdf`](report/main.pdf); its LaTeX source is in `report/`. The Colab notebook
[`notebooks/experiment.ipynb`](notebooks/experiment.ipynb) runs the whole study top to bottom: setup, CPU smoke test,
tests, data splits, stress tests, baseline / proposed / ablation pretraining, linear-probe evaluation, robustness
plots and findings.

Everything runs on any machine with Python 3 and PyTorch, on one GPU or, slowly, on CPU. Nothing depends on a
particular cluster or scheduler.

## Repository layout

```
tiny-jepa-vision-research/
|-- README.md
|-- models/modified_ijepa.py   upstream ijepa.py, only train() changed (step-limited, mask strategies, context noise)
|-- models/masks.py            the mask strategies of the report (H1), selected by configs/masks/*.yaml
|-- data/splits.py             fixed CIFAR-10 splits (10k pretrain / 2k probe-train / 1k val / 2k test)
|-- data/corruptions.py        fixed-seed occlusion and Gaussian-noise test conditions
|-- train.py                   pretrain from a YAML config
|-- evaluate.py                frozen-encoder linear probe, robustness, held-out loss, collapse metrics
|-- configs/{baseline,proposed,ablation,...}.yaml   experiments (one per run in the report)
|-- configs/masks/*.yaml       one file per mask strategy with its default arguments
|-- tests/{test_masks,test_ema,test_model,test_eval}.py
|-- results/                   splits.json, metrics.csv, report_runs/ (outputs behind every table of the report)
|-- notebooks/experiment.ipynb the Colab notebook (runnable top to bottom in a fresh runtime)
|-- report/                    technical report (LaTeX) and the scripts that build its tables and figures
`-- scripts/                   train.sh, evaluate.sh (baseline by default), plot_run.py (training graphs of a run)
```

### What changed relative to upstream

`models/modified_ijepa.py` is a copy of upstream `ijepa.py`. The classes `Encoder` and `Predictor`, the functions
`sample_ijepa_masks`, `ema_update`, `lr_warmup_cosine` and `param_groups`, and the loss formula are byte-identical.
To see every change, run:

```bash
diff ../jepa/ijepa.py models/modified_ijepa.py
```

Only `train()` changed:

1. **Step-limited training.** Upstream runs epochs over all 50k images. Here training runs a fixed number of
   optimizer steps over the given dataset, which is the fixed 10k pretraining split. A seeded reshuffle starts each
   epoch, and the warmup-cosine LR and the linear EMA schedule (0.996 → 1) span the step budget.
2. **Pluggable masks.** A `mask_fn(B, grid, rng)` from `models/masks.py` replaces the call to `sample_ijepa_masks`.
   With the `upstream` strategy the masks are exactly the original ones (this is tested). The loss is computed per
   target set, i.e. per target block, and averaged over the sets, exactly as upstream.
3. **Context noise (H3, optional).** `add_context_noise` noises only the context-encoder input. The target encoder
   still sees the clean image.

Context/target separation, stop-gradient on the target branch, and EMA-only target updates are all unchanged, and
the tests cover them.

## Setup

**Colab:** [open `notebooks/experiment.ipynb` in Colab](https://colab.research.google.com/github/alfahimmohammad/tiny-jepa-vision-research/blob/main/notebooks/experiment.ipynb)
and run all cells. The notebook clones this repository and upstream `keon/jepa` at commit `d55bab7b`, and installs `pyyaml`
and `pytest`; torch, torchvision, numpy, pandas and matplotlib are preinstalled on Colab. `STEPS = 500` is the
assignment budget and `STEPS = 15000` reproduces the report. Without a GPU the notebook runs a short CPU
demonstration instead.

**Locally**, install with `pip install -r requirements.txt`. CIFAR-10 is downloaded to `--data_root`, which
defaults to `./cifar_data`.

## Usage

Baseline (the unmodified Tiny I-JEPA), training then evaluation:

```bash
bash scripts/train.sh                                 # configs/baseline.yaml, seed 0, 15k steps -> runs/baseline_s0/
bash scripts/evaluate.sh                              # evaluates runs/baseline_s0 (+ training graphs)

bash scripts/train.sh configs/baseline.yaml 0 500     # assignment budget: 500 steps -> runs/baseline_s0_500steps/
bash scripts/evaluate.sh runs/baseline_s0_500steps
```

The same scripts run every other experiment, e.g. `bash scripts/train.sh configs/proposed.yaml 1`. They read
`DATA_ROOT` (default `./cifar_data`; CIFAR-10 is downloaded there), `RUNS` (default `./runs`), `METRICS_CSV`
(default `results/metrics.csv`) and `PYTHON` from the environment. The scripts only call:

```bash
python -m pytest -q                                   # 49 tests, CPU, ~12 s
python train.py --config configs/baseline.yaml        # -> runs/baseline_s0/ (uses a GPU if available)
python evaluate.py --run_dir runs/baseline_s0         # -> runs/baseline_s0/eval.json, appends results/metrics.csv
python train.py --config configs/proposed.yaml --seed 1 --steps 500   # seed / steps / data_root / out_dir from the CLI
```

**Outputs** in `runs/<name>_s<seed>/`:

- `ckpt_final.pt`: the final models, plus the step-0 initialisation, which serves as the random-encoder control;
- `train_log.csv`: per-step loss, lr, EMA momentum and token counts;
- `eval_curve.csv`: fixed-probe metrics every `eval_every` steps;
- `summary.json`: parameters, training time, peak memory and hardware;
- after `evaluate.sh`: `eval.json` and the training graphs `training.png`, `metrics.png`, `robustness.png`.

`evaluate.py` runs the protocol on both the pretrained encoder and its random initialisation. It writes one row per
encoder to `results/metrics.csv`.

## Configs

An experiment config sets the training arguments and points to a mask-strategy file. The strategy file supplies the
strategy's default arguments, so an experiment only names the file:

```yaml
# configs/proposed.yaml
base: baseline.yaml          # inherit every key not set here
name: proposed
mask: masks/1_large.yaml     # strategy + its default arguments
```

```yaml
# configs/masks/1_large.yaml
strategy: blocks
ctx_size: 20
tgt_size: 32
n_targets: 1
tgt_scale: [0.52, 0.62]
tgt_aspect: [0.75, 1.5]
ctx_scale: [0.85, 1.0]
overlap: true
```

`mask_args:` overrides single arguments of the strategy file, e.g. `mask_args: {max_overlap_frac: 0.5}`.

| Experiment config | Mask file (strategy) | Context / target tokens per image | Role |
|---|---|---|---|
| `baseline.yaml` | `upstream.yaml` (upstream) | ≈20 (batch minimum) / 4×≈11 | original model |
| `proposed.yaml` | `1_large.yaml` (blocks) | 20 / 32 | one large contiguous target block of 35–42 patches (H1) |
| `ablation.yaml` | `scattered.yaml` (scattered) | 20 / 6–16 blocks, 32–36 | same context, targets in many small blocks |
| `2_large_overlap.yaml` | `2_large_overlap.yaml` (blocks) | 20 / 2×16 | two large blocks (16–30 patches), may overlap |
| `2_large_nonoverlap.yaml` | `2_large_nonoverlap.yaml` (blocks) | 20 / 2×16 | two large disjoint blocks (16–20 patches) |
| `overlap_frac.yaml` | `overlap_frac.yaml` (overlap_frac) | ≤20 (batch minimum, mean 16.8) / 4×8 | 4 blocks, pairwise overlap < 25% |
| `scattered_var5.yaml` | `scattered_var5.yaml` (scattered) | batch minimum / 5 whole blocks, 10–30 | scattered without token budgets (varying counts; not in the report) |
| `ctx_noise.yaml` | `upstream.yaml` | as baseline | noise on the context-encoder input (H3) |

`n×k`: n target blocks, each predicted as its own set of k tokens (as upstream I-JEPA); `per_block: false` in a mask file
predicts one set drawn from the union of the blocks instead (the first version of these samplers). Each strategy is
documented in its YAML file, in the docstrings of `models/masks.py`, and in Sec. 4 / Table 2 of the report (block
shapes, placement and context rules). `results/h1_perblock/` compares the two versions (training graphs and final
metrics).

## Evaluation protocol (`evaluate.py`)

- **Splits.** All splits are fixed and fingerprinted; the fingerprint is `f8c34b6bbf7f644f`, saved in
  `results/splits.json`.
- **Corruptions.** They are applied in [0, 1] pixel space, before normalisation, with fixed seeds 1000–1011.
- **Probe.**
  - The frozen context encoder sees all patches and its tokens are mean-pooled. The features are standardised with
    probe-train statistics.
  - A linear probe is trained with AdamW for 100 epochs. (lr, wd) is selected on the 1k validation images only.
  - The clean-trained probe is then applied to every test condition. The test set is never used for selection.
- **Diagnostics:**
  - random-init control;
  - oracle probe, fit on corrupted probe-train images;
  - held-out latent loss on fixed validation images and masks: with the real context, with another image's context,
    and for a content-free per-position mean target;
  - mean per-feature std, effective rank, top-1 singular-value share, and mean pairwise cosine;
  - a probe trained on a mix of clean and noisy images (σ = 0.15): every epoch each probe-train image is clean or,
    with probability ½, a fresh noisy copy; (lr, wd) selected on the mean of clean and noisy validation accuracy.

## Results

Every number in the report comes from `results/report_runs/` and `results/h1_perblock/runs/` (15k steps on one
L40S). The tables and figures are rebuilt from these files with `python report/scripts/make_tables.py`,
`makefigs.py`, `fig_h1_mech.py` and `fig_appendix_curves.py` (see `report/README.md`); checkpoints are not stored
in the repository. Headline results, test accuracy (%) with the validation-selected probe:

| | clean | occ 25% | occ 40% | noise 0.15 |
|---|---|---|---|---|
| baseline (5 seeds) | 48.9 ± 0.4 | 37.3 ± 0.8 | 28.9 ± 0.4 | 22.2 ± 1.0 |
| proposed: 1_large (5 seeds) | 49.9 ± 0.3 | 39.6 ± 1.3 | 31.7 ± 0.5 | 27.5 ± 2.3 |
| ablation: scattered (seed 0) | 50.1 | 38.2 | 30.3 | 17.0 |
| context noise (seed 0) | 47.5 | — | — | 41.9 |

`results/partA/` holds the earlier Part A outputs, from the first pipeline (500/5,000-step baselines and the
supervised reference).

## Reproducibility, attribution and AI disclosure

- **Seeds:** splits 0/1, corruptions 1000–1011 (oracle probe +100), probe 0, held-out masks 12345, pretraining 0
  (1–4 for the seed studies), context noise 50000 + seed.
- **Development notebooks.** The report's runs were produced by development notebooks that are not part of this
  repository. The repository code uses the same seeded random streams for the mask samplers and the data order,
  so its masks are identical bit for bit, and `evaluate.py` reproduces the stored metrics of those runs.
- **GPU nondeterminism.** Runs can still differ slightly between GPUs, because some kernels are nondeterministic.
- **Upstream code:** keon/jepa (MIT). The model code is copied verbatim into `models/modified_ijepa.py`.
- **AI assistance:** an AI coding assistant (Claude, Anthropic) was used to build the mask strategies
  (`models/masks.py`), the helper functions (data splits, corruptions, configs, plotting and report scripts) and
  the downstream evaluation (`evaluate.py`), under the author's direction. All experiments, design decisions and
  conclusions were checked by the author.
