"""Pretrain Tiny I-JEPA from a YAML config on the fixed 10k unlabelled CIFAR-10 subset.

Usage:  python train.py --config configs/proposed.yaml [--seed 1] [--steps 500] [--data_root ./cifar_data]
Writes runs/<name>_s<seed>/: ckpt_final.pt (final + step-0 models), train_log.csv, eval_curve.csv, summary.json.
Then:   python evaluate.py --run_dir runs/<name>_s<seed>

Config files: configs/<experiment>.yaml sets the training arguments and names a mask strategy file
(`mask: masks/<strategy>.yaml`) whose arguments are the strategy's defaults; `mask_args:` overrides some of
them, and `base: <file>.yaml` inherits every key that the experiment does not set.
"""
import argparse
import csv
import json
import os

import torch
import yaml

import evaluate
from data.splits import PretrainSet, load_cifar, load_or_create_splits, pretrain_transform
from models.masks import mask_sampler
from models.modified_ijepa import MEAN, STD, train

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:                                   # inherit an experiment config
        cfg = {**load_config(os.path.join(CONFIG_DIR, cfg.pop("base"))), **cfg}
    if isinstance(cfg["mask"], str):                    # mask strategy file + its default arguments
        with open(os.path.join(CONFIG_DIR, cfg["mask"])) as f:
            cfg["mask_file"], cfg["mask"] = cfg["mask"], yaml.safe_load(f)
    cfg["mask"] = {**cfg["mask"], **(cfg.pop("mask_args", None) or {})}
    return cfg


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--data_root")
    ap.add_argument("--out_dir")
    ap.add_argument("--device")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg.update({k: v for k, v in (("seed", a.seed), ("steps", a.steps), ("data_root", a.data_root)) if v is not None})
    out_dir = a.out_dir or os.path.join(cfg["out_root"], f"{cfg['name']}_s{cfg['seed']}")
    os.makedirs(out_dir, exist_ok=True)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(cfg, indent=1))

    images, ytr, _, yte = load_cifar(cfg["data_root"])
    splits = load_or_create_splits(cfg["splits_path"], ytr, yte)
    dataset = PretrainSet(images, splits["indices"]["pretrain"], pretrain_transform(MEAN, STD))   # images only

    curve, on_eval = [], None
    if cfg["eval_every"]:                               # fixed-probe metrics during training (no selection)
        data = evaluate.EvalData(cfg["data_root"], cfg["splits_path"], device)

        def on_eval(step, ctx_enc, tgt_enc, pred):
            curve.append({"step": step, **evaluate.quick_eval(data, ctx_enc, tgt_enc, pred)})
            r = curve[-1]
            print(f"  [eval step {step}] val={r['val_acc']:.3f} clean={r['acc_clean']:.3f} "
                  f"occ40={r['acc_occ40']:.3f} noise.15={r['acc_noise0.15']:.3f} "
                  f"heldout_loss={r['heldout_loss']:.4f} rank={r['eff_rank']:.1f}", flush=True)

    out = train(dataset, steps=cfg["steps"], batch_size=cfg["batch_size"], lr=cfg["lr"], wd=cfg["wd"],
                ema_start=cfg["ema_start"], ema_end=cfg["ema_end"], mask_fn=mask_sampler(cfg["mask"]),
                ctx_noise=cfg["ctx_noise"], seed=cfg["seed"], num_workers=cfg["num_workers"], device=device,
                on_eval=on_eval, eval_every=cfg["eval_every"])

    ctx_enc, tgt_enc, pred = out["ctx_enc"], out["tgt_enc"], out["predictor"]
    torch.save({"ctx_enc": ctx_enc.state_dict(), "tgt_enc": tgt_enc.state_dict(), "predictor": pred.state_dict(),
                "init": out["init"], "config": cfg}, os.path.join(out_dir, "ckpt_final.pt"))
    write_csv(os.path.join(out_dir, "train_log.csv"), out["log"])
    if curve:
        write_csv(os.path.join(out_dir, "eval_curve.csv"), curve)
    summary = {"run": cfg["name"], "config_path": a.config, "seed": cfg["seed"], "steps": cfg["steps"],
               "batch_size": cfg["batch_size"], "splits_fingerprint": splits["fingerprint"],
               "params_trainable": sum(p.numel() for m in (ctx_enc, pred) for p in m.parameters()),
               "train_time_s": out["train_time_s"], "peak_mem_mb": out["peak_mem_mb"],
               "hardware": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
               "torch": torch.__version__, "config": cfg}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(f"done: {out['train_time_s']:.0f}s training, peak memory {out['peak_mem_mb'] or 0:.0f} MB -> {out_dir}")


if __name__ == "__main__":
    main()
