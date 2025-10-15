#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a small grid to find a good imbalance-handling setup for CNN.
It calls train_cnn.py with different balancing knobs and summarises results.

Outputs
- runs_dl/sweep/summary_sweep.csv     (overall ranking)
- per-run artifacts under runs_dl/sweep/<mode>/<run_tag>/
"""
import os, json, sys, itertools, subprocess, time
from pathlib import Path
import pandas as pd

PY = sys.executable  # use current venv Python
TRAIN = "src/training/dl/train_cnn.py"

# -------------------------------
# Global training defaults
# -------------------------------
SEQ_LEN   = 12
EPOCHS    = 30
LR        = 5e-4
CV        = 5
BATCH     = 128

# Modes to sweep (可按需删/加)
MODES = ["clean_only", "conf_weighted", "full"]

# -------------------------------
# Balancing grids (小而实用)
# -------------------------------
GRID = [
    # 1) 只采样（温和 & 稍强）
    dict(balancing="sampler", sampler_alpha=0.5, cw_beta=0.0, cw_clip=1.0, conf_alpha=1.0),
    dict(balancing="sampler", sampler_alpha=0.8, cw_beta=0.0, cw_clip=1.0, conf_alpha=1.0),

    # 2) 只类权重（温和 & 更软；并限幅）
    dict(balancing="class_weight", sampler_alpha=0.0, cw_beta=0.5, cw_clip=3.0, conf_alpha=1.0),
    dict(balancing="class_weight", sampler_alpha=0.0, cw_beta=0.3, cw_clip=2.0, conf_alpha=1.0),

    # 3) 二者皆用，但力度很软
    dict(balancing="both", sampler_alpha=0.5, cw_beta=0.3, cw_clip=2.0, conf_alpha=1.0),

    # 4) 对照：不做平衡
    dict(balancing="none", sampler_alpha=0.0, cw_beta=0.0, cw_clip=1.0, conf_alpha=1.0),
]

OUT_ROOT = Path("runs_dl") / "sweep"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_one(mode: str, cfg: dict, tag: str):
    outdir = OUT_ROOT / mode / tag
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, TRAIN,
        "--mode", mode,
        "--seq_len", str(SEQ_LEN),
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--cv", str(CV),
        "--batch_size", str(BATCH),
        "--outdir", str(outdir),

        "--balancing", cfg["balancing"],
        "--sampler_alpha", str(cfg["sampler_alpha"]),
        "--cw_beta", str(cfg["cw_beta"]),
        "--cw_clip", str(cfg["cw_clip"]),
        "--conf_alpha", str(cfg["conf_alpha"]),
    ]
    print("\n>>>", " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    (outdir / "stdout.txt").write_text(res.stdout)
    (outdir / "stderr.txt").write_text(res.stderr)
    if res.returncode != 0:
        print(f"[ERR] {tag} ({mode}) failed in {dt:.1f}s; see {outdir/'stderr.txt'}")
        return None
    print(f"[OK] {tag} ({mode}) done in {dt/60:.1f} min")
    return str(outdir)


def high_agg_f1(report_overall_json: str) -> float:
    if not os.path.isfile(report_overall_json): return float("nan")
    rep = json.load(open(report_overall_json, "r", encoding="utf-8"))
    keys = ["left_only_high", "right_only_high", "both_symmetric_high"]
    vals = []
    for k in keys:
        if k in rep and isinstance(rep[k], dict):
            vals.append(rep[k].get("f1-score", 0.0))
        else:
            vals.append(0.0)
    return sum(vals) / len(vals) if vals else float("nan")


def collect_one(outdir: str) -> dict:
    out = Path(outdir)
    metrics_csv = out / "metrics_cnn.csv"
    rep_json    = out / "report_overall.json"
    mode = out.parts[-2]
    tag  = out.parts[-1]

    row = {"mode": mode, "tag": tag, "outdir": str(out)}
    if metrics_csv.is_file():
        df = pd.read_csv(metrics_csv)
        row["f1_macro_mean"] = df["f1_macro"].mean()
        row["f1_macro_std"]  = df["f1_macro"].std()
        row["acc_mean"]      = df["accuracy"].mean()
        row["acc_std"]       = df["accuracy"].std()
    else:
        row["f1_macro_mean"] = row["f1_macro_std"] = row["acc_mean"] = row["acc_std"] = float("nan")
    row["high_agg_f1"] = high_agg_f1(str(rep_json))
    return row


def main():
    rows = []
    for mode in MODES:
        for i, cfg in enumerate(GRID, 1):
            tag = f"bal_{cfg['balancing']}_sa{cfg['sampler_alpha']}_cb{cfg['cw_beta']}_cc{cfg['cw_clip']}_ca{cfg['conf_alpha']}"
            outdir = run_one(mode, cfg, tag)
            if outdir:
                rows.append(collect_one(outdir))

    if not rows:
        print("No successful runs. Please check errors in runs_dl/sweep/**/stderr.txt")
        return

    df = pd.DataFrame(rows).sort_values(["mode","f1_macro_mean"], ascending=[True, False])
    df.to_csv(OUT_ROOT / "summary_sweep.csv", index=False)
    print("\n=== Sweep summary (top by mode) ===")
    for mode in MODES:
        d = df[df["mode"]==mode].head(5)
        if d.empty: continue
        print(f"\n[{mode}]")
        print(d[["tag","f1_macro_mean","high_agg_f1","acc_mean"]].round(4).to_string(index=False))
    print(f"\nSaved: {OUT_ROOT/'summary_sweep.csv'}")


if __name__ == "__main__":
    main()
