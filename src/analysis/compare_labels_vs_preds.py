# -*- coding: utf-8 -*-
"""
Visualise alignment between weak labels (ground truth) and model predictions (ensemble outputs),
and estimate/apply time offset between them.

Usage:
    # 原始对比（只估计 offset，不修正）
    python src/analysis/compare_labels_vs_preds.py --video Flapping_her_arms_KPuLA5LlVjg.mp4

    # 应用 offset 校正后再对齐
    python src/analysis/compare_labels_vs_preds.py --video Flapping_her_arms_KPuLA5LlVjg.mp4 --apply_offset
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import correlate

# === 自动定位项目根目录 ===
ROOT = Path(__file__).resolve().parents[2]


def encode_labels(series):
    """Convert categorical labels into integer codes (for plotting and correlation)."""
    unique = sorted(series.dropna().unique().tolist())
    mapping = {lab: i for i, lab in enumerate(unique)}
    return series.map(mapping), mapping


def estimate_offset(gt_series, pred_series):
    """Estimate the frame offset using cross-correlation."""
    gt_num = np.array(gt_series.fillna(0), dtype=int)
    pred_num = np.array(pred_series.fillna(0), dtype=int)
    n = min(len(gt_num), len(pred_num))
    gt_num, pred_num = gt_num[:n], pred_num[:n]

    corr = correlate(gt_num - gt_num.mean(), pred_num - pred_num.mean(), mode="full")
    lags = np.arange(-len(gt_num) + 1, len(pred_num))
    lag = lags[np.argmax(corr)]
    return lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Video name, e.g. Flapping_her_arms_KPuLA5LlVjg.mp4")
    ap.add_argument("--pred_csv", default=str(ROOT / "runs_dl" / "hybrid_len64_clean_finetune" / "preds_ensemble.csv"))
    ap.add_argument("--label_csv", default=str(ROOT / "data" / "labels" / "weak_labels.csv"))
    ap.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    ap.add_argument("--apply_offset", action="store_true", help="Apply estimated offset correction to align sequences")
    args = ap.parse_args()

    stem = Path(args.video).stem
    print(f"🔍 Comparing labels vs predictions for: {stem}")

    # === 载入标签 ===
    df_labels = pd.read_csv(args.label_csv)
    df_labels = df_labels[df_labels["video"].astype(str).str.contains(stem, na=False)]
    if len(df_labels) == 0:
        raise FileNotFoundError(f"No weak labels found for {stem}")
    print(f"✅ Loaded {len(df_labels)} label segments")

    # === 载入预测 ===
    df_pred = pd.read_csv(args.pred_csv)
    if "video" in df_pred.columns:
        df_pred = df_pred[df_pred["video"].astype(str).str.contains(stem, na=False)]
    if len(df_pred) == 0:
        raise FileNotFoundError(f"No predictions found for {stem}")
    print(f"✅ Loaded {len(df_pred)} predicted windows")

    # === 构造时间轴 ===
    max_frame = max(df_labels["end_frame"].max(), len(df_pred) * 64)
    t = np.arange(0, max_frame)

    # === 构造标签帧序列 ===
    label_series = pd.Series(np.nan, index=t)
    for _, row in df_labels.iterrows():
        start, end = int(row.start_frame), int(row.end_frame)
        label_series[start:end] = row.label

    # === 构造预测帧序列 ===
    pred_series = pd.Series(np.nan, index=t)
    if "start_frame" in df_pred.columns and "end_frame" in df_pred.columns:
        for _, row in df_pred.iterrows():
            start, end = int(row.start_frame), int(row.end_frame)
            pred_series[start:end] = row.pred_label
    else:
        # 兼容旧格式（按窗口填充）
        for i, row in enumerate(df_pred.itertuples()):
            start = i * 64
            end = min(start + 64, len(t))
            pred_series[start:end] = row.pred_label

    # === 编码标签以便可视化 ===
    label_encoded, map_label = encode_labels(label_series)
    pred_encoded, _ = encode_labels(pred_series)

    # === 估计偏移量 ===
    offset_frames = estimate_offset(label_encoded.fillna(0), pred_encoded.fillna(0))
    offset_time = offset_frames / args.fps
    print(f"📏 Estimated offset ≈ {offset_frames} frames ({offset_time:.2f} s)")

    # === 应用偏移校正 ===
    if args.apply_offset and offset_frames != 0:
        if offset_frames > 0:
            pred_encoded = pred_encoded.shift(offset_frames)
        else:
            label_encoded = label_encoded.shift(-offset_frames)
        print(f"🔄 Applied offset correction: shifted by {offset_frames} frames")

    # === 计算重叠率 ===
    n = min(len(label_encoded), len(pred_encoded))
    valid = (~label_encoded.isna()) & (~pred_encoded.isna())
    match = (label_encoded[valid] == pred_encoded[valid]).sum()
    overlap_ratio = match / valid.sum() if valid.sum() > 0 else 0
    print(f"📊 Overlap ratio: {overlap_ratio*100:.2f}%")

    # === 绘制时间对齐图 ===
    plt.figure(figsize=(12, 3))

    # 先画预测，再画标签，避免标签被覆盖
    plt.plot(pred_encoded, color="orange", lw=2, alpha=0.7, label="Predictions (Hybrid CNN-LSTM)")
    plt.plot(label_encoded, color="green", lw=2, linestyle="--", label="Weak labels")

    title = f"Alignment — {stem}\nOffset ≈ {offset_frames} frames ({offset_time:.2f}s), Overlap={overlap_ratio * 100:.1f}%"
    if args.apply_offset:
        title += " (offset corrected)"
    plt.title(title)
    plt.xlabel("Frame index")
    plt.ylabel("Class code")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_dir = ROOT / "runs_dl" / "alignment_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alignment_{stem}.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✅ Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
