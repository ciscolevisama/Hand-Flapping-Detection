# -*- coding: utf-8 -*-
"""
Quick diagnostic: check alignment between raw video, extracted features, and weak labels.
Usage:
    python src/analysis/check_alignment.py --video Flapping_her_arms_KPuLA5LlVjg.mp4
"""

import os, argparse, json
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === 自动定位项目根目录 ===
ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Video file name (e.g. Flapping_her_arms_KPuLA5LlVjg.mp4)")
    ap.add_argument("--label_csv", default=str(ROOT / "data" / "labels" / "weak_labels.csv"))
    ap.add_argument("--proc_dir", default=str(ROOT / "data" / "processed_angles"))
    args = ap.parse_args()

    video_name = args.video
    stem = Path(video_name).stem

    # === 1️⃣ 视频帧数与FPS ===
    video_path = ROOT / "data" / "raw_videos" / video_name
    if not video_path.exists():
        raise FileNotFoundError(f"❌ Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    print(f"\n🎞️  Video: {video_name}")
    print(f"   - Frames: {frame_count}")
    print(f"   - FPS: {fps:.2f}")
    print(f"   - Duration: {duration:.2f} s")

    # === 2️⃣ 对应特征帧数 ===
    feat_path = Path(args.proc_dir) / f"{stem}.npy"
    if not feat_path.exists():
        print(f"⚠️  Feature file not found: {feat_path}")
        feat_count = 0
    else:
        feat = np.load(feat_path)
        feat_count = feat.shape[0]
        print(f"   - Extracted feature frames: {feat_count}")

    # === 帧数差异分析 ===
    if frame_count > 0 and feat_count > 0:
        diff = frame_count - feat_count
        ratio = feat_count / frame_count
        print(f"   - Feature/Video ratio: {ratio * 100:.2f}%")
        if ratio < 0.9:
            print("⚠️  Large mismatch detected (possible NaN frame drops during extraction).")

    # === 3️⃣ 标签区间检查 ===
    df = pd.read_csv(args.label_csv)
    if "video" not in df.columns:
        raise ValueError("❌ weak_labels.csv missing 'video' column.")
    df_v = df[df["video"].str.contains(stem, na=False)]
    if len(df_v) == 0:
        print(f"⚠️  No labels found for {stem} in weak_labels.csv")
    else:
        print("\n🧾 Weak labels for this video:")
        print(df_v[["start_frame", "end_frame", "label", "confidence"]])
        label_starts = df_v["start_frame"].min()
        label_ends = df_v["end_frame"].max()
        print(
            f"   - Labelled range: {label_starts}–{label_ends} (duration ≈ {(label_ends - label_starts) / fps:.2f} s)")

        # 如果标签超过特征帧范围，说明错位
        if label_ends > feat_count:
            print(f"⚠️  Label end ({label_ends}) > feature length ({feat_count}) → misalignment likely.")

    # === 4️⃣ 简单时间轴可视化 ===
    if feat_count > 0:
        plt.figure(figsize=(8, 1))
        plt.hlines(1, 0, frame_count, color="gray", linewidth=6, label="Video frames")
        plt.hlines(1.1, 0, feat_count, color="orange", linewidth=6, label="Extracted features")
        if len(df_v) > 0:
            for _, row in df_v.iterrows():
                plt.hlines(1.2, row["start_frame"], row["end_frame"], color="green", linewidth=6)
        plt.legend()
        plt.xlabel("Frame index")
        plt.yticks([])
        plt.title(f"Alignment check: {stem}")
        plt.tight_layout()
        plt.show()

    print("\n✅ Check complete.")
    print("   If feature length << video frames → Mediapipe dropped many frames.")
    print("   If label end > feature length → labels misaligned.")
    print("   If labels appear shifted → use offset=32–64 in visualize_prediction.")


if __name__ == "__main__":
    main()
