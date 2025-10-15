import os
import sys
import pandas as pd
import numpy as np
import json

# 确保能导入 utils/segment_and_label.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from segment_and_label import label_window

# 路径
processed_dir = "../../data/processed_angles"
labels_file = "../../data/labels/weak_labels.csv"
out_dir = "../../data/analysis"
os.makedirs(out_dir, exist_ok=True)

# === 加载阈值（保持和 segment_and_label 一致） ===
thr_path = "../../data/analysis/recommended_thresholds_window.json"
with open(thr_path, "r") as f:
    thr_data = json.load(f)
THRESHOLDS = thr_data.get("corrected", thr_data)

# === 读取 weak labels ===
df = pd.read_csv(labels_file)

# === 重新用 label_window 判定一遍 ===
uncertain_flags = []
for idx, row in df.iterrows():
    video_id = row["video"]
    start, end = int(row["start_frame"]), int(row["end_frame"])
    npy_file = os.path.join(processed_dir, f"{video_id}.npy")
    if not os.path.exists(npy_file):
        uncertain_flags.append(False)
        continue

    data = np.load(npy_file)
    window = data[start:end]

    new_label, _, new_uncertain = label_window(window)  # 👈 三个返回值
    # 如果新判定和原标签不一致，或者 new_uncertain=True → 记为不确定
    uncertain_flags.append((new_label != row["label"]) or new_uncertain)

df["uncertain_check"] = uncertain_flags  # 避免和原来的 uncertain 字段冲突

# === 总体统计 ===
total_uncertain = df["uncertain_check"].sum()
rate_uncertain = total_uncertain / len(df) * 100
print(f"✅ Found {total_uncertain} uncertain windows out of {len(df)} ({rate_uncertain:.2f}%)")

# === 各类别统计 ===
summary = df.groupby("label")["uncertain_check"].agg(["count", "sum"])
summary["uncertain_rate"] = (summary["sum"] / summary["count"] * 100).round(2)

out_path = os.path.join(out_dir, "uncertain_label_stats.csv")
summary.to_csv(out_path)
print(f"✅ Saved per-class uncertainty stats to {out_path}")
print(summary)
