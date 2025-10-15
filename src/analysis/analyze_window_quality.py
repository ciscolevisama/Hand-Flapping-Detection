import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
features_path = "../../data/window_features/all_window_features.csv"
out_dir = "../../data/analysis"
Path(out_dir).mkdir(parents=True, exist_ok=True)

if not os.path.exists(features_path):
    raise FileNotFoundError(f"❌ File not found: {features_path}")

# Load data
df = pd.read_csv(features_path)

if "valid_ratio" not in df.columns:
    raise ValueError("❌ valid_ratio column not found. Did you run the updated extract_window_features.py?")

# --- 1. Per-video stats ---
stats = df.groupby("video").agg(
    num_windows=("video", "count"),
    mean_valid_ratio=("valid_ratio", "mean"),
    min_valid_ratio=("valid_ratio", "min"),
    max_valid_ratio=("valid_ratio", "max")
).reset_index()

stats_path = os.path.join(out_dir, "window_quality_stats.csv")
stats.to_csv(stats_path, index=False)
print(f"✅ Per-video stats saved to {stats_path}")

# --- 2. Global distribution of valid_ratio ---
plt.figure(figsize=(6, 4))
df["valid_ratio"].hist(bins=20)
plt.xlabel("Valid Ratio")
plt.ylabel("Number of Windows")
plt.title("Distribution of valid_ratio across all windows")
plt.grid(True, linestyle="--", alpha=0.5)

plot_path = os.path.join(out_dir, "valid_ratio_hist.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"✅ Histogram saved to {plot_path}")
