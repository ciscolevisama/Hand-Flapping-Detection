import pandas as pd
import numpy as np
import os
import json

# Paths
in_path = "../../data/window_features/all_window_features.csv"
out_dir = "../../data/analysis"
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(in_path):
    raise FileNotFoundError(f"❌ Input file not found: {in_path}. Run extract_window_features.py first.")

# Load data
df = pd.read_csv(in_path)

# Extra safety: drop low-quality windows
if "valid_ratio" in df.columns:
    before = len(df)
    df = df[df["valid_ratio"] >= 0.8].copy()
    after = len(df)
    print(f"⚠️ Dropped {before - after} low-quality windows (valid_ratio < 0.8)")

if df.empty:
    raise ValueError("❌ No valid windows remain after filtering. Check your data.")

stats = {}

# === Amplitude thresholds ===
stats["AMPLITUDE_THRESHOLD"] = float(np.percentile(np.hstack([df["A_left"], df["A_right"]]), 75))
stats["AMPLITUDE_MIN"] = float(np.percentile(np.hstack([df["A_left"], df["A_right"]]), 25))  # movement baseline

# === Frequency threshold ===
stats["FREQ_THRESHOLD"] = float(np.percentile(df["freq"], 75))

# === Symmetry thresholds ===
stats["DELTA_A_THRESHOLD"] = float(np.percentile(df["delta_A"], 75))
stats["PHASE_THRESHOLD"] = 45.0  # placeholder until phase diff is computed

# === Single-hand detection ===
stats["SINGLE_SIDE_RATIO"] = 0.7  # default value, can be tuned

# === Position thresholds ===
all_H = np.hstack([df["H_left"], df["H_right"]])
stats["E_HIGH"] = float(np.percentile(all_H, 75))  # typical high (above shoulder)
stats["E_LOW"] = float(np.percentile(all_H, 25))   # typical low (close to body)

# Save descriptive stats
stats_path = os.path.join(out_dir, "window_feature_stats.csv")
df.describe().to_csv(stats_path)

# === Add corrected thresholds (人工建议值) ===
corrected = stats.copy()
# 人工修正：让标签更符合 taxonomy 定义
corrected["DELTA_A_THRESHOLD"] = 20.0   # 对称/不对称分界，更严格
corrected["E_HIGH"] = 0.2              # 举过肩膀
corrected["E_LOW"] = 0.0             # 明显低于肩膀

out_data = {
    "raw": stats,
    "corrected": corrected
}

# Save thresholds (for rule-based labeling)
json_path = os.path.join(out_dir, "recommended_thresholds_window.json")
with open(json_path, "w") as f:
    json.dump(out_data, f, indent=4)

print(f"✅ Saved stats to {stats_path}")
print(f"✅ Recommended thresholds saved to {json_path}")
print("📊 Suggested raw values:")
for k, v in stats.items():
    print(f"   {k}: {v:.2f}")
print("📊 Corrected thresholds ready for segment_and_label.py")
