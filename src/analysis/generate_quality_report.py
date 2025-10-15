import os
import json
import pandas as pd
from pathlib import Path

# Paths
angles_dir = "../../data/processed_angles"
window_dir = "../../data/window_features"
analysis_dir = "../../data/analysis"
os.makedirs(analysis_dir, exist_ok=True)

# 1. Collect frame-level stats from extract_angles logs (if available)
#    Assume each .npy has a companion .log.json with total/valid counts
frame_stats = {}
for file in os.listdir(angles_dir):
    if file.endswith(".log.json"):
        path = os.path.join(angles_dir, file)
        with open(path, "r") as f:
            info = json.load(f)
        video = file.replace(".log.json", "")
        frame_stats[video] = info

# 2. Collect window-level stats
window_stats = {}
for file in os.listdir(window_dir):
    if file.endswith("_features.csv") and file != "all_window_features.csv":
        path = os.path.join(window_dir, file)
        df = pd.read_csv(path)
        video = file.replace("_features.csv", "")
        window_stats[video] = {"num_windows": len(df)}

# 3. Merge both stats
rows = []
videos_all = set(frame_stats.keys()) | set(window_stats.keys())

for video in sorted(videos_all):
    frames_total = frame_stats.get(video, {}).get("frames_total", None)
    frames_valid = frame_stats.get(video, {}).get("frames_valid", None)
    valid_rate = (
        0 if not frames_total else frames_valid / frames_total
    ) if frames_total else None
    num_windows = window_stats.get(video, {}).get("num_windows", 0)

    rows.append({
        "video": video,
        "frames_total": frames_total,
        "frames_valid": frames_valid,
        "valid_rate": valid_rate,
        "num_windows": num_windows
    })

# 4. Save report
df_report = pd.DataFrame(rows)
out_path = os.path.join(analysis_dir, "data_quality_report.csv")
df_report.to_csv(out_path, index=False)
print(f"✅ Quality report saved to {out_path}")
