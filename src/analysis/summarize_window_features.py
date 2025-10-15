import pandas as pd
import json
from pathlib import Path

# paths
features_path = "../../data/window_features/all_window_features.csv"
log_dir = "../../data/processed_angles"
out_dir = "../../data/analysis"
Path(out_dir).mkdir(parents=True, exist_ok=True)

# load window features
df = pd.read_csv(features_path)

# --- 1. 每个视频的窗口数 ---
stats = df.groupby("video").size().reset_index(name="num_windows")

# --- 2. 加入 log.json 信息 ---
frames_total_list = []
frames_valid_list = []
valid_rate_list = []

for vid in stats["video"]:
    log_path = Path(log_dir) / f"{Path(vid).stem}.log.json"
    if log_path.exists():
        with open(log_path, "r") as f:
            log = json.load(f)
        frames_total_list.append(log.get("frames_total", None))
        frames_valid_list.append(log.get("frames_valid", None))
        valid_rate_list.append(log.get("valid_rate", None))
    else:
        frames_total_list.append(None)
        frames_valid_list.append(None)
        valid_rate_list.append(None)

stats["frames_total"] = frames_total_list
stats["frames_valid"] = frames_valid_list
stats["valid_rate"] = valid_rate_list

# --- 3. 基本统计 ---
summary = {
    "total_videos": stats.shape[0],
    "total_windows": stats["num_windows"].sum(),
    "mean_windows": stats["num_windows"].mean(),
    "median_windows": stats["num_windows"].median(),
    "min_windows": stats["num_windows"].min(),
    "max_windows": stats["num_windows"].max(),
}

# --- 4. Top/Bottom 视频 ---
top5 = stats.sort_values("num_windows", ascending=False).head(5)
low5 = stats.sort_values("num_windows", ascending=True).head(5)

# save per-video stats
stats_path = Path(out_dir) / "video_window_stats.csv"
stats.to_csv(stats_path, index=False)

# save summary
summary_path = Path(out_dir) / "window_features_summary.txt"
with open(summary_path, "w") as f:
    f.write("=== Global Summary ===\n")
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")
    f.write("\n=== Top 5 videos (most windows) ===\n")
    f.write(top5.to_string(index=False))
    f.write("\n\n=== Bottom 5 videos (fewest windows) ===\n")
    f.write(low5.to_string(index=False))

print(f"✅ Per-video stats saved to {stats_path}")
print(f"✅ Summary report saved to {summary_path}")
