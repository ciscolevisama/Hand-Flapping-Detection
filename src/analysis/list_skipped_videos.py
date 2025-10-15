import pandas as pd
from pathlib import Path

# paths
processed_dir = Path("../../data/processed_angles")
features_path = "../../data/window_features/all_window_features.csv"
out_path = "../../data/analysis/skipped_videos.txt"

# load processed npy files
processed_videos = [p.stem for p in processed_dir.glob("*.npy")]

# load all_window_features.csv
df = pd.read_csv(features_path)
kept_videos = df["video"].unique().tolist()

# compute skipped
skipped = sorted(set(processed_videos) - set(Path(v).stem for v in kept_videos))

# save (UTF-8 to avoid GBK error)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== Skipped videos (no valid windows) ===\n")
    f.write(f"Total skipped: {len(skipped)}\n\n")
    for v in skipped:
        f.write(v + "\n")

print(f"✅ Skipped videos list saved to {out_path}")
