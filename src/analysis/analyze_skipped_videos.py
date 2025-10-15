import pandas as pd
import json
from pathlib import Path

# paths
processed_dir = Path("../../data/processed_angles")
features_path = "../../data/window_features/all_window_features.csv"
out_path = "../../data/analysis/skipped_videos_report.csv"

# load processed npy files
processed_videos = [p.stem for p in processed_dir.glob("*.npy")]

# load features csv (kept videos)
df = pd.read_csv(features_path)
kept_videos = [Path(v).stem for v in df["video"].unique()]

# find skipped
skipped = sorted(set(processed_videos) - set(kept_videos))

records = []
for vid in skipped:
    log_path = processed_dir / f"{vid}.log.json"
    reason = "unknown"
    frames_total = None
    frames_valid = None
    valid_rate = None

    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        frames_total = log.get("frames_total", None)
        frames_valid = log.get("frames_valid", None)
        valid_rate = log.get("valid_rate", None)

        if frames_total is not None:
            if frames_total < 300:
                reason = "short_video (too few frames to form windows)"
            elif valid_rate is not None and valid_rate < 0.2:
                reason = "pose_detection_failed (too many NaN frames)"
            elif frames_valid and frames_valid > 0:
                reason = "low_motion (pose detected but movement too weak)"
            else:
                reason = "unknown"
    else:
        reason = "no_log_json"

    records.append({
        "video": vid,
        "frames_total": frames_total,
        "frames_valid": frames_valid,
        "valid_rate": valid_rate,
        "reason": reason
    })

# save report
df_report = pd.DataFrame(records)
df_report.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"✅ Skipped video report saved to {out_path}")
