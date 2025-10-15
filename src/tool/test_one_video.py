import subprocess
import sys
import json
import numpy as np
from pathlib import Path

# project paths
ROOT = Path(__file__).resolve().parents[2]  # .../Flapping/
SRC = ROOT / "src" / "preprocessing"
DATA_RAW = ROOT / "data" / "raw_videos"
DATA_OUT = ROOT / "data" / "processed_angles"

def run_one(video_name: str):
    video_path = DATA_RAW / video_name
    out_path = DATA_OUT / (Path(video_name).stem + ".npy")

    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    # call extract_angles.py
    cmd = [
        sys.executable, str(SRC / "extract_angles.py"),
        "--video", str(video_path),
        "--out", str(out_path)
    ]
    print(f"▶️ Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # check log.json
    log_path = out_path.with_suffix(".log.json")
    if log_path.exists():
        with open(log_path, "r") as f:
            log = json.load(f)
        print("\n📊 Video log:")
        for k, v in log.items():
            print(f"   {k}: {v}")
    else:
        print("⚠️ No log.json found.")

    # check wrist height distribution
    if out_path.exists():
        data = np.load(out_path)
        wr_l = data[:, 6]
        wr_r = data[:, 7]
        print("\n✋ Wrist relative height stats:")
        print(f"   Left  wr_l_rel: min={np.nanmin(wr_l):.3f}, max={np.nanmax(wr_l):.3f}")
        print(f"   Right wr_r_rel: min={np.nanmin(wr_r):.3f}, max={np.nanmax(wr_r):.3f}")
    else:
        print("⚠️ No .npy file found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/utils/test_one_video.py \"Flapping her arms [KPuLA5LlVjg].mp4\"")
    else:
        run_one(sys.argv[1])
