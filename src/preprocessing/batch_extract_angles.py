import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

# Resolve repo root from this file: .../Flapping/src/preprocessing/batch_extract_angles.py
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]            # .../Flapping
SRC_DIR   = THIS_FILE.parent                # .../Flapping/src/preprocessing
EXTRACT   = SRC_DIR / "extract_angles_fixnan.py"   # absolute path to extract script

RAW_DIR = REPO_ROOT / "data" / "raw_videos"
OUT_DIR = REPO_ROOT / "data" / "processed_angles"

# Allowed extensions (case-insensitive)
ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

def collect_videos():
    vids = []
    # search recursively under data/raw_videos/**
    for p in RAW_DIR.glob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            vids.append(p)
    return sorted(vids)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = collect_videos()
    if not videos:
        print(f"❌ No videos found.\n"
              f"   Looked under: {RAW_DIR}\n"
              f"   Allowed types: {sorted(ALLOWED_EXTS)}")
        return

    print(f"Repo root     : {REPO_ROOT}")
    print(f"Input folder  : {RAW_DIR}")
    print(f"Output folder : {OUT_DIR}")
    print(f"Found {len(videos)} video(s).\n")

    for video in tqdm(videos, desc="Processing videos", unit="file"):
        out_file = OUT_DIR / (video.stem + ".npy")
        # Use the same Python interpreter running this script
        cmd = [
            sys.executable, str(EXTRACT),
            "--video", str(video),
            "--out", str(out_file)
        ]
        # Propagate non-zero return codes
        subprocess.run(cmd, check=True)

    print(f"\n✅ All done. Features saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()
