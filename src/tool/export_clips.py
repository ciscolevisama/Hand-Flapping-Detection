import os
import cv2
import pandas as pd
import argparse

def export_clip(video_path, start, end, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start <= frame_idx < end:
            out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print(f"   ✅ Saved {out_path}")


def main(args):
    df = pd.read_csv(args.labels)

    # 过滤目标类别
    target_df = df[df["label"].isin(args.classes.split(","))]

    if args.limit > 0:
        target_df = target_df.sample(args.limit, random_state=42)

    os.makedirs(args.out_dir, exist_ok=True)

    for _, row in target_df.iterrows():
        video_name = row["video"]
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        label = row["label"]

        video_path = os.path.join(args.video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            print(f"⚠️ Video file not found: {video_path}")
            continue

        out_name = f"{video_name}_{start}_{end}_{label}.mp4"
        out_path = os.path.join(args.out_dir, out_name)

        export_clip(video_path, start, end, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to weak_labels.csv")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory with original .mp4 videos")
    parser.add_argument("--out_dir", type=str, default="check_clips",
                        help="Directory to save exported clips")
    parser.add_argument("--classes", type=str, default="left_only_low,right_only_low",
                        help="Comma-separated list of classes to export")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max number of clips per run (0 = no limit)")
    args = parser.parse_args()
    main(args)
