# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

mp_pose = mp.solutions.pose
VIS_TH = 0.5  # visibility threshold

INPUT_DIR = r"D:\semester4\CD\code\Flapping\data\raw_videos"
OUTPUT_DIR = r"D:\semester4\CD\code\Flapping\data\debug_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def landmark_xy(lm, idx):
    p = lm[idx]
    return (p.x, p.y, getattr(p, "visibility", 1.0))


def analyse_video(video_path: Path):
    """Wrapper: 提取左右手腕相对高度，返回统计并画图"""
    cap = cv2.VideoCapture(str(video_path))
    pose = mp_pose.Pose()
    wr_l_rel_list, wr_r_rel_list = [], []

    frames_total = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_total += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if not results.pose_landmarks:
            wr_l_rel_list.append(np.nan)
            wr_r_rel_list.append(np.nan)
            continue
        lm = results.pose_landmarks.landmark

        # landmarks
        sh_lx, sh_ly, sh_lv = landmark_xy(lm, 11)
        sh_rx, sh_ry, sh_rv = landmark_xy(lm, 12)
        wr_lx, wr_ly, wr_lv = landmark_xy(lm, 15)
        wr_rx, wr_ry, wr_rv = landmark_xy(lm, 16)
        hip_lx, hip_ly, hip_lv = landmark_xy(lm, 23)
        hip_rx, hip_ry, hip_rv = landmark_xy(lm, 24)
        nose_x, nose_y, nose_v = landmark_xy(lm, 0)

        # check arm keypoints
        if min(sh_lv, sh_rv, wr_lv, wr_rv, nose_v) < VIS_TH:
            wr_l_rel_list.append(np.nan)
            wr_r_rel_list.append(np.nan)
            continue

        shoulder_c = [(sh_lx + sh_rx) / 2, (sh_ly + sh_ry) / 2]

        if hip_lv >= VIS_TH and hip_rv >= VIS_TH:
            hip_c = [(hip_lx + hip_rx) / 2, (hip_ly + hip_ry) / 2]
            torso_len = nose_y - hip_c[1]
        else:
            torso_len = nose_y - shoulder_c[1]

        if torso_len < 1e-3:
            wr_l_rel_list.append(np.nan)
            wr_r_rel_list.append(np.nan)
            continue

        wr_l_rel = (shoulder_c[1] - wr_ly) / torso_len
        wr_r_rel = (shoulder_c[1] - wr_ry) / torso_len
        wr_l_rel_list.append(wr_l_rel)
        wr_r_rel_list.append(wr_r_rel)

    cap.release()

    # 统计信息
    report = {
        "video": str(video_path),
        "frames_total": frames_total,
        "valid_ratio": 1 - (np.isnan(wr_l_rel_list).sum() / frames_total),
        "left_min": float(np.nanmin(wr_l_rel_list)),
        "left_max": float(np.nanmax(wr_l_rel_list)),
        "right_min": float(np.nanmin(wr_r_rel_list)),
        "right_max": float(np.nanmax(wr_r_rel_list)),
    }

    # 保存 JSON
    report_path = Path(OUTPUT_DIR) / f"{video_path.stem}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 保存图像
    plt.figure(figsize=(12, 4))
    plt.plot(wr_l_rel_list, label="Left wrist rel height")
    plt.plot(wr_r_rel_list, label="Right wrist rel height")
    plt.xlabel("Frame")
    plt.ylabel("Relative height (shoulder→torso normalized)")
    plt.legend()
    plt.title(video_path.stem)
    plt.tight_layout()
    fig_path = Path(OUTPUT_DIR) / f"{video_path.stem}_heights.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"✓ Done: {video_path.name}")
    print(f"  JSON: {report_path}")
    print(f"  PNG : {fig_path}")
    return report


def main():
    skipped_videos = [
        r"Autism Flapping and Stimming, She May Fly Away She's So Excited [R_gZqQy_Ae4].mp4",
        r"Early signs of Autism. Armflapping⧸Stimming [8Mm02aD6Pf0].mp4",
        r"Hand flapping, tip-toe walking, talking, out of the blue laughter.. stimming？ [qYeYiDxFcB0].mp4",
        r"Jacob rocking on daddy's lap, also arm flapping!!. [1vRklwIBC28].mp4",
        r"armflapping_18.mp4",
        r"flapping [7lgAK1z-Scs].mp4",
    ]
    for v in skipped_videos:
        vp = Path(INPUT_DIR) / v
        if vp.exists():
            try:
                analyse_video(vp)
            except Exception as e:
                print(f"✗ Error on {vp.name}: {e}")
        else:
            print(f"✗ Missing: {vp}")


if __name__ == "__main__":
    main()
