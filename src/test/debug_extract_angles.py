import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

mp_pose = mp.solutions.pose
VIS_TH = 0.5  # visibility threshold


def landmark_xy(lm, idx):
    p = lm[idx]
    return (p.x, p.y, getattr(p, "visibility", 1.0))


def debug_extract_angles(video_path):
    cap = cv2.VideoCapture(video_path)
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

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(wr_l_rel_list, label="Left wrist rel height")
    plt.plot(wr_r_rel_list, label="Right wrist rel height")
    plt.xlabel("Frame")
    plt.ylabel("Relative height (shoulder→torso normalized)")
    plt.legend()
    plt.title(f"Debug wrist relative heights: {Path(video_path).name}")
    plt.show()

    # Print some stats
    print(f"Frames total: {frames_total}")
    print("Left wrist: min={:.3f}, max={:.3f}".format(np.nanmin(wr_l_rel_list), np.nanmax(wr_l_rel_list)))
    print("Right wrist: min={:.3f}, max={:.3f}".format(np.nanmin(wr_r_rel_list), np.nanmax(wr_r_rel_list)))


if __name__ == "__main__":
    # 👉 改成你的视频路径
    video_file = "../../data/raw_videos/flapping [7lgAK1z-Scs].mp4"
    debug_extract_angles(video_file)
