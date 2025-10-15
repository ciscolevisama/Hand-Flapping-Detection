# src/preprocessing/extract_angles.py
import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import json
from pathlib import Path

mp_pose = mp.solutions.pose

# 可见性阈值（放宽一点提高召回）
VIS_TH = 0.30
EPS = 1e-6


def calculate_angle(a, b, c):
    """
    计算三点 a-b-c 构成的夹角（角度制）。任一向量长度过小返回 NaN。
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < EPS or nbc < EPS:
        return np.nan
    cosang = np.dot(ba, bc) / (nba * nbc)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def landmark_xy(lm, idx):
    """返回 (x, y, visibility)（MediaPipe 是归一化坐标，y 向下为正）。"""
    p = lm[idx]
    return (float(p.x), float(p.y), float(getattr(p, "visibility", 1.0)))


def extract_angles(video_path, save_path, return_dict=False, debug=False):
    """
    从视频提取每帧 8 维特征：
      0: ang_lwr  左腕角
      1: ang_rwr  右腕角
      2: ang_lel  左肘角
      3: ang_rel  右肘角
      4: ang_lax  左肩-肘-髋夹角（近似肩外展轴）
      5: ang_rax  右肩-肘-髋夹角
      6: wr_l_rel 左手腕相对高度（肩线为基准，肩上为正；以肩-髋竖直距离归一化）
      7: wr_r_rel 右手腕相对高度（同上）
    """
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.30,
        min_tracking_confidence=0.30
    )

    feats = []
    frames_total, frames_valid = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_total += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if not results.pose_landmarks:
            feats.append([np.nan] * 8)
            continue

        lm = results.pose_landmarks.landmark

        # 取关键点
        nose_x, nose_y, nose_v = landmark_xy(lm, 0)
        sh_lx, sh_ly, sh_lv = landmark_xy(lm, 11)
        sh_rx, sh_ry, sh_rv = landmark_xy(lm, 12)
        el_lx, el_ly, el_lv = landmark_xy(lm, 13)
        el_rx, el_ry, el_rv = landmark_xy(lm, 14)
        wr_lx, wr_ly, wr_lv = landmark_xy(lm, 15)
        wr_rx, wr_ry, wr_rv = landmark_xy(lm, 16)
        hip_lx, hip_ly, hip_lv = landmark_xy(lm, 23)
        hip_rx, hip_ry, hip_rv = landmark_xy(lm, 24)

        # 上肢可见性检查（尽量放宽但保证质量）
        arm_ok = all(v >= VIS_TH for v in [sh_lv, sh_rv, el_lv, el_rv, wr_lv, wr_rv])
        if not arm_ok:
            feats.append([np.nan] * 8)
            continue

        # 构造点
        shoulder_l, shoulder_r = [sh_lx, sh_ly], [sh_rx, sh_ry]
        elbow_l, elbow_r = [el_lx, el_ly], [el_rx, el_ry]
        wrist_l, wrist_r = [wr_lx, wr_ly], [wr_rx, wr_ry]
        hip_l, hip_r = [hip_lx, hip_ly], [hip_rx, hip_ry]
        shoulder_c = [(sh_lx + sh_rx) / 2.0, (sh_ly + sh_ry) / 2.0]
        hip_c = [(hip_lx + hip_rx) / 2.0, (hip_ly + hip_ry) / 2.0]
        have_sh = (sh_lv >= VIS_TH and sh_rv >= VIS_TH)
        have_hip = (hip_lv >= VIS_TH and hip_rv >= VIS_TH)

        # 躯干竖直长度：优先肩-髋，保证为正（y 向下为正，因此 hip_y > shoulder_y）
        if have_sh and have_hip:
            torso_len = max(hip_c[1] - shoulder_c[1], EPS)  # 正值
        elif have_hip and nose_v >= VIS_TH:
            torso_len = max(hip_c[1] - nose_y, EPS)        # 正值
        elif have_sh and nose_v >= VIS_TH:
            torso_len = max(nose_y - shoulder_c[1], EPS)   # 正值（退而求其次）
        else:
            torso_len = 1.0                                # 最后兜底

        # 手腕相对高度：肩上为正、肩下为负；裁剪到 [-3, 3] 以抑制异常
        wr_l_rel = (shoulder_c[1] - wrist_l[1]) / torso_len
        wr_r_rel = (shoulder_c[1] - wrist_r[1]) / torso_len
        wr_l_rel = float(np.clip(wr_l_rel, -3.0, 3.0))
        wr_r_rel = float(np.clip(wr_r_rel, -3.0, 3.0))

        # 关节角（角度制）
        ang_lel = calculate_angle(shoulder_l, elbow_l, wrist_l)  # 左肘
        ang_rel = calculate_angle(shoulder_r, elbow_r, wrist_r)  # 右肘
        # 腕部“俯仰”角：用腕点向下一个微小偏移构造近似
        ang_lwr = calculate_angle(elbow_l, wrist_l, [wrist_l[0], wrist_l[1] + 0.1])
        ang_rwr = calculate_angle(elbow_r, wrist_r, [wrist_r[0], wrist_r[1] + 0.1])
        # 肩-肘-髋的夹角（大致反映上臂外展/屈曲）
        ang_lax = calculate_angle(hip_l, shoulder_l, elbow_l) if hip_lv >= VIS_TH else np.nan
        ang_rax = calculate_angle(hip_r, shoulder_r, elbow_r) if hip_rv >= VIS_TH else np.nan

        feats.append([ang_lwr, ang_rwr, ang_lel, ang_rel, ang_lax, ang_rax, wr_l_rel, wr_r_rel])
        frames_valid += 1

    cap.release()
    pose.close()

    feats = np.array(feats, dtype=float)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, feats)

    valid_rate = 0.0 if frames_total == 0 else frames_valid / frames_total
    print(f"✅ {Path(video_path).name} | total={frames_total}, valid={frames_valid} ({valid_rate:.1%}) | saved {Path(save_path).name}")

    # 保存日志
    log = {
        "video": Path(video_path).name,
        "frames_total": frames_total,
        "frames_valid": frames_valid,
        "valid_rate": valid_rate,
        "output_file": Path(save_path).name,
        "feature_order": [
            "ang_lwr","ang_rwr","ang_lel","ang_rel","ang_lax","ang_rax",
            "wr_l_rel","wr_r_rel"
        ],
        "wr_height_semantics": "positive=above shoulder; zero≈shoulder line; negative=below shoulder",
    }
    with open(str(Path(save_path).with_suffix(".log.json")), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    if return_dict:
        return {
            "angles": feats,
            "frames_total": frames_total,
            "frames_valid": frames_valid,
            "valid_rate": valid_rate
        }
    return feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-frame angles and wrist-height features from a video.")
    parser.add_argument("--video", type=str, required=True, help="Input video path (.mp4)")
    parser.add_argument("--out", type=str, required=True, help="Output .npy file path")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    extract_angles(args.video, args.out)
