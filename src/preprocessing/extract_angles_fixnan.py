import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import json
from pathlib import Path

mp_pose = mp.solutions.pose

# 放宽可见性阈值
VIS_TH = 0.25
EPS = 1e-6
MAX_FILL_GAP = 15   # 连续多少帧可前向填充
DECAY = 0.95        # 填充值衰减系数


def calculate_angle(a, b, c):
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < EPS or nbc < EPS:
        return np.nan
    cosang = np.dot(ba, bc) / (nba * nbc)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def landmark_xy(lm, idx):
    p = lm[idx]
    return (float(p.x), float(p.y), float(getattr(p, "visibility", 1.0)))


def extract_angles(video_path, save_path, return_dict=False, debug=False):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    feats = []
    frames_total, frames_valid = 0, 0
    last_valid = None
    fill_count = 0
    miss_streak = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_total += 1

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if not results.pose_landmarks:
            miss_streak += 1
            # 前向填充
            if last_valid is not None and miss_streak <= MAX_FILL_GAP:
                feats.append(last_valid * (DECAY ** miss_streak))
                fill_count += 1
            else:
                feats.append([np.nan] * 8)
            continue

        lm = results.pose_landmarks.landmark
        # 取关键点
        ids = [0,11,12,13,14,15,16,23,24]
        xs, ys, vs = zip(*[landmark_xy(lm, i) for i in ids])
        nose_x, sh_lx, sh_rx, el_lx, el_rx, wr_lx, wr_rx, hip_lx, hip_rx = xs
        nose_y, sh_ly, sh_ry, el_ly, el_ry, wr_ly, wr_ry, hip_ly, hip_ry = ys
        nose_v, sh_lv, sh_rv, el_lv, el_rv, wr_lv, wr_rv, hip_lv, hip_rv = vs

        arm_ok = all(v >= VIS_TH for v in [sh_lv, sh_rv, el_lv, el_rv, wr_lv, wr_rv])
        if not arm_ok:
            miss_streak += 1
            if last_valid is not None and miss_streak <= MAX_FILL_GAP:
                feats.append(last_valid * (DECAY ** miss_streak))
                fill_count += 1
            else:
                feats.append([np.nan] * 8)
            continue
        miss_streak = 0  # reset

        # 计算角度特征
        shoulder_l, shoulder_r = [sh_lx, sh_ly], [sh_rx, sh_ry]
        elbow_l, elbow_r = [el_lx, el_ly], [el_rx, el_ry]
        wrist_l, wrist_r = [wr_lx, wr_ly], [wr_rx, wr_ry]
        hip_l, hip_r = [hip_lx, hip_ly], [hip_rx, hip_ry]
        shoulder_c = [(sh_lx + sh_rx) / 2.0, (sh_ly + sh_ry) / 2.0]
        hip_c = [(hip_lx + hip_rx) / 2.0, (hip_ly + hip_ry) / 2.0]

        have_sh = (sh_lv >= VIS_TH and sh_rv >= VIS_TH)
        have_hip = (hip_lv >= VIS_TH and hip_rv >= VIS_TH)
        if have_sh and have_hip:
            torso_len = max(hip_c[1] - shoulder_c[1], EPS)
        elif have_hip and nose_v >= VIS_TH:
            torso_len = max(hip_c[1] - nose_y, EPS)
        elif have_sh and nose_v >= VIS_TH:
            torso_len = max(nose_y - shoulder_c[1], EPS)
        else:
            torso_len = 1.0

        wr_l_rel = (shoulder_c[1] - wrist_l[1]) / torso_len
        wr_r_rel = (shoulder_c[1] - wrist_r[1]) / torso_len
        wr_l_rel = float(np.clip(wr_l_rel, -3.0, 3.0))
        wr_r_rel = float(np.clip(wr_r_rel, -3.0, 3.0))

        ang_lel = calculate_angle(shoulder_l, elbow_l, wrist_l)
        ang_rel = calculate_angle(shoulder_r, elbow_r, wrist_r)
        ang_lwr = calculate_angle(elbow_l, wrist_l, [wrist_l[0], wrist_l[1] + 0.1])
        ang_rwr = calculate_angle(elbow_r, wrist_r, [wrist_r[0], wrist_r[1] + 0.1])
        ang_lax = calculate_angle(hip_l, shoulder_l, elbow_l) if hip_lv >= VIS_TH else np.nan
        ang_rax = calculate_angle(hip_r, shoulder_r, elbow_r) if hip_rv >= VIS_TH else np.nan

        feat = np.array([ang_lwr, ang_rwr, ang_lel, ang_rel, ang_lax, ang_rax, wr_l_rel, wr_r_rel], dtype=float)
        feats.append(feat)
        last_valid = feat
        frames_valid += 1

    cap.release()
    pose.close()

    feats = np.array(feats, dtype=float)
    nan_ratio = np.isnan(feats).mean()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, feats)

    valid_rate = 0 if frames_total == 0 else frames_valid / frames_total
    print(f"✅ {Path(video_path).name} | total={frames_total}, valid={frames_valid} ({valid_rate:.1%}) | "
          f"nan_rate={nan_ratio:.1%} | fills={fill_count} | saved {Path(save_path).name}")

    # 写日志
    log = {
        "video": Path(video_path).name,
        "frames_total": frames_total,
        "frames_valid": frames_valid,
        "valid_rate": valid_rate,
        "nan_ratio": nan_ratio,
        "fill_count": fill_count,
        "params": {"VIS_TH": VIS_TH, "MAX_FILL_GAP": MAX_FILL_GAP, "DECAY": DECAY},
        "output_file": Path(save_path).name,
        "feature_order": [
            "ang_lwr","ang_rwr","ang_lel","ang_rel","ang_lax","ang_rax","wr_l_rel","wr_r_rel"
        ],
    }
    with open(str(Path(save_path).with_suffix(".log.json")), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    if return_dict:
        return {"angles": feats, "frames_total": frames_total,
                "frames_valid": frames_valid, "valid_rate": valid_rate,
                "nan_ratio": nan_ratio, "fill_count": fill_count}
    return feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-frame angles with NaN handling & forward-fill.")
    parser.add_argument("--video", type=str, required=True, help="Input video path (.mp4)")
    parser.add_argument("--out", type=str, required=True, help="Output .npy file path")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    extract_angles(args.video, args.out)
