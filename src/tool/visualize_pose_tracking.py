import cv2
import mediapipe as mp
import numpy as np
import argparse
from pathlib import Path

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


def draw_pose(frame, results):
    h, w, _ = frame.shape
    frame_draw = frame.copy()

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        sh_l, sh_r = lm[11], lm[12]
        wr_l, wr_r = lm[15], lm[16]
        nose = lm[0]
        sh_c = ((sh_l.x + sh_r.x) / 2, (sh_l.y + sh_r.y) / 2)

        mp_drawing.draw_landmarks(
            frame_draw, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
        )

        cv2.circle(frame_draw, (int(sh_c[0] * w), int(sh_c[1] * h)), 6, (0, 255, 0), -1)
        cv2.putText(frame_draw, "shoulder_c", (int(sh_c[0]*w)+5, int(sh_c[1]*h)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.circle(frame_draw, (int(nose.x * w), int(nose.y * h)), 6, (0, 0, 255), -1)
        cv2.putText(frame_draw, "nose", (int(nose.x*w)+5, int(nose.y*h)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    return frame_draw


def visualize_pose(video_path, skip=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    playing = True
    current_frame = 0

    # === 初始化 MediaPipe ===
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    # === 创建窗口和滑块 ===
    win_name = "Pose Visualization (Press SPACE to pause/resume, Q to quit)"
    cv2.namedWindow(win_name)

    def on_trackbar(val):
        nonlocal current_frame
        current_frame = val
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    cv2.createTrackbar("Frame", win_name, 0, total_frames - 1, on_trackbar)

    while cap.isOpened():
        if playing:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Frame", win_name, current_frame)

            # 跳帧逻辑
            if (current_frame - 1) % skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame_draw = draw_pose(frame, results)

            text = f"Frame: {current_frame}/{total_frames}"
            cv2.putText(frame_draw, text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow(win_name, frame_draw)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # 空格：暂停/继续
            playing = not playing
        elif key == 81:  # ←
            current_frame = max(0, current_frame - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == 83:  # →
            current_frame = min(total_frames - 1, current_frame + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    cap.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--skip", type=int, default=1, help="Frame skip step for faster playback")
    args = parser.parse_args()

    visualize_pose(args.video, skip=args.skip)
