import numpy as np
import os
import pandas as pd
from pathlib import Path

# 阈值（可调）
AMPLITUDE_THRESHOLD = 25   # 区分 low / high
MOVEMENT_THRESHOLD = 10    # 区分 flap / no_flap
SYMMETRY_THRESHOLD = 20    # 区分 symmetric / asymmetric
WINDOW_SIZE = 12           # 窗口长度（帧数）

LABELS = {
    0: "no_flap",
    1: "left_only_low",
    2: "left_only_high",
    3: "right_only_low",
    4: "right_only_high",
    5: "both_symmetric_low",
    6: "both_symmetric_high",
    7: "both_asymmetric"
}

def classify_window(window):
    """对一个窗口进行分类，返回 (label, confidence)"""
    mean = np.mean(window, axis=0)
    var = np.var(window, axis=0)

    left_amp = mean[4]
    right_amp = mean[5]

    left_move = var[0] + var[2]
    right_move = var[1] + var[3]

    # no_flap
    if left_move < MOVEMENT_THRESHOLD and right_move < MOVEMENT_THRESHOLD:
        return 0, 0.9  # 高置信度

    # left only
    if left_move > MOVEMENT_THRESHOLD and right_move < MOVEMENT_THRESHOLD:
        if left_amp > AMPLITUDE_THRESHOLD:
            return 2, 0.8
        else:
            return 1, 0.8

    # right only
    if right_move > MOVEMENT_THRESHOLD and left_move < MOVEMENT_THRESHOLD:
        if right_amp > AMPLITUDE_THRESHOLD:
            return 4, 0.8
        else:
            return 3, 0.8

    # both hands
    if left_move > MOVEMENT_THRESHOLD and right_move > MOVEMENT_THRESHOLD:
        if abs(left_amp - right_amp) < SYMMETRY_THRESHOLD:
            if max(left_amp, right_amp) > AMPLITUDE_THRESHOLD:
                return 6, 0.7
            else:
                return 5, 0.7
        else:
            return 7, 0.6  # 不对称，置信度较低

    return 0, 0.5  # 默认 no_flap，低置信度

def segment_video(features, window_size=WINDOW_SIZE):
    segments = []
    num_windows = len(features) // window_size

    for i in range(num_windows):
        window = features[i*window_size:(i+1)*window_size]
        label, conf = classify_window(window)
        segments.append((i*window_size, (i+1)*window_size, label, conf))

    return segments

if __name__ == "__main__":
    processed_dir = "../../data/processed_angles"
    save_dir = "../../data/auto_labels"
    os.makedirs(save_dir, exist_ok=True)

    all_dfs = []  # 用来收集所有视频的结果

    for file in os.listdir(processed_dir):
        if file.endswith(".npy"):
            path = os.path.join(processed_dir, file)
            features = np.load(path)

            segments = segment_video(features)

            df = pd.DataFrame([{
                "video_name": Path(file).stem,
                "start_frame": seg[0],
                "end_frame": seg[1],
                "label": LABELS[seg[2]],
                "confidence": seg[3]
            } for seg in segments])

            # 保存单个视频的结果
            save_path = os.path.join(save_dir, Path(file).stem + "_segments.csv")
            df.to_csv(save_path, index=False)
            print(f"✅ Segmented {file}, {len(segments)} segments saved to {save_path}")

            all_dfs.append(df)

    # 合并所有结果，输出总表
    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_save_path = os.path.join(save_dir, "all_segments.csv")
        all_df.to_csv(all_save_path, index=False)
        print(f"📊 All results saved to {all_save_path}, total {len(all_df)} segments")
