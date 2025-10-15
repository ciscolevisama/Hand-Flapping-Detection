import numpy as np
import os
import pandas as pd
from pathlib import Path

def summarize_processed(processed_dir, save_path=None):
    records = []

    for file in os.listdir(processed_dir):
        if file.endswith(".npy"):
            path = os.path.join(processed_dir, file)
            data = np.load(path)

            # 成功帧数
            success_frames = data.shape[0]

            # 注意：无法直接知道视频总帧数，这里默认 processed = success_frames
            # 如果你以后保存了总帧数，可以替换这里
            total_frames = success_frames
            missing = total_frames - success_frames
            missing_rate = missing / total_frames if total_frames > 0 else 0

            records.append({
                "video": Path(file).stem,
                "total_frames": total_frames,
                "success_frames": success_frames,
                "missing_rate": round(missing_rate, 4)
            })

    df = pd.DataFrame(records)
    print(df)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n✅ Summary saved to {save_path}")

    return df

if __name__ == "__main__":
    processed_dir = "../../data/processed_angles"
    save_path = "../../data/processed_summary.csv"
    summarize_processed(processed_dir, save_path)
