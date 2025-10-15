import numpy as np, pandas as pd, os

angles_dir = "data/processed_angles"
df = pd.read_csv("data/labels/weak_labels.csv")
vid = df["video"].unique()[0]  # 随便选一个
arr = np.load(os.path.join(angles_dir, f"{vid}.npy"))
print(f"{vid}: total frames={len(arr)}, dim={arr.shape[1]}")

# 检查前几个标签窗口
for _, r in df[df["video"]==vid].head(10).iterrows():
    s, e = int(r["start_frame"]), int(r["end_frame"])
    if e > len(arr): print(f"[WARN] {vid} end_frame {e} > total {len(arr)}")
    seg = arr[s:e]
    print(f"{vid}: {r['label']} | len={len(seg)} | H_left mean={seg[:,0].mean():.2f}")

#
# import numpy as np, matplotlib.pyplot as plt
#
# vid = "10_yr_severe_autistic_I7fdv1q9-m8"
# arr = np.load(f"data/processed_angles/{vid}.npy")
# plt.plot(arr[:,6], label="H_left (rel)")
# plt.plot(arr[:,7], label="H_right (rel)")
# plt.legend(); plt.show()