import pandas as pd
import matplotlib.pyplot as plt
import os

label_file = "../../data/labels/weak_labels.csv"
save_dir = "../../data/analysis"
os.makedirs(save_dir, exist_ok=True)

def summarize_labels(label_file, save_dir):
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"❌ Label file not found: {label_file}")

    df = pd.read_csv(label_file)

    # 统计标签数量
    label_counts = df["label"].value_counts().sort_values(ascending=False)
    label_percent = label_counts / len(df) * 100
    label_conf = df.groupby("label")["confidence"].mean()
    summary = pd.DataFrame({
        "count": label_counts,
        "percent": label_percent.round(2),
        "avg_conf": label_conf.round(2)
    })

    # 保存结果
    csv_path = os.path.join(save_dir, "label_distribution.csv")
    summary.to_csv(csv_path)
    print(f"✅ Saved label distribution to {csv_path}")

    # 柱状图
    plt.figure(figsize=(8, 5))
    label_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "label_distribution.png"))
    plt.close()
    print(f"✅ Saved label distribution plot to {save_dir}/label_distribution.png")

    return summary

if __name__ == "__main__":
    summary = summarize_labels(label_file, save_dir)
    print(summary)
