import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# taxonomy 保持一致
CLASS_NAMES = [
    "no_flap",
    "left_only_low",
    "left_only_high",
    "right_only_low",
    "right_only_high",
    "both_symmetric_high",
    "both_asymmetric"
]

def load_reports(root_dir):
    reports = {}
    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if fn.startswith("report_") and fn.endswith(".json"):
                path = os.path.join(dirpath, fn)
                with open(path, "r") as f:
                    reports[os.path.relpath(path, root_dir)] = json.load(f)
    return reports

def analyse_report(report):
    """从 classification_report json 提取 per-class 指标"""
    per_class = {}
    for cls in CLASS_NAMES:
        if cls in report:
            metrics = report[cls]
            per_class[cls] = {
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1-score", 0.0),
                "support": metrics.get("support", 0)
            }
        else:
            per_class[cls] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    return per_class

def plot_per_class_f1(per_class, out_file, title="Per-class F1"):
    df = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "class"})
    plt.figure(figsize=(10, 6))
    sns.barplot(x="class", y="f1", data=df, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1-score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def main(args):
    reports = load_reports(args.report_dir)
    all_stats = []

    for name, report in reports.items():
        per_class = analyse_report(report)
        for cls, metrics in per_class.items():
            all_stats.append({
                "report": name,
                "class": cls,
                **metrics
            })

        # 画柱状图
        out_plot = os.path.join(args.out_dir, f"f1_{os.path.basename(name).replace('.json','')}.png")
        plot_per_class_f1(per_class, out_plot, title=f"F1 per class ({name})")

    df = pd.DataFrame(all_stats)
    df.to_csv(os.path.join(args.out_dir, "error_report.csv"), index=False)
    print("✅ Saved per-class stats to error_report.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_dir", type=str, required=True,
                        help="root directory containing report_*.json")
    parser.add_argument("--out_dir", type=str, default="analysis_out",
                        help="directory to save analysis results")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
