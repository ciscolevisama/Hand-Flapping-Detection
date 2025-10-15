import os
import json
import pandas as pd

# 定义输入和输出目录
RUNS_DIR = "../../runs"
OUT_PATH = os.path.join(RUNS_DIR, "baseline_results.csv")

# 三个实验模式
modes = ["full", "clean", "confw"]
models = ["rf", "svm"]

rows = []

for mode in modes:
    for model in models:
        report_path = os.path.join(RUNS_DIR, mode, f"report_{model}.json")
        if not os.path.exists(report_path):
            print(f"⚠️ Missing report: {report_path}")
            continue

        with open(report_path, "r") as f:
            report = json.load(f)

        # 提取整体指标（macro avg 和 accuracy）
        f1_macro = report["macro avg"]["f1-score"]
        acc = report["accuracy"]

        rows.append({
            "mode": mode,
            "model": model.upper(),
            "f1_macro": round(f1_macro, 3),
            "accuracy": round(acc, 3)
        })

# 保存结果
df = pd.DataFrame(rows, columns=["mode","model","f1_macro","accuracy"])
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ 汇总完成，结果已保存到 {OUT_PATH}")
print(df)
