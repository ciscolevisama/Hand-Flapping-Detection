"""
evaluate_all_models.py
-------------------------------------------------
Collects and summarises model evaluation metrics
for both ML and DL architectures under 5-fold CV.
Generates a single CSV (Table 4.1) for dissertation.
-------------------------------------------------
Author: Wang Yida
Date: 2025-10
"""

import os
import json
import pandas as pd
import numpy as np
import glob

# -----------------------------
# Utility: Safe file loading
# -----------------------------
def safe_load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================
# 1. Aggregate baseline (RF / SVM) results from CSV file
# ============================================================
def summarize_baselines(csv_path="runs/baseline_results.csv"):
    if not os.path.exists(csv_path):
        print(f"[WARN] Baseline results file not found: {csv_path}")
        return pd.DataFrame(columns=["Model", "Accuracy_mean", "Accuracy_std", "F1_mean", "F1_std"])

    df = pd.read_csv(csv_path)
    # 只保留最关键的几种模式（你可调整）
    df = df[df["mode"].isin(["clean_only", "conf_weighted", "clean", "full"])]

    grouped = df.groupby("model")[["accuracy", "f1_macro"]].agg(["mean", "std"])
    grouped.columns = ["Accuracy_mean", "Accuracy_std", "F1_mean", "F1_std"]
    grouped = grouped.reset_index().rename(columns={"model": "Model"})
    print("✅ Baseline models summarised:\n", grouped)
    return grouped


# ============================================================
# 2. Aggregate DL models (CNN, LSTM, Hybrid)
# ============================================================
def summarize_dl_models(model_dirs=None):
    if model_dirs is None:
        model_dirs = {
            "CNN1D": "runs_dl/cnn_len64_clean_finetune",
            "BiLSTM": "runs_dl/lstm_len64_clean_finetune",
            "Hybrid CNN–LSTM": "runs_dl/hybrid_len64_clean_finetune",
        }

    records = []

    for model, path in model_dirs.items():
        metrics = []
        for i in range(1, 6):  # 5 folds
            # 支持不同命名，例如 report_fold1.json、metrics_fold_1.json
            pattern = os.path.join(path, f"*fold*{i}*.json")
            files = glob.glob(pattern)
            if not files:
                print(f"[WARN] No report file found for {model} fold {i}")
                continue
            fpath = files[0]  # 取第一个匹配文件
            m = safe_load_json(fpath)
            if m:
                metrics.append(m)


        if not metrics:
            print(f"[WARN] No metric files found for {model}")
            continue

        def avg(key):
            vals = []
            for m in metrics:
                # 直接匹配键（accuracy、f1_macro等）
                if key in m:
                    vals.append(m[key])
                    continue
                # 支持 classification_report 结构
                if "macro avg" in m and key in ["f1_macro", "precision_macro", "recall_macro"]:
                    vals.append(m["macro avg"]["f1-score"] if "f1" in key else m["macro avg"][key.split("_")[0]])
                    continue
                # 支持 weighted avg（可选）
                if "weighted avg" in m and key in ["f1_weighted"]:
                    vals.append(m["weighted avg"]["f1-score"])
            return np.nanmean(vals), np.nanstd(vals)

        acc_mean, acc_std = avg("accuracy")
        f1_mean, f1_std = avg("f1_macro")

        records.append({
            "Model": model,
            "Accuracy_mean": round(acc_mean, 3),
            "Accuracy_std": round(acc_std, 3),
            "F1_mean": round(f1_mean, 3),
            "F1_std": round(f1_std, 3)
        })

    df = pd.DataFrame(records)
    print("✅ Deep learning models summarised:\n", df)
    return df


# ============================================================
# 3. Merge & Export (Final Table 4.1)
# ============================================================
def merge_results(df_ml, df_dl, output_path="runs/final_table4_1.csv"):
    df_all = pd.concat([df_ml, df_dl], ignore_index=True)
    df_all = df_all[["Model", "Accuracy_mean", "Accuracy_std", "F1_mean", "F1_std"]]
    df_all = df_all.round(3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, index=False)
    print(f"\n📊 Final Table 4.1 summary exported to: {output_path}")
    print(df_all)
    return df_all


# ============================================================
# 4. Main
# ============================================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print(" Evaluating all models and compiling Table 4.1 ...")
    print("--------------------------------------------------")

    df_ml = summarize_baselines("runs/baseline_results.csv")
    df_dl = summarize_dl_models()

    final_df = merge_results(df_ml, df_dl)

    print("\n✅ Done. Use this CSV for Table 4.1 in Chapter 4.")
    print("Add the following footnote in your dissertation:")
    print("  *Note: Values represent mean ± standard deviation across five folds.*")
