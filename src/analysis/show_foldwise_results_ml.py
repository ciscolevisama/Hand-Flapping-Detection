import os
import pandas as pd

# ------------------------------------------------------
# Define the ML metrics files (update paths if necessary)
# ------------------------------------------------------
ml_paths = {
    "Random Forest (RF)": "runs/clean/metrics_rf.csv",
    "Support Vector Machine (SVM)": "runs/clean/metrics_svm.csv"
}

records = []

for model, path in ml_paths.items():
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        continue

    df = pd.read_csv(path)
    if "f1_macro" not in df.columns:
        print(f"[WARN] f1_macro not found in {path}, available columns: {df.columns}")
        continue

    # Extract values
    fold_values = df["f1_macro"].round(3).tolist()
    record = {"Model": model}
    for i, val in enumerate(fold_values, start=1):
        record[f"Fold {i}"] = val
    record["Mean"] = round(df["f1_macro"].mean(), 3)
    record["Std"] = round(df["f1_macro"].std(), 3)
    records.append(record)

# Create dataframe
summary = pd.DataFrame(records)

if not summary.empty:
    print("\n✅ Fold-wise F1 scores (ML models):")
    print(summary.to_string(index=False))
    out_path = "runs/foldwise_f1_ml_summary.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\n📊 Saved to {out_path}")
else:
    print("\n⚠️ No valid ML results found. Please check file paths or CSV content.")
