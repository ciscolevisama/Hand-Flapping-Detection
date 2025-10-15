import json, os, pandas as pd

# ======================================================
# Define your model directories
# ======================================================
model_dirs = {
    "CNN1D": "runs_dl/cnn_len64_clean_finetune",
    "Bi-LSTM": "runs_dl/lstm_len64_clean_finetune",
    "Hybrid CNN–LSTM": "runs_dl/hybrid_len64_clean_finetune"
}

records = []

for model, path in model_dirs.items():
    fold_values = []
    for i in range(1, 6):  # folds 1–5
        fpath = os.path.join(path, f"report_fold{i}.json")
        if not os.path.exists(fpath):
            print(f"[WARN] Missing file: {fpath}")
            continue

        with open(fpath, "r") as f:
            metrics = json.load(f)

        # Extract macro-F1 from nested structure
        try:
            f1 = metrics["macro avg"]["f1-score"]
        except KeyError:
            f1 = None

        if f1 is not None:
            fold_values.append(f1)
        else:
            print(f"[WARN] Could not extract F1 from {fpath}")

    if fold_values:
        record = {"Model": model}
        for i, val in enumerate(fold_values, start=1):
            record[f"Fold {i}"] = round(val, 3)
        record["Mean"] = round(sum(fold_values)/len(fold_values), 3)
        record["Std"] = round(pd.Series(fold_values).std(), 3)
        records.append(record)
    else:
        print(f"[WARN] No valid results for {model}")

df = pd.DataFrame(records)

if not df.empty:
    print("\n✅ Fold-wise F1 scores:")
    print(df.to_string(index=False))
    out_path = "runs/foldwise_f1_summary.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n📊 Saved to {out_path}")
else:
    print("\n⚠️ No valid F1 data found. Please check file structure again.")
