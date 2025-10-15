#!/usr/bin/env python3
import argparse, os, json, joblib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TAXONOMY = [
    "no_flap",
    "left_only_low", "left_only_high",
    "right_only_low", "right_only_high",
    "both_symmetric_low", "both_symmetric_high",
    "both_asymmetric",
]

def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    non_feats = {"video", "start_frame", "end_frame", "label", "confidence", "uncertain"}
    num_cols = df.select_dtypes(include=["number", "bool", "float", "int"]).columns.tolist()
    feats = [c for c in num_cols if c not in non_feats]
    if not feats:
        raise ValueError("No numeric feature columns detected. Please check your features CSV.")
    return feats

def load_and_merge(features_path: str, labels_path: str, id_col=None) -> pd.DataFrame:
    fx = pd.read_csv(features_path)
    lb = pd.read_csv(labels_path)

    # 对齐列名
    if "start" in fx.columns and "end" in fx.columns:
        fx = fx.rename(columns={"start": "start_frame", "end": "end_frame"})

    required = {"video", "start_frame", "end_frame"}
    if not required.issubset(fx.columns) or not required.issubset(lb.columns):
        raise KeyError(f"Both files must have {required}, but got features={fx.columns}, labels={lb.columns}")

    df = fx.merge(lb, on=["video", "start_frame", "end_frame"], how="inner", validate="one_to_one")

    # 补充字段并修复 NaN
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    else:
        df["confidence"] = df["confidence"].fillna(1.0)   # 自动修复 NaN

    if "uncertain" not in df.columns:
        df["uncertain"] = 0
    df["uncertain"] = df["uncertain"].astype(int).clip(lower=0, upper=1)

    return df

def filter_mode(df: pd.DataFrame, mode: str):
    if mode == "full":
        sample_weight = np.ones(len(df), dtype=float)
        return df, sample_weight
    elif mode == "clean_only":
        clean = df[df["uncertain"] == 0].copy()
        sample_weight = np.ones(len(clean), dtype=float)
        return clean, sample_weight
    elif mode == "conf_weighted":
        sw = df["confidence"].astype(float).values
        return df, sw
    else:
        raise ValueError(f"Unknown mode: {mode}")

def compute_class_weights(y: np.ndarray, labels: List[str]) -> dict:
    values, counts = np.unique(y, return_counts=True)
    n = len(y); k = len(labels)
    weights = {}
    for val, cnt in zip(values, counts):
        weights[val] = n / (k * cnt)
    for lab in labels:
        weights.setdefault(lab, 1.0)
    return weights

def plot_confusion_matrix(cm, classes, outpath):
    fig = plt.figure(figsize=(8,6))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title('Normalised Confusion Matrix')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

def train_and_eval(X, y, classes, outdir, model_name, model, sample_weight):
    os.makedirs(outdir, exist_ok=True)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = splitter.split(X, y)

    metrics_rows = []
    y_true_all, y_pred_all = [], []
    fold = 0
    for train_idx, test_idx in splits:
        fold += 1
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None

        clf = model
        clf.fit(Xtr, ytr, **({"sample_weight": sw_tr} if sw_tr is not None else {}))
        yhat = clf.predict(Xte)
        y_true_all.extend(yte); y_pred_all.extend(yhat)
        f1_macro = f1_score(yte, yhat, average="macro", zero_division=0)
        f1_micro = f1_score(yte, yhat, average="micro", zero_division=0)
        acc = accuracy_score(yte, yhat)
        metrics_rows.append({"fold": fold, "f1_macro": f1_macro, "f1_micro": f1_micro, "accuracy": acc})

    y_true_all = np.array(y_true_all); y_pred_all = np.array(y_pred_all)
    report = classification_report(y_true_all, y_pred_all, labels=classes, output_dict=True, zero_division=0)
    with open(os.path.join(outdir, f"report_{model_name}.json"), "w") as f:
        json.dump(report, f, indent=2)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes, normalize="true")
    plot_confusion_matrix(cm, classes, os.path.join(outdir, f"cm_{model_name}.png"))
    pd.DataFrame(metrics_rows).to_csv(os.path.join(outdir, f"metrics_{model_name}.csv"), index=False)
    joblib.dump(model, os.path.join(outdir, f"model_{model_name}.pkl"))
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to all_window_features.csv")
    ap.add_argument("--labels", required=True, help="Path to weak_labels.csv")
    ap.add_argument("--mode", choices=["full", "clean_only", "conf_weighted"], default="full")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--svm-kernel", default="linear", choices=["linear","rbf"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_and_merge(args.features, args.labels)

    if "label" not in df.columns:
        raise KeyError("Expected 'label' column in labels CSV")
    df["label"] = df["label"].astype(str)
    classes = sorted(df["label"].unique().tolist(), key=lambda x: TAXONOMY.index(x) if x in TAXONOMY else 999)

    df_f, sample_weight = filter_mode(df, args.mode)
    feat_cols = infer_feature_columns(df_f)
    X = df_f[feat_cols].to_numpy(dtype=float)
    y = df_f["label"].to_numpy()

    class_weights = compute_class_weights(y, classes)

    # === Random Forest ===
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        class_weight=class_weights,
        random_state=args.seed,
    )
    rf_sw = sample_weight * np.vectorize(class_weights.get)(y) if sample_weight is not None else None
    report_rf = train_and_eval(X, y, classes, args.outdir, "rf", rf, rf_sw)

    # === SVM === (手动标准化)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel=args.svm_kernel, class_weight=class_weights, probability=False, random_state=args.seed)
    report_svm = train_and_eval(X_scaled, y, classes, args.outdir, "svm", svm, sample_weight)

    summary = {
        "mode": args.mode,
        "n_samples": int(len(df_f)),
        "n_features": len(feat_cols),
        "classes": classes,
        "feature_columns": feat_cols,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
