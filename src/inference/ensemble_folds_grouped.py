# -*- coding: utf-8 -*-
"""
Cross-fold ensemble inference for DL models (CNN1D / LSTM1D / CNN-LSTM)
Final version with per-video grouped predictions and local frame reset.

Usage:
python src/inference/ensemble_folds_grouped.py ^
  --arch hybrid ^
  --model_dir runs_dl/hybrid_len64_clean_finetune ^
  --mode clean_only ^
  --seq_len 64 ^
  --labels_csv data/labels/weak_labels.csv ^
  --processed_dir data/processed_angles
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, accuracy_score

# === 自动定位项目根目录 ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from training.dl.train_cnn import (
    TAXONOMY, HIGH_CLASSES,
    FlapDataset, compute_norm_stats_24,
    stratified_group_kfold,
    set_seed, calc_high_agg_f1
)

def build_model(arch: str, in_ch: int, n_classes: int, device):
    """根据架构构建模型实例"""
    if arch == "cnn":
        from training.dl.train_cnn import CNN1D
        return CNN1D(in_ch=in_ch, n_classes=n_classes).to(device)
    elif arch == "lstm":
        from training.dl.train_lstm import LSTM1D
        return LSTM1D(in_ch=in_ch, n_classes=n_classes).to(device)
    elif arch == "hybrid":
        from training.dl.train_hybrid import HybridCNNLSTM
        return HybridCNNLSTM(in_ch=in_ch, n_classes=n_classes).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

@torch.no_grad()
def ensemble_predict(models, loader, device):
    """使用多个模型进行 softmax 平均推理"""
    if not models:
        raise RuntimeError("No models were loaded for ensembling.")
    all_logits = []
    for model in models:
        model.eval()
        logits_all = []
        for xb, _, _ in loader:
            xb = xb.to(device)
            logits = F.softmax(model(xb), dim=1).cpu().numpy()
            logits_all.append(logits)
        all_logits.append(np.concatenate(logits_all, axis=0))
    probs = np.mean(np.stack(all_logits, axis=0), axis=0)  # (N, C)
    preds = np.argmax(probs, axis=1)
    return preds, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["cnn", "lstm", "hybrid"], default="cnn")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--processed_dir", default="data/processed_angles")
    ap.add_argument("--labels_csv", default="data/labels/weak_labels.csv")
    ap.add_argument("--mode", choices=["clean_only","full","conf_weighted"], default="clean_only")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 载入标签 ===
    df = pd.read_csv(args.labels_csv)
    if args.mode == "clean_only":
        df = df[df["uncertain"].fillna(0).astype(int) == 0].copy()
        df["confidence"] = 1.0
    elif args.mode == "full":
        df = df.copy(); df["confidence"] = 1.0
    else:
        df["confidence"] = df["confidence"].fillna(1.0).astype(float)

    classes = [c for c in TAXONOMY if c in df["label"].unique().tolist()]
    label2id = {c: i for i, c in enumerate(classes)}
    print(f"[INFO] Ensemble over {args.cv} folds | arch={args.arch} | classes={classes}")

    fold_metrics = []
    all_true, all_pred, all_probs = [], [], []

    # === Cross-validation folds ===
    for fold, (_, va_idx) in enumerate(stratified_group_kfold(df, label2id, args.cv, args.seed), 1):
        df_va = df.iloc[va_idx].reset_index(drop=True)
        mean24, std24, _ = compute_norm_stats_24(df_va, args.processed_dir, args.seq_len)
        ds_va = FlapDataset(df_va, args.processed_dir, args.seq_len, label2id, mean24, std24)
        dl_va = torch.utils.data.DataLoader(ds_va, batch_size=128, shuffle=False)

        # === 载入所有折模型 ===
        models = []
        for k in range(1, args.cv + 1):
            path = os.path.join(args.model_dir, f"best_fold{k}.pt")
            if os.path.isfile(path):
                m = build_model(args.arch, in_ch=24, n_classes=len(classes), device=device)
                state = torch.load(path, map_location=device)
                m.load_state_dict(state)
                models.append(m)
        print(f"[Fold {fold}] Loaded {len(models)} models for ensemble.")
        if not models:
            continue

        y_true = df_va["label"].map(label2id).to_numpy()
        y_pred, probs = ensemble_predict(models, dl_va, device)

        f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        rep  = classification_report(
            y_true, y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True, zero_division=0
        )
        high_f1 = calc_high_agg_f1(rep, classes)
        print(f"[Fold {fold}] F1_macro={f1m:.4f}, HighF1={high_f1:.4f}, Acc={acc:.4f}")

        fold_metrics.append({"fold": fold, "f1_macro": f1m, "highF1": high_f1, "acc": acc})
        all_true += y_true.tolist()
        all_pred += y_pred.tolist()
        all_probs.append(probs)

        # === 每个视频独立保存预测（重置帧号） ===
        for vid, df_vid in df_va.groupby("video"):
            local_idx = list(range(len(df_vid)))
            preds_vid = [classes[y_pred[i - df_va.index[0]]] for i in df_vid.index]
            probs_vid = [float(np.max(probs[i - df_va.index[0]])) for i in df_vid.index]

            df_out = pd.DataFrame({
                "video": [vid] * len(df_vid),
                "start_frame": [i * args.seq_len for i in local_idx],
                "end_frame": [(i + 1) * args.seq_len for i in local_idx],
                "true_label": df_vid["label"].tolist(),
                "pred_label": preds_vid,
                "confidence": probs_vid
            })

            out_path_vid = os.path.join(args.model_dir, f"{vid}_preds.csv")
            df_out.to_csv(out_path_vid, index=False)
        print(f"   ↳ Saved per-video predictions (reset frame index) for fold {fold}")

    # === 汇总性能 ===
    dfm = pd.DataFrame(fold_metrics)
    dfm.to_csv(os.path.join(args.model_dir, "ensemble_metrics.csv"), index=False)
    print("\n📊 Mean metrics over folds:")
    print(dfm.mean(numeric_only=True))

    rep_all = classification_report(
        all_true, all_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True, zero_division=0
    )
    highF1_all = calc_high_agg_f1(rep_all, classes)
    overall = {
        "f1_macro": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "highF1": highF1_all,
        "acc": accuracy_score(all_true, all_pred)
    }
    with open(os.path.join(args.model_dir, "ensemble_report.json"), "w", encoding="utf-8") as f:
        json.dump({"overall": overall, "per_class": rep_all}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All per-video prediction CSVs saved under: {args.model_dir}")

if __name__ == "__main__":
    main()
