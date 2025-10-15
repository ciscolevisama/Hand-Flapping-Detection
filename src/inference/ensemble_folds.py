# -*- coding: utf-8 -*-
"""
Cross-fold ensemble inference for DL models (CNN1D / LSTM1D / CNN-LSTM).

Usage examples:
# Hybrid
python src/inference/ensemble_folds.py ^
  --arch hybrid ^
  --model_dir runs_dl\hybrid_len64_clean_finetune ^
  --mode clean_only ^
  --seq_len 64 ^
  --labels_csv data/labels/weak_labels.csv ^
  --processed_dir data/processed_angles
"""

import sys, os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt

# === 自动定位项目根目录 ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# 导入通用训练组件
from training.dl.train_cnn import (
    TAXONOMY, HIGH_CLASSES,
    FlapDataset, compute_norm_stats_24,
    stratified_group_kfold,
    set_seed, calc_high_agg_f1
)


def build_model(arch: str, in_ch: int, n_classes: int, device):
    """根据 --arch 选择并实例化模型。"""
    if arch == "cnn":
        from training.dl.train_cnn import CNN1D
        m = CNN1D(in_ch=in_ch, n_classes=n_classes).to(device)
    elif arch == "lstm":
        from training.dl.train_lstm import LSTM1D
        m = LSTM1D(in_ch=in_ch, n_classes=n_classes).to(device)
    elif arch == "hybrid":
        from training.dl.train_hybrid import HybridCNNLSTM
        m = HybridCNNLSTM(in_ch=in_ch, n_classes=n_classes).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


@torch.no_grad()
def ensemble_predict(models, loader, device):
    """对一个 DataLoader，用多个模型输出 softmax 概率并平均。"""
    if len(models) == 0:
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
    ap.add_argument("--arch", choices=["cnn", "lstm", "hybrid"], default="cnn", help="Model architecture")
    ap.add_argument("--model_dir", required=True, help="Directory containing best_fold*.pt")
    ap.add_argument("--processed_dir", default="data/processed_angles")
    ap.add_argument("--labels_csv", default="data/labels/weak_labels.csv")
    ap.add_argument("--mode", choices=["clean_only", "full", "conf_weighted"], default="clean_only")
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
    all_true, all_pred = [], []
    all_probs = []

    # === 逐折推理 ===
    for fold, (_, va_idx) in enumerate(stratified_group_kfold(df, label2id, args.cv, args.seed), 1):
        df_va = df.iloc[va_idx].reset_index(drop=True)

        # 归一化统计（与训练逻辑一致）
        mean24, std24, _ = compute_norm_stats_24(df_va, args.processed_dir, args.seq_len)
        ds_va = FlapDataset(df_va, args.processed_dir, args.seq_len, label2id, mean24, std24)
        dl_va = torch.utils.data.DataLoader(ds_va, batch_size=128, shuffle=False)

        # 加载模型权重
        models = []
        for k in range(1, args.cv + 1):
            path = os.path.join(args.model_dir, f"best_fold{k}.pt")
            if os.path.isfile(path):
                m = build_model(args.arch, in_ch=24, n_classes=len(classes), device=device)
                state = torch.load(path, map_location=device)
                m.load_state_dict(state)
                models.append(m)
            else:
                print(f"[WARN] Missing {path}")
        print(f"[Fold {fold}] Loaded {len(models)} models for ensemble.")

        if len(models) == 0:
            raise RuntimeError(f"No checkpoint found in {args.model_dir} (expected best_fold*.pt).")

        # === 推理与评估 ===
        y_true = df_va["label"].map(label2id).to_numpy()
        y_pred, probs = ensemble_predict(models, dl_va, device)

        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        rep = classification_report(
            y_true, y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True, zero_division=0
        )
        high_f1 = calc_high_agg_f1(rep, classes)
        print(f"[Fold {fold}] Ensemble: f1_macro={f1m:.4f}, highF1={high_f1:.4f}, acc={acc:.4f}")

        fold_metrics.append({"fold": fold, "f1_macro": f1m, "highF1": high_f1, "acc": acc})
        all_true += y_true.tolist()
        all_pred += y_pred.tolist()
        all_probs.append(probs)

    # === 汇总指标 ===
    dfm = pd.DataFrame(fold_metrics)
    dfm.to_csv(os.path.join(args.model_dir, "ensemble_metrics.csv"), index=False)
    print("\n📊 Mean metrics over folds:")
    print(dfm.mean(numeric_only=True))

    # === 汇总总体报告 ===
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

    # === 绘制总览柱状图 ===
    plt.figure(figsize=(6, 4))
    plt.bar(["F1_macro", "HighF1", "Acc"], [overall["f1_macro"], overall["highF1"], overall["acc"]])
    plt.ylim(0, 1)
    plt.title(f"{args.arch.upper()} Ensemble Performance")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, "ensemble_summary.png"))

    # === 保存逐样本预测结果（含置信度） ===
    all_probs = np.concatenate(all_probs, axis=0)
    # === 保存逐样本预测结果（包含 video + start/end frame + confidence） ===
    video_col = []
    start_frames, end_frames = [], []
    for fold, (_, va_idx) in enumerate(stratified_group_kfold(df, label2id, args.cv, args.seed), 1):
        df_va = df.iloc[va_idx]
        video_col += df_va["video"].tolist()
        if "start_frame" in df_va.columns and "end_frame" in df_va.columns:
            start_frames += df_va["start_frame"].tolist()
            end_frames += df_va["end_frame"].tolist()
        else:
            start_frames += [None] * len(df_va)
            end_frames += [None] * len(df_va)

    out_df = pd.DataFrame({
        "video": video_col[:len(all_pred)],
        "start_frame": start_frames[:len(all_pred)],
        "end_frame": end_frames[:len(all_pred)],
        "true_label": [classes[i] for i in all_true],
        "pred_label": [classes[i] for i in all_pred],
        "confidence": np.max(all_probs, axis=1)[:len(all_pred)]
    })

    out_path = os.path.join(args.model_dir, "preds_ensemble.csv")
    out_df.to_csv(out_path, index=False)
    print(f"💾 Saved per-sample predictions (grouped by video) to {out_path}")

    out_path = os.path.join(args.model_dir, "preds_ensemble.csv")
    out_df.to_csv(out_path, index=False)
    print(f"💾 Saved per-sample predictions to {out_path}")

    print(f"\n✅ Ensemble results saved to {args.model_dir}")


if __name__ == "__main__":
    main()
