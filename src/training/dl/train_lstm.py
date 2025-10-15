# -*- coding: utf-8 -*-
"""
BiLSTM training for 8-class hand-flapping form classification.

与 train_cnn.py 的数据管线与参数保持一致：
- NaN-safe 归一化、分组分层 CV、欠拟合/崩溃监测
- 失衡处理：sampler / class_weight / both；支持 Focal Loss
- 支持置信度加权 conf_alpha
- --freeze 微调：none / early2(冻结LSTM第0层±反向) / encoder(冻结全部LSTM，仅训练FC)
- --resume_from "runs_xxx\\best_fold{fold}.pt" 继续训练

快速示例（与 CNN 一致的设置）：
python src\\training\\dl\\train_lstm.py ^
  --mode clean_only ^
  --seq_len 64 --epochs 30 --lr 5e-4 --cv 5 ^
  --balancing both --sampler_alpha 0.7 --cw_beta 0.5 --cw_clip 3 ^
  --loss focal --gamma 2.0 --max_nan_ratio 0.6 ^
  --dropout 0.3 --hidden_size 128 --num_layers 1 ^
  --outdir runs_dl\\lstm_len64_clean_dropout03
"""

import os, json, argparse, warnings, math, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False

warnings.filterwarnings("ignore")

# ------------------------ taxonomy ------------------------
TAXONOMY = [
    "no_flap",
    "left_only_low","left_only_high",
    "right_only_low","right_only_high",
    "both_symmetric_low","both_symmetric_high",
    "both_asymmetric",
]
HIGH_CLASSES = {"left_only_high","right_only_high","both_symmetric_high"}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_json(obj, path):
    with open(path,"w",encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)

# ------------------------ grouped stratified CV ------------------------
def stratified_group_kfold(df: pd.DataFrame, label2id: Dict[str,int], cv:int, seed:int, max_tries:int=50):
    y = df["label"].map(label2id).to_numpy()
    groups = df["video"].astype(str).to_numpy()
    uniq_groups = pd.unique(groups).tolist()
    def ok_train_has_all(tr_idx):
        ys = set(df.iloc[tr_idx]["label"].map(label2id).tolist())
        return len(ys) == len(set(label2id.values()))
    if cv == 1:
        rng = np.random.default_rng(seed)
        for _ in range(max_tries):
            vids = uniq_groups.copy(); rng.shuffle(vids)
            cut = max(1, int(len(vids)*0.8))
            tr = set(vids[:cut]); va = set(vids[cut:])
            tr_idx = df.index[df["video"].astype(str).isin(tr)].to_numpy()
            va_idx = df.index[df["video"].astype(str).isin(va)].to_numpy()
            if ok_train_has_all(tr_idx): yield tr_idx, va_idx; return
        raise RuntimeError("Failed to build a split with TRAIN covering all labels.")
    else:
        if HAS_SGK:
            sp = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=seed)
            for tr,va in sp.split(np.zeros_like(y), y, groups=groups):
                if ok_train_has_all(tr): yield tr,va
            return
        rng = np.random.default_rng(seed)
        for _ in range(max_tries):
            vids = uniq_groups.copy(); rng.shuffle(vids)
            g2hist = {}
            for g in vids:
                idx = np.where(groups==g)[0]
                hist = np.bincount(y[idx], minlength=len(label2id))
                g2hist[g] = hist
            fold_vids = [[] for _ in range(cv)]
            fold_hist = [np.zeros(len(label2id), dtype=np.int64) for _ in range(cv)]
            for g in vids:
                costs = []
                for k in range(cv):
                    new_hist = fold_hist[k] + g2hist[g]
                    cost = np.linalg.norm(new_hist - new_hist.mean())
                    costs.append((cost, k))
                k_best = sorted(costs)[0][1]
                fold_vids[k_best].append(g); fold_hist[k_best] += g2hist[g]
            all_ok = True
            for k in range(cv):
                va = set(fold_vids[k]); tr = set(vids) - va
                tr_idx = df.index[df["video"].astype(str).isin(tr)].to_numpy()
                if not ok_train_has_all(tr_idx): all_ok=False; break
            if not all_ok: continue
            for k in range(cv):
                va = set(fold_vids[k]); tr = set(vids) - va
                tr_idx = df.index[df["video"].astype(str).isin(tr)].to_numpy()
                va_idx = df.index[df["video"].astype(str).isin(va)].to_numpy()
                yield tr_idx, va_idx
            return
        raise RuntimeError("Failed to build grouped stratified folds (fallback).")

# ------------------------ features: base+Δ+Δ² -> 24ch ------------------------
def make_features(x: np.ndarray, seq_len: int) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] < 8:
        raise ValueError(f"Expected x shape [T,8+], got {x.shape}")
    T, D = x.shape
    if T != seq_len:
        old = np.linspace(0, 1, T, dtype=np.float32)
        new = np.linspace(0, 1, seq_len, dtype=np.float32)
        xr = np.zeros((seq_len, D), dtype=np.float32)
        for d in range(D):
            col = x[:, d].astype(np.float32)
            mask = np.isfinite(col)
            if mask.sum() >= 2:
                xr[:, d] = np.interp(new, old[mask], col[mask])
            elif mask.sum() == 1:
                xr[:, d] = col[mask][0]
            else:
                xr[:, d] = np.nan
    else:
        xr = x.astype(np.float32)
    delta  = np.diff(xr, axis=0, prepend=xr[0:1])
    delta2 = np.diff(delta, axis=0, prepend=delta[0:1])
    feats = np.concatenate([xr[:, :8], delta[:, :8], delta2[:, :8]], axis=1)  # (L,24)
    return feats.astype(np.float32)

def _nanfix_timewise(x24: np.ndarray) -> np.ndarray:
    L, C = x24.shape
    idx = np.arange(L)
    for c in range(C):
        col = x24[:, c]
        if np.isnan(col).any():
            mask = np.isfinite(col)
            if mask.any():
                col[~mask] = np.interp(idx[~mask], idx[mask], col[mask])
            else:
                col[:] = 0.0
            x24[:, c] = col
    return x24

def load_angles(processed_dir: str, video: str) -> np.ndarray:
    base = os.path.join(processed_dir, f"{video}")
    for ext in (".npy", ".npz"):
        p = base + ext
        if os.path.isfile(p):
            if ext == ".npy":
                return np.load(p)
            else:
                with np.load(p) as z:
                    for k in ("arr", "data", "angles"):
                        if k in z: return z[k]
    raise FileNotFoundError(f"Missing processed angles for video={video}")

def window_nan_ratio(arr8: np.ndarray, s: int, e: int) -> float:
    seg = arr8[s:e, :8]
    return float(np.isnan(seg).sum()) / max(1, seg.size)

def compute_norm_stats_24(df_sub: pd.DataFrame, processed_dir: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray, Dict[str,int]]:
    cache: Dict[str, np.ndarray] = {}
    rows = []
    for _, r in df_sub.iterrows():
        video = str(r["video"])
        if video not in cache:
            cache[video] = load_angles(processed_dir, video).astype(np.float32)
        vid = cache[video]
        s = int(r["start_frame"])
        e = int(r["end_frame"]) if not pd.isna(r["end_frame"]) else s + seq_len
        s = max(0, min(s, len(vid)-1)); e = max(s+1, min(e, len(vid)))
        x = vid[s:e, :8]
        rows.append(make_features(x, seq_len))
    M = np.concatenate(rows, axis=0)
    n_total = int(M.size); n_nan = int(np.isnan(M).sum())
    mean24 = np.nanmean(M, axis=0, keepdims=True).astype(np.float32)
    std24  = np.nanstd(M,  axis=0, keepdims=True).astype(np.float32)
    std24  = np.clip(std24, 1e-6, None)
    counters = {"norm_total_vals": n_total, "norm_nan_vals": n_nan,
                "mean_has_nan": int(np.isnan(mean24).sum()), "std_has_nan": int(np.isnan(std24).sum())}
    return mean24, std24, counters

# ------------------------ dataset ------------------------
class FlapDataset(Dataset):
    def __init__(self, df, processed_dir, seq_len, label2id, mean24, std24):
        self.df = df.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.seq_len = seq_len
        self.label2id = label2id
        self.mean24 = mean24.astype(np.float32)
        self.std24  = np.clip(std24.astype(np.float32), 1e-6, None)
        self._cache: Dict[str, np.ndarray] = {}

    def _load_video(self, video: str) -> np.ndarray:
        if video not in self._cache:
            self._cache[video] = load_angles(self.processed_dir, video).astype(np.float32)
        return self._cache[video]

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        vid_arr = self._load_video(str(r["video"]))
        s = int(r["start_frame"]); e = int(r["end_frame"]) if not pd.isna(r["end_frame"]) else s + self.seq_len
        s = max(0, min(s, len(vid_arr)-1)); e = max(s+1, min(e, len(vid_arr)))
        x = vid_arr[s:e, :8]
        x24 = make_features(x, self.seq_len)
        x24 = _nanfix_timewise(x24)
        x24 = (x24 - self.mean24) / self.std24
        x24 = np.nan_to_num(x24, nan=0.0, posinf=0.0, neginf=0.0)
        x_ch_first = torch.from_numpy(x24.T.astype(np.float32))  # (24, L)
        y = torch.tensor(self.label2id[str(r["label"])], dtype=torch.long)
        conf = torch.tensor(float(r.get("confidence", 1.0)), dtype=torch.float32)
        return x_ch_first, y, conf

# ------------------------ model & loss ------------------------
class BiLSTM(nn.Module):
    def __init__(self, in_dim: int = None, in_ch: int = None, hidden_size: int = 128,
                 num_layers: int = 1, n_classes: int = 8,
                 bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        if in_ch is not None and in_dim is None:
            in_dim = in_ch  # 兼容 ensemble 脚本
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, n_classes)

    def forward(self, x):               # x: (B, C, L)
        x = x.transpose(1, 2)           # -> (B, L, C)
        out, _ = self.lstm(x)           # (B, L, D)
        h = out.mean(dim=1)             # temporal average pooling
        h = self.head_dropout(h)
        return self.fc(h)
LSTM1D = BiLSTM

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, reduction='none'):
        super().__init__()
        self.weight = weight; self.gamma = gamma; self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    def forward(self, logits, target):
        ce = self.ce(logits, target)  # (B,)
        with torch.no_grad():
            pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

@torch.no_grad()
def evaluate(model, loader, device, class_names: List[str]):
    model.eval(); y_true=[]; y_pred=[]
    for xb, yb, _ in loader:
        logits = model(xb.to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_pred += pred; y_true += yb.numpy().tolist()
    uniq = sorted(list(set(y_true + y_pred)))
    used_names = [class_names[i] for i in uniq]
    rep = classification_report(y_true, y_pred, labels=uniq, target_names=used_names,
                                output_dict=True, zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return rep, f1m, acc, (y_true, y_pred)

def calc_high_agg_f1(report_dict: Dict[str, Dict], class_names: List[str]) -> float:
    f1s = []
    for cname in class_names:
        if cname in HIGH_CLASSES and cname in report_dict:
            f1s.append(float(report_dict[cname]["f1-score"]))
    return float(np.mean(f1s)) if f1s else 0.0

def apply_freeze(model: nn.Module, mode: str) -> int:
    n_frozen = 0
    def freeze_param(p):
        nonlocal n_frozen
        p.requires_grad = False
        n_frozen += p.numel()

    if mode == "none":
        pass
    elif mode == "early2":
        # 冻结 LSTM 第0层（含反向）
        for name, p in model.lstm.named_parameters():
            if ("l0" in name):  # covers l0 和 l0_reverse
                freeze_param(p)
    elif mode in ("encoder", "all"):
        for p in model.lstm.parameters(): freeze_param(p)
    else:
        print(f"[WARN] Unknown freeze mode '{mode}', fallback to 'none'")
    return n_frozen

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed_angles")
    ap.add_argument("--labels_csv",   default="data/labels/weak_labels.csv")
    ap.add_argument("--mode", choices=["full","clean_only","conf_weighted"], default="clean_only")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    # imbalance
    ap.add_argument("--balancing", choices=["none","sampler","class_weight","both"], default="none")
    ap.add_argument("--sampler_alpha", type=float, default=0.5)
    ap.add_argument("--cw_beta", type=float, default=0.5)
    ap.add_argument("--cw_clip", type=float, default=3.0)
    ap.add_argument("--conf_alpha", type=float, default=1.0)
    # loss
    ap.add_argument("--loss", choices=["ce","focal"], default="ce")
    ap.add_argument("--gamma", type=float, default=2.0)
    # data hygiene
    ap.add_argument("--max_nan_ratio", type=float, default=0.6)
    # model
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    # fine-tune
    ap.add_argument("--freeze", choices=["none","early2","encoder","all"], default="none")
    ap.add_argument("--resume_from", type=str, default="", help=r'Pattern path e.g. runs\xxx\best_fold{fold}.pt')
    args = ap.parse_args()

    set_seed(args.seed); ensure_dir(args.outdir)

    # save arch meta 方便 ensemble 读取
    save_json({
        "arch": "lstm",
        "in_dim": 24,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "bidirectional": True,
        "dropout": args.dropout
    }, os.path.join(args.outdir, "arch.json"))

    df = pd.read_csv(args.labels_csv)
    req = {"video","start_frame","end_frame","label"}
    if not req.issubset(df.columns): raise KeyError(f"labels_csv must contain {req}, got {df.columns}")

    if args.mode == "clean_only":
        df = df[df["uncertain"].fillna(0).astype(int) == 0].copy()
        df["confidence"] = 1.0
    elif args.mode == "full":
        df = df.copy(); df["confidence"] = 1.0
    else:
        df["confidence"] = df["confidence"].fillna(1.0).astype(float)

    classes = [c for c in TAXONOMY if c in df["label"].unique().tolist()]
    label2id = {c:i for i,c in enumerate(classes)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device={device} | classes={classes} | N={len(df)}")
    print("[classes]", classes)
    print("[label2id]", label2id)

    fold_rows=[]; all_true=[]; all_pred=[]

    for fold, (tr_idx, va_idx) in enumerate(stratified_group_kfold(df, label2id, args.cv, args.seed), 1):
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        tr_counts = df_tr["label"].value_counts().reindex(classes, fill_value=0).to_dict()
        va_counts = df_va["label"].value_counts().reindex(classes, fill_value=0).to_dict()
        print(f"[Fold {fold}] TRAIN counts:", tr_counts)
        print(f"[Fold {fold}]  VAL  counts:", va_counts)
        if any(v==0 for v in tr_counts.values()):
            raise RuntimeError(f"[Fold {fold}] TRAIN is missing classes — splitting failed.")

        # drop high-NaN windows
        drop_tr, keep_tr = 0, []
        cache = {}
        for i, r in df_tr.iterrows():
            vid = str(r["video"])
            if vid not in cache: cache[vid] = load_angles(args.processed_dir, vid).astype(np.float32)
            arr = cache[vid]
            s = int(r["start_frame"]); e = int(r["end_frame"]) if not pd.isna(r["end_frame"]) else s + args.seq_len
            s = max(0, min(s, len(arr)-1)); e = max(s+1, min(e, len(arr)))
            ratio = window_nan_ratio(arr, s, e)
            if ratio <= args.max_nan_ratio: keep_tr.append(i)
            else: drop_tr += 1
        if drop_tr: print(f"[Fold {fold}] Dropped {drop_tr} train windows for NaN ratio > {args.max_nan_ratio}")
        df_tr = df_tr.iloc[keep_tr].reset_index(drop=True)

        # norm stats (train only)
        mean24, std24, counters = compute_norm_stats_24(df_tr, args.processed_dir, args.seq_len)
        print(f"[Fold {fold}] Norm counters: {counters}")
        print(f"[Fold {fold}] mean has NaN? {np.isnan(mean24).sum()} | std has NaN? {np.isnan(std24).sum()}")

        ds_tr = FlapDataset(df_tr, args.processed_dir, args.seq_len, label2id, mean24, std24)
        ds_va = FlapDataset(df_va, args.processed_dir, args.seq_len, label2id, mean24, std24)

        y_tr = df_tr["label"].map(label2id).to_numpy()
        vals, cnts = np.unique(y_tr, return_counts=True)
        k = len(classes)
        cnt_map = {int(v): int(c) for v,c in zip(vals, cnts)}

        # class weights
        if args.balancing in ("class_weight","both"):
            median_cnt = float(np.median(cnts)) if len(cnts)>0 else 1.0
            cw = np.array([ (median_cnt / cnt_map.get(i, median_cnt)) ** args.cw_beta for i in range(k) ], dtype=np.float32)
            cw = np.clip(cw, 1.0/args.cw_clip, args.cw_clip)
            cls_w_t = torch.tensor(cw, dtype=torch.float32, device=device)
            print(f"[Fold {fold}] class_weight ON | beta={args.cw_beta} clip={args.cw_clip} | weights={cw.round(3).tolist()}")
        else:
            cls_w_t = None
            print(f"[Fold {fold}] class_weight OFF")

        # loss
        if args.loss == "focal":
            criterion = FocalLoss(weight=cls_w_t, gamma=args.gamma, reduction='none')
            print(f"[Fold {fold}] loss=FocalLoss(gamma={args.gamma})")
        else:
            criterion = nn.CrossEntropyLoss(weight=cls_w_t, reduction='none')
            print(f"[Fold {fold}] loss=CrossEntropy")

        # sampler / loader
        pin = device.type == "cuda"
        if args.balancing in ("sampler","both"):
            sw = np.array([ (1.0 / cnt_map[int(y)]) ** args.sampler_alpha for y in y_tr ], dtype=np.float32)
            sampler = WeightedRandomSampler(sw.tolist(), num_samples=len(sw), replacement=True)
            dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=pin)
            print(f"[Fold {fold}] sampler ON | alpha={args.sampler_alpha}")
        else:
            dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
            print(f"[Fold {fold}] sampler OFF (shuffle=True)")
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

        print(f"[Fold {fold}] conf_alpha={args.conf_alpha}")

        # model / opt
        model = BiLSTM(in_dim=24, hidden_size=args.hidden_size, num_layers=args.num_layers,
                       n_classes=len(classes), bidirectional=True, dropout=args.dropout).to(device)

        # resume
        if args.resume_from:
            ckpt_path = args.resume_from.format(fold=fold)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[Fold {fold}] Loaded checkpoint: {ckpt_path}")

        # freeze
        n_frozen = apply_freeze(model, args.freeze)
        print(f"[Fold {fold}] Freeze mode='{args.freeze}' | frozen params={n_frozen}")

        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)

        best = -1.0; patience=10; bad=0
        for ep in range(1, args.epochs+1):
            model.train(); total = 0.0; ep_pred=[]
            for xb, yb, conf in dl_tr:
                xb = xb.to(device); yb = yb.to(device); conf = conf.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss_vec = criterion(logits, yb)
                weight = torch.ones_like(conf) if args.conf_alpha == 0 else conf.pow(args.conf_alpha)
                loss = (loss_vec * weight).mean()
                loss.backward(); opt.step()
                total += float(loss) * xb.size(0)
                ep_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
            tr_loss = total / max(1, len(dl_tr.dataset))
            tr_dist = np.bincount(ep_pred, minlength=len(classes)).tolist()
            if sum(tr_dist) > 0 and (max(tr_dist) / sum(tr_dist) > 0.95):
                print(f"[Fold {fold}] [WARN] training collapse detected (pred_dist≈{tr_dist}).")

            rep_va, f1m, acc, _ = evaluate(model, dl_va, device, classes)
            high_f1 = calc_high_agg_f1(rep_va, classes)
            print(f"[Fold {fold}] Epoch {ep:02d}/{args.epochs} | loss={tr_loss:.4f} | f1_macro={f1m:.4f} | highF1={high_f1:.4f} | acc={acc:.4f} | pred_dist_tr={tr_dist}")

            if f1m > best:
                best = f1m; bad = 0
                torch.save(model.state_dict(), os.path.join(args.outdir, f"best_fold{fold}.pt"))
            else:
                bad += 1
                if bad >= patience:
                    print(f"[Fold {fold}] Early stopping at epoch {ep}."); break

        # final eval
        model.load_state_dict(torch.load(os.path.join(args.outdir, f"best_fold{fold}.pt"), map_location=device))
        rep_va, f1m, acc, (yt, yp) = evaluate(model, dl_va, device, classes)
        high_f1 = calc_high_agg_f1(rep_va, classes)
        pred_dist = np.bincount(yp, minlength=len(classes)).tolist()
        print(f"[Fold {fold}] final f1_macro={f1m:.4f} highF1={high_f1:.4f} acc={acc:.4f} | pred_dist={pred_dist}")

        fold_rows.append({"fold": fold, "f1_macro": f1m, "highF1": high_f1, "accuracy": acc})
        all_true += yt; all_pred += yp
        save_json(rep_va, os.path.join(args.outdir, f"report_fold{fold}.json"))

    # overall
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.outdir, "metrics_lstm.csv"), index=False)
    uniq = sorted(list(set(all_true + all_pred)))
    used_names = [classes[i] for i in uniq]
    rep_all = classification_report(all_true, all_pred, labels=uniq, target_names=used_names,
                                    output_dict=True, zero_division=0)
    save_json(rep_all, os.path.join(args.outdir, "report_overall.json"))
    overall_pred_dist = np.bincount(all_pred, minlength=len(classes)).tolist()
    print(f"[OVERALL] pred_dist={overall_pred_dist}")
    print("[DONE] Saved:", args.outdir)

if __name__ == "__main__":
    main()
