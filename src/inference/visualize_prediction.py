# -*- coding: utf-8 -*-
"""
Visualise a video with overlaid predictions, auto-fit to screen (safe DPI scaling, no cropping).
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]

def resolve_path(p: str) -> Path:
    pth = Path(p)
    if pth.exists():
        return pth
    cand = ROOT / p
    return cand if cand.exists() else pth


def get_screen_info():
    """Get physical screen size, safe usable height (minus taskbar), and DPI scaling factor."""
    sw, sh, dpi_scale = 1920, 1080, 1.0
    usable_h = sh
    if platform.system().lower() == "windows":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            sw = int(user32.GetSystemMetrics(0))
            sh = int(user32.GetSystemMetrics(1))
            # 获取DPI比例（缩放倍数）
            try:
                awareness = ctypes.c_int()
                ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
                dpi_scale = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
            except Exception:
                dpi_scale = 1.0
            # 任务栏高度估计（约5%屏幕）
            usable_h = int(sh * 0.94)
        except Exception:
            pass
    else:
        # macOS/Linux fallback
        try:
            import tkinter as tk
            r = tk.Tk(); r.withdraw()
            sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
            r.destroy()
        except Exception:
            pass
    return sw, sh, usable_h, dpi_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--offset", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--max_screen_ratio", type=float, default=None,  # 可手动覆盖
                    help="Manually override screen usage ratio (0–1). Leave blank for auto.")
    args = ap.parse_args()

    # === 路径解析 ===
    video_path = resolve_path(args.video)
    pred_csv = resolve_path(args.pred_csv)
    if not video_path.exists():
        raise FileNotFoundError(f"❌ Video not found: {video_path}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"❌ Prediction CSV not found: {pred_csv}")

    # === 屏幕信息 ===
    sw, sh, usable_h, dpi = get_screen_info()
    print(f"🖥️ Screen: {sw}x{sh} usable={usable_h}px  |  DPI scale≈{dpi:.2f}")

    # === 加载预测 ===
    preds = pd.read_csv(pred_csv)
    stem = video_path.stem
    if "video" in preds.columns and not {"start_frame", "end_frame"}.issubset(preds.columns):
        preds = preds[preds["video"].astype(str).str.contains(stem, na=False)].reset_index(drop=True)
    if len(preds) == 0:
        raise RuntimeError(f"❌ No predictions for '{stem}' in {pred_csv}")

    use_range = {"start_frame", "end_frame"}.issubset(preds.columns)
    if use_range:
        start_frames = preds["start_frame"].astype(int).to_numpy()
        end_frames = preds["end_frame"].astype(int).to_numpy()

    # === 打开视频 ===
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌ Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # === 自动计算安全缩放比例 ===
    # 默认情况下，会留出任务栏与DPI缩放的余量
    if args.max_screen_ratio is None:
        safe_ratio = min(0.85 / dpi, 0.9)  # 自动安全值
    else:
        safe_ratio = float(args.max_screen_ratio)
    scale = min((sw * safe_ratio) / vw, (usable_h * safe_ratio) / vh)
    disp_w, disp_h = int(vw * scale), int(vh * scale)

    font_scale = max(0.6, 0.9 * (disp_h / 720.0))
    thickness = max(1, int(2 * (disp_h / 720.0)))
    top_pad = max(30, int(24 * (disp_h / 720.0)))
    bot_pad_y = disp_h - max(20, int(24 * (disp_h / 720.0)))

    cv2.namedWindow("Hand Flapping Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Flapping Detection", disp_w, disp_h)

    print(f"🎬 {video_path.name}: {vw}x{vh}@{fps:.1f} → display {disp_w}x{disp_h} (ratio={safe_ratio:.2f})")
    print("ℹ️ Press 'q' or ESC to quit.")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        q = max(frame_id - args.offset, 0)
        if use_range:
            i = int(np.searchsorted(start_frames, q, side="right") - 1)
            i = int(np.clip(i, 0, len(preds)-1))
        else:
            i = int(np.clip(q // args.seq_len, 0, len(preds)-1))

        row = preds.iloc[i]
        label = str(row.get("pred_label", "NA"))
        conf = float(row.get("confidence", 1.0))

        # 缩放帧
        disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # 绘制半透明背景
        cv2.rectangle(disp, (0, 0), (disp_w, int(top_pad * 1.8)), (0, 0, 0), -1)
        cv2.rectangle(disp, (0, disp_h - int(top_pad * 1.8)), (disp_w, disp_h), (0, 0, 0), -1)

        # 构建预测文字
        pred_text = f"Predicted: {label} ({conf:.2f})"

        # 计算文本尺寸（宽度）
        (text_w, text_h), _ = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 如果太宽（超过画面宽度90%），则自动缩小字体或换行
        max_width = int(disp_w * 0.9)
        if text_w > max_width:
            # 尝试缩放字体
            scale_factor = max_width / text_w
            adj_font_scale = font_scale * scale_factor
            if adj_font_scale < 0.6:
                # 太小则改为换行
                mid = len(pred_text) // 2
                line1, line2 = pred_text[:mid], pred_text[mid:]
                y1, y2 = top_pad, top_pad + int(text_h * 1.5)
                cv2.putText(disp, line1, (20, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 0), thickness)
                cv2.putText(disp, line2, (20, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 0), thickness)
            else:
                cv2.putText(disp, pred_text, (20, top_pad),
                            cv2.FONT_HERSHEY_SIMPLEX, adj_font_scale, (0, 255, 0), thickness)
        else:
            # 正常情况直接显示
            cv2.putText(disp, pred_text, (20, top_pad),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        cv2.putText(disp, "Press 'q' or ESC to quit",
                    (20, bot_pad_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255),
                    max(1, thickness - 1))

        cv2.imshow("Hand Flapping Detection", disp)
        key = cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF
        if key in [ord('q'), 27]:
            break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    main()
