"""Nhan dien dieu kien xau tren frame video/camera (chi OpenCV + numpy).

Heuristic dua tren thong ke kenh mau/do sang — du nhanh de chay moi frame
video tren CPU, du on dinh de quyet dinh co ap buoc tang cuong nao.
1 frame co the mang nhieu nhan cung luc (vd dem + mua).

Nguong duoc chon theo kinh nghiem thuc te tren frame CCTV ngoai troi,
khong phai toi uu tuyet doi — chinh trong cac hang so *_THRESH neu can.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

NIGHT_L_MEAN = 55.0
NIGHT_DARK_RATIO = 0.35
GLARE_BRIGHT_RATIO = 0.12
FOG_CONTRAST_STD = 35.0
FOG_SAT_MEAN = 70.0
FOG_DARK_MEAN = 90.0
BLUR_LAPLACIAN_VAR = 80.0
RAIN_TOPHAT_RATIO = 0.004
RAIN_TOPHAT_THRESH = 30
BACKLIGHT_BORDER_GAP = 40.0
BACKLIGHT_BRIGHT_RATIO = 0.05


def frame_metrics(frame_bgr: np.ndarray) -> Dict[str, float]:
    """Do 1 lan, tai su dung cho ca classify + enhance (tranh tinh lap lai)."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0].astype(np.float64)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2].astype(np.float64)
    s_chan = hsv[:, :, 1].astype(np.float64)

    h, w = l_chan.shape
    total = float(h * w)
    dark_ratio = float(np.mean(v_chan < 40.0))
    bright_ratio = float(np.mean(v_chan > 235.0))

    # Do sang vung trung tam so voi vien (phat hien nguoc sang).
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    center_mean = float(np.mean(l_chan[cy0:cy1, cx0:cx1]))
    border_pixels = np.concatenate([
        l_chan[:cy0, :].ravel(), l_chan[cy1:, :].ravel(),
        l_chan[cy0:cy1, :cx0].ravel(), l_chan[cy0:cy1, cx1:].ravel(),
    ])
    border_mean = float(np.mean(border_pixels)) if border_pixels.size else center_mean

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Vệt mưa: white top-hat với kernel dọc + ngang, lấy max (mưa thường xiên).
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    k_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    streaks = np.maximum(
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_vert),
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_horz),
    )
    rain_ratio = float(np.mean(streaks > RAIN_TOPHAT_THRESH))

    # Dark channel (min tren 3 kenh BGR + blur nhe) cho phat hien suong mu.
    dark = np.min(frame_bgr.astype(np.float64), axis=2)
    dark = cv2.blur(dark, (15, 15))
    dark_mean = float(np.mean(dark))

    return {
        "l_mean": float(np.mean(l_chan)),
        "l_std": float(np.std(l_chan)),
        "sat_mean": float(np.mean(s_chan)),
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "center_mean": center_mean,
        "border_mean": border_mean,
        "lap_var": lap_var,
        "rain_ratio": rain_ratio,
        "dark_mean": dark_mean,
        "total_px": total,
    }


def classify_conditions(
    frame_bgr: np.ndarray, metrics: Optional[Dict[str, float]] = None
) -> List[str]:
    """Tra ve danh sach nhan dieu kien: night, rain, glare, fog, backlight,
    blur. Rong thi frame binh thuong (pipeline tu doi thanh 'normal')."""
    if metrics is None:
        metrics = frame_metrics(frame_bgr)

    conds: List[str] = []
    if metrics["l_mean"] < NIGHT_L_MEAN and metrics["dark_ratio"] > NIGHT_DARK_RATIO:
        conds.append("night")
    if metrics["rain_ratio"] > RAIN_TOPHAT_RATIO:
        conds.append("rain")
    if metrics["bright_ratio"] > GLARE_BRIGHT_RATIO:
        conds.append("glare")
    if (
        metrics["l_std"] < FOG_CONTRAST_STD
        and metrics["sat_mean"] < FOG_SAT_MEAN
        and metrics["dark_mean"] > FOG_DARK_MEAN
    ):
        conds.append("fog")
    if (
        "night" not in conds
        and metrics["border_mean"] - metrics["center_mean"] > BACKLIGHT_BORDER_GAP
        and metrics["bright_ratio"] > BACKLIGHT_BRIGHT_RATIO
    ):
        conds.append("backlight")
    # Anh toi/mu tu nhien co lap_var thap — bo qua nhan blur de tranh bao sai.
    if "night" not in conds and "fog" not in conds and metrics["lap_var"] < BLUR_LAPLACIAN_VAR:
        conds.append("blur")
    return conds
