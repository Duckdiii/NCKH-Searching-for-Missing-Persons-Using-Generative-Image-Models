"""Ghep nhan dien dieu kien + tang cuong thanh 1 ham duy nhat cho video/camera.

Thu tu ap dung co chu dich: khu mu -> giam mua -> can sang (toi/nguoc sang/
choi) -> can bang trang (tai dung Shades of Gray san co) -> khu mo.
Can bang trang va cac buoc deu bo qua anh xam de bao toan tong goc.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.preprocessing import conditions as cond
from src.preprocessing import enhance as enh
from src.utils.face_enhancement import apply_white_balance, is_effectively_grayscale

METRICS_MAX_DIM = 480  # do metric tren anh thu nho (phan loai khong can full-res)


def _small_for_metrics(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= METRICS_MAX_DIM:
        return frame_bgr
    scale = METRICS_MAX_DIM / float(longest)
    return cv2.resize(
        frame_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_video_frame(
    frame_bgr: np.ndarray,
    enabled: bool = True,
    apply_wb: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Tien xu ly 1 frame BGR. Tra ve (frame_da_xu_ly, report) voi report gồm:
    conditions (list nhan: normal/night/rain/glare/fog/backlight/blur),
    applied (list buoc da ap), metrics (thong ke lam tron)."""
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    if not enabled:
        return frame_bgr, {"conditions": ["passthrough"], "applied": [], "metrics": {}}

    metrics = cond.frame_metrics(_small_for_metrics(frame_bgr))
    found = cond.classify_conditions(frame_bgr, metrics)

    out = frame_bgr
    applied: List[str] = []
    if "fog" in found:
        out = enh.dehaze_simple(out)
        applied.append("dehaze")
    if "rain" in found:
        out = enh.suppress_rain(out)
        applied.append("derain")
    if "night" in found:
        out = enh.lift_night(out)
        applied.append("night_lift")
    elif "backlight" in found:
        out = enh.fix_backlight(out)
        applied.append("backlight_fix")
    if "glare" in found:
        out = enh.compress_glare(out)
        applied.append("glare_fix")

    if apply_wb:
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        if not is_effectively_grayscale(out_rgb):
            out = cv2.cvtColor(apply_white_balance(out_rgb), cv2.COLOR_RGB2BGR)
            applied.append("white_balance")

    if "blur" in found:
        out = enh.deblur(out)
        applied.append("deblur")

    labels = found if found else ["normal"]
    metrics_round = {k: round(float(v), 3) for k, v in metrics.items() if k != "total_px"}
    return out, {"conditions": labels, "applied": applied, "metrics": metrics_round}
