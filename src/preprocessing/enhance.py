"""Cac buoc tang cuong nhe theo tung dieu kien (chi OpenCV, do thuc te
~180ms/frame 960px CPU truong hop xau nhat; frame sach chi mat ~30ms metric).

Muc tieu: dua frame ve mien sang/tuong phan gan voi anh dieu kien tot de
InsightFace detect + embedding on dinh hon. Day la xu ly nhe (khu mua DL,
siêu phan giai... nam ngoai pham vi) — khong bien anh xau thanh anh dep,
chi giam tac hai cua dieu kien xau.
"""

import cv2
import numpy as np


def _apply_to_l_channel(image_bgr: np.ndarray, fn) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = fn(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _gamma_lut(gamma: float) -> np.ndarray:
    """Bang tra gamma: out = (in/255)^gamma. gamma < 1 lam sang (dem, nguoc
    sang), gamma > 1 lam toi."""
    table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)])
    return np.clip(table, 0, 255).astype(np.uint8)


def lift_night(image_bgr: np.ndarray) -> np.ndarray:
    """Troi toi/ban dem: nang gamma + CLAHE kenh L + giam nhieu nhe."""
    out = _apply_to_l_channel(image_bgr, lambda l: cv2.LUT(l, _gamma_lut(0.6)))
    out = _apply_to_l_channel(out, lambda l: cv2.createCLAHE(clipLimit=2.0).apply(l))
    return cv2.bilateralFilter(out, d=5, sigmaColor=50, sigmaSpace=50)


def fix_backlight(image_bgr: np.ndarray) -> np.ndarray:
    """Nguoc sang (mat toi, nen sang): CLAHE manh + nang gamma vua."""
    out = _apply_to_l_channel(image_bgr, lambda l: cv2.createCLAHE(clipLimit=2.5).apply(l))
    return _apply_to_l_channel(out, lambda l: cv2.LUT(l, _gamma_lut(0.8)))


def compress_glare(image_bgr: np.ndarray) -> np.ndarray:
    """Nang choi/chay sang: nen vung highlight (V>200) + CLAHE nhe."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_f = v.astype(np.float32)
    over = v_f > 200.0
    v_f[over] = 200.0 + (v_f[over] - 200.0) * 0.5
    v = np.clip(v_f, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    return _apply_to_l_channel(out, lambda l: cv2.createCLAHE(clipLimit=1.5).apply(l))


def dehaze_simple(image_bgr: np.ndarray, omega: float = 0.9) -> np.ndarray:
    """Khu suong mu: dark-channel prior rut gon (He et al. 2009), transmission
    lam min bang box blur thay guided filter de chay nhanh."""
    img = image_bgr.astype(np.float64) / 255.0
    dark = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(dark, kernel)

    flat_dark = dark.ravel()
    flat_img = img.reshape(-1, 3)
    n_atm = max(1, int(flat_dark.size * 0.001))
    idx = np.argpartition(flat_dark, -n_atm)[-n_atm:]
    atm = np.max(flat_img[idx], axis=0)  # anh sang khi quyen A
    atm = np.maximum(atm, 0.2)

    trans = 1.0 - omega * dark / max(float(np.max(atm)), 1e-6)
    trans = cv2.blur(trans, (15, 15))
    trans = np.clip(trans, 0.1, 1.0)[:, :, np.newaxis]

    recovered = (img - atm) / trans + atm
    return (np.clip(recovered, 0.0, 1.0) * 255.0).astype(np.uint8)


def suppress_rain(image_bgr: np.ndarray) -> np.ndarray:
    """Giam vet mua: loc song phuong giu bien + unsharp khoi phuc chi tiet."""
    smooth = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=75, sigmaSpace=75)
    blur = cv2.GaussianBlur(smooth, (0, 0), 2.0)
    return cv2.addWeighted(smooth, 1.5, blur, -0.5, 0)


def deblur(image_bgr: np.ndarray) -> np.ndarray:
    """Anh mo: unsharp masking nhe."""
    blur = cv2.GaussianBlur(image_bgr, (0, 0), 2.0)
    return cv2.addWeighted(image_bgr, 1.6, blur, -0.6, 0)
