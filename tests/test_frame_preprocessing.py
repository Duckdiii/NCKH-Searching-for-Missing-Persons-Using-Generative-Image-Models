"""Test cho src/preprocessing (tien xu ly frame video/camera).

Dung anh tong hop nho (160x120) de chay nhanh, khong can model.
Chi assert membership ("night" in conds) thay vi equality vi 1 frame
co the mang nhieu nhan cung luc (vd mua trang + choi sang).
"""

import cv2
import numpy as np
import pytest

from src.preprocessing import preprocess_video_frame
from src.preprocessing import conditions as cond
from src.preprocessing import enhance as enh

W, H = 160, 120


def _blocks(seed: int = 0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(80, 180, size=(H // 10, W // 10, 3), dtype=np.uint8)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    return np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def _l_mean(img: np.ndarray) -> float:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return float(np.mean(lab[:, :, 0]))


def test_normal_frame_has_no_condition():
    img = _blocks()
    assert cond.classify_conditions(img) == []


def test_night_detected_and_lifted():
    img = _blocks(scale=0.2)
    assert "night" in cond.classify_conditions(img)
    out = enh.lift_night(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert _l_mean(out) > _l_mean(img)


def test_glare_detected_and_compressed():
    img = _blocks()
    img[10:50, 10:150] = 255  # ~21% pixel chay sang
    assert "glare" in cond.classify_conditions(img)
    hsv_before = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    out = enh.compress_glare(img)
    hsv_after = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[:, :, 2]
    assert float(np.mean(hsv_after > 235)) < float(np.mean(hsv_before > 235))


def test_rain_detected():
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    img[:, ::7] = 255  # vet mua doc trang
    assert "rain" in cond.classify_conditions(img)
    out = enh.suppress_rain(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_fog_detected_and_dehazed():
    rng = np.random.default_rng(1)
    noise = rng.integers(-8, 9, size=(H, W, 3)).astype(np.int16)
    img = np.clip(np.full((H, W, 3), 170, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    assert "fog" in cond.classify_conditions(img)
    out = enh.dehaze_simple(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)
    lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float64)
    assert float(np.std(lab_out)) > float(np.std(lab_in))


def test_blur_detected_and_sharpened():
    rng = np.random.default_rng(2)
    small = rng.integers(0, 256, size=(H // 10, W // 10, 3), dtype=np.uint8)
    sharp = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    tiny = cv2.resize(sharp, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    img = cv2.resize(tiny, (W, H), interpolation=cv2.INTER_CUBIC)  # mo tu nhien
    assert "blur" in cond.classify_conditions(img)
    out = enh.deblur(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_pipeline_report_and_passthrough():
    img = _blocks(scale=0.2)
    out, report = preprocess_video_frame(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert "night" in report["conditions"]
    assert len(report["applied"]) > 0
    assert "l_mean" in report["metrics"]

    out2, report2 = preprocess_video_frame(img, enabled=False)
    assert report2["conditions"] == ["passthrough"]
    assert np.array_equal(out2, img)

    with pytest.raises(ValueError):
        preprocess_video_frame(None)
