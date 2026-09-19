import numpy as np
import pytest
from PIL import Image

from src.utils.face_enhancement import (
    apply_adaptive_padding,
    apply_white_balance,
    apply_white_balance_from_point,
    run_codeformer,
)


def test_apply_adaptive_padding_adds_border_when_needed():
    # Small dummy image (100x100) -> triggers padding
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    padded, was_padded = apply_adaptive_padding(img, pad_ratio=0.20)
    assert was_padded is True
    # 100 + 20*2 = 140
    assert padded.shape == (140, 140, 3)


def test_apply_white_balance_normalizes_colors():
    # Sepia/tinted image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # Red
    img[:, :, 1] = 160  # Green
    img[:, :, 2] = 80   # Blue

    wb_out, info = apply_white_balance(img, return_info=True)
    assert wb_out.shape == (100, 100, 3)
    assert "raw_gains" in info
    assert "clamped_gains" in info
    # Gains should boost blue (underrepresented) and reduce red
    assert info["clamped_gains"][2] > 1.0


def test_run_codeformer_fallback():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    # Should safely fallback to returning original image if CLI is not present
    out = run_codeformer(img)
    assert out.shape == img.shape


def test_preprocess_face_image_pipeline():
    from src.utils.face_enhancement import preprocess_face_image
    # Create 100x100 dummy face image with keypoints
    img_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 150
    kps = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32)

    preprocessed_bgr, updated_kps = preprocess_face_image(img_bgr, kps=kps)
    # Since image is small (< 300px), it should be padded by 20% on each side -> 140x140
    assert preprocessed_bgr.shape == (140, 140, 3)
    assert updated_kps is not None
    # Keypoints should be offset by pad_w=20, pad_h=20
    np.testing.assert_allclose(updated_kps, kps + np.array([20, 20]))


def test_is_effectively_grayscale_distinguishes_gray_and_color():
    from src.utils.face_enhancement import is_effectively_grayscale

    # Pure grayscale image (R=G=B)
    gray_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert is_effectively_grayscale(gray_img, threshold=6.0) is True

    # Slightly noisy monochrome image (diff < 6.0)
    noisy_gray = gray_img.copy()
    noisy_gray[:, :, 0] += 2
    noisy_gray[:, :, 2] -= 2
    assert is_effectively_grayscale(noisy_gray, threshold=6.0) is True

    # Clear color image (diff > 6.0)
    color_img = gray_img.copy()
    color_img[:, :, 0] = 200
    color_img[:, :, 2] = 50
    assert is_effectively_grayscale(color_img, threshold=6.0) is False


def test_preprocess_face_image_with_direct_bbox_and_grayscale():
    from src.utils.face_enhancement import preprocess_face_image

    # 400x400 monochrome image with face covering >85% (occupancy trigger)
    gray_img_bgr = np.ones((400, 400, 3), dtype=np.uint8) * 120
    bbox = (20.0, 20.0, 380.0, 380.0)  # 360/400 = 90% occupancy
    kps = np.array([[150, 150], [250, 150], [200, 220], [170, 280], [230, 280]], dtype=np.float32)

    preprocessed_bgr, updated_kps = preprocess_face_image(gray_img_bgr, kps=kps, bbox=bbox)
    # Should pad 20% = 80px on each side -> 400 + 160 = 560x560
    assert preprocessed_bgr.shape == (560, 560, 3)
    np.testing.assert_allclose(updated_kps, kps + np.array([80, 80]))
    # For grayscale, colors should remain completely neutral (R=G=B)
    assert np.all(preprocessed_bgr[:, :, 0] == preprocessed_bgr[:, :, 1])
    assert np.all(preprocessed_bgr[:, :, 1] == preprocessed_bgr[:, :, 2])


def test_apply_white_balance_from_point():
    # Sepia/tinted image: Red=200, Green=160, Blue=80
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 160
    img[:, :, 2] = 80

    out, info = apply_white_balance_from_point(img, click_x=50, click_y=50, return_info=True)
    assert out.shape == (100, 100, 3)
    assert info["click_point"] == (50, 50)
    # Gain should boost blue (underrepresented) and lower red
    assert info["clamped_gains"][2] > 1.0
    assert info["clamped_gains"][0] < 1.0


def test_apply_white_balance_from_point_bounds_and_safety_clamp():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Pure red -> extreme imbalance

    # Click outside bounds should safely clamp to (99, 99)
    out, info = apply_white_balance_from_point(img, click_x=150, click_y=-10, return_info=True)
    assert out.shape == (100, 100, 3)
    assert info["click_point"] == (99, 0)
    # Gains must be within safety clamp [0.75, 1.30]
    for g in info["clamped_gains"]:
        assert 0.75 <= g <= 1.30
    # Dynamic alpha should be active (alpha < 1.0) because of extreme color shift
    assert info["alpha"] <= 1.0


def test_preprocess_face_image_manual_options():
    from src.utils.face_enhancement import preprocess_face_image

    img_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 150
    kps = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32)

    # 1. Padding disabled -> should keep 100x100
    no_pad, kps_no_pad = preprocess_face_image(img_bgr, kps=kps, padding_enabled=False)
    assert no_pad.shape == (100, 100, 3)
    np.testing.assert_allclose(kps_no_pad, kps)

    # 2. White balance from point
    tinted_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    tinted_bgr[:, :, 0] = 80   # B
    tinted_bgr[:, :, 1] = 160  # G
    tinted_bgr[:, :, 2] = 200  # R
    out_bgr, _ = preprocess_face_image(
        tinted_bgr,
        kps=kps,
        padding_enabled=False,
        white_balance_enabled=True,
        wb_point=(50, 50),
        fidelity_weight=0.7
    )
    assert out_bgr.shape == (100, 100, 3)


