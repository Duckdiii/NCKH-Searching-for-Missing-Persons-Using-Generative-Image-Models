import numpy as np
import pytest
from PIL import Image

from src.utils.face_enhancement import apply_adaptive_padding, apply_white_balance, run_codeformer


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
