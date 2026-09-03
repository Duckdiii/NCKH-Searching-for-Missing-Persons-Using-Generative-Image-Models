"""
Test nhỏ cho AgeEstimator (MiVOLO) - chạy độc lập.

Cần checkpoint MiVOLO đã tải sẵn về đường dẫn trong configs/config.yaml
(yolov8x_person_face.pt + mivolo_imdb.pth.tar). Chạy tay:  pytest tests/test_age_estimator.py -v

KHÔNG so sánh với ground truth ở đây - việc đánh giá độ chính xác (FG-NET/Nhánh A) làm riêng
trên Colab. Test này chỉ xác nhận: (1) estimate() trả về int hợp lý 0-120, (2) raise ValueError
rõ ràng khi ảnh không tồn tại.
"""

import glob
import os

import pytest
import yaml

from src.utils.age_estimator import AgeEstimator, resolve_initial_age

CONFIG_PATH = "configs/config.yaml"
FFHQ_DIR = "D:/Data/project/nckh/ffhq_aging_150_samples"
LABELS_CSV = f"{FFHQ_DIR}/sampled_labels.csv"


def _make_estimator() -> AgeEstimator:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["age_estimator"]
    for key in ("detector_checkpoint", "age_checkpoint"):
        if not os.path.isfile(cfg[key]):
            pytest.skip(f"Thiếu checkpoint MiVOLO: {cfg[key]}")
    return AgeEstimator(cfg["detector_checkpoint"], cfg["age_checkpoint"], cfg["device"])


def test_estimate_returns_reasonable_int():
    """estimate() trên 1 ảnh mẫu bất kỳ -> int trong [0, 120]."""
    estimator = _make_estimator()
    image_path = sorted(glob.glob(os.path.join(FFHQ_DIR, "*.png")))[0]

    age = estimator.estimate(image_path)

    assert isinstance(age, int)
    assert 0 <= age <= 120


def test_estimate_raises_valueerror_on_missing_file():
    """Đường dẫn ảnh không tồn tại -> ValueError rõ ràng (không phải FileNotFoundError trần)."""
    estimator = _make_estimator()

    with pytest.raises(ValueError):
        estimator.estimate("D:/khong/ton/tai/anh_khong_co_that.png")


# ===== Test resolve_initial_age() - 3 nguồn IA tách biệt (không cần model, chạy nhanh) =====


def test_resolve_initial_age_manual_returns_exact_value():
    """mode='manual' phải trả về ĐÚNG giá trị nhập tay, không làm tròn/biến đổi gì."""
    assert resolve_initial_age(mode="manual", manual_age=8) == 8


def test_resolve_initial_age_manual_raises_on_out_of_range():
    """manual_age ngoài [0,120] phải raise ValueError, không âm thầm clamp lại."""
    with pytest.raises(ValueError):
        resolve_initial_age(mode="manual", manual_age=150)


def test_resolve_initial_age_manual_raises_when_missing_manual_age():
    """mode='manual' mà không truyền manual_age phải raise ValueError."""
    with pytest.raises(ValueError):
        resolve_initial_age(mode="manual")


def test_resolve_initial_age_csv_raises_on_unknown_image_number():
    """mode='csv' với image_number không tồn tại trong CSV phải raise ValueError - KHÔNG
    được trả về None hay 0."""
    with pytest.raises(ValueError):
        resolve_initial_age(mode="csv", labels_csv=LABELS_CSV, image_number=999999999)


def test_resolve_initial_age_mivolo_raises_when_missing_age_estimator():
    """mode='mivolo' thiếu age_estimator phải raise ValueError - không được tự tạo estimator
    ngầm hay bỏ qua."""
    with pytest.raises(ValueError):
        resolve_initial_age(mode="mivolo", image_path=f"{FFHQ_DIR}/01366.png")


def test_resolve_initial_age_invalid_mode_raises():
    """mode không thuộc {csv, manual, mivolo} phải raise ValueError rõ ràng."""
    with pytest.raises(ValueError):
        resolve_initial_age(mode="khong_ton_tai")
