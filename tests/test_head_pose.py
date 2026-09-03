"""
Test nhỏ cho head_pose (add-on) - CHỈ kiểm tra hành vi CƠ HỌC (đúng kiểu dữ liệu, xử lý
input suy biến), KHÔNG kiểm tra độ CHÍNH XÁC của góc ước lượng - việc đó thuộc về
scripts/validate_head_pose.py (so với ground truth thật trong sampled_labels.csv), vì đây
là số đo độ chính xác của 1 phép xấp xỉ hình học, không phải hợp đồng đúng/sai cố định của
code. Chạy tay bằng: pytest tests/test_head_pose.py -v

Landmark "chính diện" dưới đây là SỐ THẬT lấy từ insightface chạy trên ảnh 63374.png (FFHQ) -
ảnh này có ground-truth head_yaw=-0.18°, head_pitch=0.33°, head_roll=0.50° trong
sampled_labels.csv (gần 0 nhất trong 140 ảnh) - KHÔNG bịa tọa độ tay, vì lần đầu viết test
này đã tự bịa landmark và vô tình tạo ra input không đại diện ảnh chính diện thật (phát hiện
khi chạy thử: pitch ra 172° - sai hoàn toàn), gây hiểu lầm là code lỗi trong khi lỗi ở test.
"""

import numpy as np

from src.utils.head_pose import check_image_quality, estimate_head_pose

IMAGE_SHAPE = (256, 256, 3)

# Landmark THẬT lấy tu insightface.detect_faces() tren anh 63374.png (xem docstring tren).
FRONTAL_LANDMARKS = np.array(
    [
        [98.078445, 120.911514],
        [160.157516, 119.940224],
        [133.197830, 152.754288],
        [101.358292, 178.353943],
        [157.122040, 177.221161],
    ]
)

# 5 diem moc suy bien - trung nhau hoan toan, khong mang thong tin hinh hoc gi.
DEGENERATE_LANDMARKS = np.array([[128.0, 128.0]] * 5)

# Landmark bat doi xung cuc doan (mat trai/phai lech han sang 1 ben) - mo phong mat quay
# nghieng rat manh, de kiem tra nhanh canh bao yaw lon co kich hoat khong.
EXTREME_YAW_LANDMARKS = np.array(
    [
        [230.0, 120.0],
        [245.0, 118.0],
        [220.0, 150.0],
        [225.0, 178.0],
        [240.0, 176.0],
    ]
)


def test_estimate_head_pose_returns_three_floats_for_frontal_landmarks():
    """Input 5 điểm mốc thật (ảnh chính diện) phải trả về đúng 1 tuple 3 số thực, không None."""
    pose = estimate_head_pose(FRONTAL_LANDMARKS, IMAGE_SHAPE)

    assert pose is not None
    assert len(pose) == 3
    assert all(isinstance(a, float) for a in pose)


def test_estimate_head_pose_returns_none_for_degenerate_landmarks():
    """5 điểm mốc trùng nhau hoàn toàn (suy biến, không mang thông tin hình học) -> trả về
    None, KHÔNG raise exception. Chặn tường minh bằng kiểm tra độ trải rộng (xem FIX trong
    head_pose.py) - đã xác nhận qua chạy thử: SOLVEPNP_EPNP KHÔNG tự phát hiện case này."""
    pose = estimate_head_pose(DEGENERATE_LANDMARKS, IMAGE_SHAPE)

    assert pose is None


def test_check_image_quality_warns_on_extreme_yaw():
    """Landmark bất đối xứng cực đoan (mô phỏng mặt quay nghiêng rất mạnh) phải sinh cảnh
    báo yaw, dù độ tin cậy detection cao."""
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

    warnings = check_image_quality(image, EXTREME_YAW_LANDMARKS, detection_confidence=0.99)

    assert any("nghiêng (quay ngang)" in w for w in warnings)


def test_check_image_quality_warns_on_low_confidence():
    """Độ tin cậy phát hiện thấp phải sinh ra cảnh báo tương ứng, dù landmark chính diện."""
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

    warnings = check_image_quality(image, FRONTAL_LANDMARKS, detection_confidence=0.5)

    assert any("tin cậy" in w for w in warnings)


def test_check_image_quality_warns_on_degenerate_pose():
    """Landmark suy biến (không ước lượng được góc) phải sinh cảnh báo riêng cho việc đó."""
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

    warnings = check_image_quality(image, DEGENERATE_LANDMARKS, detection_confidence=0.99)

    assert any("góc nghiêng" in w for w in warnings)
