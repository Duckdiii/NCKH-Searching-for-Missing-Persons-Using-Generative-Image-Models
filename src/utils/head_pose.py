"""
Cảnh báo chất lượng ảnh đầu vào (add-on) - ước lượng góc nghiêng khuôn mặt (yaw/pitch/roll)
từ 5 điểm mốc (landmarks) qua cv2.solvePnP, để cảnh báo người dùng khi ảnh nghiêng nhiều
hoặc độ tin cậy phát hiện thấp (có thể bị che 1 phần) - KHÔNG chặn cứng, chỉ cảnh báo, vì
happy path không có UI xử lý ảnh xấu (đây là bước "khuyên", không phải "bắt buộc").

ĐÃ CHẠY scripts/validate_head_pose.py thật trên cả 140 ảnh FFHQ (so với head_yaw/head_pitch/
head_roll thật trong sampled_labels.csv). Kết quả thật (không phải ước lượng):
    MAE yaw:   5.40°  (median 3.89°, max 38.55°)  - kha tin cay
    MAE pitch: 14.60° (median 10.78°, max 71.17°) - SAI SO LON, kem tin cay hon han yaw/roll
    MAE roll:  2.20°  (median 1.79°, max 8.91°)   - kha tin cay
    False positive o nguong mac dinh: 2/140 anh bi bao "nghieng yaw" sai (that ra <30°),
    10/140 anh bi bao "nghieng pitch" sai (that ra <25°) - tat ca 140 anh FFHQ dung de test
    deu duoc gan nhan la khong nghieng qua muc trong CSV goc.

QUYẾT ĐỊNH: MAE pitch (14.6°) vượt quá nửa giá trị ngưỡng cảnh báo (25°) - sai số quá lớn để
tin dùng. ĐÃ BỎ cảnh báo pitch khỏi check_image_quality() (xem bên dưới) - CHỈ còn giữ cảnh
báo yaw + roll (MAE 5.4°/2.2° - đủ tin cậy) + độ tin cậy detection. estimate_head_pose() vẫn
tính và trả về pitch (không đổi API, không phá vỡ giá trị trả về) để scripts/validate_head_pose.py
vẫn đo được MAE pitch cho mục đích tham khảo/theo dõi sau này nếu model 3D điểm mốc được tinh
chỉnh lại - chỉ KHÔNG dùng pitch để cảnh báo người dùng nữa.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

# Mo hinh 5 diem moc 3D chuan (don vi mm, xap xi) dung trong cac tutorial OpenCV head-pose:
# mat trai, mat phai, mui, khoe mieng trai, khoe mieng phai - THEO DUNG THU TU insightface
# tra ve trong Face.kps (chuan alignment ArcFace 5 diem).
MODEL_3D_POINTS = np.array(
    [
        [-30.0, 30.0, -30.0],
        [30.0, 30.0, -30.0],
        [0.0, 0.0, 0.0],
        [-25.0, -30.0, -20.0],
        [25.0, -30.0, -20.0],
    ],
    dtype=np.float64,
)

YAW_THRESHOLD_DEFAULT = 30
PITCH_THRESHOLD_DEFAULT = 25
CONF_THRESHOLD_DEFAULT = 0.9


def estimate_head_pose(
    landmarks_2d: np.ndarray, image_shape: Tuple[int, int]
) -> Optional[Tuple[float, float, float]]:
    """Ước lượng (yaw, pitch, roll) độ, từ 5 điểm mốc 2D (mảng 5x2, cùng thứ tự
    MODEL_3D_POINTS) + kích thước ảnh (dùng dựng camera matrix xấp xỉ pinhole - focal length
    = chiều rộng ảnh, principal point = tâm ảnh; đây là xấp xỉ vì không có thông số ống kính
    camera thật, đặt trần hạn chế độ chính xác bất kể chỉnh ngưỡng thế nào).

    Trả về None (KHÔNG raise) nếu solvePnP không hội tụ, hoặc nếu landmark suy biến (5 điểm
    gần như trùng nhau - không mang thông tin hình học, xem FIX bên dưới) - đây là input
    "cảnh báo mềm" cho UI (check_image_quality), không phải lỗi pipeline cứng cần dừng
    chương trình."""
    # FIX (phát hiện qua chạy thử thật): SOLVEPNP_EPNP (dùng bên dưới) KHÔNG tự phát hiện
    # input suy biến (5 điểm trùng nhau) như kỳ vọng ban đầu - nó vẫn "thành công" về mặt số
    # học nhưng cho ra góc vô nghĩa (vd 172°) vì hoàn toàn không có thông tin hình học thật.
    # Chặn tường minh: nếu độ trải rộng giữa các điểm quá nhỏ so với kích thước ảnh, coi là
    # suy biến, trả None ngay - không tin solvePnP tự báo lỗi trong trường hợp này.
    spread = np.ptp(landmarks_2d, axis=0)  # (max-min) theo x va y
    if np.max(spread) < 1.0:
        return None

    h, w = image_shape[:2]
    camera_matrix = np.array(
        [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64
    )

    # FIX (phát hiện qua chạy thử thật): flag mặc định (SOLVEPNP_ITERATIVE) dùng DLT để dựng
    # ước lượng ban đầu, đòi hỏi >=6 điểm - với OpenCV 5.0.0 cài trên máy, 5 điểm mốc chuẩn
    # (đúng số insightface cung cấp) làm solvePnP crash với lỗi "DLT algorithm needs at least
    # 6 points", không phải chỉ trả success=False như code giả định ban đầu. Dùng
    # SOLVEPNP_EPNP - thuật toán EPnP hoạt động đúng với N>=4 điểm, không cần DLT.
    try:
        success, rotation_vec, _ = cv2.solvePnP(
            MODEL_3D_POINTS,
            landmarks_2d.astype(np.float64),
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj = np.hstack((rotation_mat, np.zeros((3, 1))))
    euler_angles = cv2.decomposeProjectionMatrix(proj)[6].flatten()
    pitch, yaw, roll = float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])

    # FIX (phát hiện qua chạy thử thật với ảnh chính diện thật, ground-truth pitch≈0): pitch
    # trả về ~175° thay vì ~0° - đây là ambiguity kinh điển của cv2.decomposeProjectionMatrix
    # (rotation matrix có thể phân rã thành (yaw,pitch,roll) hoặc (yaw±180, 180-pitch, roll±180)
    # cùng biểu diễn 1 phép quay, và hàm này thường chọn nhánh pitch gần ±180 khi pitch thật
    # gần 0). Chuẩn hoá pitch về khoảng [-90, 90] - cách sửa tiêu chuẩn cho quirk này.
    if pitch > 90:
        pitch = 180 - pitch
    elif pitch < -90:
        pitch = -180 - pitch

    return yaw, pitch, roll


def check_image_quality(
    image: np.ndarray,
    landmarks_5pt: np.ndarray,
    detection_confidence: float,
    yaw_threshold: float = YAW_THRESHOLD_DEFAULT,
    conf_threshold: float = CONF_THRESHOLD_DEFAULT,
) -> List[str]:
    """Trả về list cảnh báo tiếng Việt (rỗng nếu ảnh ổn) - KHÔNG raise, đây là text tư vấn
    cho UI (app.py), không phải cổng chặn pipeline. Kiểm tra 2 việc độc lập:
      1. Góc nghiêng YAW (quay ngang) qua estimate_head_pose() - nếu không tính được góc
         (None), cảnh báo riêng, không suy ra ảnh xấu. KHÔNG cảnh báo theo PITCH (xem docstring
         module ở đầu file - MAE pitch đo thật 14.6° quá lớn so với ngưỡng 25°, không đủ tin
         cậy để cảnh báo người dùng, dù estimate_head_pose() vẫn tính pitch cho mục đích khác).
      2. Độ tin cậy phát hiện khuôn mặt (detection_confidence, từ insightface det_score) -
         thấp có thể do ảnh bị che 1 phần, mờ, hoặc góc quá lệch khiến model không chắc chắn."""
    warnings: List[str] = []

    pose = estimate_head_pose(landmarks_5pt, image.shape)
    if pose is None:
        warnings.append("Không ước lượng được góc nghiêng khuôn mặt")
    else:
        yaw, _pitch, _roll = pose
        if abs(yaw) > yaw_threshold:
            warnings.append(f"Ảnh nghiêng (quay ngang) khá lớn ({abs(yaw):.0f}°)")

    if detection_confidence < conf_threshold:
        warnings.append(f"Độ tin cậy phát hiện thấp ({detection_confidence:.2f}) — có thể bị che 1 phần")

    return warnings
