"""
Align ảnh khuôn mặt THEO ĐÚNG công thức gốc của dataset FFHQ (NVIDIA) - KHÔNG dùng template
ArcFace/InsightFace (khác tỷ lệ, gây lệch khung hình so với 140 ảnh Module 1 đã học - xem
docstring trong app.py phần "FIX align").

Nguồn công thức: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
(hàm recreate_aligned_images, dòng 287-318) - README chính thức của repo này ghi rõ chạy
`download_ffhq.py --align` sẽ "reproduce exact replicas" của ảnh đã align trong dataset, tức
đây là công thức toán đóng, xác định, công khai - KHÔNG phải suy đoán/đo đạc xấp xỉ.

Khác biệt DUY NHẤT so với bản gốc: bản gốc dùng 68 điểm mốc dlib (eye_left/eye_right =
TRUNG BÌNH 6 điểm contour mỗi mắt). Ở đây dùng 5 điểm insightface (Face.kps) - vốn đã là
1 điểm TÂM mắt có sẵn cho mỗi mắt (không cần tính trung bình lại) - và 2 điểm khóe miệng
tương ứng đúng eye_to_eye/eye_to_mouth 4-điểm mà công thức gốc thực sự cần (điểm mũi thứ 5
của insightface KHÔNG được công thức gốc dùng đến, bỏ qua).

GIỚI HẠN: bản gốc NVIDIA còn có bước "shrink + crop + pad phản chiếu làm mờ biên" (dòng
320-367 file gốc) để xử lý ảnh "in the wild" cực lớn mà vùng cần align vượt ra ngoài khung ảnh
gốc. Bản rút gọn ở đây BỎ QUA bước đó (dùng cv2.BORDER_REFLECT đơn giản thay thế) - chấp nhận
được vì app.py xử lý ảnh người dùng upload thông thường (không phải ảnh "in the wild" cực lớn
như dataset gốc thu thập), nhưng cần biết đây là 1 đơn giản hoá so với bản đầy đủ.
"""

import cv2
import numpy as np


def _ffhq_quad(eye_left: np.ndarray, eye_right: np.ndarray, mouth_left: np.ndarray, mouth_right: np.ndarray) -> np.ndarray:
    """Dựng "quad" (hình vuông xoay định vị vùng align) - ĐÚNG NGUYÊN VĂN công thức gốc
    NVlabs/ffhq-dataset (download_ffhq.py dòng 287-303), chỉ thay 4 điểm đầu vào từ dlib
    68-điểm sang insightface 5-điểm (xem docstring module)."""
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * np.array([-1, 1])
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * np.array([-1, 1])
    c0 = eye_avg + eye_to_mouth * 0.1

    # Thu tu dung PIL.Image.QUAD mong doi: tren-trai, duoi-trai, duoi-phai, tren-phai.
    return np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])


def align_to_ffhq(image_bgr: np.ndarray, kps: np.ndarray, output_size: int = 512) -> np.ndarray:
    """Align ảnh về output_size x output_size theo ĐÚNG công thức gốc FFHQ (xem module
    docstring). `kps`: 5 điểm insightface (mắt trái, mắt phải, mũi, khoé miệng trái, khoé
    miệng phải) - CHỈ dùng 4 điểm mắt+miệng, bỏ qua điểm mũi (công thức gốc không cần).

    `quad` (4 điểm) tạo thành 1 hình vuông xoay (x⊥y, |x|=|y|, đảm bảo bởi chính công thức
    dựng x/y ở trên) - nên map 3/4 góc bằng cv2.getAffineTransform là đủ chính xác tuyệt đối
    (góc thứ 4 tự động đúng vì là hình vuông, không phải tứ giác bất kỳ)."""
    eye_left, eye_right = kps[0], kps[1]
    mouth_left, mouth_right = kps[3], kps[4]

    quad = _ffhq_quad(eye_left, eye_right, mouth_left, mouth_right).astype(np.float32)
    dst = np.array(
        [[0, 0], [0, output_size], [output_size, output_size], [output_size, 0]], dtype=np.float32
    )

    M = cv2.getAffineTransform(quad[:3], dst[:3])
    return cv2.warpAffine(image_bgr, M, (output_size, output_size), borderMode=cv2.BORDER_REFLECT)
