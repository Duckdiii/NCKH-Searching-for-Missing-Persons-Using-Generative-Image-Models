"""
Test nhỏ cho ffhq_align (add-on) - xác nhận align_to_ffhq() gần đúng "bất biến" trên ảnh đã
align sẵn: align lại 1 ảnh FFHQ đã align phải cho kết quả GẦN NHƯ Y HỆT bản gốc (chỉ lệch do
insightface 5-điểm và dlib 68-điểm gốc là 2 detector khác nhau, không phải lỗi công thức).
Chạy tay bằng: pytest tests/test_ffhq_align.py -v
"""

import os

import cv2
import numpy as np

from src.search.embedding import FaceEmbedder
from src.utils.ffhq_align import align_to_ffhq

FFHQ_DIR = "D:/Data/project/nckh/ffhq_aging_150_samples"


def test_align_to_ffhq_is_near_identity_on_already_aligned_image():
    """Ảnh FFHQ đã align sẵn (01366.png), align lại bằng align_to_ffhq() phải cho:
    - Chiều cao khuôn mặt trong khung gần đúng tỷ lệ gốc (~76% khung, sai số vài %)
    - Sai lệch pixel trung bình nhỏ (< 15/255) - không phải xáo trộn hoàn toàn ảnh."""
    embedder = FaceEmbedder(ctx_id=-1)
    image_path = f"{FFHQ_DIR}/01366.png"
    img = cv2.imread(image_path)
    faces = embedder.detect_faces(image_path)
    face = faces[0]

    aligned = align_to_ffhq(img, face.kps, output_size=256)

    assert aligned.shape == (256, 256, 3)

    # FaceEmbedder.detect_faces() chi nhan duong dan file - ghi ra file tam de detect lai.
    tmp_path = "outputs/_test_ffhq_align_tmp.png"
    cv2.imwrite(tmp_path, aligned)
    try:
        faces2 = embedder.detect_faces(tmp_path)
    finally:
        os.remove(tmp_path)

    assert len(faces2) >= 1
    face2 = faces2[0]

    face_h_original = face.bbox[3] - face.bbox[1]
    face_h_aligned = face2.bbox[3] - face2.bbox[1]
    assert abs(face_h_aligned / 256 - face_h_original / 256) < 0.05  # lech duoi 5 diem %

    diff = np.abs(img.astype(np.int32) - aligned.astype(np.int32))
    assert diff.mean() < 15.0
