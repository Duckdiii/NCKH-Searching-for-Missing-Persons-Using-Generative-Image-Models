"""
Test nhỏ cho Module 4 (InsightFace Embedding) - chạy độc lập, không phụ thuộc module nào khác.

Lưu ý: cần tải model buffalo_l lần đầu (qua mạng, insightface tự động tải về
~/.insightface/models/). Chạy tay bằng: pytest tests/test_embedding.py -v
"""

import glob
import os
import shutil
import tempfile

import numpy as np

from src.search.embedding import FaceEmbedder

FFHQ_DIR = "D:/Data/project/nckh/ffhq_aging_150_samples"


def _sample_image_paths(n: int = 3):
    """Lấy n ảnh bất kỳ trong dataset FFHQ (đều có mặt thật) để test."""
    candidates = sorted(glob.glob(os.path.join(FFHQ_DIR, "*.png")))
    assert len(candidates) >= n, f"Cần ít nhất {n} ảnh trong {FFHQ_DIR}"
    return candidates[:n]


def test_embed_returns_normalized_512dim_vector():
    """Kiểm tra embed() trả về vector 512-dim, đã chuẩn hoá L2 (norm xấp xỉ 1)."""
    embedder = FaceEmbedder(ctx_id=-1)
    image_path = _sample_image_paths(1)[0]

    embedding = embedder.embed(image_path)

    assert embedding.shape == (512,)
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-3


def test_build_gallery_on_small_folder():
    """Copy 3 ảnh thật vào 1 folder tạm, kiểm tra build_gallery trả về đúng 3 embedding + 3
    label khớp tên file, không có failed_files (vì cả 3 ảnh đều hợp lệ)."""
    embedder = FaceEmbedder(ctx_id=-1)
    sample_paths = _sample_image_paths(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for path in sample_paths:
            shutil.copy(path, tmp_dir)

        embeddings, identity_labels, failed_files = embedder.build_gallery(tmp_dir)

        assert len(embeddings) == 3
        assert len(identity_labels) == 3
        assert failed_files == []
        assert set(identity_labels) == {
            os.path.splitext(os.path.basename(p))[0] for p in sample_paths
        }


def test_build_gallery_reports_failed_files_without_raising():
    """Kiểm tra hành vi 'không dừng cứng, nhưng không âm thầm bỏ qua': trộn 1 ảnh thật + 1
    file rác (không phải ảnh hợp lệ) vào cùng folder - build_gallery phải chạy xong, trả về
    đúng 1 embedding thành công VÀ liệt kê rõ file lỗi trong failed_files."""
    embedder = FaceEmbedder(ctx_id=-1)
    sample_path = _sample_image_paths(1)[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.copy(sample_path, tmp_dir)
        bad_path = os.path.join(tmp_dir, "broken.png")
        with open(bad_path, "wb") as f:
            f.write(b"khong phai file anh that")

        embeddings, identity_labels, failed_files = embedder.build_gallery(tmp_dir)

        assert len(embeddings) == 1
        assert len(identity_labels) == 1
        assert failed_files == [bad_path]


def test_detect_faces_returns_full_face_objects_including_kps_and_bbox():
    """detect_faces() (add-on cho app.py) phải trả về list Face THÔ, có đủ .bbox, .kps (5
    điểm mốc), .det_score, .normed_embedding - không chỉ mỗi embedding như embed()."""
    embedder = FaceEmbedder(ctx_id=-1)
    image_path = _sample_image_paths(1)[0]

    faces = embedder.detect_faces(image_path)

    assert len(faces) >= 1
    face = faces[0]
    assert face.bbox.shape == (4,)
    assert face.kps.shape == (5, 2)
    assert isinstance(face.det_score, (float, np.floating))
    assert face.normed_embedding.shape == (512,)
