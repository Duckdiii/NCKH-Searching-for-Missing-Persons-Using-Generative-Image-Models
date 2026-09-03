"""
Test nhỏ cho Module 5 (FAISS Search) - chạy độc lập, không cần GPU/model gì cả (chỉ là phép
toán vector thuần túy). Chạy tay bằng: pytest tests/test_faiss_search.py -v
"""

import numpy as np
import pytest

from src.search.faiss_index import build_index, search


def _random_normalized_vectors(n: int, dim: int = 512, seed: int = 0):
    """Sinh n vector ngẫu nhiên đã chuẩn hoá L2 (norm = 1), dùng làm embedding giả cho test."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [v for v in vectors]


def test_search_returns_exact_match_as_top1():
    """Build index từ 5 vector ngẫu nhiên, dùng chính 1 trong số đó làm query - phải ra đúng
    identity đó ở vị trí top-1 với score gần 1.0 (giống hệt chính nó)."""
    embeddings = _random_normalized_vectors(5)
    labels = [f"id_{i}" for i in range(5)]
    index = build_index(embeddings)

    results = search(index, embeddings[2], labels, k=3)

    assert results[0][0] == "id_2"
    assert results[0][1] > 0.999


def test_search_k_larger_than_gallery_size():
    """Nếu k > số lượng identity trong gallery, FAISS trả về -1 cho các vị trí thừa - hàm
    search phải lọc bỏ, chỉ trả về đúng số kết quả thực tế có."""
    embeddings = _random_normalized_vectors(3)
    labels = [f"id_{i}" for i in range(3)]
    index = build_index(embeddings)

    results = search(index, embeddings[0], labels, k=10)

    assert len(results) == 3


def test_build_index_raises_on_unnormalized_embeddings():
    """Vector chưa chuẩn hoá L2 (norm != 1) phải bị build_index() chặn lại ngay, không được
    âm thầm build index rồi cho ra kết quả sai lệch về sau."""
    embeddings = [np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)]  # norm = 5, không phải 1

    with pytest.raises(ValueError):
        build_index(embeddings)


def test_search_raises_on_mismatched_labels_length():
    """identity_labels lệch độ dài với số vector trong index phải bị chặn ngay từ đầu, tránh
    map nhầm nhãn mà không báo lỗi."""
    embeddings = _random_normalized_vectors(5)
    index = build_index(embeddings)
    wrong_labels = ["id_0", "id_1", "id_2"]  # chỉ có 3, thiếu 2

    with pytest.raises(ValueError):
        search(index, embeddings[0], wrong_labels, k=3)
