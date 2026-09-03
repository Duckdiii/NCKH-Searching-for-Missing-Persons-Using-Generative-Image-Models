"""
Module 5 - FAISS Search

Tìm kiếm khuôn mặt bằng FAISS IndexFlatIP (exact search, KHÔNG dùng approximate index -
dữ liệu nhỏ, ưu tiên chính xác tuyệt đối theo yêu cầu đề tài). Gồm 2 hàm thuần túy (không giữ
state) - người gọi (main.py) tự quản lý việc giữ lại `index` giữa các lần gọi search().
"""

from typing import List, Tuple

import faiss
import numpy as np

L2_NORM_TOLERANCE = 1e-3


def _validate_l2_normalized(matrix: np.ndarray) -> None:
    """Kiểm tra tất cả vector trong matrix đã được chuẩn hoá L2 (norm ~ 1). IndexFlatIP tính
    inner product THÔ - chỉ tương đương cosine similarity khi vector đã chuẩn hoá. Nếu bỏ qua
    kiểm tra này, index vẫn build được, search vẫn chạy, KHÔNG báo lỗi gì - nhưng thứ hạng kết
    quả sẽ sai lệch âm thầm (đúng dạng lỗi "chạy được nhưng sai" nguy hiểm nhất)."""
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=L2_NORM_TOLERANCE):
        bad_idx = np.where(np.abs(norms - 1.0) > L2_NORM_TOLERANCE)[0]
        raise ValueError(
            f"Có {len(bad_idx)} embedding CHƯA được chuẩn hoá L2 (norm != 1), ví dụ vị trí "
            f"{bad_idx[:5].tolist()} có norm {norms[bad_idx[:5]].tolist()}. IndexFlatIP chỉ "
            f"tương đương cosine similarity khi embedding đã normalize - kiểm tra lại đầu vào "
            f"từ Module 4 (FaceEmbedder.embed() phải trả về normed_embedding)."
        )


def build_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatIP:
    """Build FAISS index từ list embedding (KHÔNG dùng approximate index - dữ liệu nhỏ, ưu
    tiên chính xác tuyệt đối). Kiểm tra L2-norm trước khi add, đảm bảo inner product của
    IndexFlatIP tương đương cosine similarity. Trả về index đã add xong."""
    if not embeddings:
        raise ValueError("embeddings rỗng - không thể build index.")

    matrix = np.stack(embeddings).astype(np.float32)
    _validate_l2_normalized(matrix)

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index


def search(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    identity_labels: List[str],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Tìm k identity gần nhất với query_embedding trong index, trả về list (identity, score)
    sắp theo score giảm dần (score = cosine similarity, vì embedding đã normalize L2 - càng
    gần 1 càng giống).

    identity_labels PHẢI cùng độ dài và cùng thứ tự với embeddings đã đưa vào build_index()
    (FAISS không tự lưu nhãn, chỉ trả về vị trí số nguyên) - nếu lệch độ dài, raise ValueError
    NGAY thay vì để lỗi IndexError khó hiểu hoặc tệ hơn là map nhầm nhãn mà không crash."""
    if len(identity_labels) != index.ntotal:
        raise ValueError(
            f"identity_labels có {len(identity_labels)} phần tử nhưng index có {index.ntotal} "
            f"vector - 2 danh sách này PHẢI khớp nhau (cùng độ dài, cùng thứ tự với lúc "
            f"build_index()). Lệch nhau sẽ gây map nhầm nhãn mà không báo lỗi."
        )

    query = query_embedding.reshape(1, -1).astype(np.float32)
    _validate_l2_normalized(query)

    scores, indices = index.search(query, k)

    results: List[Tuple[str, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS trả về -1 nếu index có ít hơn k phần tử
            continue
        results.append((identity_labels[idx], float(score)))
    return results
