"""
Rejection threshold (add-on) - quyết định cuối cùng có CHẤP NHẬN kết quả search hay không.
Đây là lớp an toàn quan trọng nhất trong 5 lớp add-on: nếu điểm không đủ tin cậy, hệ thống
PHẢI báo "không tìm thấy đủ tin cậy" thay vì trả bừa ra 1 identity gần giống nhất trong
gallery - với bài toán tìm người thất lạc, 1 kết quả sai được trình bày như "đã tìm thấy"
nguy hiểm hơn nhiều so với việc thẳng thắn báo không chắc chắn.
"""

from typing import Dict, List, Optional, Tuple

# 0.6 la nguong ArcFace pho bien duoc coi la "cung 1 nguoi" (cosine similarity tren embedding
# da L2-normalize, 1 PHEP SO SANH MAT-VOI-MAT DON LE). Co the chinh qua configs/config.yaml
# (search.rejection_threshold).
REJECTION_THRESHOLD_DEFAULT = 0.6


def apply_rejection_threshold(
    scores_per_identity_raw: Dict[str, List[float]],
    final_scores_ensembled: Dict[str, float],
    threshold: float = REJECTION_THRESHOLD_DEFAULT,
) -> Tuple[bool, Optional[str], float]:
    """Dùng ensemble (final_scores_ensembled, output của ensemble_search_results - PHẢI đã
    sort giảm dần) để CHỌN top candidate (xếp hạng - vẫn giữ đúng lợi ích chống identity chỉ
    "spike" 1 target_age của ensemble). Nhưng QUYẾT ĐỊNH accept/reject dựa trên điểm THÔ CAO
    NHẤT của chính candidate đó (lấy từ scores_per_identity_raw), KHÔNG dùng điểm ensemble đã
    bị pha loãng.

    FIX (phát hiện qua chạy thật main.py, xem chi tiết trong ensemble.py): threshold 0.6 hiệu
    chỉnh cho 1 phép so sánh mặt-với-mặt đơn lẻ - điểm ensemble (trung bình qua nhiều target_age
    có chất lượng khác nhau, target_age càng xa tuổi gốc điểm càng thấp theo quy luật đã biết)
    tự nhiên thấp hơn điểm thô tốt nhất rất nhiều, khiến kể cả identity đúng 100% cũng bị từ
    chối oan nếu so thẳng điểm ensemble với threshold. Ensemble chỉ dùng để XẾP HẠNG (chọn ai
    là top-1), còn quyết định accept/reject dựa trên bằng chứng THÔ tốt nhất mà candidate đó
    từng đạt được - đúng bản chất câu hỏi threshold cần trả lời: "có ít nhất 1 lần so khớp đủ
    tin cậy không", không phải "trung bình các lần so khớp có đủ tin cậy không".

    Raise ValueError nếu final_scores_ensembled rỗng - đây là lỗi dữ liệu upstream (ensemble/
    search không có gì để xét), khác hẳn với "có candidate nhưng điểm thấp" (accepted=False
    bình thường) - không được lẫn lộn 2 trường hợp này."""
    if not final_scores_ensembled:
        raise ValueError("final_scores_ensembled rỗng - không có candidate nào để xét rejection threshold.")

    top_identity = next(iter(final_scores_ensembled))  # da sort giam dan -> phan tu dau la top-1 theo ensemble
    best_raw_score = max(scores_per_identity_raw[top_identity])

    if best_raw_score >= threshold:
        return True, top_identity, best_raw_score
    return False, None, best_raw_score
