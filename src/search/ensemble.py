"""
Ensemble nhiều target_age (add-on) - tổng hợp kết quả search RIÊNG của từng target_age
(Module 5) để XẾP HẠNG candidate nào đáng tin nhất qua nhiều target_age, đồng thời giữ lại
điểm THÔ (chưa pha loãng) của từng candidate để nơi khác (rejection.py) tự quyết định
accept/reject - KHÔNG dùng chung 1 con số cho cả 2 việc (xem FIX trong docstring bên dưới).

Không đụng vào build_index()/search() của Module 5 - chỉ tổng hợp OUTPUT của search(), gọi
sau khi đã search xong cho từng target_age.
"""

from typing import Dict, List, Tuple


def ensemble_search_results(
    search_results_per_age: Dict[int, List[Tuple[str, float]]]
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Gộp kết quả search() của nhiều target_age, trả về (final_scores, scores_per_identity):

      final_scores:         {identity: điểm trung bình}, sort giảm dần - DÙNG ĐỂ XẾP HẠNG
                             candidate nào đáng tin nhất qua nhiều target_age.
      scores_per_identity:  {identity: [danh sách điểm THÔ qua từng target_age nó xuất hiện]} -
                             KHÔNG bị pha loãng bởi mẫu số cố định, dùng để QUYẾT ĐỊNH
                             accept/reject ở rejection.py (xem FIX bên dưới).

    Công thức final_scores: final_score(identity) = tổng score identity đó đạt được qua các
    target_age XUẤT HIỆN, CHIA CHO tổng số target_age đã query (mẫu số CỐ ĐỊNH =
    len(search_results_per_age), KHÔNG PHẢI số lần identity đó thực sự xuất hiện). Identity
    vắng mặt ở 1 target_age coi như đóng góp 0 điểm cho age đó, không phải bị loại khỏi mẫu số.
    Mục đích: PHẠT identity chỉ trúng đúng 1 target_age (dù điểm ở age đó rất cao), THƯỞNG
    identity xuất hiện NHẤT QUÁN qua nhiều target_age.

    FIX (phát hiện qua chạy thật main.py): điểm final_scores này KHÔNG được dùng để so với
    rejection threshold - threshold 0.6 được hiệu chỉnh cho 1 phép so sánh mặt-với-mặt đơn lẻ,
    trong khi final_scores là trung bình của nhiều phép so sánh có chất lượng KHÁC NHAU (target
    age càng xa tuổi gốc, điểm càng thấp theo quy luật đã biết - không phải dấu hiệu sai
    identity). Kể cả identity đúng 100% cũng hiếm khi vượt threshold sau khi bị pha loãng kiểu
    này. Do đó hàm này giờ trả thêm scores_per_identity (điểm thô, chưa pha loãng) để
    rejection.py tự lấy max mà xét threshold - ensemble chỉ còn đảm nhiệm việc XẾP HẠNG, không
    còn đảm nhiệm việc QUYẾT ĐỊNH accept/reject.

    Raise ValueError nếu search_results_per_age rỗng (không có gì để ensemble - lỗi cấu hình/
    dữ liệu upstream, không phải hành vi hợp lệ trả về dict rỗng)."""
    if not search_results_per_age:
        raise ValueError("search_results_per_age rỗng - không có kết quả nào để ensemble.")

    num_target_ages = len(search_results_per_age)

    scores_per_identity: Dict[str, List[float]] = {}
    for top_k in search_results_per_age.values():
        for identity, score in top_k:
            scores_per_identity.setdefault(identity, []).append(score)

    final_scores = {
        identity: sum(scores) / num_target_ages for identity, scores in scores_per_identity.items()
    }
    final_scores_sorted = dict(sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True))

    return final_scores_sorted, scores_per_identity
