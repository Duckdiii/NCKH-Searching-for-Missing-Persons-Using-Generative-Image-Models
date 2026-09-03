"""
Test nhỏ cho rejection threshold (add-on) - chạy độc lập, không cần GPU/model gì cả (chỉ là
phép so sánh trên dict thuần túy). Chạy tay bằng: pytest tests/test_rejection.py -v
"""

import pytest

from src.search.rejection import apply_rejection_threshold


def test_rejection_accepts_when_best_raw_score_above_threshold_even_if_ensemble_below():
    """Đây CHÍNH LÀ case đã phát hiện qua chạy thật main.py và cần fix: identity đúng có điểm
    thô giảm dần theo target_age (0.6177/0.2956/0.1821) - điểm ENSEMBLE (trung bình = 0.365)
    thấp hơn threshold 0.6, nhưng điểm THÔ CAO NHẤT (0.6177) vẫn >= threshold -> PHẢI accept,
    không được từ chối oan chỉ vì điểm ensemble bị pha loãng."""
    scores_per_identity_raw = {"01366": [0.6177, 0.2956, 0.1821]}
    final_scores_ensembled = {"01366": (0.6177 + 0.2956 + 0.1821) / 3}  # = 0.365, < 0.6

    accepted, top_identity, top_score = apply_rejection_threshold(
        scores_per_identity_raw, final_scores_ensembled, threshold=0.6
    )

    assert accepted is True
    assert top_identity == "01366"
    assert top_score == pytest.approx(0.6177)


def test_rejection_rejects_when_best_raw_score_below_threshold():
    """Điểm thô cao nhất của top candidate vẫn dưới ngưỡng -> accepted=False, top_identity=None
    (KHÔNG được trả identity không đáng tin), nhưng top_score (điểm thô cao nhất) vẫn trả về
    để tham khảo."""
    scores_per_identity_raw = {"id_a": [0.3, 0.2], "id_b": [0.1]}
    final_scores_ensembled = {"id_a": 0.25, "id_b": 0.1}

    accepted, top_identity, top_score = apply_rejection_threshold(
        scores_per_identity_raw, final_scores_ensembled, threshold=0.6
    )

    assert accepted is False
    assert top_identity is None
    assert top_score == pytest.approx(0.3)


def test_rejection_boundary_score_equal_to_threshold_is_accepted():
    """Điểm thô cao nhất bằng đúng ngưỡng (biên) -> accepted=True (>=, không phải >)."""
    scores_per_identity_raw = {"id_a": [0.6]}
    final_scores_ensembled = {"id_a": 0.6}

    accepted, top_identity, top_score = apply_rejection_threshold(
        scores_per_identity_raw, final_scores_ensembled, threshold=0.6
    )

    assert accepted is True
    assert top_identity == "id_a"


def test_rejection_uses_ensemble_only_to_pick_top_candidate():
    """Ensemble vẫn phải là nơi CHỌN ai là top-1 (chống identity chỉ 'spike' 1 lần) - hàm
    rejection KHÔNG được tự ý chọn lại candidate khác dựa trên điểm thô, chỉ lấy max điểm thô
    của ĐÚNG candidate mà ensemble đã xếp hạng cao nhất."""
    # id_spike co diem tho cao nhat toan cuc (0.9), nhung ensemble xep id_consistent len top
    # (dung nhu test_ensemble_averages_across_all_target_ages_not_just_appearances).
    scores_per_identity_raw = {"id_consistent": [0.5, 0.5, 0.5], "id_spike": [0.9]}
    final_scores_ensembled = {"id_consistent": 0.5, "id_spike": 0.3}  # da sort giam dan

    accepted, top_identity, top_score = apply_rejection_threshold(
        scores_per_identity_raw, final_scores_ensembled, threshold=0.6
    )

    # Phai xet id_consistent (top theo ensemble), KHONG duoc nham sang id_spike du diem tho
    # cua id_spike (0.9) cao hon.
    assert top_identity is None or top_score == pytest.approx(0.5)
    assert accepted is False  # max([0.5,0.5,0.5]) = 0.5 < 0.6


def test_rejection_raises_on_empty_final_scores():
    """final_scores_ensembled rỗng phải raise ValueError - khác hẳn case 'có candidate nhưng
    điểm thấp'."""
    with pytest.raises(ValueError):
        apply_rejection_threshold({}, {})
