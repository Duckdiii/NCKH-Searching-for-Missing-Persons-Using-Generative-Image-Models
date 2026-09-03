"""
Test nhỏ cho ensemble (add-on) - chạy độc lập, không cần GPU/model gì cả (chỉ là phép toán
trên dict/list thuần túy). Chạy tay bằng: pytest tests/test_ensemble.py -v
"""

import pytest

from src.search.ensemble import ensemble_search_results


def test_ensemble_averages_across_all_target_ages_not_just_appearances():
    """Identity xuất hiện đủ 3/3 target_age với điểm thấp hơn phải THẮNG (về final_scores -
    dùng để XẾP HẠNG) identity chỉ xuất hiện 1/3 target_age với điểm cao hơn - đúng mục đích
    thiết kế mẫu số cố định (không phải trung bình cộng theo số lần xuất hiện thực tế)."""
    search_results_per_age = {
        30: [("id_consistent", 0.5), ("id_spike", 0.9)],
        50: [("id_consistent", 0.5)],
        70: [("id_consistent", 0.5)],
    }

    final_scores, _ = ensemble_search_results(search_results_per_age)

    # id_spike: 0.9 / 3 = 0.3 ; id_consistent: (0.5+0.5+0.5) / 3 = 0.5
    assert final_scores["id_consistent"] == pytest.approx(0.5)
    assert final_scores["id_spike"] == pytest.approx(0.3)
    assert final_scores["id_consistent"] > final_scores["id_spike"]


def test_ensemble_sorts_descending_by_final_score():
    """Kết quả final_scores trả về phải sort giảm dần theo điểm."""
    search_results_per_age = {
        30: [("low", 0.1), ("high", 0.8)],
        50: [("low", 0.1), ("high", 0.8)],
    }

    final_scores, _ = ensemble_search_results(search_results_per_age)

    assert list(final_scores.keys()) == ["high", "low"]


def test_ensemble_also_returns_raw_scores_per_identity():
    """scores_per_identity phải giữ NGUYÊN VẸN danh sách điểm thô (chưa chia mẫu số) của từng
    identity - đây là giá trị rejection.py cần để không bị pha loãng bởi ensemble.

    Dùng đúng số liệu thật đã đo trong dự án (case đúng identity, điểm giảm dần theo khoảng
    cách target_age tới tuổi gốc): 0.6177/0.2956/0.1821 ở target_age 30/50/70."""
    search_results_per_age = {
        30: [("01366", 0.6177)],
        50: [("01366", 0.2956)],
        70: [("01366", 0.1821)],
    }

    final_scores, scores_per_identity = ensemble_search_results(search_results_per_age)

    assert scores_per_identity["01366"] == [0.6177, 0.2956, 0.1821]
    assert final_scores["01366"] == pytest.approx((0.6177 + 0.2956 + 0.1821) / 3)
    assert max(scores_per_identity["01366"]) == pytest.approx(0.6177)


def test_ensemble_raises_on_empty_input():
    """search_results_per_age rỗng phải raise ValueError, không âm thầm trả dict rỗng."""
    with pytest.raises(ValueError):
        ensemble_search_results({})
