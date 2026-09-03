"""
Test nhỏ cho deduplicate (add-on, chưa nối vào main/app - chỉ chuẩn bị cho Nhánh B video) -
chạy độc lập, không cần GPU/model gì cả (chỉ là phép toán trên list/dict thuần túy).
Chạy tay bằng: pytest tests/test_deduplicate.py -v
"""

from src.search.deduplicate import deduplicate_by_identity


def test_dedup_keeps_max_score_within_time_window():
    """3 detection cùng identity/camera, cách nhau < time_window - chỉ giữ lại đúng 1 bản ghi
    có score cao nhất, KHÔNG lấy trung bình."""
    detections = [
        {"identity": "01366", "camera_id": "cam1", "timestamp": 0.0, "score": 0.5},
        {"identity": "01366", "camera_id": "cam1", "timestamp": 1.0, "score": 0.9},
        {"identity": "01366", "camera_id": "cam1", "timestamp": 1.8, "score": 0.6},
    ]

    result = deduplicate_by_identity(detections, time_window=2.0)

    assert len(result) == 1
    assert result[0]["score"] == 0.9


def test_dedup_keeps_separate_records_outside_time_window():
    """Cùng identity/camera nhưng khoảng cách thời gian LỚN HƠN time_window - phải giữ
    RIÊNG 2 bản ghi, không gộp làm 1."""
    detections = [
        {"identity": "01366", "camera_id": "cam1", "timestamp": 0.0, "score": 0.8},
        {"identity": "01366", "camera_id": "cam1", "timestamp": 10.0, "score": 0.7},
    ]

    result = deduplicate_by_identity(detections, time_window=2.0)

    assert len(result) == 2


def test_dedup_treats_different_cameras_as_independent():
    """Cùng identity, cùng timestamp, nhưng KHÁC camera_id - không được coi là trùng nhau."""
    detections = [
        {"identity": "01366", "camera_id": "cam1", "timestamp": 5.0, "score": 0.8},
        {"identity": "01366", "camera_id": "cam2", "timestamp": 5.0, "score": 0.7},
    ]

    result = deduplicate_by_identity(detections, time_window=2.0)

    assert len(result) == 2


def test_dedup_chains_consecutive_frames_spanning_longer_than_window():
    """1 chuỗi frame liên tiếp trải dài hơn time_window (nhưng mỗi bước cách nhau < window)
    phải được gộp thành 1 nhóm duy nhất - đúng kiểu 'chained', không phải bucket cố định."""
    detections = [
        {"identity": "01366", "camera_id": "cam1", "timestamp": 0.0, "score": 0.5},
        {"identity": "01366", "camera_id": "cam1", "timestamp": 1.5, "score": 0.9},
        {"identity": "01366", "camera_id": "cam1", "timestamp": 3.0, "score": 0.4},
        {"identity": "01366", "camera_id": "cam1", "timestamp": 4.5, "score": 0.6},
    ]

    result = deduplicate_by_identity(detections, time_window=2.0)

    assert len(result) == 1
    assert result[0]["score"] == 0.9
