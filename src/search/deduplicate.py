"""
De-duplicate theo identity (add-on cho Nhánh B - video, ĐANG HOÃN) - gom các detection của
CÙNG 1 identity, CÙNG 1 camera, gần nhau về thời gian thành 1 nhóm, chỉ giữ lại bản ghi có
score CAO NHẤT làm đại diện.

CHƯA được nối vào main.py hay app.py - chỉ Nhánh A (ảnh tĩnh) đang được làm, Nhánh B (video/
camera) chưa implement. Hàm này viết + test độc lập để sẵn sàng dùng khi làm tới Nhánh B.
"""

from typing import List

TIME_WINDOW_DEFAULT = 2.0


def deduplicate_by_identity(
    detections: List[dict], time_window: float = TIME_WINDOW_DEFAULT
) -> List[dict]:
    """Gom các detection liên tiếp của cùng 1 người thành 1 bản ghi đại diện.

    detections: list dict, mỗi dict BẮT BUỘC có 4 khoá: "identity" (str), "camera_id" (str/
    int), "timestamp" (float, giây), "score" (float). Đây là data shape MỚI, chưa có nơi nào
    khác trong code hiện tại sinh ra - sẽ do pipeline Nhánh B (video) tạo ra sau này.

    Thuật toán: sort theo timestamp tăng dần, rồi duyệt tuần tự - 1 detection được gộp vào 1
    nhóm đang mở nếu CÙNG identity, CÙNG camera_id, VÀ cách timestamp của bản ghi CUỐI CÙNG
    vừa thêm vào nhóm đó (không phải bản ghi đầu nhóm, cũng không phải bản ghi điểm cao nhất)
    chưa quá time_window giây - đây là kiểu "chuỗi liên tiếp" (chained/single-linkage), không
    phải chia bucket thời gian cố định hay so với 1 mốc điểm-cao-nhất cố định, để 1 chuỗi
    frame trải dài (vd 10 giây, cách nhau đều 2 giây) được gộp liền mạch thành 1 nhóm thay vì
    bị cắt/gộp sai tùy theo mốc so sánh rơi vào đâu.

    Trong mỗi nhóm, CHỈ GIỮ bản ghi có score CAO NHẤT làm đại diện (tính SAU KHI đã gộp nhóm
    xong, không dùng làm mốc chaining) - KHÔNG lấy trung bình, vì các frame liên tiếp của
    cùng 1 khoảnh khắc là CÙNG 1 bằng chứng (người đó đi ngang camera), không phải nhiều bằng
    chứng độc lập cộng dồn được như nhiều target_age khác nhau (khác hẳn
    ensemble_search_results ở src/search/ensemble.py).

    Trả về list đã dedupe, sort theo timestamp tăng dần (deterministic)."""
    groups: List[List[dict]] = []

    for det in sorted(detections, key=lambda d: d["timestamp"]):
        matched_group = None
        for group in groups:
            last_added = group[-1]  # cac phan tu duoc them theo dung thu tu timestamp tang dan
            if (
                det["identity"] == last_added["identity"]
                and det["camera_id"] == last_added["camera_id"]
                and abs(det["timestamp"] - last_added["timestamp"]) < time_window
            ):
                matched_group = group
                break

        if matched_group is not None:
            matched_group.append(det)
        else:
            groups.append([det])

    deduped = [max(group, key=lambda d: d["score"]) for group in groups]
    return sorted(deduped, key=lambda d: d["timestamp"])
