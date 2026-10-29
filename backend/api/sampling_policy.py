"""Chọn tốc độ lấy mẫu camera (khung/giây) từ hình học camera và chuyển động người.

Module thuần (không DB/GPU), xem docs/thuat-toan-quen-va-lay-mau.md.

Ba ràng buộc, fps khuyến nghị = max(ràng buộc dưới) rồi kẹp vào giới hạn trên:

1. Phủ (coverage): một người ở trong vùng nhận diện được T giây. Cần lấy đủ n
   khung để với xác suất ``confidence`` có ít nhất ``needed_usable`` khung dùng
   được (mặt đủ thẳng, đủ nét) — n là nghịch đảo phân phối nhị thức với tỷ lệ
   khung dùng được ``usable_ratio``. fps_coverage = n / T.
2. Bám vết (tracking): tracker ghép khung bằng IoU >= ngưỡng t. Người đi ngang
   dịch Δ = v/fps mét mỗi mẫu; hai bbox cùng cỡ lệch ngang Δ có
   IoU = (w − Δ)/(w + Δ) với w = bề rộng mặt (mét). Giữ IoU >= t cần
   Δ/w <= (1 − t)/(1 + t)  ⇒  fps_tracking = v·(1 + t) / (w·(1 − t)).
   Kết quả KHÔNG phụ thuộc khoảng cách camera (pixel/mét triệt tiêu).
   Người đi về phía camera: bbox phóng to s = (d + Δ)/d, IoU = 1/s² nên cần
   s <= 1/√t  ⇒  fps_tracking = v / (d_gần · (1/√t − 1)).
3. Giới hạn trên: fps tối đa phiên camera, fps gốc của camera, ngân sách GPU.

Nếu ràng buộc dưới vượt giới hạn trên, plan vẫn trả fps khả thi kèm cảnh báo
(ví dụ: dưới fps_tracking thì mỗi khung thành một tracklet riêng).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

from backend.api.camera_tracks import IOU_MATCH_THRESHOLD, TRACK_TIMEOUT_SEC

# Bề rộng mặt người trưởng thành ~14–16cm; dùng 0.16m (trẻ em nhỏ hơn → khó hơn).
FACE_WIDTH_M = 0.16
# Ảnh crop/embedding còn dùng được khi mặt >= ~48px (ArcFace suy giảm rõ dưới ~40px).
MIN_FACE_PX = 48
# Giới hạn sample_fps của API start camera (schemas.CaptureStartRequest).
SESSION_FPS_MAX = 10.0


@dataclass(frozen=True)
class CameraGeometry:
    hfov_deg: float = 90.0          # góc nhìn ngang (ống 2.8mm phổ biến ~90–100°)
    image_width_px: int = 1920      # độ rộng khung hình gốc
    motion: str = "crossing"        # "crossing" đi ngang | "approaching" đi về phía camera
    distance_m: float = 4.0         # crossing: khoảng cách vuông góc từ camera tới lối đi
    near_m: float = 1.5             # approaching: điểm gần nhất còn thấy mặt
    far_m: float = 8.0              # approaching: điểm xa nhất người bắt đầu đi vào khung
    walk_speed_mps: float = 1.4     # tốc độ thiết kế (đi bộ ~1.2–1.4; đi nhanh ~2.0)
    camera_fps: float = 25.0        # fps gốc của luồng camera
    edge_margin: float = 0.1        # bỏ mỗi mép khung 10% (méo ống kính, mặt bị cắt)
    shutter_s: float = 1.0 / 50     # thời gian phơi sáng (ước lượng nhòe chuyển động)


@dataclass(frozen=True)
class SamplingTargets:
    face_width_m: float = FACE_WIDTH_M
    min_face_px: float = MIN_FACE_PX
    usable_ratio: float = 0.35      # tỷ lệ khung lấy mẫu dùng được (chính diện, nét)
    needed_usable: int = 3          # selector giữ tối đa 3 ứng viên/tracklet
    confidence: float = 0.95
    iou_threshold: float = IOU_MATCH_THRESHOLD
    fps_min: float = 0.5
    fps_max: float = SESSION_FPS_MAX
    fps_budget: Optional[float] = None   # fps GPU chia cho camera này (None = không giới hạn)
    miss_tolerance: int = 2         # số mẫu liên tiếp được phép mất mặt trước khi đóng track
    crops_per_tracklet: float = 2.0  # trung bình thực tế (tối đa 3)
    bytes_per_crop: float = 17.4 * 1024  # p95 đo bằng scripts/bench_crop_jpeg.py (256px Q85)
    people_per_hour: Optional[float] = None


def field_width_m(distance_m: float, hfov_deg: float) -> float:
    return 2.0 * distance_m * math.tan(math.radians(hfov_deg) / 2.0)


def face_pixels(distance_m: float, geom: CameraGeometry, face_width_m: float = FACE_WIDTH_M) -> float:
    return face_width_m * geom.image_width_px / field_width_m(distance_m, geom.hfov_deg)


def max_recognition_distance_m(geom: CameraGeometry, targets: SamplingTargets) -> float:
    """Khoảng cách xa nhất mà mặt còn >= min_face_px."""
    return (targets.face_width_m * geom.image_width_px
            / (targets.min_face_px * 2.0 * math.tan(math.radians(geom.hfov_deg) / 2.0)))


def hfov_for_distance(distance_m: float, geom: CameraGeometry, targets: SamplingTargets) -> float:
    """Góc nhìn ngang lớn nhất để mặt ở ``distance_m`` vẫn đạt min_face_px."""
    half = math.atan(targets.face_width_m * geom.image_width_px
                     / (2.0 * targets.min_face_px * distance_m))
    return math.degrees(2.0 * half)


def binomial_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) với X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if n < k:
        return 0.0
    below = sum(math.comb(n, i) * p ** i * (1.0 - p) ** (n - i) for i in range(k))
    return max(0.0, 1.0 - below)


def samples_needed(k: int, p: float, confidence: float, n_max: int = 1000) -> int:
    """n nhỏ nhất để P(ít nhất k khung dùng được trong n khung) >= confidence."""
    if not 0.0 < p <= 1.0:
        raise ValueError("usable_ratio phải trong (0, 1].")
    for n in range(max(k, 1), n_max + 1):
        if binomial_tail(n, k, p) >= confidence:
            return n
    return n_max


def tracking_fps(geom: CameraGeometry, targets: SamplingTargets) -> float:
    t = targets.iou_threshold
    if not 0.0 < t < 1.0:
        raise ValueError("iou_threshold phải trong (0, 1).")
    v = geom.walk_speed_mps
    if geom.motion == "approaching":
        return v / (geom.near_m * (1.0 / math.sqrt(t) - 1.0))
    return v * (1.0 + t) / (targets.face_width_m * (1.0 - t))


def plan_from_dict(d: dict) -> dict:
    """Tạo plan từ dict phẳng (body API): khóa thuộc CameraGeometry hoặc SamplingTargets."""
    geo_keys = CameraGeometry.__dataclass_fields__.keys()
    tgt_keys = SamplingTargets.__dataclass_fields__.keys()
    unknown = set(d) - set(geo_keys) - set(tgt_keys)
    if unknown:
        raise ValueError(f"Tham số không hỗ trợ: {sorted(unknown)}")
    geom = CameraGeometry(**{k: v for k, v in d.items() if k in geo_keys and v is not None})
    targets = SamplingTargets(**{k: v for k, v in d.items() if k in tgt_keys and v is not None})
    return plan_sampling(geom, targets)


def plan_sampling(geom: CameraGeometry, targets: SamplingTargets = SamplingTargets()) -> dict:
    """Tính fps khuyến nghị + giải thích từng ràng buộc. Không có side effect."""
    if geom.motion not in ("crossing", "approaching"):
        raise ValueError("motion phải là 'crossing' hoặc 'approaching'.")
    if geom.walk_speed_mps <= 0 or geom.hfov_deg <= 0 or geom.hfov_deg >= 180:
        raise ValueError("walk_speed_mps > 0 và 0 < hfov_deg < 180.")
    warnings: list[str] = []
    d_max = max_recognition_distance_m(geom, targets)

    if geom.motion == "crossing":
        d = geom.distance_m
        width = field_width_m(d, geom.hfov_deg)
        face_px = face_pixels(d, geom, targets.face_width_m)
        usable = d <= d_max
        path_m = max(0.0, width * (1.0 - 2.0 * geom.edge_margin) - targets.face_width_m)
        if not usable:
            warnings.append(
                f"Mặt chỉ ~{face_px:.0f}px ở {d:.1f}m (cần >= {targets.min_face_px:.0f}px): "
                f"giảm khoảng cách xuống <= {d_max:.1f}m hoặc dùng ống kính hẹp hơn "
                f"(góc nhìn <= {hfov_for_distance(d, geom, targets):.0f}°).")
    else:
        width = field_width_m(geom.near_m, geom.hfov_deg)
        face_px = face_pixels(geom.near_m, geom, targets.face_width_m)
        far_eff = min(geom.far_m, d_max)
        path_m = max(0.0, far_eff - geom.near_m)
        usable = path_m > 0
        if geom.far_m > d_max:
            warnings.append(
                f"Từ {d_max:.1f}m trở ra mặt nhỏ hơn {targets.min_face_px:.0f}px: "
                f"vùng dùng được chỉ {geom.near_m:.1f}–{far_eff:.1f}m.")
        if not usable:
            warnings.append("Không có đoạn đường nào mặt đủ lớn để nhận diện.")

    dwell_s = path_m / geom.walk_speed_mps if path_m > 0 else 0.0
    n_need = samples_needed(targets.needed_usable, targets.usable_ratio, targets.confidence)
    fps_cov = n_need / dwell_s if dwell_s > 0 else float("inf")
    fps_trk = tracking_fps(geom, targets)

    caps = {"fps_max": targets.fps_max, "camera_fps": geom.camera_fps}
    if targets.fps_budget is not None:
        caps["budget"] = targets.fps_budget
    cap_name, cap = min(caps.items(), key=lambda kv: kv[1])
    need_name, need = max((("coverage", fps_cov), ("tracking", fps_trk)), key=lambda kv: kv[1])
    if not usable or dwell_s <= 0:
        # Không có đoạn nào nhận diện được: lấy mẫu dày chỉ tốn GPU.
        fps, binding = targets.fps_min, "unusable"
    elif need <= cap:
        fps, binding = max(need, targets.fps_min), need_name
        if fps == targets.fps_min and targets.fps_min > need:
            binding = "fps_min"
    else:
        fps, binding = cap, cap_name

    samples_pass = fps * dwell_s
    track_ok = fps >= fps_trk
    tracklets_pass = 1.0 if track_ok else max(1.0, samples_pass)
    coverage_conf = binomial_tail(int(math.floor(samples_pass)), targets.needed_usable,
                                  targets.usable_ratio)
    if usable and fps < fps_cov:
        warnings.append(
            f"Chỉ ~{samples_pass:.1f} khung/lượt đi qua: xác suất có đủ "
            f"{targets.needed_usable} ảnh tốt ~{coverage_conf:.0%} "
            f"(mục tiêu {targets.confidence:.0%}).")
    if usable and not track_ok:
        warnings.append(
            f"fps {fps:.1f} < {fps_trk:.1f} cần để bám vết: mỗi khung sẽ thành một "
            f"tracklet riêng (~{tracklets_pass:.0f} tracklet/lượt). Ưu tiên đặt camera "
            f"nhìn người đi tới (approaching) hoặc dùng tracker dự đoán chuyển động.")
    blur_ratio = geom.walk_speed_mps * geom.shutter_s / targets.face_width_m
    if blur_ratio > 0.1:
        warnings.append(
            f"Nhòe chuyển động ~{blur_ratio:.0%} bề rộng mặt (phơi {geom.shutter_s * 1000:.0f}ms): "
            f"giảm thời gian phơi sáng xuống <= "
            f"{0.1 * targets.face_width_m / geom.walk_speed_mps * 1000:.0f}ms nếu ảnh mờ.")

    timeout = max(TRACK_TIMEOUT_SEC, (targets.miss_tolerance + 1) / fps)
    out = {
        "fps": round(fps, 2),
        "binding": binding,
        "usable": usable,
        "field_width_m": round(width, 2),
        "face_px": round(face_px, 1),
        "max_recognition_distance_m": round(d_max, 2),
        "path_m": round(path_m, 2),
        "dwell_s": round(dwell_s, 2),
        "samples_needed": n_need,
        "fps_coverage": round(fps_cov, 2) if math.isfinite(fps_cov) else None,
        "fps_tracking": round(fps_trk, 2),
        "samples_per_pass": round(samples_pass, 1),
        "coverage_confidence": round(coverage_conf, 3),
        "tracklets_per_pass": round(tracklets_pass, 1),
        "motion_blur_ratio": round(blur_ratio, 3),
        "tracker_timeout_sec": round(timeout, 2),
        "bytes_per_pass": round(tracklets_pass * targets.crops_per_tracklet * targets.bytes_per_crop),
        "warnings": warnings,
        "inputs": {"geometry": asdict(geom), "targets": asdict(targets)},
    }
    if targets.people_per_hour:
        per_day = targets.people_per_hour * 24.0
        out["tracklets_per_day"] = round(per_day * tracklets_pass)
        out["bytes_per_day"] = round(per_day * out["bytes_per_pass"])
    return out
