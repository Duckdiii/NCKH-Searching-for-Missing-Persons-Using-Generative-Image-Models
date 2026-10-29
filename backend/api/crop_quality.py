"""Đánh giá chất lượng ảnh mặt (FIQA) để chọn crop tốt nhất trong một tracklet.

Module thuần (numpy/OpenCV), xem docs/thuat-toan-chon-crop.md.

Mỗi khung có mặt được chấm 7 thành phần trong [0, 1]:

    resolution     khoảng cách hai mắt (px) — thước đo độ phân giải dùng được cho nhận dạng
    sharpness      1 − chỉ số mờ Crete et al. (2007) trên vùng mặt chuẩn hóa 64×64
    pose           yaw / pitch / roll ước lượng từ 5 điểm mốc InsightFace
    exposure       độ sáng, tương phản, tỷ lệ pixel cháy/tối trên vùng mặt
    completeness   phần bbox nằm ngoài khung hình (mặt bị cắt)
    recognizability  norm embedding ArcFace chưa chuẩn hóa (tương quan khả năng nhận dạng)
    detection      det_score của detector

Điểm tổng là trung bình nhân có trọng số  Q = Π s_i^{w_i}  (Σw = 1): một thành phần
rất tệ kéo cả ảnh xuống, khác cộng tuyến tính (ảnh rất mờ nhưng to vẫn thắng).
Thành phần không đo được (thiếu landmark, thiếu norm) bị bỏ và chuẩn hóa lại trọng số.

Loại cứng (gate, Q = 0): mặt nghiêng hẳn, mặt bị cắt > 15%, landmark vô lý.

Chọn crop: điểm chọn = Q × độ nhất quán danh tính (cosine tới tâm embedding của
tracklet, có trọng số Q) — chống tracker nhảy sang người khác giữa chừng.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

# Trọng số hiệu chuẩn bằng ERC trên FG-NET (scripts/eval_crop_quality.py): norm
# embedding dự báo khớp đúng tốt nhất; góc mặt vừa phải ArcFace chịu được nên chỉ
# dùng trọng số nhỏ + gate nghiêng hẳn. Độ phân giải giữ trọng số dù FG-NET (ảnh
# lớn) không đo được, vì camera thực tế có nhiều mặt nhỏ.
WEIGHTS = {
    "recognizability": 0.30,
    "sharpness": 0.20,
    "resolution": 0.15,
    "exposure": 0.10,
    "completeness": 0.10,
    "pose": 0.10,
    "detection": 0.05,
}
FLOOR = 0.01                 # tránh log(0); một thành phần 0 vẫn kéo Q về rất thấp
IOD_LOW_PX, IOD_GOOD_PX = 10.0, 36.0   # ArcFace 112×112 căn chỉnh có IOD ≈ 35px
SHARP_SIZE = 64              # chuẩn hóa vùng mặt trước khi đo mờ (so sánh được giữa các cỡ)
# Ảnh mặt thật có 1 − blur_index khoảng 0.3–0.7 (FG-NET) → giãn thang về [0, 1].
SHARP_LOW, SHARP_GOOD = 0.25, 0.65
# Vỡ khối nén: tỷ lệ chênh lệch tại biên lưới 8px. Ảnh JPEG/H.264 bình thường ~1.5–2;
# nén mạnh (Q15) ~4, rất mạnh (Q5) ~8–13 → bắt đầu phạt ở 2, độ nét về 0 ở 6.
BLOCK_OK, BLOCK_BAD = 2.0, 6.0
# Nhiễu cảm biến (σ Immerkær): ảnh rõ ~0.5; camera thiếu sáng 5–20. Nhiễu cũng bị đo
# mờ hiểu nhầm là chi tiết → phạt độ nét từ σ = 3, về 0 ở σ = 15.
NOISE_OK, NOISE_BAD = 3.0, 15.0
# Phơi sáng: độ sáng TB vùng mặt trong [70, 200] là đủ sáng (không phạt).
EXPOSURE_DARK, EXPOSURE_BRIGHT = 70.0, 200.0
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
# Mặt chính diện theo template ArcFace: mũi nằm ~49% từ đường mắt tới đường miệng.
PITCH_CENTER = 0.49
YAW_SIGMA, PITCH_SIGMA = 0.25, 0.25
YAW_GATE = 0.45              # |yaw_ratio| lớn hơn → nghiêng hẳn (profile)
TRUNCATION_GATE = 0.15
# Norm embedding ArcFace buffalo_l (w600k_r50) chưa chuẩn hóa — hiệu chuẩn trên FG-NET
# bằng scripts/eval_crop_quality.py; ảnh khó nhận dạng có norm thấp.
NORM_LOW, NORM_GOOD = 18.0, 25.0       # FG-NET: p5 = 20.3, p50 = 22.7, p95 = 25.6
# Nhất quán danh tính: cosine tới tâm tracklet.
CONSIST_LOW, CONSIST_GOOD = 0.25, 0.55
CONSIST_MIN_OBS = 3
CONSIST_FLOOR = 0.1          # dưới mức này: gần như chắc chắn không cùng người


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _ramp(x: float, lo: float, hi: float) -> float:
    return _clip01((x - lo) / (hi - lo))


def blur_crete(gray: np.ndarray) -> float:
    """Chỉ số mờ không tham chiếu Crete et al. (2007): 0 = nét, 1 = rất mờ.

    Làm mờ lại ảnh bằng lọc trung bình 1×9 theo từng hướng; ảnh nét mất nhiều biến
    thiên giữa pixel kề nhau, ảnh vốn đã mờ thì gần như không đổi.
    """
    f = gray.astype(np.float64)
    if f.ndim != 2 or min(f.shape) < 4:
        return 1.0
    scores = []
    # cv2.blur nhận (rộng, cao): (1, 9) làm mờ dọc → so chênh lệch theo trục hàng (0).
    for ksize, axis in (((1, 9), 0), ((9, 1), 1)):
        b = cv2.blur(f, ksize)
        df = np.abs(np.diff(f, axis=axis))
        db = np.abs(np.diff(b, axis=axis))
        s_f = float(df.sum())
        s_v = float(np.maximum(0.0, df - db).sum())
        scores.append(1.0 if s_f <= 1e-9 else (s_f - s_v) / s_f)
    return _clip01(max(scores))


def pose_from_kps(kps: Any) -> Optional[dict]:
    """yaw_ratio, pitch_ratio, roll_deg, iod_px từ 5 điểm (mắt T/P, mũi, miệng T/P)."""
    if kps is None:
        return None
    pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 5:
        return None
    le, re, nose, lm, rm = pts[:5]
    d = re - le
    iod = float(np.hypot(*d))
    if iod < 1.0:
        return {"iod_px": iod, "yaw_ratio": 0.0, "pitch_ratio": 0.0, "roll_deg": 0.0,
                "plausible": False}
    roll = math.atan2(d[1], d[0])
    c, s = math.cos(-roll), math.sin(-roll)
    eye_mid = (le + re) / 2.0

    def rot(p):
        q = p - eye_mid
        return np.array([c * q[0] - s * q[1], s * q[0] + c * q[1]])

    n, m = rot(nose), rot((lm + rm) / 2.0)          # mắt giữa = gốc tọa độ sau xoay
    mouth_drop = m[1]
    pitch_ratio = n[1] / mouth_drop if mouth_drop > 1e-6 else 0.0
    # Lệch ngang của mũi so với trục giữa mặt (mắt-giữa → miệng-giữa) tại độ cao mũi.
    yaw_ratio = (n[0] - m[0] * pitch_ratio) / iod
    plausible = 0.4 * iod < mouth_drop < 2.0 * iod and 0.05 < pitch_ratio < 0.95
    return {"iod_px": iod, "yaw_ratio": float(yaw_ratio), "pitch_ratio": float(pitch_ratio),
            "roll_deg": math.degrees(roll), "plausible": bool(plausible)}


def pose_score(pose: dict) -> float:
    yaw = math.exp(-(pose["yaw_ratio"] / YAW_SIGMA) ** 2)
    pitch = math.exp(-((pose["pitch_ratio"] - PITCH_CENTER) / PITCH_SIGMA) ** 2)
    roll = 1.0 - 0.15 * min(1.0, abs(pose["roll_deg"]) / 45.0)   # căn chỉnh sửa được roll
    return yaw * pitch * roll


def exposure_score(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    g = gray.astype(np.float64)
    mean, std = float(g.mean()), float(g.std())
    clipped = float(((g < 6) | (g > 249)).mean())
    # Dải sáng chấp nhận được: da đủ sáng ~150–200 KHÔNG bị phạt (bản cũ lấy tâm 125
    # nên chấm ảnh tối cao hơn ảnh gốc sáng). Chỉ phạt khi quá tối / quá sáng.
    if mean < EXPOSURE_DARK:
        level = math.exp(-((EXPOSURE_DARK - mean) / 35.0) ** 2)
    elif mean > EXPOSURE_BRIGHT:
        level = math.exp(-((mean - EXPOSURE_BRIGHT) / 25.0) ** 2)
    else:
        level = 1.0
    return (level
            * math.sqrt(min(1.0, std / 35.0))            # tương phản thấp (tối, mờ sương)
            * (1.0 - min(1.0, clipped / 0.25)))          # cháy sáng / đen kịt


def truncation(bbox: list[float], frame_shape: tuple) -> float:
    """Tỷ lệ diện tích bbox nằm ngoài khung hình."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    area = max(1e-9, (x2 - x1) * (y2 - y1))
    ix = max(0.0, min(x2, w) - max(x1, 0.0))
    iy = max(0.0, min(y2, h) - max(y1, 0.0))
    return _clip01(1.0 - ix * iy / area)


def blockiness(gray: np.ndarray, off_x: int = 0, off_y: int = 0, block: int = 8) -> float:
    """Tỷ lệ chênh lệch tại biên lưới khối nén (JPEG/H.264) so với vị trí khác.

    ~1 = không vỡ khối; > 1.3 = vỡ khối rõ. ``off_x/off_y`` là tọa độ góc ROI trong
    frame gốc để căn đúng lưới khối của bộ mã hóa.
    """
    g = gray.astype(np.float64)
    if g.ndim != 2 or min(g.shape) < 2 * block + 2:
        return 1.0
    ratios = []
    for axis, off in ((1, off_x), (0, off_y)):
        d = np.abs(np.diff(g, axis=axis)).mean(axis=1 - axis)   # chênh lệch TB theo vị trí
        pos = (np.arange(d.size) + off + 1) % block == 0         # biên giữa pixel 8k-1 và 8k
        if pos.sum() == 0 or (~pos).sum() == 0:
            continue
        other = float(d[~pos].mean())
        ratios.append(float(d[pos].mean()) / other if other > 1e-9 else 1.0)
    return max(ratios) if ratios else 1.0


def noise_sigma(gray: np.ndarray) -> float:
    """Ước lượng độ lệch chuẩn nhiễu Gauss (Immerkær 1996, 'Fast noise variance estimation')."""
    g = gray.astype(np.float64)
    if g.ndim != 2 or min(g.shape) < 3:
        return 0.0
    h, w = g.shape
    conv = cv2.filter2D(g, -1, _IMMERKAER)[1:-1, 1:-1]
    return math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2)) * float(np.abs(conv).sum())


def inner_face_box(frame_shape: tuple, bbox: list[float], keep: float = 0.8):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    mx, my = (x2 - x1) * (1 - keep) / 2, (y2 - y1) * (1 - keep) / 2
    ix1, iy1 = max(0, int(x1 + mx)), max(0, int(y1 + my))
    ix2, iy2 = min(w, int(math.ceil(x2 - mx))), min(h, int(math.ceil(y2 - my)))
    return ix1, iy1, ix2, iy2


def inner_face(frame_bgr: np.ndarray, bbox: list[float], keep: float = 0.8) -> np.ndarray:
    """Vùng giữa bbox (bỏ viền nền/tóc), ảnh xám — view tạm, không giữ lại."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    mx, my = (x2 - x1) * (1 - keep) / 2, (y2 - y1) * (1 - keep) / 2
    ix1, iy1, ix2, iy2 = inner_face_box(frame_bgr.shape, bbox, keep)
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    roi = frame_bgr[iy1:iy2, ix1:ix2]
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi


@dataclass
class QualityReport:
    total: float
    gate: Optional[str]
    components: dict
    pose: Optional[dict] = None
    blur_index: float = 1.0
    blockiness: float = 1.0
    noise_sigma: float = 0.0

    def as_dict(self) -> dict:
        out = {"total": round(self.total, 4), "gate": self.gate,
               **{k: round(v, 4) for k, v in self.components.items()},
               "blur_index": round(self.blur_index, 4),
               "blockiness": round(self.blockiness, 3),
               "noise_sigma": round(self.noise_sigma, 2)}
        if self.pose:
            out.update({k: round(v, 3) if isinstance(v, float) else v
                        for k, v in self.pose.items()})
        return out


def combine(components: dict, weights: dict = WEIGHTS) -> float:
    used = {k: w for k, w in weights.items() if k in components}
    total_w = sum(used.values())
    if total_w <= 0:
        return 0.0
    return math.exp(sum(w * math.log(max(FLOOR, components[k])) for k, w in used.items())
                    / total_w)


def assess_face(frame_bgr: np.ndarray, bbox: list[float], *, kps: Any = None,
                det_score: Optional[float] = None,
                embedding_norm: Optional[float] = None) -> QualityReport:
    """Chấm chất lượng một khuôn mặt trong frame gốc (tọa độ frame)."""
    comps: dict[str, float] = {}
    gate: Optional[str] = None
    pose = pose_from_kps(kps)
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if pose is not None:
        iod = pose["iod_px"]
        comps["pose"] = pose_score(pose)
        if not pose["plausible"]:
            gate = "landmark_implausible"
        elif abs(pose["yaw_ratio"]) > YAW_GATE:
            gate = "profile"
    else:
        iod = 0.38 * max(0.0, x2 - x1)        # không có landmark: IOD ≈ 38% bề rộng bbox
    comps["resolution"] = _ramp(iod, IOD_LOW_PX, IOD_GOOD_PX)

    gray = inner_face(frame_bgr, bbox)
    blur_idx, block, noise = 1.0, 1.0, 0.0
    if gray.size:
        norm = cv2.resize(gray, (SHARP_SIZE, SHARP_SIZE),
                          interpolation=cv2.INTER_AREA if gray.shape[0] > SHARP_SIZE
                          else cv2.INTER_LINEAR)
        blur_idx = blur_crete(norm)
        # Cạnh khối nén bị Crete hiểu nhầm là chi tiết nét → phạt theo độ vỡ khối,
        # đo trên ROI gốc (trước resize) căn theo lưới khối của frame.
        ix1, iy1, _, _ = inner_face_box(frame_bgr.shape, bbox)
        block = blockiness(gray, ix1, iy1)
        noise = noise_sigma(gray)
        block_pen = 1.0 - _ramp(block, BLOCK_OK, BLOCK_BAD)
        noise_pen = 1.0 - _ramp(noise, NOISE_OK, NOISE_BAD)
        comps["sharpness"] = (_ramp(1.0 - blur_idx, SHARP_LOW, SHARP_GOOD)
                              * block_pen * noise_pen)
        comps["exposure"] = exposure_score(gray)
    else:
        comps["sharpness"] = comps["exposure"] = 0.0

    trunc = truncation(bbox, frame_bgr.shape)
    comps["completeness"] = _clip01(1.0 - 2.0 * trunc)
    if trunc > TRUNCATION_GATE and gate is None:
        gate = "truncated"
    if embedding_norm is not None:
        comps["recognizability"] = _ramp(embedding_norm, NORM_LOW, NORM_GOOD)
    if det_score is not None:
        comps["detection"] = _ramp(det_score, 0.5, 0.9)

    total = 0.0 if gate else combine(comps)
    return QualityReport(total=total, gate=gate, components=comps, pose=pose,
                         blur_index=blur_idx, blockiness=block, noise_sigma=noise)


# --------------------------------------------------- nhất quán danh tính -----

@dataclass
class IdentityCentroid:
    """Tâm embedding có trọng số chất lượng của một tracklet — O(1) RAM."""
    sum_vec: Optional[np.ndarray] = None
    weight: float = 0.0
    count: int = 0

    def add(self, emb: Optional[np.ndarray], quality: float) -> None:
        if emb is None or quality <= 0.2:
            return
        v = np.asarray(emb, dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-9)
        self.sum_vec = v * quality if self.sum_vec is None else self.sum_vec + v * quality
        self.weight += quality
        self.count += 1

    def consistency(self, emb: Optional[np.ndarray]) -> float:
        """1 = đúng người của tracklet; thấp = có thể là người khác/crop hỏng."""
        if emb is None or self.sum_vec is None or self.count < CONSIST_MIN_OBS:
            return 1.0
        c = self.sum_vec / (np.linalg.norm(self.sum_vec) + 1e-9)
        v = np.asarray(emb, dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-9)
        return max(CONSIST_FLOOR, _ramp(float(np.dot(c, v)), CONSIST_LOW, CONSIST_GOOD))


def selection_score(quality: float, consistency: float) -> float:
    return quality * consistency


@dataclass
class RankedCrop:
    index: int
    quality: float
    consistency: float
    score: float
    gate: Optional[str] = None
    detail: dict = field(default_factory=dict)


def rank_faces(items: list[dict]) -> list[RankedCrop]:
    """Xếp hạng một tập ảnh mặt (offline: ảnh tải lên, thư mục crop).

    items: [{frame_bgr, bbox, kps?, det_score?, embedding?, embedding_norm?}].
    Hai lượt: chấm Q → dựng tâm danh tính từ ảnh tốt → nhân độ nhất quán.
    """
    reports = [assess_face(it["frame_bgr"], it["bbox"], kps=it.get("kps"),
                           det_score=it.get("det_score"),
                           embedding_norm=it.get("embedding_norm")) for it in items]
    centroid = IdentityCentroid()
    for it, r in zip(items, reports):
        centroid.add(it.get("embedding"), r.total)
    ranked = []
    for i, (it, r) in enumerate(zip(items, reports)):
        c = centroid.consistency(it.get("embedding"))
        ranked.append(RankedCrop(i, r.total, c, selection_score(r.total, c), r.gate,
                                 r.as_dict()))
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked
