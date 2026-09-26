"""P1 — Tracker local trong camera + selector 1–3 crop/tracklet (crop-only).

Theo docs/kien-truc-da-camera-tiet-kiem-bo-nho.md §4:
- Tracking bbox mặt trong từng camera bằng IoU (+ embedding khi mơ hồ).
  ByteTrack là tham khảo association trong một camera, không giải ID xuyên camera.
- Mỗi tracklet giữ tối đa 3 ứng viên trong RAM: ảnh rõ nhất gần chính diện
  + tối đa 2 ảnh góc khác hữu ích. Đánh giá độ nét, kích thước mặt thực,
  che khuất, phơi sáng, góc mặt; confidence detector KHÔNG đủ.
- Crop phải copy vùng mặt (tránh NumPy view giữ tham chiếu toàn frame).
- Chỉ thay mẫu khi tốt hơn đáng kể hoặc thêm góc hữu ích; không thay vì
  cosine dao động nhẹ. Người đứng lâu không sinh tracklet mới mỗi phút:
  checkpoint cùng tracklet + hard timeout + quota.
- Encode JPEG cạnh dài tối đa 256px, giữ tỉ lệ, chỉ downscale (không upscale),
  Q85 mặc định để benchmark Q80/85/90.

Module thuần (không DB/storage): RAM-bounded, deterministic, test được.
Persistence crop-only nằm ở ingest.persist_camera_tracklet().
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

# --- Thông số khởi điểm cần đo (doc §4.2) ---
CROP_MAX_LONG_EDGE = 256
CROP_JPEG_QUALITY = 85
MAX_CANDIDATES_PER_TRACKLET = 3
# Thay mẫu chỉ khi quality tốt hơn đáng kể (tránh churn vì cosine dao động).
REPLACE_QUALITY_MARGIN = 0.08
# Góc khác hữu ích: embedding cosine < ngưỡng này so với mọi mẫu đã giữ.
NOVEL_ANGLE_COSINE = 0.92
# Tracker.
IOU_MATCH_THRESHOLD = 0.3
TRACK_TIMEOUT_SEC = 3.0
TRACK_HARD_TIMEOUT_SEC = 300.0
# Tốc độ thay slot sớm tối thiểu (doc §4.1: giới hạn tốc độ thay slot).
CHECKPOINT_MIN_INTERVAL_SEC = 60.0
MAX_OPEN_TRACKS_PER_CAMERA = 20
COOLDOWN_SEC = 2.0
# Quota camera/ngày (doc §4.1): chặn tracker đứt sinh tracklet vô hạn.
MAX_TRACKLETS_PER_CAMERA_PER_DAY = 2000


def bbox_iou(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def estimate_blur(gray: np.ndarray) -> float:
    """Variance of Laplacian — proxy độ nét, càng cao càng nét."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_exposure(gray: np.ndarray) -> float:
    """Điểm phơi sáng [0,1]: mean gần 0/255 hoặc std thấp => kém."""
    if gray.size == 0:
        return 0.0
    mean = float(gray.mean())
    std = float(gray.std())
    # Lý tưởng mean ~110-160, std đủ lớn.
    mean_score = max(0.0, 1.0 - abs(mean - 135.0) / 135.0)
    std_score = min(1.0, std / 60.0)
    return 0.6 * mean_score + 0.4 * std_score


def estimate_frontal(kps: Any, bbox: list[float]) -> float:
    """Proxy góc mặt [0,1] từ landmark đối xứng; thiếu kps => 0.5 trung tính."""
    try:
        if kps is None:
            return 0.5
        pts = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 5:
            return 0.5
        x1, _, x2, _ = (float(v) for v in bbox)
        w = max(1.0, x2 - x1)
        # 5-point: mắt trái/phải, mũi, miệng trái/phải.
        eye_dx = abs(float(pts[0][0]) - float(pts[1][0])) / w
        mouth_dx = abs(float(pts[3][0]) - float(pts[4][0])) / w
        nose_cx = (float(pts[2][0]) - x1) / w  # 0.5 = chính diện
        sym = max(0.0, 1.0 - abs(nose_cx - 0.5) * 4.0)
        size_ok = min(1.0, (eye_dx + mouth_dx) * 2.0)
        return 0.6 * sym + 0.4 * size_ok
    except Exception:
        return 0.5


def score_candidate(face_area: float, blur: float, exposure: float,
                    frontal: float, det_score: float) -> float:
    """Điểm chất lượng tổng hợp. det_score chỉ là 1 thành phần nhỏ."""
    area_term = min(1.0, math.log1p(max(0.0, face_area) / 2000.0) / 4.0)
    blur_term = min(1.0, blur / 300.0)
    return (
        0.30 * blur_term
        + 0.25 * area_term
        + 0.20 * frontal
        + 0.15 * exposure
        + 0.10 * max(0.0, min(1.0, det_score))
    )


def copy_crop(frame_bgr: np.ndarray, bbox: list[float],
              pad_ratio: float = 0.15) -> np.ndarray:
    """Cắt crop có padding, COPY để không giữ tham chiếu toàn frame.

    Giữ đủ vùng quanh mặt để căn chỉnh lại (doc §4.2). Trả bản copy liên tục.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1p, y1p = x1 - bw * pad_ratio, y1 - bh * pad_ratio
    x2p, y2p = x2 + bw * pad_ratio, y2 + bh * pad_ratio
    ix1, iy1 = max(0, int(x1p)), max(0, int(y1p))
    ix2, iy2 = min(w, int(math.ceil(x2p))), min(h, int(math.ceil(y2p)))
    if ix2 <= ix1 or iy2 <= iy1:
        ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return np.ascontiguousarray(frame_bgr[0:1, 0:1].copy())
    return np.ascontiguousarray(frame_bgr[iy1:iy2, ix1:ix2].copy())


def encode_crop_jpeg(crop_bgr: np.ndarray,
                     quality: int = CROP_JPEG_QUALITY) -> tuple[bytes, dict]:
    """Resize cạnh dài về ≤256 (chỉ downscale) + encode JPEG.

    Trả (bytes, provenance {codec, quality, size_src, size_dst}).
    """
    h, w = crop_bgr.shape[:2]
    longest = max(h, w)
    dst = crop_bgr
    if longest > CROP_MAX_LONG_EDGE:
        scale = CROP_MAX_LONG_EDGE / float(longest)
        dst = cv2.resize(crop_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    dh, dw = dst.shape[:2]
    ok, buf = cv2.imencode(".jpg", dst,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
                            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
    if not ok:
        raise IOError("Encode crop JPEG thất bại.")
    prov = {"codec": "jpeg", "quality": int(quality),
            "size_src": [int(w), int(h)], "size_dst": [int(dw), int(dh)],
            "resize": "area-downscale" if dst is not crop_bgr else "none",
            "preprocessing_version": "camera_crop_v1"}
    return bytes(buf), prov


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Candidate:
    crop_bgr: np.ndarray  # bản copy, không view frame
    quality: float
    blur: float
    exposure: float
    frontal: float
    face_area: float
    det_score: float
    bbox: list[float]
    embedding: Optional[np.ndarray] = None
    observed_at: float = 0.0
    sha256: str = ""


@dataclass
class TrackState:
    local_id: str
    track_id: int
    last_bbox: list[float]
    last_seen_monotonic: float
    started_monotonic: float
    observation_count: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    seen_hashes: set[str] = field(default_factory=set)
    closed: bool = False
    last_embedding: Optional[np.ndarray] = None
    # P-checkpoint (§4.1): lưu sớm theo slot có revision, giới hạn tốc độ.
    slot_revision: int = 0
    last_checkpoint_monotonic: float = 0.0
    checkpoint_pending: bool = False


class CameraTracker:
    """Tracker local 1 camera: IoU greedy + embedding tie-break, bounded RAM.

    - Không tự tạo ID xuyên camera (doc §2: dedupe dùng sau association riêng).
    - Người đứng lâu: checkpoint cùng tracklet, hard timeout + quota chặn tăng vô hạn.
    - Tracker đứt: cooldown lượt quay lại + TTL + quota camera/ngày.
    """

    def __init__(self, *,
                 iou_threshold: float = IOU_MATCH_THRESHOLD,
                 timeout_sec: float = TRACK_TIMEOUT_SEC,
                 hard_timeout_sec: float = TRACK_HARD_TIMEOUT_SEC,
                 max_open: int = MAX_OPEN_TRACKS_PER_CAMERA,
                 cooldown_sec: float = COOLDOWN_SEC,
                 max_tracklets_per_day: int = MAX_TRACKLETS_PER_CAMERA_PER_DAY,
                 checkpoint_min_interval_sec: float = CHECKPOINT_MIN_INTERVAL_SEC,
                 now: Optional[float] = None):
        self.iou_threshold = iou_threshold
        self.timeout_sec = timeout_sec
        self.hard_timeout_sec = hard_timeout_sec
        self.max_open = max_open
        self.cooldown_sec = cooldown_sec
        self.max_tracklets_per_day = max_tracklets_per_day
        self.checkpoint_min_interval_sec = checkpoint_min_interval_sec
        self._tracks: dict[str, TrackState] = {}
        self._next_id = 1
        self._created_today = 0
        self._day = time.strftime("%Y-%m-%d")
        self._recently_closed: list[tuple[list[float], float]] = []
        self.dropped_quota = 0
        self._now0 = now

    def _now(self) -> float:
        return self._now0 if self._now0 is not None else time.monotonic()

    def set_now(self, value: float) -> None:
        self._now0 = value

    def _roll_day(self) -> None:
        day = time.strftime("%Y-%m-%d")
        if day != self._day:
            self._day = day
            self._created_today = 0

    def open_tracks(self) -> list[TrackState]:
        return [t for t in self._tracks.values() if not t.closed]

    def update(self, frame_bgr: np.ndarray,
               faces: list[dict], now: Optional[float] = None) -> dict[str, list[dict]]:
        """Gán detection -> track. faces: [{bbox_orig, det_score, embedding, kps}].

        Trả {local_id: [face,...]}. Frame không crop ở đây — selector quyết định
        sau (tránh giữ crop mỗi frame).
        """
        t = now if now is not None else self._now()
        self._roll_day()
        self._expire(t)
        assignment: dict[str, list[dict]] = {}
        unmatched: list[dict] = []
        open_tracks = self.open_tracks()
        used: set[str] = set()
        for face in faces:
            bbox = [float(v) for v in face["bbox_orig"]]
            best_id, best_score = None, self.iou_threshold
            for tr in open_tracks:
                if tr.local_id in used:
                    continue
                score = bbox_iou(bbox, tr.last_bbox)
                # Tie-break bằng embedding khi IoU mơ hồ (0.2–0.5).
                if 0.15 < score < 0.55 and face.get("embedding") is not None \
                        and tr.last_embedding is not None:
                    try:
                        a = np.asarray(face["embedding"], dtype=np.float64)
                        b = np.asarray(tr.last_embedding, dtype=np.float64)
                        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                        score += 0.1 * max(0.0, cos)
                    except Exception:
                        pass
                if score >= best_score:
                    best_score, best_id = score, tr.local_id
            if best_id is None:
                unmatched.append(face)
            else:
                used.add(best_id)
                assignment.setdefault(best_id, []).append(face)
        # Mở track mới cho unmatched (kèm cooldown + quota).
        for face in unmatched:
            bbox = [float(v) for v in face["bbox_orig"]]
            if self._in_cooldown(bbox, t):
                continue
            if self._created_today >= self.max_tracklets_per_day:
                self.dropped_quota += 1
                continue
            if len(self.open_tracks()) >= self.max_open:
                # Evict track cũ nhất (không tăng cache vô hạn).
                oldest = min(self.open_tracks(), key=lambda tr: tr.last_seen_monotonic)
                oldest.closed = True
            lid = f"t{self._next_id}"
            self._next_id += 1
            self._tracks[lid] = TrackState(
                local_id=lid, track_id=self._next_id,
                last_bbox=bbox, last_seen_monotonic=t, started_monotonic=t)
            self._created_today += 1
            assignment.setdefault(lid, []).append(face)
        # Cập nhật trạng thái + selector ứng viên.
        for lid, flist in assignment.items():
            tr = self._tracks[lid]
            for face in flist:
                tr.observation_count += 1
                tr.last_bbox = [float(v) for v in face["bbox_orig"]]
                tr.last_seen_monotonic = t
                if face.get("embedding") is not None:
                    try:
                        tr.last_embedding = np.asarray(face["embedding"], dtype=np.float64)
                    except Exception:
                        pass
                self._consider_candidate(tr, frame_bgr, face, t)
            # Hard timeout: checkpoint CÙNG tracklet (§4.1) thay vì cắt tracklet
            # mới mỗi N phút cho người đứng lâu. Slot có revision, giới hạn tốc
            # độ thay slot; cửa sổ hard timeout khởi động lại sau checkpoint.
            if (t - tr.started_monotonic > self.hard_timeout_sec
                    and tr.candidates
                    and t - tr.last_checkpoint_monotonic >= self.checkpoint_min_interval_sec):
                tr.checkpoint_pending = True
                tr.slot_revision += 1
                tr.last_checkpoint_monotonic = t
                tr.started_monotonic = t
        return assignment

    def _in_cooldown(self, bbox: list[float], t: float) -> bool:
        self._recently_closed = [(b, ts) for b, ts in self._recently_closed
                                 if t - ts < self.cooldown_sec * 10]
        for b, ts in self._recently_closed:
            if t - ts < self.cooldown_sec and bbox_iou(bbox, b) > 0.5:
                return True
        return False

    def _expire(self, t: float) -> list[TrackState]:
        expired = []
        for tr in self._tracks.values():
            if not tr.closed and t - tr.last_seen_monotonic > self.timeout_sec:
                tr.closed = True
                self._recently_closed.append((list(tr.last_bbox), t))
                expired.append(tr)
        # Giữ RAM bounded: chỉ nhớ tối đa max_open track đã đóng gần nhất.
        closed = [tr for tr in self._tracks.values() if tr.closed]
        if len(closed) > self.max_open * 2:
            for tr in sorted(closed, key=lambda x: x.last_seen_monotonic)[:len(closed) - self.max_open * 2]:
                del self._tracks[tr.local_id]
        return expired

    def pop_closed_ready(self, t: Optional[float] = None) -> list[TrackState]:
        """Lấy track vừa đóng (có ứng viên) để persist, rồi quên để giải phóng RAM."""
        now = t if t is not None else self._now()
        self._expire(now)
        ready = [tr for tr in self._tracks.values()
                 if tr.closed and tr.candidates]
        for tr in ready:
            del self._tracks[tr.local_id]
        return ready

    def pop_checkpoints(self) -> list[TrackState]:
        """Lấy track mở đến kỳ checkpoint sớm (§4.1): trả ứng viên rồi XÓA crop
        khỏi RAM (giữ seen_hashes để tiếp tục lọc trùng), track ở lại mở."""
        out = []
        for tr in self._tracks.values():
            if not tr.closed and tr.checkpoint_pending and tr.candidates:
                tr.checkpoint_pending = False
                out.append(tr)
        return out

    def release_checkpoint_candidates(self, tr: TrackState) -> None:
        """Giải phóng crop RAM sau khi checkpoint đã persist."""
        tr.candidates = []

    def close_all(self, t: Optional[float] = None) -> list[TrackState]:
        """Đóng mọi track mở (dùng khi stream reconnect: timeline phải thể hiện
        khoảng trống thay vì nốiTracklet qua đoạn mất hình, §5)."""
        now = t if t is not None else self._now()
        out = []
        for tr in self._tracks.values():
            if not tr.closed:
                tr.closed = True
                self._recently_closed.append((list(tr.last_bbox), now))
                if tr.candidates:
                    out.append(tr)
        return out

    def _consider_candidate(self, tr: TrackState, frame_bgr: np.ndarray,
                            face: dict, t: float) -> None:
        bbox = [float(v) for v in face["bbox_orig"]]
        x1, y1, x2, y2 = bbox
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        # Bỏ mặt quá nhỏ (không đủ khả năng nhận dạng, tiết kiệm crop).
        if area < 40 * 40:
            return
        crop = copy_crop(frame_bgr, bbox)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = estimate_blur(gray)
        exposure = estimate_exposure(gray)
        frontal = estimate_frontal(face.get("kps"), bbox)
        det = float(face.get("det_score", 0.0))
        quality = score_candidate(area, blur, exposure, frontal, det)
        emb = None
        if face.get("embedding") is not None:
            try:
                emb = np.asarray(face["embedding"], dtype=np.float64)
            except Exception:
                emb = None
        cand = Candidate(crop_bgr=crop, quality=quality, blur=blur,
                         exposure=exposure, frontal=frontal, face_area=area,
                         det_score=det, bbox=bbox, embedding=emb, observed_at=t)
        # Lọc trùng byte trong tracklet (SHA-256 trên crop resize nhỏ).
        try:
            tiny = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
            cand.sha256 = sha256_bytes(tiny.tobytes())
        except Exception:
            cand.sha256 = ""
        if cand.sha256 and cand.sha256 in tr.seen_hashes:
            return
        if len(tr.candidates) < MAX_CANDIDATES_PER_TRACKLET:
            # Thêm góc mới hữu ích hoặc chất lượng cao hơn.
            if self._is_novel_angle(tr, emb) or not tr.candidates \
                    or quality > min(c.quality for c in tr.candidates):
                tr.candidates.append(cand)
                if cand.sha256:
                    tr.seen_hashes.add(cand.sha256)
                tr.candidates.sort(key=lambda c: c.quality, reverse=True)
                tr.candidates = tr.candidates[:MAX_CANDIDATES_PER_TRACKLET]
            return
        # Đã đủ 3: chỉ thay khi tốt hơn đáng kể hoặc góc mới hữu ích.
        worst = min(tr.candidates, key=lambda c: c.quality)
        is_novel = self._is_novel_angle(tr, emb)
        if quality > worst.quality + REPLACE_QUALITY_MARGIN or (
                is_novel and quality > worst.quality):
            tr.candidates.remove(worst)
            tr.candidates.append(cand)
            if cand.sha256:
                tr.seen_hashes.add(cand.sha256)
            tr.candidates.sort(key=lambda c: c.quality, reverse=True)

    def _is_novel_angle(self, tr: TrackState, emb: Optional[np.ndarray]) -> bool:
        if emb is None or not tr.candidates:
            return True
        try:
            a = emb / (np.linalg.norm(emb) + 1e-9)
            for c in tr.candidates:
                if c.embedding is None:
                    continue
                b = np.asarray(c.embedding, dtype=np.float64)
                b = b / (np.linalg.norm(b) + 1e-9)
                if float(np.dot(a, b)) >= NOVEL_ANGLE_COSINE:
                    return False
            return True
        except Exception:
            return True
