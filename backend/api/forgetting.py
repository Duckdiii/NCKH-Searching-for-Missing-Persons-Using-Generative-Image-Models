"""Quên dần dữ liệu camera theo đường cong quên (Ebbinghaus), có củng cố và áp lực ổ đĩa.

Module bổ sung cho retention.py (TTL cố định). Xem docs/thuat-toan-quen-va-lay-mau.md.

Đơn vị nhớ là một tracklet (một lượt người đi qua một camera). Độ nhớ:

    R(Δt) = exp(−Δt / S)

- Δt: số ngày kể từ lần CUỐI người đó được thấy (theo danh tính toàn cục nếu
  tracklet đã được liên kết, không thì theo chính tracklet).
- S (độ bền, ngày) = S0 · g_củng_cố · g_chất_lượng · g_áp_lực
    g_củng_cố  = 1 + α · ln(số ngày khác nhau người đó xuất hiện)   (lặp lại → nhớ lâu)
    g_chất_lượng = 0.5 + q, q ∈ [0, 1] chất lượng ảnh tốt nhất        (ảnh tồi → quên sớm)
    g_áp_lực  = 1 khi ổ dưới ngưỡng cảnh báo, giảm tuyến tính tới
                pressure_min_factor ở ngưỡng dừng                    (gần đầy → quên nhanh)

Hai bậc quên (tính ra thời điểm, kẹp bởi sàn/trần):
    LƯỢC (thin):  R < thin_threshold   → chỉ giữ crop tốt nhất của tracklet
    QUÊN (forget): R < forget_threshold → xóa mọi crop + đánh dấu tracklet expired
    t_lược = S·ln(1/θ_lược),  t_quên = clamp(S·ln(1/θ_quên), sàn, trần)

Ngoại lệ không bao giờ quên tự động:
    - crop từng KHỚP truy vấn tìm người (search_results.accepted_by_threshold)
      trong ``hold_days`` gần nhất — là bằng chứng cho hồ sơ;
    - crop đang làm mẫu định danh (identity_exemplars) — theo policy exemplar riêng.

An toàn giống retention.py: mặc định chỉ lập kế hoạch (dry run); xóa thật chỉ khi
FORGETTING_ENABLED=true, và việc xóa đi qua deletion_ledger (tombstone → GC).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from backend.api.retention import STOP_RATIO, WARN_RATIO

logger = logging.getLogger(__name__)

KEEP, THIN, FORGET, HOLD = "keep", "thin", "forget", "hold"


@dataclass(frozen=True)
class ForgettingParams:
    base_stability_days: float = 3.0   # S0: một lần thấy, chất lượng trung bình
    reinforce_alpha: float = 0.7       # mức tăng độ bền theo số ngày xuất hiện
    thin_threshold: float = 0.5        # R dưới mức này → lược còn 1 crop
    forget_threshold: float = 0.1      # R dưới mức này → quên hẳn
    floor_days: float = 7.0            # giữ tối thiểu (người nhà thường báo mất tích sau vài ngày)
    cap_days: float = 30.0             # giữ tối đa với người lạ chưa từng khớp truy vấn
    hold_days: float = 180.0           # crop đã khớp truy vấn: giữ sau lần khớp cuối
    pressure_min_factor: float = 0.4   # độ bền còn 40% khi ổ chạm ngưỡng dừng
    warn_ratio: float = WARN_RATIO
    stop_ratio: float = STOP_RATIO

    @classmethod
    def from_env(cls) -> "ForgettingParams":
        def f(name: str, default: float) -> float:
            raw = (os.environ.get(name, "") or "").strip()
            try:
                return float(raw) if raw else default
            except ValueError:
                logger.warning("Bỏ qua %s=%r (không phải số).", name, raw)
                return default
        d = cls()
        p = cls(
            base_stability_days=f("FORGET_BASE_STABILITY_DAYS", d.base_stability_days),
            reinforce_alpha=f("FORGET_REINFORCE_ALPHA", d.reinforce_alpha),
            thin_threshold=f("FORGET_THIN_THRESHOLD", d.thin_threshold),
            forget_threshold=f("FORGET_THRESHOLD", d.forget_threshold),
            floor_days=f("FORGET_FLOOR_DAYS", d.floor_days),
            cap_days=f("FORGET_CAP_DAYS", d.cap_days),
            hold_days=f("FORGET_HOLD_DAYS", d.hold_days),
            pressure_min_factor=f("FORGET_PRESSURE_MIN_FACTOR", d.pressure_min_factor),
        )
        p.validate()
        return p

    def validate(self) -> None:
        if self.base_stability_days <= 0:
            raise ValueError("base_stability_days phải > 0.")
        if not 0.0 < self.forget_threshold < self.thin_threshold < 1.0:
            raise ValueError("Cần 0 < forget_threshold < thin_threshold < 1.")
        if not 0.0 < self.floor_days <= self.cap_days:
            raise ValueError("Cần 0 < floor_days <= cap_days.")
        if not 0.0 < self.pressure_min_factor <= 1.0:
            raise ValueError("pressure_min_factor phải trong (0, 1].")


def enabled() -> bool:
    return (os.environ.get("FORGETTING_ENABLED", "false") or "").strip().lower() == "true"


def pressure_factor(usage_ratio: Optional[float], p: ForgettingParams) -> float:
    if usage_ratio is None or usage_ratio <= p.warn_ratio:
        return 1.0
    span = max(p.stop_ratio - p.warn_ratio, 1e-9)
    frac = min(1.0, (usage_ratio - p.warn_ratio) / span)
    return 1.0 - (1.0 - p.pressure_min_factor) * frac


def stability_days(sighting_days: int, quality: float,
                   usage_ratio: Optional[float], p: ForgettingParams) -> float:
    n = max(1, int(sighting_days))
    q = min(1.0, max(0.0, float(quality)))
    return (p.base_stability_days
            * (1.0 + p.reinforce_alpha * math.log(n))
            * (0.5 + q)
            * pressure_factor(usage_ratio, p))


def strength(age_days: float, s_days: float) -> float:
    return math.exp(-max(0.0, age_days) / s_days)


def schedule(s_days: float, p: ForgettingParams, usage_ratio: Optional[float] = None) -> dict:
    """Thời điểm lược/quên (ngày kể từ lần thấy cuối) cho một độ bền S.

    Sàn chỉ áp cho bậc quên (lược vẫn giữ ảnh tốt nhất nên an toàn). Khi ổ đã chạm
    ngưỡng dừng, trần cũng co lại theo cùng hệ số áp lực nhưng không dưới sàn.
    """
    cap = max(p.floor_days, p.cap_days * pressure_factor(usage_ratio, p))
    forget_at = min(cap, max(p.floor_days, s_days * math.log(1.0 / p.forget_threshold)))
    thin_at = min(forget_at, s_days * math.log(1.0 / p.thin_threshold))
    return {"thin_after_days": round(thin_at, 2), "forget_after_days": round(forget_at, 2)}


@dataclass
class MemoryItem:
    tracklet_id: str
    last_seen: datetime                 # lần thấy cuối (danh tính hoặc tracklet)
    sighting_days: int = 1              # số ngày khác nhau danh tính xuất hiện
    quality: float = 0.5                # chất lượng crop tốt nhất (0–1)
    last_match: Optional[datetime] = None   # lần cuối crop của tracklet khớp truy vấn


def decide(item: MemoryItem, now: datetime, p: ForgettingParams,
           usage_ratio: Optional[float] = None) -> dict:
    age = (now - item.last_seen).total_seconds() / 86400.0
    s = stability_days(item.sighting_days, item.quality, usage_ratio, p)
    sch = schedule(s, p, usage_ratio)
    out = {
        "tracklet_id": item.tracklet_id,
        "age_days": round(age, 2),
        "stability_days": round(s, 2),
        "strength": round(strength(age, s), 3),
        **sch,
    }
    if item.last_match is not None:
        held = (now - item.last_match).total_seconds() / 86400.0 < p.hold_days
        if held:
            return {**out, "stage": HOLD, "reason": "đã khớp truy vấn tìm người"}
    if age >= sch["forget_after_days"]:
        return {**out, "stage": FORGET, "reason": "hết độ nhớ"}
    if age >= sch["thin_after_days"]:
        return {**out, "stage": THIN, "reason": "độ nhớ thấp: giữ ảnh tốt nhất"}
    return {**out, "stage": KEEP, "reason": "còn nhớ"}


# ---------------------------------------------------------------- DB ---------

_ITEMS_SQL = """
WITH trk AS (
    SELECT t.id, t.last_seen_at, a.global_id
    FROM face_media.tracklets t
    LEFT JOIN face_media.identity_assignments a
      ON a.tracklet_id = t.id AND a.valid_to IS NULL
    WHERE t.status = 'closed'
),
ident AS (
    SELECT a.global_id, max(t.last_seen_at) AS last_seen,
           count(DISTINCT date_trunc('day', t.last_seen_at)) AS days
    FROM face_media.identity_assignments a
    JOIN face_media.tracklets t ON t.id = a.tracklet_id
    WHERE a.valid_to IS NULL
    GROUP BY a.global_id
),
q AS (
    SELECT d.tracklet_id,
           max(coalesce((d.quality->>'track_quality')::real, d.confidence)) AS quality
    FROM face_media.face_detections d
    WHERE d.tracklet_id IS NOT NULL
    GROUP BY d.tracklet_id
),
held AS (
    SELECT d.tracklet_id, max(r.created_at) AS last_match
    FROM face_media.search_results r
    JOIN face_media.face_crops c ON c.id = r.candidate_crop_id
    JOIN face_media.face_detections d ON d.id = c.detection_id
    WHERE r.accepted_by_threshold AND d.tracklet_id IS NOT NULL
    GROUP BY d.tracklet_id
)
SELECT trk.id, coalesce(ident.last_seen, trk.last_seen_at), coalesce(ident.days, 1),
       coalesce(q.quality, 0.5), held.last_match
FROM trk
LEFT JOIN ident ON ident.global_id = trk.global_id
LEFT JOIN q ON q.tracklet_id = trk.id
LEFT JOIN held ON held.tracklet_id = trk.id
-- Cũ nhất trước; tracklet đang được giữ (đã khớp truy vấn) xếp cuối để không
-- chiếm hết batch mãi mãi.
ORDER BY (held.last_match IS NOT NULL), 2
LIMIT %s
"""

_CROPS_SQL = """
SELECT c.id, a.storage_key, a.byte_size, d.tracklet_id,
       coalesce((d.quality->>'track_quality')::real, d.confidence) AS q
FROM face_media.face_crops c
JOIN face_media.face_detections d ON d.id = c.detection_id
JOIN face_media.assets a ON a.id = c.asset_id
LEFT JOIN face_media.deletion_ledger l ON l.crop_id = c.id AND l.done = false
LEFT JOIN face_media.identity_exemplars e ON e.crop_id = c.id
WHERE d.tracklet_id = ANY(%s) AND l.id IS NULL AND e.crop_id IS NULL
ORDER BY d.tracklet_id, q DESC NULLS LAST
"""


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def plan_forgetting(conn, *, now: Optional[datetime] = None, batch: int = 500,
                    params: Optional[ForgettingParams] = None,
                    usage_ratio: Optional[float] = None) -> dict:
    """Lập kế hoạch quên (không ghi gì). Trả quyết định từng tracklet + crop cần xóa."""
    p = params or ForgettingParams.from_env()
    now = _aware(now or datetime.now(timezone.utc))
    with conn.cursor() as cur:
        cur.execute(_ITEMS_SQL, (batch,))
        rows = cur.fetchall()
    decisions = [
        decide(MemoryItem(str(r[0]), _aware(r[1]), int(r[2] or 1), float(r[3] or 0.0),
                          _aware(r[4]) if r[4] else None), now, p, usage_ratio)
        for r in rows
    ]
    act = {d["tracklet_id"]: d["stage"] for d in decisions if d["stage"] in (THIN, FORGET)}
    crops_thin: list[dict] = []
    crops_forget: list[dict] = []
    if act:
        with conn.cursor() as cur:
            cur.execute(_CROPS_SQL, (list(act),))
            crop_rows = cur.fetchall()
        kept_best: set[str] = set()
        for crop_id, key, size, tid, _q in crop_rows:   # đã sắp q giảm dần theo tracklet
            tid = str(tid)
            c = {"crop_id": str(crop_id), "storage_key": key, "byte_size": int(size or 0),
                 "tracklet_id": tid}
            if act[tid] == FORGET:
                crops_forget.append(c)
            elif tid in kept_best:
                crops_thin.append(c)
            else:
                kept_best.add(tid)      # crop tốt nhất của tracklet được lược: giữ lại
    counts = {s: 0 for s in (KEEP, THIN, FORGET, HOLD)}
    for d in decisions:
        counts[d["stage"]] += 1
    return {
        "now": now.isoformat(),
        "usage_ratio": usage_ratio,
        "pressure_factor": round(pressure_factor(usage_ratio, p), 3),
        "params": asdict(p),
        "counts": counts,
        "crops_thin": crops_thin,
        "crops_forget": crops_forget,
        "tracklets_forget": [t for t, s in act.items() if s == FORGET],
        "bytes_to_free": sum(c["byte_size"] for c in crops_thin + crops_forget),
        "decisions": decisions,
    }


def apply_forgetting(conn, plan: dict, *, dry_run: bool = True) -> dict:
    """Thực thi kế hoạch: tombstone crop qua ledger (GC xóa sau) + expire tracklet."""
    from backend.api.retention import tombstone_crops
    result = {"dry_run": dry_run, "thinned": 0, "forgotten": 0, "tracklets_expired": 0,
              "bytes_to_free": plan.get("bytes_to_free", 0)}
    if dry_run:
        result.update(thinned=len(plan["crops_thin"]), forgotten=len(plan["crops_forget"]),
                      tracklets_expired=len(plan["tracklets_forget"]))
        return result
    if not enabled():
        raise PermissionError("Đặt FORGETTING_ENABLED=true để quên thật (mặc định chỉ dry run).")
    if plan["crops_thin"]:
        result["thinned"] = tombstone_crops(conn, plan["crops_thin"], "forget_thin")
    if plan["crops_forget"]:
        result["forgotten"] = tombstone_crops(conn, plan["crops_forget"], "forget_full")
    if plan["tracklets_forget"]:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE face_media.tracklets SET status = 'expired', expires_at = now()"
                " WHERE id = ANY(%s) AND status = 'closed'",
                (plan["tracklets_forget"],),
            )
            result["tracklets_expired"] = cur.rowcount or 0
    return result
