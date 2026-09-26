"""P0/P1 — Repository tracklets (đoạn theo dõi trong 1 camera/source).

Tương thích DB chưa migrate 005: mọi hàm bắt lỗi thiếu bảng/cột và raise
RuntimeError rõ ràng để caller fallback (không crash luồng chính).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .errors import ConflictError, NotFoundError
from .media import _fk_violation, _unique_violation, new_id


def _missing_table(exc: Exception) -> bool:
    text = str(exc).lower()
    return "tracklets" in text and ("does not exist" in text or "undefinedtable" in text
                                    or "không tồn tại" in text)


def create_tracklet(
    conn, *, source_id: str, camera_id: Optional[str] = None,
    local_track_id: str, status: str = "open",
    started_at: Optional[str] = None, ended_at: Optional[str] = None,
    last_seen_at: Optional[str] = None, observation_count: int = 0,
    quality_summary: Optional[Any] = None, expires_at: Optional[str] = None,
    tracklet_id: Optional[str] = None,
) -> str:
    if status not in ("open", "closed", "expired"):
        raise ValueError(f"status tracklet không hợp lệ: {status!r}.")
    tracklet_id = tracklet_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.tracklets
                    (id, source_id, camera_id, local_track_id, status,
                     started_at, ended_at, last_seen_at, observation_count,
                     quality_summary, expires_at)
                VALUES (%s, %s, %s, %s, %s,
                        COALESCE(%s::timestamptz, now()), %s::timestamptz,
                        COALESCE(%s::timestamptz, now()), %s, %s::jsonb, %s::timestamptz)
                """,
                (tracklet_id, source_id, camera_id, local_track_id, status,
                 started_at, ended_at, last_seen_at, observation_count,
                 json.dumps(quality_summary or {}), expires_at),
            )
    except Exception as exc:
        if _missing_table(exc):
            raise RuntimeError("DB chưa migrate 005 (thiếu face_media.tracklets).") from exc
        if _unique_violation(exc):
            raise ConflictError("Tracklet đã tồn tại (trùng source/local_track_id).") from exc
        if _fk_violation(exc):
            raise ValueError("source_id/camera_id tham chiếu không tồn tại.") from exc
        raise
    return tracklet_id


def close_tracklet(conn, tracklet_id: str, *, ended_at: Optional[str] = None,
                   observation_count: Optional[int] = None,
                   quality_summary: Optional[Any] = None) -> None:
    try:
        with conn.cursor() as cur:
            sets, vals = ["status = 'closed'"], []
            if ended_at is not None:
                sets.append("ended_at = %s::timestamptz")
                vals.append(ended_at)
            else:
                sets.append("ended_at = COALESCE(ended_at, now())")
            sets.append("last_seen_at = now()")
            if observation_count is not None:
                sets.append("observation_count = %s")
                vals.append(observation_count)
            if quality_summary is not None:
                sets.append("quality_summary = %s::jsonb")
                vals.append(json.dumps(quality_summary))
            vals.append(tracklet_id)
            cur.execute(
                f"UPDATE face_media.tracklets SET {', '.join(sets)} WHERE id = %s",
                vals,
            )
            if cur.rowcount == 0:
                raise NotFoundError(f"Tracklet không tồn tại: {tracklet_id}")
    except Exception as exc:
        if _missing_table(exc):
            raise RuntimeError("DB chưa migrate 005 (thiếu face_media.tracklets).") from exc
        raise


def get_tracklet(conn, tracklet_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_id, camera_id, local_track_id, status,
                   started_at, ended_at, last_seen_at, observation_count,
                   quality_summary, expires_at, created_at
            FROM face_media.tracklets WHERE id = %s
            """,
            (tracklet_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise NotFoundError(f"Tracklet không tồn tại: {tracklet_id}")
    keys = ("tracklet_id", "source_id", "camera_id", "local_track_id", "status",
            "started_at", "ended_at", "last_seen_at", "observation_count",
            "quality_summary", "expires_at", "created_at")
    out = dict(zip(keys, row))
    for k in ("started_at", "ended_at", "last_seen_at", "expires_at", "created_at"):
        out[k] = str(out[k]) if out[k] else None
    return out


def touch_tracklet(conn, tracklet_id: str, *, source_id: str,
                   camera_id: Optional[str] = None, local_track_id: str,
                   status: str = "open",
                   last_seen_at: Optional[str] = None,
                   observation_count: Optional[int] = None,
                   quality_summary: Optional[Any] = None) -> None:
    """Checkpoint (§4.1): tạo hàng tracklet nếu chưa có, nếu có thì cập nhật
    trạng thái/observation mà KHÔNG đóng (upsert theo id quyết định)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.tracklets
                    (id, source_id, camera_id, local_track_id, status,
                     last_seen_at, observation_count, quality_summary)
                VALUES (%s, %s, %s, %s, %s,
                        COALESCE(%s::timestamptz, now()), %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    last_seen_at = now(),
                    observation_count = GREATEST(
                        face_media.tracklets.observation_count,
                        EXCLUDED.observation_count),
                    quality_summary = EXCLUDED.quality_summary
                """,
                (tracklet_id, source_id, camera_id, local_track_id, status,
                 last_seen_at, observation_count or 0,
                 json.dumps(quality_summary or {})),
            )
    except Exception as exc:
        if _missing_table(exc):
            raise RuntimeError("DB chưa migrate 005 (thiếu face_media.tracklets).") from exc
        raise
