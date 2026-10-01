"""P2 — Repository global identities / assignments / exemplars / topology.

Mọi hàm chịu được DB chưa migrate 006: raise RuntimeError rõ ràng để caller
fallback (giữ unresolved) thay vì crash luồng ingest.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .errors import ConflictError, NotFoundError
from .media import _fk_violation, _unique_violation, new_id


def _missing(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("global_identities" in text or "identity_assignments" in text
            or "identity_exemplars" in text or "camera_topology" in text) and (
        "does not exist" in text or "undefinedtable" in text)


def create_identity(conn, *, status: str = "open",
                    last_seen_at: Optional[str] = None,
                    expires_at: Optional[str] = None,
                    identity_id: Optional[str] = None) -> str:
    if status not in ("open", "merged", "split", "archived", "unresolved"):
        raise ValueError(f"status identity không hợp lệ: {status!r}.")
    identity_id = identity_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.global_identities
                    (id, status, last_seen_at, expires_at)
                VALUES (%s, %s, %s::timestamptz, %s::timestamptz)
                """,
                (identity_id, status, last_seen_at, expires_at),
            )
    except Exception as exc:
        if _missing(exc):
            raise RuntimeError("DB chưa migrate 006 (thiếu bảng identity).") from exc
        if _unique_violation(exc):
            raise ConflictError("Identity đã tồn tại.") from exc
        raise
    return identity_id


def bump_revision(conn, identity_id: str, *, status: Optional[str] = None,
                  last_seen_at: Optional[str] = None) -> int:
    with conn.cursor() as cur:
        if status is not None:
            cur.execute(
                """
                UPDATE face_media.global_identities
                SET revision = revision + 1, status = %s,
                    last_seen_at = COALESCE(%s::timestamptz, last_seen_at)
                WHERE id = %s RETURNING revision
                """,
                (status, last_seen_at, identity_id),
            )
        else:
            cur.execute(
                """
                UPDATE face_media.global_identities
                SET revision = revision + 1,
                    last_seen_at = COALESCE(%s::timestamptz, last_seen_at)
                WHERE id = %s RETURNING revision
                """,
                (last_seen_at, identity_id),
            )
        row = cur.fetchone()
    if row is None:
        raise NotFoundError(f"Identity không tồn tại: {identity_id}")
    return int(row[0])


def get_identity(conn, identity_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, status, revision, last_seen_at, expires_at, created_at
            FROM face_media.global_identities WHERE id = %s
            """,
            (identity_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise NotFoundError(f"Identity không tồn tại: {identity_id}")
    out = dict(zip(("identity_id", "status", "revision", "last_seen_at",
                    "expires_at", "created_at"), row))
    for k in ("last_seen_at", "expires_at", "created_at"):
        out[k] = str(out[k]) if out[k] else None
    return out


def list_identities(conn, *, status: Optional[str] = None,
                    limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    with conn.cursor() as cur:
        if status is None:
            cur.execute("SELECT count(*) FROM face_media.global_identities")
            total = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT id, status, revision, last_seen_at, expires_at, created_at
                FROM face_media.global_identities
                ORDER BY created_at DESC LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
        else:
            cur.execute(
                "SELECT count(*) FROM face_media.global_identities WHERE status = %s",
                (status,),
            )
            total = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT id, status, revision, last_seen_at, expires_at, created_at
                FROM face_media.global_identities WHERE status = %s
                ORDER BY created_at DESC LIMIT %s OFFSET %s
                """,
                (status, limit, offset),
            )
        items = []
        for r in cur.fetchall():
            items.append({
                "identity_id": r[0], "status": r[1], "revision": int(r[2]),
                "last_seen_at": str(r[3]) if r[3] else None,
                "expires_at": str(r[4]) if r[4] else None,
                "created_at": str(r[5]),
            })
    return items, total


def add_assignment(conn, *, tracklet_id: str, global_id: str,
                   score: float, margin: Optional[float] = None,
                   evidence: Optional[Any] = None, reason: Optional[str] = None,
                   model_name: str, model_version: str,
                   preprocessing_version: str,
                   topology_version: Optional[int] = None,
                   assignment_id: Optional[str] = None) -> str:
    """Gán hiện hành cho 1 tracklet. Tracklet đã có assignment hiện hành →
    ConflictError (caller phải supersede thay vì ghi đè)."""
    assignment_id = assignment_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.identity_assignments
                    (id, tracklet_id, global_id, score, margin, evidence, reason,
                     model_name, model_version, preprocessing_version,
                     topology_version)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
                """,
                (assignment_id, tracklet_id, global_id, float(score), margin,
                 json.dumps(evidence or {}), reason, model_name, model_version,
                 preprocessing_version, topology_version),
            )
    except Exception as exc:
        if _missing(exc):
            raise RuntimeError("DB chưa migrate 006 (thiếu bảng identity).") from exc
        if _unique_violation(exc):
            raise ConflictError(
                f"Tracklet {tracklet_id} đã có assignment hiện hành.") from exc
        if _fk_violation(exc):
            raise ValueError("tracklet_id/global_id tham chiếu không tồn tại.") from exc
        raise
    return assignment_id


def supersede_assignment(conn, assignment_id: str, *,
                         new_global_id: Optional[str] = None,
                         reason: Optional[str] = None,
                         score: Optional[float] = None,
                         evidence: Optional[Any] = None) -> str:
    """Sửa gán nhầm: đóng assignment cũ (valid_to=now), mở assignment mới giữ
    cùng tracklet. Trả assignment mới."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tracklet_id, global_id, model_name, model_version,
                   preprocessing_version, topology_version
            FROM face_media.identity_assignments
            WHERE id = %s AND valid_to IS NULL
            """,
            (assignment_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise NotFoundError(f"Assignment hiện hành không tồn tại: {assignment_id}")
        tracklet_id, old_global, mn, mv, pv, tv = row
        target = new_global_id or old_global
        new_id_ = new_id()
        cur.execute(
            """
            UPDATE face_media.identity_assignments
            SET valid_to = now(), superseded_by = %s WHERE id = %s
            """,
            (new_id_, assignment_id),
        )
        cur.execute(
            """
            INSERT INTO face_media.identity_assignments
                (id, tracklet_id, global_id, score, evidence, reason,
                 model_name, model_version, preprocessing_version,
                 topology_version, revision, superseded_by)
            SELECT %s, tracklet_id, %s, %s, %s::jsonb, %s,
                   model_name, model_version, preprocessing_version,
                   topology_version, revision + 1, NULL
            FROM face_media.identity_assignments WHERE id = %s
            """,
            (new_id_, target, float(score) if score is not None else 0.0,
             json.dumps(evidence or {"fix": reason or "manual"}),
             reason, assignment_id),
        )
        _ = (mn, mv, pv, tv)
    return new_id_


def set_exemplar(conn, *, global_id: str, embedding_space: str, slot: int,
                 crop_id: Optional[str] = None,
                 embedding_id: Optional[str] = None,
                 weight: float = 1.0) -> None:
    if slot not in (0, 1, 2):
        raise ValueError("slot exemplar phải thuộc {0,1,2}.")
    # Doc §3: ảnh sinh (FADING) là query có provenance — không đưa vào quan sát
    # camera, không tự cập nhật mẫu ID. Exemplar chỉ nhận embedding của crop
    # quan sát purpose=search; embedding gắn generated_image_id bị từ chối.
    if embedding_id is not None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.crop_id, e.generated_image_id, c.purpose
                FROM face_media.face_embeddings e
                LEFT JOIN face_media.face_crops c ON c.id = e.crop_id
                WHERE e.id = %s
                """,
                (embedding_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"embedding_id không tồn tại: {embedding_id}")
            emb_crop, emb_gen, purpose = row
            if emb_gen is not None:
                raise ValueError(
                    "Từ chối exemplar từ ảnh tạo sinh (FADING là query, "
                    "không phải quan sát camera).")
            if purpose is not None and purpose != "search":
                raise ValueError(
                    f"Exemplar chỉ nhận crop quan sát purpose=search (nhận {purpose!r}).")
            if crop_id is not None and emb_crop is not None \
                    and str(emb_crop) != str(crop_id):
                raise ValueError("crop_id không khớp embedding_id.")
            if crop_id is None:
                crop_id = str(emb_crop) if emb_crop else None
    elif crop_id is not None:
        with conn.cursor() as cur:
            cur.execute("SELECT purpose FROM face_media.face_crops WHERE id = %s",
                        (crop_id,))
            row = cur.fetchone()
            if row is not None and row[0] != "search":
                raise ValueError(
                    f"Exemplar chỉ nhận crop quan sát purpose=search (nhận {row[0]!r}).")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.identity_exemplars
                    (global_id, embedding_space, slot, crop_id, embedding_id, weight)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (global_id, embedding_space, slot) DO UPDATE SET
                    crop_id = EXCLUDED.crop_id,
                    embedding_id = EXCLUDED.embedding_id,
                    weight = EXCLUDED.weight,
                    created_at = now()
                """,
                (global_id, embedding_space, slot, crop_id, embedding_id, weight),
            )
    except Exception as exc:
        if _missing(exc):
            raise RuntimeError("DB chưa migrate 006 (thiếu bảng identity).") from exc
        if _fk_violation(exc):
            raise ValueError("global_id/crop_id/embedding_id tham chiếu không tồn tại.") from exc
        raise


def get_exemplars(conn, global_id: str,
                  embedding_space: Optional[str] = None) -> list[dict]:
    with conn.cursor() as cur:
        if embedding_space is None:
            cur.execute(
                """
                SELECT global_id, embedding_space, slot, crop_id, embedding_id,
                       weight, created_at
                FROM face_media.identity_exemplars WHERE global_id = %s ORDER BY slot
                """,
                (global_id,),
            )
        else:
            cur.execute(
                """
                SELECT global_id, embedding_space, slot, crop_id, embedding_id,
                       weight, created_at
                FROM face_media.identity_exemplars
                WHERE global_id = %s AND embedding_space = %s ORDER BY slot
                """,
                (global_id, embedding_space),
            )
        return [dict(zip(("global_id", "embedding_space", "slot", "crop_id",
                           "embedding_id", "weight", "created_at"), r))
                for r in cur.fetchall()]


def delete_exemplar(conn, *, global_id: str, embedding_space: str, slot: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM face_media.identity_exemplars
            WHERE global_id = %s AND embedding_space = %s AND slot = %s
            """,
            (global_id, embedding_space, slot),
        )


def clear_exemplars(conn, global_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM face_media.identity_exemplars WHERE global_id = %s",
            (global_id,),
        )
        return int(cur.rowcount or 0)


def crops_of_tracklet(conn, tracklet_id: str) -> list[str]:
    """Crop thuộc 1 tracklet (qua detection.tracklet_id). Thiếu cột → []."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id FROM face_media.face_crops c
                JOIN face_media.face_detections d ON d.id = c.detection_id
                WHERE d.tracklet_id = %s
                """,
                (tracklet_id,),
            )
            return [str(r[0]) for r in cur.fetchall()]
    except Exception:
        return []


def recent_candidates(conn, *, embedding_space: str, since: Optional[str] = None,
                      site: Optional[str] = None, camera_id: Optional[str] = None,
                      limit: int = 50) -> list[dict]:
    """Ứng viên ID gần đây cho association: exemplar + assignment mới nhất.

    Topology/site là gợi ý mềm thu hẹp vùng tìm (doc §5.2): ứng viên cùng
    camera hoặc có liên kết topology với camera hiện tại được xếp trước; khi
    thiếu topology vẫn trả rộng hơn để caller re-rank (không ép gộp).
    """
    cols = ("global_id", "slot", "crop_id", "embedding_id", "score",
            "valid_from", "camera_id", "source_id", "src_camera")
    topo_order = """
                (t.camera_id = %(cam)s OR EXISTS (
                    SELECT 1 FROM face_media.camera_topology tp
                    WHERE tp.site = %(site)s AND (
                        (tp.camera_a = %(cam)s AND tp.camera_b = t.camera_id)
                        OR (tp.camera_b = %(cam)s AND tp.camera_a = t.camera_id)))
                ) DESC,"""
    base_where = """
                WHERE e.embedding_space = %(space)s
                  AND g.status IN ('open', 'unresolved')
                  AND (%(since)s::timestamptz IS NULL
                       OR a.valid_from >= %(since)s::timestamptz)"""
    base_select = """
                SELECT e.global_id, e.slot, e.crop_id, e.embedding_id,
                       a.score, a.valid_from, t.camera_id, t.source_id,
                       s.camera_id AS src_camera
                FROM face_media.identity_exemplars e
                JOIN face_media.global_identities g ON g.id = e.global_id
                LEFT JOIN face_media.identity_assignments a
                  ON a.global_id = e.global_id AND a.valid_to IS NULL
                LEFT JOIN face_media.tracklets t ON t.id = a.tracklet_id
                LEFT JOIN face_media.sources s ON s.id = t.source_id"""
    params = {"space": embedding_space, "since": since, "limit": limit,
              "site": site or "default", "cam": camera_id}
    queries = []
    if camera_id:
        queries.append((base_select + base_where +
                        " ORDER BY " + topo_order +
                        " a.valid_from DESC NULLS LAST LIMIT %(limit)s", params))
    queries.append((base_select + base_where +
                    " ORDER BY a.valid_from DESC NULLS LAST LIMIT %(limit)s",
                    {"space": embedding_space, "since": since, "limit": limit}))
    last_exc = None
    for sql, prm in queries:
        try:
            with conn.cursor() as cur:
                cur.execute(sql, prm)
                rows = cur.fetchall()
            return [dict(zip(cols, r)) for r in rows]
        except Exception as exc:
            last_exc = exc
            text = str(exc).lower()
            if "camera_topology" in text or "tracklets" in text:
                continue  # DB chưa migrate 005/006 → thử query rộng hơn
            if _missing(exc):
                return []
            raise
    if last_exc is not None and _missing(last_exc):
        return []
    raise last_exc if last_exc is not None else RuntimeError("unreachable")


def get_timeline(conn, global_id: str, *, gap_sec: float = 600.0) -> dict:
    """Timeline các tracklet của 1 ID theo thời gian, thể hiện khoảng trống.

    Mỗi segment kèm crop đại diện tốt nhất (crop_id/crop_key/score) để viewer
    hiển thị bằng chứng (doc §3: xem crop, timeline và xác nhận).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.id, a.tracklet_id, a.score, a.valid_from, a.valid_to,
                   t.camera_id, t.started_at, t.ended_at, t.last_seen_at,
                   t.observation_count
            FROM face_media.identity_assignments a
            JOIN face_media.tracklets t ON t.id = a.tracklet_id
            WHERE a.global_id = %s ORDER BY t.started_at
            """,
            (global_id,),
        )
        rows = cur.fetchall()
    segments = []
    for r in rows:
        segments.append({
            "assignment_id": r[0], "tracklet_id": r[1], "score": float(r[2]),
            "valid_from": str(r[3]) if r[3] else None,
            "valid_to": str(r[4]) if r[4] else None,
            "camera_id": r[5],
            "started_at": str(r[6]) if r[6] else None,
            "ended_at": str(r[7]) if r[7] else None,
            "last_seen_at": str(r[8]) if r[8] else None,
            "observation_count": int(r[9] or 0),
            "evidence_crop_id": None, "evidence_crop_key": None,
            "evidence_score": None,
        })
    # Crop đại diện/tracklet: quality det_score cao nhất (best-effort).
    for seg in segments:
        try:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT c.id, a.storage_key, d.confidence
                        FROM face_media.face_crops c
                        JOIN face_media.assets a ON a.id = c.asset_id
                        JOIN face_media.face_detections d ON d.id = c.detection_id
                        LEFT JOIN face_media.deletion_ledger l
                          ON l.crop_id = c.id AND l.done = false
                        WHERE d.tracklet_id = %s AND l.id IS NULL
                        ORDER BY d.confidence DESC LIMIT 1
                        """,
                        (seg["tracklet_id"],),
                    )
                except Exception as exc:
                    if "tracklet_id" in str(exc) or "deletion_ledger" in str(exc):
                        cur.execute(
                            """
                            SELECT c.id, a.storage_key, d.confidence
                            FROM face_media.face_crops c
                            JOIN face_media.assets a ON a.id = c.asset_id
                            JOIN face_media.face_detections d ON d.id = c.detection_id
                            WHERE d.track_id = %s
                            ORDER BY d.confidence DESC LIMIT 1
                            """,
                            (seg["tracklet_id"],),
                        )
                    else:
                        raise
                row = cur.fetchone()
                if row:
                    seg["evidence_crop_id"] = str(row[0])
                    seg["evidence_crop_key"] = str(row[1])
                    seg["evidence_score"] = float(row[2])
        except Exception:
            continue
    # Khoảng trống: suy từ ended_at → started_at kế tiếp (ISO compare best-effort).
    gaps = []
    try:
        from datetime import datetime as _dt
        times = []
        for s in segments:
            a = _dt.fromisoformat(s["started_at"]) if s["started_at"] else None
            b = _dt.fromisoformat(s["ended_at"] or s["last_seen_at"] or s["started_at"]) \
                if (s["ended_at"] or s["last_seen_at"] or s["started_at"]) else None
            times.append((a, b))
        for i in range(1, len(times)):
            prev_end, cur_start = times[i - 1][1], times[i][0]
            if prev_end and cur_start:
                dt = (cur_start - prev_end).total_seconds()
                if dt > gap_sec:
                    gaps.append({"after_segment": i - 1, "gap_sec": round(dt, 1),
                                 "note": "không thấy mặt — không bảo đảm nối ID liên tục"})
    except Exception:
        pass
    return {"global_id": global_id, "segments": segments, "gaps": gaps}


def upsert_topology(conn, *, site: str = "default",
                    camera_a: Optional[str] = None,
                    camera_b: Optional[str] = None,
                    travel_sec_min: Optional[float] = None,
                    travel_sec_typical: Optional[float] = None,
                    travel_sec_max: Optional[float] = None,
                    overlapping: bool = False, version: int = 1,
                    note: Optional[str] = None) -> str:
    topo_id = new_id()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO face_media.camera_topology
                (id, site, camera_a, camera_b, travel_sec_min,
                 travel_sec_typical, travel_sec_max, overlapping, version, note)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (topo_id, site, camera_a, camera_b, travel_sec_min,
             travel_sec_typical, travel_sec_max, overlapping, version, note),
        )
    return topo_id


def get_topology(conn, *, site: str = "default") -> list[dict]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, site, camera_a, camera_b, travel_sec_min,
                       travel_sec_typical, travel_sec_max, overlapping,
                       version, note, created_at
                FROM face_media.camera_topology WHERE site = %s ORDER BY version DESC
                """,
                (site,),
            )
            return [dict(zip(("topology_id", "site", "camera_a", "camera_b",
                               "travel_sec_min", "travel_sec_typical",
                               "travel_sec_max", "overlapping", "version",
                               "note", "created_at"), r)) for r in cur.fetchall()]
    except Exception as exc:
        if _missing(exc):
            return []
        raise
