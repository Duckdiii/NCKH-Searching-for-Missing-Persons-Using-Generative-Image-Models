"""T07–T11 — Repository search: cameras, ingestion_runs, search_runs/results,
danh sách sources/crops phân trang.

Mọi query tham số hóa; hàm nhận ``conn`` do caller mở từ pool (1 transaction
nghiệp vụ). Purpose search/reference vẫn do endpoint truyền rõ.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .errors import ConflictError, NotFoundError
from .media import _fk_violation, _unique_violation, new_id


# ---------------------------------------------------------------- cameras ---
def create_camera(
    conn, *, name: str, location: Optional[str] = None,
    connection_secret_ref: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> str:
    """T09: chỉ giữ connection_secret_ref (tham chiếu secret), không lưu URL
    hay mật khẩu camera vào DB."""
    if not (name or "").strip():
        raise ValueError("Camera cần tên.")
    camera_id = camera_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.cameras (id, name, location, connection_secret_ref)
                VALUES (%s, %s, %s, %s)
                """,
                (camera_id, name.strip(), location, connection_secret_ref),
            )
    except Exception as exc:
        if _unique_violation(exc):
            raise ConflictError("Camera đã tồn tại (trùng id).") from exc
        raise
    return camera_id


def get_camera(conn, camera_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, location, connection_secret_ref, created_at
            FROM face_media.cameras WHERE id = %s
            """,
            (camera_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise NotFoundError(f"Camera không tồn tại: {camera_id}")
    return {
        "camera_id": row[0], "name": row[1], "location": row[2],
        "has_connection_ref": bool(row[3]), "created_at": str(row[4]),
    }


def list_cameras(conn, *, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM face_media.cameras")
        total = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT id, name, location,
                   (connection_secret_ref IS NOT NULL) AS has_ref, created_at
            FROM face_media.cameras ORDER BY created_at DESC LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        items = [
            {"camera_id": r[0], "name": r[1], "location": r[2],
             "has_connection_ref": bool(r[3]), "created_at": str(r[4])}
            for r in cur.fetchall()
        ]
    return items, total


# ----------------------------------------------------------- ingestion_runs ---
_RUN_STATUSES = ("pending", "running", "done", "error", "canceled")


def create_ingestion_run(
    conn, *, source_id: str, fps_target: float = 3.0, max_frames: int = 90,
    run_id: Optional[str] = None,
) -> str:
    run_id = run_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.ingestion_runs
                    (id, source_id, status, fps_target, max_frames)
                VALUES (%s, %s, 'pending', %s, %s)
                """,
                (run_id, source_id, fps_target, max_frames),
            )
    except Exception as exc:
        if _fk_violation(exc):
            raise ValueError("source_id không tồn tại.") from exc
        raise
    return run_id


def update_ingestion_run(conn, run_id: str, **fields: Any) -> None:
    allowed = {
        "status", "frames_sampled", "faces_found", "frames_total",
        "duration_sec", "truncated", "error_message",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if "status" in updates and updates["status"] not in _RUN_STATUSES:
        raise ValueError(f"status ingestion không hợp lệ: {updates['status']!r}.")
    if updates.get("status") in ("done", "error", "canceled"):
        updates["finished_at"] = "now()"
    if not updates:
        return
    set_clause = ", ".join(
        f"{k} = now()" if k == "finished_at" and v == "now()" else f"{k} = %s"
        for k, v in updates.items()
    )
    values = [v for k, v in updates.items() if not (k == "finished_at" and v == "now()")]
    values.append(run_id)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE face_media.ingestion_runs SET {set_clause} WHERE id = %s",
            values,
        )
        if cur.rowcount == 0:
            raise NotFoundError(f"Ingestion run không tồn tại: {run_id}")


def get_ingestion_run(conn, run_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_id, status, fps_target, max_frames,
                   frames_sampled, faces_found, frames_total, duration_sec,
                   truncated, error_message, created_at, finished_at
            FROM face_media.ingestion_runs WHERE id = %s
            """,
            (run_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise NotFoundError(f"Ingestion run không tồn tại: {run_id}")
    keys = (
        "run_id", "source_id", "status", "fps_target", "max_frames",
        "frames_sampled", "faces_found", "frames_total", "duration_sec",
        "truncated", "error_message", "created_at", "finished_at",
    )
    out = dict(zip(keys, row))
    out["created_at"] = str(out["created_at"])
    out["finished_at"] = str(out["finished_at"]) if out["finished_at"] else None
    return out


# -------------------------------------------------------------- search_runs ---
def create_search_run(
    conn, *, query_kind: str, model_name: str, model_version: str,
    threshold: float,
    query_crop_id: Optional[str] = None,
    scope_source_id: Optional[str] = None,
    generation_job_id: Optional[str] = None,
    preprocessing_version: str = "video_preprocess_v1",
    index_version: str = "adhoc",
    parameters: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> str:
    if query_kind not in ("generated_set", "crop"):
        raise ValueError(f"query_kind không hợp lệ: {query_kind!r}.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold phải trong [0, 1].")
    run_id = run_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.search_runs
                    (id, query_kind, query_crop_id, scope_source_id,
                     generation_job_id, model_name, model_version,
                     preprocessing_version, index_version, threshold, parameters)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    run_id, query_kind, query_crop_id, scope_source_id,
                    generation_job_id, model_name, model_version,
                    preprocessing_version, index_version, threshold,
                    json.dumps(parameters or {}),
                ),
            )
    except Exception as exc:
        if _fk_violation(exc):
            raise ValueError("query/scope/job tham chiếu không tồn tại.") from exc
        raise
    return run_id


def add_search_result(
    conn, *, run_id: str, candidate_crop_id: str, score: float, rank: int,
    best_generated_image_id: Optional[str] = None,
    accepted_by_threshold: bool = False,
    result_id: Optional[str] = None,
) -> str:
    if rank < 1:
        raise ValueError("rank phải >= 1.")
    result_id = result_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.search_results
                    (id, run_id, candidate_crop_id, best_generated_image_id,
                     score, rank, accepted_by_threshold)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, candidate_crop_id) DO NOTHING
                RETURNING id
                """,
                (
                    result_id, run_id, candidate_crop_id,
                    best_generated_image_id, float(score), rank,
                    accepted_by_threshold,
                ),
            )
            row = cur.fetchone()
            return row[0] if row else result_id
    except Exception as exc:
        if _fk_violation(exc):
            raise ValueError("run/candidate/generated tham chiếu không tồn tại.") from exc
        raise


def confirm_search_result(
    conn, *, result_id: str, confirmed: bool,
) -> dict:
    """T11: xác nhận của con người — tách biệt similarity (accepted_by_threshold)."""
    with conn.cursor() as cur:
        if confirmed:
            cur.execute(
                """
                UPDATE face_media.search_results
                SET human_confirmed = true, confirmed_at = now()
                WHERE id = %s
                """,
                (result_id,),
            )
        else:
            cur.execute(
                """
                UPDATE face_media.search_results
                SET human_confirmed = false, confirmed_at = NULL
                WHERE id = %s
                """,
                (result_id,),
            )
        if cur.rowcount == 0:
            raise NotFoundError(f"Kết quả không tồn tại: {result_id}")
    return {"result_id": result_id, "human_confirmed": confirmed}


def get_search_run_detail(conn, run_id: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, query_kind, query_crop_id, scope_source_id,
                   generation_job_id, model_name, model_version,
                   preprocessing_version, index_version, threshold,
                   parameters, created_at
            FROM face_media.search_runs WHERE id = %s
            """,
            (run_id,),
        )
        run = cur.fetchone()
        if run is None:
            raise NotFoundError(f"Search run không tồn tại: {run_id}")
        cur.execute(
            """
            SELECT r.id, r.candidate_crop_id, a.storage_key,
                   r.best_generated_image_id, r.score, r.rank,
                   r.accepted_by_threshold, r.human_confirmed,
                   f.frame_index, f.offset_ms, f.captured_at,
                   s.id, s.kind, s.camera_id
            FROM face_media.search_results r
            JOIN face_media.face_crops c ON c.id = r.candidate_crop_id
            JOIN face_media.assets a ON a.id = c.asset_id
            JOIN face_media.face_detections d ON d.id = c.detection_id
            JOIN face_media.frames f ON f.id = d.frame_id
            JOIN face_media.sources s ON s.id = f.source_id
            WHERE r.run_id = %s ORDER BY r.rank
            """,
            (run_id,),
        )
        results = [
            {
                "result_id": x[0], "candidate_crop_id": x[1], "crop_key": x[2],
                "best_generated_image_id": x[3], "score": float(x[4]), "rank": x[5],
                "accepted_by_threshold": bool(x[6]), "human_confirmed": bool(x[7]),
                "frame_index": int(x[8]), "offset_ms": int(x[9]),
                "captured_at": str(x[10]) if x[10] else None,
                "source_id": x[11], "source_kind": x[12], "camera_id": x[13],
            }
            for x in cur.fetchall()
        ]
    keys = (
        "run_id", "query_kind", "query_crop_id", "scope_source_id",
        "generation_job_id", "model_name", "model_version",
        "preprocessing_version", "index_version", "threshold",
        "parameters", "created_at",
    )
    out = dict(zip(keys, run))
    out["created_at"] = str(out["created_at"])
    out["results"] = results
    return out


# ---------------------------------------------------- sources/crops paging ---
def _source_storage_policy(conn) -> bool:
    """True khi cột sources.storage_policy tồn tại (đã migrate 005)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'face_media' AND table_name = 'sources'
                  AND column_name = 'storage_policy'
                """
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def list_sources(
    conn, *, purpose: Optional[str] = None, limit: int = 50, offset: int = 0,
) -> tuple[list[dict], int]:
    if purpose is not None and purpose not in ("reference", "search"):
        raise ValueError(f"purpose không hợp lệ: {purpose!r}.")
    has_policy = _source_storage_policy(conn)
    policy_col = ", storage_policy" if has_policy else ""
    with conn.cursor() as cur:
        if purpose is None:
            cur.execute("SELECT count(*) FROM face_media.sources")
            total = int(cur.fetchone()[0])
            cur.execute(
                f"""
                SELECT id, purpose, kind, original_asset_id, camera_id,
                       started_at, ended_at, created_at{policy_col}
                FROM face_media.sources ORDER BY created_at DESC LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
        else:
            cur.execute(
                "SELECT count(*) FROM face_media.sources WHERE purpose = %s", (purpose,)
            )
            total = int(cur.fetchone()[0])
            cur.execute(
                f"""
                SELECT id, purpose, kind, original_asset_id, camera_id,
                       started_at, ended_at, created_at{policy_col}
                FROM face_media.sources WHERE purpose = %s
                ORDER BY created_at DESC LIMIT %s OFFSET %s
                """,
                (purpose, limit, offset),
            )
        items = []
        for r in cur.fetchall():
            item = {
                "source_id": r[0], "purpose": r[1], "kind": r[2],
                "original_asset_id": r[3], "camera_id": r[4],
                "started_at": str(r[5]) if r[5] else None,
                "ended_at": str(r[6]) if r[6] else None,
                "created_at": str(r[7]),
            }
            if has_policy:
                item["storage_policy"] = r[8]
            else:
                # Fallback: camera kind coi như crop_only sau P0, còn lại full.
                item["storage_policy"] = "crop_only" if r[2] == "camera" else "full"
            items.append(item)
    return items, total


def list_source_crops(
    conn, *, source_id: str, limit: int = 50, offset: int = 0,
) -> tuple[list[dict], int]:
    """Crop quan sát thuộc nguồn (join trực tiếp, không phụ thuộc view).

    P0: dùng LEFT JOIN asset frame (metadata-only có asset_id NULL) để không
    làm mất kết quả crop-only; trả frame_available để viewer không hứa bối cảnh.
    P4: ẩn crop đã tombstone (ledger pending) — thiếu bảng ledger thì bỏ qua.
    """
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT count(*) FROM face_media.face_crops c
                JOIN face_media.face_detections d ON d.id = c.detection_id
                JOIN face_media.frames f ON f.id = d.frame_id
                LEFT JOIN face_media.deletion_ledger l
                  ON l.crop_id = c.id AND l.done = false
                WHERE f.source_id = %s AND l.id IS NULL
                """,
                (source_id,),
            )
            total = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT c.id, a.storage_key, c.method,
                       d.x1, d.y1, d.x2, d.y2, d.confidence,
                       f.frame_index, f.offset_ms, f.captured_at, f.id,
                       fa.storage_key
                FROM face_media.face_crops c
                JOIN face_media.assets a ON a.id = c.asset_id
                JOIN face_media.face_detections d ON d.id = c.detection_id
                JOIN face_media.frames f ON f.id = d.frame_id
                LEFT JOIN face_media.assets fa ON fa.id = f.asset_id
                LEFT JOIN face_media.deletion_ledger l
                  ON l.crop_id = c.id AND l.done = false
                WHERE f.source_id = %s AND l.id IS NULL
                ORDER BY f.frame_index, c.created_at LIMIT %s OFFSET %s
                """,
                (source_id, limit, offset),
            )
        except Exception as exc:
            if "deletion_ledger" not in str(exc):
                raise
            cur.execute(
                """
                SELECT count(*) FROM face_media.face_crops c
                JOIN face_media.face_detections d ON d.id = c.detection_id
                JOIN face_media.frames f ON f.id = d.frame_id
                WHERE f.source_id = %s
                """,
                (source_id,),
            )
            total = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT c.id, a.storage_key, c.method,
                       d.x1, d.y1, d.x2, d.y2, d.confidence,
                       f.frame_index, f.offset_ms, f.captured_at, f.id,
                       fa.storage_key
                FROM face_media.face_crops c
                JOIN face_media.assets a ON a.id = c.asset_id
                JOIN face_media.face_detections d ON d.id = c.detection_id
                JOIN face_media.frames f ON f.id = d.frame_id
                LEFT JOIN face_media.assets fa ON fa.id = f.asset_id
                WHERE f.source_id = %s
                ORDER BY f.frame_index, c.created_at LIMIT %s OFFSET %s
                """,
                (source_id, limit, offset),
            )
        items = [
            {
                "crop_id": r[0], "crop_key": r[1], "method": r[2],
                "bbox": [float(r[3]), float(r[4]), float(r[5]), float(r[6])],
                "det_score": float(r[7]), "frame_index": int(r[8]),
                "offset_ms": int(r[9]),
                "captured_at": str(r[10]) if r[10] else None,
                "frame_id": r[11],
                # P0 viewer: frame file chỉ tồn tại ở luồng full.
                "frame_available": r[12] is not None,
                "frame_key": r[12],
            }
            for r in cur.fetchall()
        ]
    return items, total
