"""T12 — Sessions bền vững, lịch sử phân trang, reconcile job gián đoạn.

- sessions: metadata + crop đã chọn (không serialize numpy/Face object).
- reconcile_interrupted: job/run đang dở khi process chết → error 'interrupted'
  (không tự chạy lại diffusion khi chưa có chiến lược resume).
"""

from __future__ import annotations

from typing import Any, Optional


def upsert_session(conn, *, session_id: str, **fields: Any) -> None:
    """Ghi/cập nhật metadata session (best-effort từ router)."""
    allowed = {
        "source_id", "frame_id", "chosen_face_index", "chosen_detection_id",
        "current_crop_id", "gender_word", "initial_age", "photo_year",
        "file_name",
    }
    data = {k: v for k, v in fields.items() if k in allowed}
    data["session_id"] = session_id
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM face_media.sessions WHERE id = %s", (session_id,))
        exists = cur.fetchone() is not None
        if not exists:
            cols = ", ".join(["id"] + list(data.keys() - {"session_id"}))
            vals = [session_id] + [data[k] for k in data.keys() - {"session_id"}]
            placeholders = ", ".join(["%s"] * len(vals))
            cur.execute(
                f"INSERT INTO face_media.sessions ({cols}) VALUES ({placeholders})",
                vals)
        elif data:
            updates = [k for k in data if k != "session_id"]
            if updates:
                set_clause = ", ".join(f"{k} = %s" for k in updates)
                cur.execute(
                    f"UPDATE face_media.sessions SET {set_clause}, "
                    "updated_at = now() WHERE id = %s",
                    [data[k] for k in updates] + [session_id])


def get_session_record(conn, session_id: str) -> Optional[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.id, s.source_id, s.frame_id, s.chosen_face_index,
                   s.chosen_detection_id, s.current_crop_id, s.gender_word,
                   s.initial_age, s.photo_year, s.file_name,
                   s.created_at, s.updated_at, a.storage_key
            FROM face_media.sessions s
            LEFT JOIN face_media.sources src ON src.id = s.source_id
            LEFT JOIN face_media.assets a ON a.id = src.original_asset_id
            WHERE s.id = %s
            """,
            (session_id,))
        row = cur.fetchone()
    if row is None:
        return None
    keys = (
        "session_id", "source_id", "frame_id", "chosen_face_index",
        "chosen_detection_id", "current_crop_id", "gender_word",
        "initial_age", "photo_year", "file_name",
        "created_at", "updated_at", "original_key",
    )
    out = dict(zip(keys, row))
    out["created_at"] = str(out["created_at"])
    out["updated_at"] = str(out["updated_at"])
    return out


def list_generation_jobs(
    conn, *, status: Optional[str] = None, limit: int = 50, offset: int = 0,
) -> tuple[list[dict], int]:
    if status is not None and status not in ("pending", "running", "done", "error"):
        raise ValueError(f"status không hợp lệ: {status!r}.")
    with conn.cursor() as cur:
        if status is None:
            cur.execute("SELECT count(*) FROM face_media.generation_jobs")
            total = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT j.id, j.input_crop_id, j.session_id, j.status,
                       j.model_name, j.model_version, j.initial_age,
                       j.error_message, j.created_at, j.finished_at,
                       count(gi.id) AS variants
                FROM face_media.generation_jobs j
                LEFT JOIN face_media.generated_images gi ON gi.job_id = j.id
                GROUP BY j.id ORDER BY j.created_at DESC LIMIT %s OFFSET %s
                """,
                (limit, offset))
        else:
            cur.execute(
                "SELECT count(*) FROM face_media.generation_jobs WHERE status = %s",
                (status,))
            total = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT j.id, j.input_crop_id, j.session_id, j.status,
                       j.model_name, j.model_version, j.initial_age,
                       j.error_message, j.created_at, j.finished_at,
                       count(gi.id) AS variants
                FROM face_media.generation_jobs j
                LEFT JOIN face_media.generated_images gi ON gi.job_id = j.id
                WHERE j.status = %s
                GROUP BY j.id ORDER BY j.created_at DESC LIMIT %s OFFSET %s
                """,
                (status, limit, offset))
        items = []
        for r in cur.fetchall():
            items.append({
                "job_id": r[0], "input_crop_id": r[1], "session_id": r[2],
                "status": r[3], "model_name": r[4], "model_version": r[5],
                "initial_age": r[6], "error_message": r[7],
                "created_at": str(r[8]),
                "finished_at": str(r[9]) if r[9] else None,
                "variant_count": int(r[10])})
    return items, total


def list_search_runs(
    conn, *, scope_source_id: Optional[str] = None,
    limit: int = 50, offset: int = 0,
) -> tuple[list[dict], int]:
    with conn.cursor() as cur:
        filt, params = "", []
        if scope_source_id is not None:
            filt, params = "WHERE scope_source_id = %s", [scope_source_id]
        cur.execute(f"SELECT count(*) FROM face_media.search_runs {filt}", params)
        total = int(cur.fetchone()[0])
        cur.execute(
            f"""
            SELECT r.id, r.query_kind, r.scope_source_id, r.generation_job_id,
                   r.threshold, r.created_at, count(s.id) AS results,
                   count(s.id) FILTER (WHERE s.human_confirmed) AS confirmed
            FROM face_media.search_runs r
            LEFT JOIN face_media.search_results s ON s.run_id = r.id
            {filt}
            GROUP BY r.id ORDER BY r.created_at DESC LIMIT %s OFFSET %s
            """,
            params + [limit, offset])
        items = [{
            "run_id": r[0], "query_kind": r[1], "scope_source_id": r[2],
            "generation_job_id": r[3], "threshold": float(r[4]),
            "created_at": str(r[5]), "result_count": int(r[6]),
            "confirmed_count": int(r[7])} for r in cur.fetchall()]
    return items, total


def reconcile_interrupted(conn) -> dict:
    """Đánh dấu job/run đang dở khi process chết (không tự chạy lại)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE face_media.generation_jobs
            SET status = 'error',
                error_message = 'interrupted: process dừng khi job đang chạy; '
                                'không tự chạy lại (xem T12).',
                finished_at = now()
            WHERE status IN ('pending', 'running')
            """)
        jobs = cur.rowcount
        cur.execute(
            """
            UPDATE face_media.ingestion_runs
            SET status = 'error',
                error_message = 'interrupted: process dừng khi run đang chạy.',
                finished_at = now()
            WHERE status IN ('pending', 'running')
            """)
        runs = cur.rowcount
    return {"generation_jobs": jobs, "ingestion_runs": runs}
