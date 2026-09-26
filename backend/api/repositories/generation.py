"""T02 — Repository generation_jobs / generated_images / face_embeddings / idempotency.

- generation_jobs chỉ nhận crop purpose=reference (service kiểm tra trước,
  FK ghép (input_crop_id, input_purpose) chặn ở DB).
- generated_images unique theo (job, target_age, variant_index): nhiều biến
  thể cùng tuổi không ghi đè nhau; retry dùng cùng variant_index → Conflict
  và trả về bản đã có thay vì tạo trùng.
- face_embeddings unique theo (crop|generated, model, version, preprocessing):
  retry embed cùng cấu hình không tạo trùng vector (ON CONFLICT DO NOTHING).
- idempotency_keys (migration 002): claim request key trước khi làm việc nặng;
  retry cùng key trả kết quả đã lưu.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from . import validation as v
from .errors import ConflictError, NotFoundError, PurposeError
from .media import _fk_violation, _unique_violation, get_crop_purpose, new_id


# ------------------------------------------------------- generation_jobs ---
_JOB_STATUSES = ("pending", "running", "done", "error")


def create_generation_job(
    conn,
    *,
    input_crop_id: str,
    model_name: str,
    model_version: str,
    status: str = "pending",
    session_id: Optional[str] = None,
    initial_age: Optional[int] = None,
    parameters: Optional[Any] = None,
    job_id: Optional[str] = None,
) -> str:
    v.check_model_meta(model_name, model_version)
    if status not in _JOB_STATUSES:
        raise ValueError(f"status phải thuộc {list(_JOB_STATUSES)}.")
    if initial_age is not None and not (0 <= initial_age <= 120):
        raise ValueError("initial_age phải trong [0, 120].")
    # Chặn crop search làm input tạo sinh ngay ở service (DB chặn lại bằng FK).
    crop_purpose = get_crop_purpose(conn, input_crop_id)
    if crop_purpose != "reference":
        raise PurposeError(
            "generation_jobs chỉ nhận crop reference; "
            f"crop {input_crop_id} thuộc luồng {crop_purpose!r}."
        )
    job_id = job_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.generation_jobs
                    (id, input_crop_id, input_purpose, session_id, status,
                     model_name, model_version, initial_age, parameters)
                VALUES (%s, %s, 'reference', %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    job_id,
                    input_crop_id,
                    session_id,
                    status,
                    model_name,
                    model_version,
                    initial_age,
                    json.dumps(parameters or {}),
                ),
            )
    except Exception as exc:
        if _unique_violation(exc):
            raise ConflictError("Generation job đã tồn tại (trùng id).") from exc
        if _fk_violation(exc):
            raise ValueError("input_crop_id không tồn tại.") from exc
        raise
    return job_id


def set_job_status(
    conn, job_id: str, status: str, error_message: Optional[str] = None
) -> None:
    if status not in _JOB_STATUSES:
        raise ValueError(f"status phải thuộc {list(_JOB_STATUSES)}.")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE face_media.generation_jobs
            SET status = %s, error_message = %s,
                finished_at = CASE WHEN %s IN ('done', 'error')
                                   THEN now() ELSE finished_at END
            WHERE id = %s
            """,
            (status, error_message, status, job_id),
        )
        if cur.rowcount == 0:
            raise NotFoundError(f"Job không tồn tại: {job_id}")


# ------------------------------------------------------ generated_images ---
def create_generated_image(
    conn,
    *,
    job_id: str,
    asset_id: str,
    target_age: int,
    variant_index: int = 0,
    seed: Optional[int] = None,
    parameters: Optional[Any] = None,
    generated_id: Optional[str] = None,
) -> str:
    if not (0 <= target_age <= 120):
        raise ValueError("target_age phải trong [0, 120].")
    if variant_index < 0:
        raise ValueError("variant_index phải >= 0.")
    generated_id = generated_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.generated_images
                    (id, job_id, asset_id, target_age, variant_index,
                     seed, parameters)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    generated_id,
                    job_id,
                    asset_id,
                    target_age,
                    variant_index,
                    seed,
                    json.dumps(parameters or {}),
                ),
            )
    except Exception as exc:
        if _unique_violation(exc):
            raise ConflictError(
                f"Ảnh tạo sinh đã tồn tại (job/target_age/variant trùng): "
                f"{job_id}/{target_age}/{variant_index}."
            ) from exc
        if _fk_violation(exc):
            raise ValueError("job_id/asset_id tham chiếu không tồn tại.") from exc
        raise
    return generated_id


# -------------------------------------------------------- face_embeddings ---
def create_embedding(
    conn,
    *,
    model_name: str,
    model_version: str,
    preprocessing_version: str,
    dimensions: int,
    values: list[float],
    normalized: bool,
    crop_id: Optional[str] = None,
    generated_image_id: Optional[str] = None,
    embedding_id: Optional[str] = None,
) -> str:
    v.check_model_meta(model_name, model_version)
    if not preprocessing_version.strip():
        raise ValueError("Cần preprocessing_version để tránh trộn không gian vector.")
    if (crop_id is None) == (generated_image_id is None):
        raise ValueError("Embedding phải gắn đúng 1 trong crop_id/generated_image_id.")
    v.check_vector(values, dimensions, normalized)
    embedding_id = embedding_id or new_id()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO face_media.face_embeddings
                (id, crop_id, generated_image_id, model_name, model_version,
                 preprocessing_version, dimensions, "values", normalized)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                embedding_id,
                crop_id,
                generated_image_id,
                model_name,
                model_version,
                preprocessing_version,
                dimensions,
                values,
                normalized,
            ),
        )
        if cur.rowcount == 0:
            # Retry cùng (entity, model, version, preprocessing) → trả id đã có.
            if crop_id is not None:
                cur.execute(
                    """
                    SELECT id FROM face_media.face_embeddings
                    WHERE crop_id = %s AND model_name = %s
                      AND model_version = %s AND preprocessing_version = %s
                    """,
                    (crop_id, model_name, model_version, preprocessing_version),
                )
            else:
                cur.execute(
                    """
                    SELECT id FROM face_media.face_embeddings
                    WHERE generated_image_id = %s AND model_name = %s
                      AND model_version = %s AND preprocessing_version = %s
                    """,
                    (
                        generated_image_id,
                        model_name,
                        model_version,
                        preprocessing_version,
                    ),
                )
            row = cur.fetchone()
            return row[0] if row else embedding_id
    return embedding_id


# ------------------------------------------------------------ idempotency ---
def claim_idempotency_key(
    conn, *, key: str, scope: str, result: Optional[Any] = None
) -> Optional[dict]:
    """Giữ key cho thao tác mới. Trả None nếu giữ thành công; trả result cũ
    nếu key đã tồn tại (retry an toàn — caller dùng lại, không làm lại)."""
    if not key.strip():
        raise ValueError("Idempotency key rỗng.")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO face_media.idempotency_keys (key, scope, result)
            VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (key) DO NOTHING
            RETURNING key
            """,
            (key, scope, json.dumps(result or {})),
        )
        if cur.fetchone() is not None:
            return None
        cur.execute(
            "SELECT result FROM face_media.idempotency_keys WHERE key = %s", (key,)
        )
        row = cur.fetchone()
    existing = row[0] if row else {}
    return existing if isinstance(existing, dict) else {"result": existing}


def save_idempotency_result(conn, *, key: str, result: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE face_media.idempotency_keys SET result = %s::jsonb WHERE key = %s",
            (json.dumps(result or {}), key),
        )
