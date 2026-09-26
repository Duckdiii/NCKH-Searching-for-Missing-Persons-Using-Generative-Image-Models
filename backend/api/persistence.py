"""T04–T06 — Ghi face_media best-effort từ luồng reference.

Nguyên tắc:
- Khi DB + storage sẵn sàng: ghi đầy đủ asset → source → frame → detection →
  crop → generation_job → generated_images, giữ lineage IDs trong RAM state.
- Khi DB không sẵn sàng (thiếu DATABASE_URL, mất kết nối): trả None, endpoint
  giữ nguyên hành vi cũ (file tạm + RAM). Không bao giờ làm hỏng luồng chính.
- Thứ tự: ping DB trước → ghi file → transaction DB; DB lỗi → dọn file mồ côi.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


def db_ping() -> bool:
    """True khi pool kết nối được DB (SELECT 1). Mọi lỗi → False, không raise."""
    try:
        from backend.api.database import get_pool

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as exc:
        logger.warning("face_media DB không sẵn sàng: %s: %s", type(exc).__name__, exc)
        return False


def _clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(bbox[0]), float(width)))
    y1 = max(0.0, min(float(bbox[1]), float(height)))
    x2 = max(0.0, min(float(bbox[2]), float(width)))
    y2 = max(0.0, min(float(bbox[3]), float(height)))
    if x2 <= x1:
        x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(height), y1 + 1.0)
    return (x1, y1, x2, y2)


def persist_reference_upload(
    *,
    image_bytes: bytes,
    filename: str,
    content_type: Optional[str],
    faces: list,
    image_width: int,
    image_height: int,
    detector_name: str = "insightface",
    detector_version: str = "buffalo_l",
) -> Optional[dict]:
    """T04: lưu ảnh gốc reference + detections. Trả None khi DB không sẵn sàng."""
    if not db_ping():
        return None
    try:
        from backend.api import repositories as repo
        from backend.api.storage import (
            PREFIX_REFERENCE_ORIGINALS,
            build_key,
            check_image_decodable,
            get_storage,
            sha256_bytes,
            sniff_mime,
        )

        width, height = check_image_decodable(image_bytes)
        mime = (content_type or "").split(";")[0].strip().lower()
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            mime = sniff_mime(image_bytes, "image/png")
        digest = sha256_bytes(image_bytes)
        _ = (image_width, image_height, filename)  # dims thực lấy từ decode

        source_id = str(uuid.uuid4())
        asset_id = str(uuid.uuid4())
        frame_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        key = build_key(PREFIX_REFERENCE_ORIGINALS, asset_id, mime)
        storage = get_storage()
        storage.put_bytes(image_bytes, key, mime_type=mime)
        try:
            with get_pool_conn() as conn:
                repo.create_asset(
                    conn, storage_key=key, media_type="image", mime_type=mime,
                    sha256=digest, byte_size=len(image_bytes),
                    width=width, height=height, role="original_image",
                    asset_id=asset_id,
                )
                repo.create_source(
                    conn, purpose="reference", kind="image",
                    original_asset_id=asset_id, source_id=source_id,
                )
                repo.create_frame(
                    conn, purpose="reference", source_id=source_id,
                    asset_id=asset_id, frame_index=0, offset_ms=0,
                    still_image=True, frame_id=frame_id,
                )
                detection_ids = []
                for index, face in enumerate(faces):
                    bbox = _clamp_bbox(list(face.bbox), width, height)
                    kps = getattr(face, "kps", None)
                    landmarks = (
                        [[float(x), float(y)] for x, y in kps] if kps is not None else None
                    )
                    detection_ids.append(
                        repo.create_detection(
                            conn, purpose="reference", frame_id=frame_id,
                            detector_name=detector_name,
                            detector_version=detector_version,
                            run_id=run_id, face_index=index, bbox=bbox,
                            confidence=float(face.det_score),
                            frame_width=width, frame_height=height,
                            landmarks=landmarks,
                            quality={"det_score": float(face.det_score)},
                        )
                    )
        except Exception:
            storage.delete_quiet(key)
            raise
        return {
            "source_id": source_id,
            "asset_id": asset_id,
            "frame_id": frame_id,
            "run_id": run_id,
            "detection_ids": detection_ids,
            "storage_key": key,
        }
    except Exception as exc:
        logger.warning("persist_reference_upload bỏ qua: %s: %s", type(exc).__name__, exc)
        return None


def persist_crop_revision(
    *,
    purpose: str,
    detection_id: str,
    cropped_bgr,
    method: str,
    preprocessing: Optional[dict] = None,
    transform_to_source: Optional[dict] = None,
) -> Optional[dict]:
    """T05: lưu revision crop mới (append-only). Trả None khi DB không sẵn sàng."""
    if detection_id is None or not db_ping():
        return None
    try:
        import cv2

        from backend.api import repositories as repo
        from backend.api.storage import (
            PREFIX_REFERENCE_CROPS,
            PREFIX_SEARCH_CROPS,
            build_key,
            get_storage,
            sha256_bytes,
        )

        ok, buf = cv2.imencode(".png", cropped_bgr)
        if not ok:
            raise IOError("cv2.imencode crop thất bại — không tạo asset.")
        data = bytes(buf)
        prefix = PREFIX_REFERENCE_CROPS if purpose == "reference" else PREFIX_SEARCH_CROPS
        crop_id = str(uuid.uuid4())
        asset_id = str(uuid.uuid4())
        key = build_key(prefix, asset_id, "image/png")
        storage = get_storage()
        storage.put_bytes(data, key, mime_type="image/png")
        try:
            height, width = cropped_bgr.shape[:2]
            with get_pool_conn() as conn:
                repo.create_asset(
                    conn, storage_key=key, media_type="image",
                    mime_type="image/png", sha256=sha256_bytes(data),
                    byte_size=len(data), width=int(width), height=int(height),
                    role="crop", asset_id=asset_id,
                )
                repo.create_crop(
                    conn, purpose=purpose, detection_id=detection_id,
                    asset_id=asset_id, method=method,
                    preprocessing=preprocessing or {},
                    transform_to_source=transform_to_source,
                    crop_id=crop_id,
                )
        except Exception:
            storage.delete_quiet(key)
            raise
        return {
            "crop_id": crop_id,
            "asset_id": asset_id,
            "storage_key": key,
            "url": storage.get_access_url(key),
        }
    except Exception as exc:
        logger.warning("persist_crop_revision bỏ qua: %s: %s", type(exc).__name__, exc)
        return None


def create_generation_job_record(
    *,
    job_id: str,
    input_crop_id: str,
    session_id: Optional[str],
    initial_age: Optional[int],
    model_name: str = "stable-diffusion-v1-5",
    model_version: str = "fading-specialized-unet",
    parameters: Optional[dict] = None,
) -> Optional[str]:
    """T06: tạo generation_jobs (pending). Trả job_id hoặc None."""
    try:
        from backend.api import repositories as repo

        with get_pool_conn() as conn:
            repo.create_generation_job(
                conn, input_crop_id=input_crop_id, model_name=model_name,
                model_version=model_version, status="pending",
                session_id=session_id, initial_age=initial_age,
                parameters=parameters or {}, job_id=job_id,
            )
        return job_id
    except Exception as exc:
        logger.warning("create_generation_job_record bỏ qua: %s: %s", type(exc).__name__, exc)
        return None


def update_job_record(
    job_id: Optional[str], status: str, error_message: Optional[str] = None
) -> None:
    """Best-effort: cập nhật trạng thái job trong DB (không raise)."""
    if not job_id:
        return
    try:
        from backend.api import repositories as repo

        with get_pool_conn() as conn:
            repo.set_job_status(conn, job_id, status, error_message)
    except Exception as exc:
        logger.warning("update_job_record bỏ qua: %s: %s", type(exc).__name__, exc)


def persist_generated_outputs(
    *,
    job_id: str,
    edited_images: dict,
    parameters: Optional[dict] = None,
) -> list[dict]:
    """T06: lưu từng ảnh tạo sinh (asset + generated_images).

    Mỗi (target_age, variant_index) một bản ghi — nhiều biến thể cùng tuổi
    không ghi đè nhau. Trả danh sách variant đã lưu (có thể rỗng).
    """
    import uuid
    from backend.api import repositories as repo
    from backend.api.storage import (get_storage, build_key, sniff_mime,
        sha256_bytes, check_image_decodable, PREFIX_REFERENCE_GENERATED)
    storage = get_storage()
    staged = []
    commit_started = False
    try:
        for age_key, path in (edited_images or {}).items():
            age = int(age_key)
            with open(path, "rb") as handle:
                data = handle.read()
            width, height = check_image_decodable(data)
            mime = sniff_mime(data, "image/png")
            asset_id, generated_id = str(uuid.uuid4()), str(uuid.uuid4())
            key = build_key(PREFIX_REFERENCE_GENERATED, asset_id, mime)
            storage.put_bytes(data, key, mime_type=mime)
            item = dict(id=generated_id, asset_id=asset_id, key=key, target_age=age,
                mime=mime, digest=sha256_bytes(data), size=len(data), width=width, height=height)
            staged.append(item)
            item["url"] = storage.get_access_url(key)
        with get_pool_conn() as conn:
            for item in staged:
                repo.create_asset(conn, storage_key=item["key"], media_type="image",
                    mime_type=item["mime"], sha256=item["digest"], byte_size=item["size"],
                    width=item["width"], height=item["height"], role="generated", asset_id=item["asset_id"])
                repo.create_generated_image(conn, job_id=job_id, asset_id=item["asset_id"],
                    target_age=item["target_age"], variant_index=0,
                    parameters=parameters or {}, generated_id=item["id"])
            commit_started = True
        return [dict(id=i["id"], target_age=i["target_age"], variant_index=0,
                     seed=None, image_url=i["url"]) for i in staged]
    except Exception as exc:
        if commit_started:
            # A lost COMMIT response is not proof of rollback. Never delete files
            # potentially referenced by committed rows; leave them for reconciliation.
            logger.error("Generated output commit outcome uncertain; retaining %s staged files", len(staged))
            return []
        for item in staged:
            storage.delete_quiet(item["key"])
        logger.warning("persist_generated_outputs rollback: %s", type(exc).__name__)
        return []


def get_generation_job_detail(job_id: str) -> Optional[dict]:
    """Đọc job + variants từ DB cho API xem lại sau restart. None nếu không có."""
    try:
        from backend.api.storage import get_storage

        with get_pool_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, input_crop_id, status, model_name, model_version,
                           initial_age, parameters, error_message,
                           created_at, finished_at
                    FROM face_media.generation_jobs WHERE id = %s
                    """,
                    (job_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cur.execute(
                    """
                    SELECT gi.id, gi.target_age, gi.variant_index, gi.seed,
                           a.storage_key
                    FROM face_media.generated_images gi
                    JOIN face_media.assets a ON a.id = gi.asset_id
                    WHERE gi.job_id = %s
                    ORDER BY gi.target_age, gi.variant_index
                    """,
                    (job_id,),
                )
                variants = [
                    {
                        "id": r[0],
                        "target_age": r[1],
                        "variant_index": r[2],
                        "seed": r[3],
                        "image_url": get_storage().get_access_url(r[4]),
                    }
                    for r in cur.fetchall()
                ]
        keys = (
            "job_id", "input_crop_id", "status", "model_name", "model_version",
            "initial_age", "parameters", "error_message", "created_at", "finished_at",
        )
        detail = dict(zip(keys, row))
        detail["variants"] = variants
        created, finished = detail.get("created_at"), detail.get("finished_at")
        detail["created_at"] = str(created) if created else None
        detail["finished_at"] = str(finished) if finished else None
        return detail
    except Exception as exc:
        logger.warning("get_generation_job_detail bỏ qua: %s: %s", type(exc).__name__, exc)
        return None


def get_pool_conn():  # type: ignore[no-untyped-def]
    from backend.api.database import get_pool

    return get_pool().connection()


def save_session_record(session, **overrides) -> None:  # type: ignore[no-untyped-def]
    """T12: ghi metadata session + crop đã chọn (best-effort, không raise)."""
    try:
        from backend.api import repositories as repo
        from backend.api.database import get_pool

        data = {
            "source_id": session.source_id,
            "frame_id": session.frame_id,
            "chosen_detection_id": session.chosen_detection_id,
            "current_crop_id": session.current_crop_id,
            "gender_word": session.gender_word,
            "initial_age": session.initial_age,
            "photo_year": session.photo_year,
            "file_name": session.file_name,
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        with get_pool().connection() as conn:
            repo.upsert_session(conn, session_id=session.session_id, **data)
    except Exception as exc:
        logger.warning("save_session_record bỏ qua: %s: %s", type(exc).__name__, exc)


def restore_session(session_id: str):  # type: ignore[no-untyped-def]
    """T12: dựng lại RAM session từ DB + file storage (detect lại ảnh gốc).

    Trả SessionState hoặc None (không có RAM/DB record, thiếu file/model).
    """
    from backend.api.session_store import get_session, save_session

    cached = get_session(session_id)
    if cached is not None:
        return cached
    try:
        import cv2
        import numpy as np

        from backend.api import repositories as repo
        from backend.api.database import get_pool
        from backend.api.dependencies import get_embedder
        from backend.api.session_store import SessionState
        from backend.api.storage import get_storage

        with get_pool().connection() as conn:
            record = repo.get_session_record(conn, session_id)
        if record is None or not record.get("original_key"):
            return None
        with get_storage().open(record["original_key"]) as handle:
            data = handle.read()
        image_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = get_embedder().detect_faces(image_bgr)
        state = SessionState(
            session_id=session_id, image_bgr=image_bgr, image_rgb=image_rgb,
            faces=faces, gender_word=record.get("gender_word") or "man",
            initial_age=record.get("initial_age"),
            photo_year=record.get("photo_year"),
            file_name=record.get("file_name") or "",
            source_id=record.get("source_id"), frame_id=record.get("frame_id"),
            detection_ids=[],
            chosen_detection_id=record.get("chosen_detection_id"),
            current_crop_id=record.get("current_crop_id"))
        index = record.get("chosen_face_index")
        if (isinstance(index, int) and 0 <= index < len(faces)
                and record.get("chosen_detection_id")):
            state.chosen_face = faces[index]
            state.detection_ids = []  # detection IDs gốc không dựng lại được
        # Dựng lại file crop legacy cho UI từ crop bền vững (nếu có).
        if state.current_crop_id:
            with get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT a.storage_key FROM face_media.face_crops c
                        JOIN face_media.assets a ON a.id = c.asset_id
                        WHERE c.id = %s
                        """,
                        (state.current_crop_id,))
                    row = cur.fetchone()
            if row is not None:
                import os as _os

                with get_storage().open(row[0]) as handle:
                    crop_data = handle.read()
                crop_bgr = cv2.imdecode(
                    np.frombuffer(crop_data, np.uint8), cv2.IMREAD_COLOR)
                if crop_bgr is not None:
                    _os.makedirs("outputs/app_uploads", exist_ok=True)
                    legacy = _os.path.join(
                        "outputs", "app_uploads", f"{session_id}_crop.png")
                    if cv2.imwrite(legacy, crop_bgr):
                        state.cropped_path = legacy
        save_session(state)
        return state
    except Exception as exc:
        logger.warning("restore_session bỏ qua (%s): %s: %s",
                       session_id, type(exc).__name__, exc)
        return None
