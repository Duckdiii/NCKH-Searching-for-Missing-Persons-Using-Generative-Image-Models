"""T02 — Repository chuỗi reference/search: assets → sources → frames → detections → crops.

- Mọi query tham số hóa (%s), không dựng SQL bằng f-string với dữ liệu đầu vào.
- Hàm nhận ``conn`` do caller mở từ ``DatabasePool.connection()``: nhiều bước
  trong cùng khối with = 1 transaction nghiệp vụ (rollback không để FK dở dang).
- Purpose truyền rõ từng hàm, giữ FK ghép (id, purpose) luôn cùng luồng.
- Crop là append-only: mỗi lần chỉnh crop tạo revision mới (chỉ INSERT,
  không UPDATE asset/bbox của bản cũ).
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from . import validation as v
from .errors import ConflictError, PurposeError

try:
    from psycopg import errors as _pg_errors
except Exception:  # pragma: no cover - cho phép import khi chưa cài psycopg
    _pg_errors = None  # type: ignore[assignment]


def _unique_violation(exc: Exception) -> bool:
    if _pg_errors is not None and isinstance(exc, _pg_errors.UniqueViolation):
        return True
    return "unique" in str(exc).lower() or "duplicate" in str(exc).lower()


def _fk_violation(exc: Exception) -> bool:
    if _pg_errors is not None and isinstance(
        exc, _pg_errors.ForeignKeyViolation
    ):
        return True
    text = str(exc).lower()
    return "foreign key" in text or "violates foreign key" in text


def new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------- assets ---
def create_asset(
    conn,
    *,
    storage_key: str,
    media_type: str,
    mime_type: str,
    sha256: str,
    byte_size: int,
    width: int,
    height: int,
    role: str = "crop",
    asset_id: Optional[str] = None,
) -> str:
    v.check_asset_role(media_type, role)
    v.check_sha256(sha256)
    if byte_size <= 0 or width <= 0 or height <= 0:
        raise ValueError("byte_size/width/height phải > 0.")
    if not storage_key or storage_key.startswith("/") or ".." in storage_key:
        raise ValueError("storage_key phải tương đối, không chứa '..'.")
    asset_id = asset_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.assets
                    (id, storage_key, media_type, mime_type, sha256,
                     byte_size, width, height)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    asset_id,
                    storage_key,
                    media_type,
                    mime_type,
                    sha256,
                    byte_size,
                    width,
                    height,
                ),
            )
    except Exception as exc:
        if _unique_violation(exc):
            raise ConflictError(
                f"Asset đã tồn tại (storage_key trùng): {storage_key}"
            ) from exc
        raise
    return asset_id


# --------------------------------------------------------------- sources ---
def create_source(
    conn,
    *,
    purpose: str,
    kind: str,
    original_asset_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    source_id: Optional[str] = None,
    # P0: chính sách lưu ('full' | 'crop_only'); fallback khi DB chưa migrate 005.
    storage_policy: str = "full",
) -> str:
    """Tạo nguồn. Endpoint phải truyền purpose cố định (không tin frontend)."""
    v.require_purpose(purpose, ("reference", "search"))
    if kind not in ("image", "video", "camera"):
        raise ValueError(f"kind không hợp lệ: {kind!r}.")
    if purpose == "reference" and kind != "image":
        raise PurposeError("Nguồn reference chỉ được là ảnh (kind='image').")
    if storage_policy not in ("full", "crop_only"):
        raise ValueError(f"storage_policy không hợp lệ: {storage_policy!r}.")
    if storage_policy == "crop_only" and not (purpose == "search" and kind == "camera"):
        raise ValueError("storage_policy='crop_only' chỉ cho nguồn search/camera.")
    source_id = source_id or new_id()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO face_media.sources
                        (id, purpose, kind, original_asset_id, camera_id,
                         started_at, ended_at, storage_policy)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        source_id,
                        purpose,
                        kind,
                        original_asset_id,
                        camera_id,
                        started_at,
                        ended_at,
                        storage_policy,
                    ),
                )
            except Exception as exc:
                if "storage_policy" in str(exc):
                    cur.execute(
                        """
                        INSERT INTO face_media.sources
                            (id, purpose, kind, original_asset_id, camera_id,
                             started_at, ended_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            source_id,
                            purpose,
                            kind,
                            original_asset_id,
                            camera_id,
                            started_at,
                            ended_at,
                        ),
                    )
                else:
                    raise
    except Exception as exc:
        if _unique_violation(exc):
            raise ConflictError("Source đã tồn tại (trùng id).") from exc
        if _fk_violation(exc):
            raise ValueError("original_asset_id/camera_id tham chiếu không tồn tại.") from exc
        raise
    return source_id


# ---------------------------------------------------------------- frames ---
def create_frame(
    conn,
    *,
    purpose: str,
    source_id: str,
    asset_id: Optional[str],
    frame_index: int,
    offset_ms: int,
    captured_at: Optional[str] = None,
    still_image: bool = False,
    frame_id: Optional[str] = None,
    # P0 camera crop-only: metadata-only (asset_id=None) + provenance.
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    received_at: Optional[str] = None,
    source_timestamp: Optional[str] = None,
    timestamp_uncertainty_ms: Optional[int] = None,
) -> str:
    v.require_purpose(purpose, ("reference", "search"))
    if still_image:
        v.check_static_frame(frame_index, offset_ms)
    if frame_index < 0 or offset_ms < 0:
        raise ValueError("frame_index/offset_ms phải >= 0.")
    if asset_id is None:
        # Metadata-only chỉ cho luồng search (camera crop-only); reference
        # vẫn giữ hợp đồng frame pixels đầy đủ để lineage tạo sinh hoạt động.
        if purpose != "search":
            raise ValueError("Frame metadata-only (asset_id NULL) chỉ cho luồng search.")
        if not source_width or not source_height:
            raise ValueError("Frame metadata-only cần source_width/source_height.")
    frame_id = frame_id or new_id()
    try:
        with conn.cursor() as cur:
            # Tương thích DB chưa/chạy migration 005: thử cột mới, fallback cũ.
            try:
                cur.execute(
                    """
                    INSERT INTO face_media.frames
                        (id, purpose, source_id, asset_id, frame_index,
                         offset_ms, captured_at, source_width, source_height,
                         received_at, source_timestamp, timestamp_uncertainty_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        frame_id,
                        purpose,
                        source_id,
                        asset_id,
                        frame_index,
                        offset_ms,
                        captured_at,
                        source_width,
                        source_height,
                        received_at,
                        source_timestamp,
                        timestamp_uncertainty_ms,
                    ),
                )
            except Exception as exc:
                if "source_width" in str(exc) or "received_at" in str(exc):
                    cur.execute(
                        """
                        INSERT INTO face_media.frames
                            (id, purpose, source_id, asset_id, frame_index,
                             offset_ms, captured_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            frame_id,
                            purpose,
                            source_id,
                            asset_id,
                            frame_index,
                            offset_ms,
                            captured_at,
                        ),
                    )
                else:
                    raise
    except Exception as exc:
        if _unique_violation(exc):
            raise ConflictError("Frame đã tồn tại (trùng source/frame_index).") from exc
        if _fk_violation(exc):
            raise ValueError("source_id/asset_id hoặc purpose ghép không khớp.") from exc
        raise
    return frame_id


# ------------------------------------------------------------ detections ---
def create_detection(
    conn,
    *,
    purpose: str,
    frame_id: str,
    detector_name: str,
    detector_version: str,
    run_id: str,
    face_index: int,
    bbox: tuple[float, float, float, float],
    confidence: float,
    frame_width: int,
    frame_height: int,
    landmarks: Optional[Any] = None,
    track_id: Optional[str] = None,
    quality: Optional[Any] = None,
    detection_id: Optional[str] = None,
    # P1: FK tracklet (nullable để tương thích hàng cũ / DB chưa migrate 005).
    tracklet_id: Optional[str] = None,
) -> str:
    import json

    v.require_purpose(purpose, ("reference", "search"))
    v.check_model_meta(detector_name, detector_version)
    v.check_bbox(*bbox, frame_width, frame_height)
    if face_index < 0:
        raise ValueError("face_index phải >= 0.")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence phải trong [0, 1].")
    detection_id = detection_id or new_id()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO face_media.face_detections
                        (id, purpose, frame_id, detector_name, detector_version,
                         run_id, face_index, x1, y1, x2, y2, confidence,
                         landmarks, track_id, quality, tracklet_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s::jsonb, %s, %s::jsonb, %s)
                    """,
                    (
                        detection_id,
                        purpose,
                        frame_id,
                        detector_name,
                        detector_version,
                        run_id,
                        face_index,
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                        float(confidence),
                        json.dumps(landmarks) if landmarks is not None else None,
                        track_id,
                        json.dumps(quality if quality is not None else {}),
                        tracklet_id,
                    ),
                )
            except Exception as exc:
                if "tracklet_id" in str(exc):
                    cur.execute(
                        """
                        INSERT INTO face_media.face_detections
                            (id, purpose, frame_id, detector_name, detector_version,
                             run_id, face_index, x1, y1, x2, y2, confidence,
                             landmarks, track_id, quality)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s::jsonb, %s, %s::jsonb)
                        """,
                        (
                            detection_id,
                            purpose,
                            frame_id,
                            detector_name,
                            detector_version,
                            run_id,
                            face_index,
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                            float(confidence),
                            json.dumps(landmarks) if landmarks is not None else None,
                            track_id,
                            json.dumps(quality if quality is not None else {}),
                        ),
                    )
                else:
                    raise
    except Exception as exc:
        if _unique_violation(exc):
            raise ConflictError("Detection đã tồn tại (trùng frame/run/face_index).") from exc
        if _fk_violation(exc):
            raise ValueError("frame_id hoặc purpose ghép không khớp.") from exc
        raise
    return detection_id


# ----------------------------------------------------------------- crops ---
def create_crop(
    conn,
    *,
    purpose: str,
    detection_id: str,
    asset_id: str,
    method: str,
    preprocessing: Optional[Any] = None,
    transform_to_source: Optional[Any] = None,
    crop_id: Optional[str] = None,
) -> str:
    """Append-only: mỗi lần chỉnh crop/align/restore tạo revision mới."""
    import json

    v.require_purpose(purpose, ("reference", "search"))
    if method not in v.CROP_METHODS:
        raise ValueError(f"method phải thuộc {list(v.CROP_METHODS)}.")
    crop_id = crop_id or new_id()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.face_crops
                    (id, purpose, detection_id, asset_id, method,
                     preprocessing, transform_to_source)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                """,
                (
                    crop_id,
                    purpose,
                    detection_id,
                    asset_id,
                    method,
                    json.dumps(preprocessing or {}),
                    json.dumps(transform_to_source)
                    if transform_to_source is not None
                    else None,
                ),
            )
    except Exception as exc:
        if _unique_violation(exc):
            raise ConflictError("Crop đã tồn tại (trùng id hoặc asset).") from exc
        if _fk_violation(exc):
            raise ValueError("detection_id/asset_id hoặc purpose ghép không khớp.") from exc
        raise
    return crop_id


def get_crop_purpose(conn, crop_id: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT purpose FROM face_media.face_crops WHERE id = %s", (crop_id,)
        )
        row = cur.fetchone()
    if row is None:
        from .errors import NotFoundError

        raise NotFoundError(f"Crop không tồn tại: {crop_id}")
    return row[0]
