"""T07/T08 — Nạp gallery search độc lập (không cần job tạo sinh).

- Ảnh: POST nhiều file, kết quả từng ảnh (done / no_face / error).
- Video: tạo nguồn search/video độc lập + ingestion run nền (tiến độ, hủy,
  retry). Verify sau nhận source_id, không xử lý lại video.
- Xem nguồn (phân trang, lọc purpose) và crop thuộc nguồn.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.api import repositories as repo
from backend.api.database import get_pool
from backend.api.ingest import (
    MAX_FRAMES,
    FPS_TARGET,
    get_ingest_embedder,
    ingest_image_bytes,
    ingest_sampled_frames,
    persist_video_source,
    sample_video_frames,
)
from backend.api.persistence import db_ping
from backend.api.schemas import (
    CropListResponse,
    ImageIngestItem,
    ImageIngestResponse,
    IngestedCrop,
    IngestionRunStatus,
    SourceCropItem,
    SearchSourceItem,
    SourceListResponse,
    VideoIngestResponse,
)
from backend.api.storage import get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search-sources"])

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024

_WORKERS: dict[str, dict] = {}
_WORKERS_LOCK = threading.Lock()


def _require_db() -> None:
    if not db_ping():
        raise HTTPException(
            status_code=503,
            detail="Database face_media không sẵn sàng — không thể nạp gallery search.",
        )


@router.post("/search-sources/images", response_model=ImageIngestResponse)
async def ingest_images(files: list[UploadFile] = File(...)):
    """T07: nạp 1..N ảnh search, mỗi ảnh một nguồn search/image riêng."""
    _require_db()
    if not files:
        raise HTTPException(status_code=400, detail="Không có file nào.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Tối đa 20 ảnh mỗi lần nạp.")
    from backend.api.dependencies import get_config

    preprocess_on = bool(get_config().get("video_preprocess", {}).get("enabled", True))
    embedder = get_ingest_embedder()
    items: list[ImageIngestItem] = []
    for upload in files:
        data = await upload.read()
        if not data:
            items.append(ImageIngestItem(
                filename=upload.filename or "?", status="error",
                error="File không có dữ liệu."))
            continue
        if len(data) > MAX_IMAGE_BYTES:
            items.append(ImageIngestItem(
                filename=upload.filename or "?", status="error",
                error=f"Ảnh vượt quá {MAX_IMAGE_BYTES // (1024 * 1024)}MB."))
            continue
        result = ingest_image_bytes(
            data=data, filename=upload.filename or "upload",
            content_type=upload.content_type, embedder=embedder,
            preprocess_on=preprocess_on,
        )
        if result is None:
            items.append(ImageIngestItem(
                filename=upload.filename or "?", status="error",
                error="Không ghi được vào face_media (DB/storage lỗi)."))
        elif result["faces_found"] == 0:
            items.append(ImageIngestItem(
                filename=upload.filename or "?", status="no_face",
                source_id=result["source_id"], faces_found=0,
                conditions=result.get("conditions", {}),
                error="Không phát hiện khuôn mặt nào trong ảnh."))
        else:
            items.append(ImageIngestItem(
                filename=upload.filename or "?", status="done",
                source_id=result["source_id"],
                faces_found=result["faces_found"],
                crops=[IngestedCrop(**c) for c in result["crops"]],
                conditions=result.get("conditions", {})))
    return ImageIngestResponse(items=items)


def _video_worker(
    *, run_id: str, source_id: str, video_path: str,
    fps_target: float, max_frames: int, preprocess_on: bool,
    cancel: threading.Event,
) -> None:
    from backend.api.dependencies import get_config  # noqa: F401 (giữ context)

    try:
        with get_pool().connection() as conn:
            repo.update_ingestion_run(conn, run_id, status="running")
        embedder = get_ingest_embedder()
        frames, meta = sample_video_frames(
            video_path, fps_target=fps_target, max_frames=max_frames,
            cancel=cancel,
        )
        if cancel.is_set():
            with get_pool().connection() as conn:
                repo.update_ingestion_run(
                    conn, run_id, status="canceled",
                    frames_sampled=len(frames),
                    frames_total=meta["frames_total"],
                    duration_sec=meta["duration_sec"],
                    truncated=meta["truncated"])
            return
        # Nạp theo batch để tiến độ cập nhật dần và cancel phản hồi nhanh.
        faces_total = 0
        processed = 0
        batch = 15
        for start in range(0, len(frames), batch):
            if cancel.is_set():
                break
            part = ingest_sampled_frames(
                source_id=source_id, frames=frames[start:start + batch],
                embedder=embedder, preprocess_on=preprocess_on,
                run_id=run_id, cancel=cancel,
            )
            faces_total += part["faces_found"]
            processed += len(frames[start:start + batch])
            with get_pool().connection() as conn:
                repo.update_ingestion_run(
                    conn, run_id, frames_sampled=processed,
                    faces_found=faces_total)
        final = "canceled" if cancel.is_set() else "done"
        with get_pool().connection() as conn:
            repo.update_ingestion_run(
                conn, run_id, status=final,
                frames_sampled=processed, faces_found=faces_total,
                frames_total=meta["frames_total"],
                duration_sec=meta["duration_sec"], truncated=meta["truncated"])
    except Exception as exc:
        logger.exception("ingest video run %s lỗi", run_id)
        try:
            with get_pool().connection() as conn:
                repo.update_ingestion_run(
                    conn, run_id, status="error",
                    error_message=f"{type(exc).__name__}: {exc}"[:2000])
        except Exception:
            pass
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass
        with _WORKERS_LOCK:
            _WORKERS.pop(run_id, None)


@router.post("/search-sources/videos", response_model=VideoIngestResponse,
             status_code=202)
async def ingest_video(
    file: UploadFile = File(...),
    fps_target: float = Query(FPS_TARGET, gt=0, le=10),
    max_frames: int = Query(MAX_FRAMES, gt=0, le=2000),
):
    """T08: nạp video thành nguồn search độc lập + run nền (202)."""
    _require_db()
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng video không hỗ trợ. Hỗ trợ: {', '.join(VIDEO_EXTENSIONS)}.",
        )
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="File video không có dữ liệu.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=400, detail="Video vượt quá 200MB.")
    mime = (file.content_type or "").split(";")[0].strip().lower() or "video/mp4"
    saved = persist_video_source(video_bytes=video_bytes, ext=ext, mime_type=mime)
    if saved is None:
        raise HTTPException(status_code=500, detail="Không lưu được video vào face_media.")
    from backend.api.dependencies import get_config

    preprocess_on = bool(get_config().get("video_preprocess", {}).get("enabled", True))
    with get_pool().connection() as conn:
        run_id = repo.create_ingestion_run(
            conn, source_id=saved["source_id"],
            fps_target=fps_target, max_frames=max_frames)
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(video_bytes)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    cancel = threading.Event()
    thread = threading.Thread(
        target=_video_worker,
        kwargs={"run_id": run_id, "source_id": saved["source_id"],
                "video_path": tmp_path, "fps_target": fps_target,
                "max_frames": max_frames, "preprocess_on": preprocess_on,
                "cancel": cancel},
        daemon=True,
    )
    with _WORKERS_LOCK:
        _WORKERS[run_id] = {"cancel": cancel, "thread": thread}
    thread.start()
    return VideoIngestResponse(
        source_id=saved["source_id"], run_id=run_id, status="running")


@router.get("/search-sources", response_model=SourceListResponse)
def list_search_sources(
    purpose: Optional[str] = Query(None, description="reference | search"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """T07/T12: xem nguồn đã nạp (phân trang, lọc purpose)."""
    _require_db()
    try:
        with get_pool().connection() as conn:
            items, total = repo.list_sources(
                conn, purpose=purpose, limit=limit, offset=offset)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return SourceListResponse(
        items=[SearchSourceItem(**item) for item in items],
        total=total, limit=limit, offset=offset)


@router.get("/search-sources/{source_id}/crops", response_model=CropListResponse)
def list_source_crops_view(
    source_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """T07: crop quan sát thuộc nguồn (phân trang)."""
    _require_db()
    storage = get_storage()
    with get_pool().connection() as conn:
        items, total = repo.list_source_crops(
            conn, source_id=source_id, limit=limit, offset=offset)
    out = []
    for item in items:
        out.append(SourceCropItem(
            crop_id=item["crop_id"], crop_key=item["crop_key"],
            crop_url=storage.get_access_url(item["crop_key"]),
            method=item["method"], bbox=item["bbox"],
            det_score=item["det_score"], frame_index=item["frame_index"],
            offset_ms=item["offset_ms"], captured_at=item["captured_at"],
            frame_id=item["frame_id"]))
    return CropListResponse(items=out, total=total, limit=limit, offset=offset)


@router.get("/ingestion-runs/{run_id}", response_model=IngestionRunStatus)
def get_run_status(run_id: str):
    """T08: tiến độ nạp video (frames_sampled/faces_found/truncated)."""
    _require_db()
    try:
        with get_pool().connection() as conn:
            return IngestionRunStatus(**repo.get_ingestion_run(conn, run_id))
    except repo.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/ingestion-runs/{run_id}/cancel", response_model=IngestionRunStatus)
def cancel_run(run_id: str):
    """T08: hủy run đang chạy (worker dừng giữa batch, ghi canceled)."""
    _require_db()
    with _WORKERS_LOCK:
        worker = _WORKERS.get(run_id)
        if worker is not None:
            worker["cancel"].set()
            return get_run_status(run_id)
    try:
        with get_pool().connection() as conn:
            current = repo.get_ingestion_run(conn, run_id)
            if current["status"] in ("pending", "running"):
                repo.update_ingestion_run(conn, run_id, status="canceled")
            return IngestionRunStatus(**repo.get_ingestion_run(conn, run_id))
    except repo.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/ingestion-runs/{run_id}/retry", response_model=VideoIngestResponse,
             status_code=202)
def retry_run(run_id: str):
    """T08: chạy lại run lỗi/hủy trên cùng source (xóa frame cũ, run mới)."""
    _require_db()
    with get_pool().connection() as conn:
        try:
            current = repo.get_ingestion_run(conn, run_id)
        except repo.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if current["status"] not in ("error", "canceled", "done"):
            raise HTTPException(
                status_code=409, detail="Chỉ retry run đã done/error/canceled.")
        source_id = current["source_id"]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT storage_key FROM face_media.assets WHERE id = ("
                "SELECT original_asset_id FROM face_media.sources WHERE id = %s)",
                (source_id,))
            row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=400, detail="Nguồn thiếu video gốc.")
        new_run_id = repo.create_ingestion_run(
            conn, source_id=source_id, fps_target=current["fps_target"],
            max_frames=current["max_frames"])
    storage = get_storage()
    try:
        with storage.open(row[0]) as handle:
            video_bytes = handle.read()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="File video gốc đã mất.")
    from backend.api.dependencies import get_config

    preprocess_on = bool(get_config().get("video_preprocess", {}).get("enabled", True))
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as handle:
        handle.write(video_bytes)
    # Xóa frame/detection/crop cũ của nguồn để retry không nhân dữ liệu.
    _clear_source_frames(source_id)
    cancel = threading.Event()
    thread = threading.Thread(
        target=_video_worker,
        kwargs={"run_id": new_run_id, "source_id": source_id,
                "video_path": tmp_path,
                "fps_target": current["fps_target"],
                "max_frames": current["max_frames"],
                "preprocess_on": preprocess_on, "cancel": cancel},
        daemon=True,
    )
    with _WORKERS_LOCK:
        _WORKERS[new_run_id] = {"cancel": cancel, "thread": thread}
    thread.start()
    return VideoIngestResponse(
        source_id=source_id, run_id=new_run_id, status="running")


def _clear_source_frames(source_id: str) -> dict:
    """Xóa toàn bộ frame/detection/crop (+ asset file) của nguồn, giữ nguyên
    video gốc và source. Dùng cho retry ingest."""
    storage = get_storage()
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, fa.storage_key, ff.id, fa2.storage_key,
                       e.id
                FROM face_media.frames ff
                JOIN face_media.assets fa2 ON fa2.id = ff.asset_id
                LEFT JOIN face_media.face_detections d ON d.frame_id = ff.id
                LEFT JOIN face_media.face_crops c ON c.detection_id = d.id
                LEFT JOIN face_media.assets fa ON fa.id = c.asset_id
                LEFT JOIN face_media.face_embeddings e ON e.crop_id = c.id
                WHERE ff.source_id = %s
                """,
                (source_id,))
            rows = cur.fetchall()
            crop_ids = [r[0] for r in rows if r[0]]
            frame_ids = list({r[2] for r in rows})
            file_keys = [k for r in rows for k in (r[1], r[3]) if k]
            if crop_ids:
                cur.execute(
                    "DELETE FROM face_media.face_embeddings WHERE crop_id = ANY(%s)",
                    (crop_ids,))
                cur.execute(
                    "DELETE FROM face_media.face_crops WHERE id = ANY(%s)", (crop_ids,))
                cur.execute(
                    "DELETE FROM face_media.assets WHERE storage_key = ANY(%s)",
                    ([r[1] for r in rows if r[1]],))
            cur.execute(
                "DELETE FROM face_media.face_detections WHERE frame_id = ANY(%s)",
                (frame_ids,))
            cur.execute(
                "DELETE FROM face_media.frames WHERE id = ANY(%s)", (frame_ids,))
            cur.execute(
                "DELETE FROM face_media.assets WHERE storage_key = ANY(%s)",
                ([r[3] for r in rows if r[3]],))
    for key in file_keys:
        storage.delete_quiet(key)
    return {"frames": len(frame_ids), "crops": len(crop_ids)}
