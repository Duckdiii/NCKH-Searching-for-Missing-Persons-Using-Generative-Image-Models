"""Đối soát bổ sung bằng video (sau bước Kết quả FADING).

Hai đường vào (T08):
- file video mới: nạp thành nguồn search/video độc lập (face_media) rồi đối
  chiếu; khi DB không sẵn sàng giữ nguyên luồng legacy (file tạm + RAM).
- source_id đã nạp: đối chiếu lại mà KHÔNG xử lý lại video, không lưu trùng.
Mọi lượt đối chiếu có DB đều lưu search_runs/search_results (T11): query là
tập ảnh tạo sinh, candidate là crop quan sát, best_age liên kết ảnh tạo sinh
cụ thể; accepted theo ngưỡng tách biệt xác nhận con người.
"""

from src.utils.cancellation import checkpoint, TaskCancelled
from backend.api.session_lifecycle import job_operation

import glob
import os
import tempfile
import uuid
from collections import Counter
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.api.dependencies import get_config, get_video_embedder
from backend.api.ingest import (
    DETECT_MAX_DIM,
    FPS_TARGET,
    MAX_FRAMES,
    MIN_DET_SCORE,
    TOP_KEEP,
    _crop_bbox,
    _downscale_for_detect,
    cosine_sim,
    ingest_sampled_frames,
    persist_video_source,
    sample_video_frames,
)
from backend.api.schemas import VideoFaceMatch, VideoVerifyResponse
from backend.api.session_store import get_job
from backend.api.persistence import db_ping
from src.preprocessing import preprocess_video_frame

router = APIRouter(prefix="/api/jobs", tags=["video-verify"])

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
MAX_VIDEO_BYTES = 200 * 1024 * 1024

__all__ = [
    "cosine_sim", "_crop_bbox", "_downscale_for_detect",
    "_resolve_edited_paths",
]


def _resolve_edited_paths(job_id: str, edited_images: dict) -> dict:
    """Map URL /outputs/... trong job result về filesystem path.

    Ưu tiên đường dẫn suy từ URL; nếu file thiếu thì fallback quét
    outputs/jobs/{job_id}/age_*.png (đúng nơi job_runner ghi ảnh edit).
    """
    resolved = {}
    for age_key, url in (edited_images or {}).items():
        try:
            age = int(age_key)
        except (TypeError, ValueError):
            continue
        fs_path = None
        if isinstance(url, str) and "outputs/" in url.replace("\\", "/"):
            norm = url.replace("\\", "/")
            rel = norm[norm.index("outputs/"):]
            cand = os.path.join(*rel.split("/"))
            if os.path.exists(cand):
                fs_path = cand
        if fs_path is None:
            for ext in (".png", ".jpg", ".jpeg"):
                cand = os.path.join("outputs", "jobs", job_id, f"age_{age}{ext}")
                if os.path.exists(cand):
                    fs_path = cand
                    break
        if fs_path is not None:
            resolved[age] = fs_path
    if not resolved:
        job_dir = os.path.join("outputs", "jobs", job_id)
        for fpath in sorted(glob.glob(os.path.join(job_dir, "age_*.*"))):
            base = os.path.splitext(os.path.basename(fpath))[0]  # age_30
            try:
                resolved[int(base.split("_")[1])] = fpath
            except (IndexError, ValueError):
                continue
    return resolved


def _resolve_reference(job_id: str) -> tuple[dict, dict]:
    """Tìm edited_images của job (RAM trước, đĩa sau). Raise HTTPException."""
    job = get_job(job_id)
    raw_urls: dict = {}
    if job and job.status == "done" and job.result and job.result.get("edited_images"):
        raw_urls = dict(job.result["edited_images"])
    if not raw_urls:
        for fpath in sorted(glob.glob(os.path.join("outputs", "jobs", job_id, "age_*.*"))):
            base = os.path.splitext(os.path.basename(fpath))[0]  # age_30
            try:
                age = int(base.split("_")[1])
            except (IndexError, ValueError):
                continue
            raw_urls[age] = f"/outputs/jobs/{job_id}/{os.path.basename(fpath)}"
    if job is not None and job.status != "done" and not raw_urls:
        raise HTTPException(
            status_code=400,
            detail="Job chưa hoàn tất sinh ảnh (edited_images). Hãy chạy pipeline FADING xong trước.",
        )
    if not raw_urls:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy job (kể cả trên đĩa). Hãy chạy lại pipeline FADING rồi thử lại.",
        )
    return job, raw_urls


def _embed_references(job_id: str, raw_urls: dict, embedder) -> tuple[dict, dict]:
    edited_paths = _resolve_edited_paths(job_id, raw_urls)
    if not edited_paths:
        raise HTTPException(status_code=400, detail="Không tìm thấy file ảnh FADING đã sinh của job.")
    ref_embeddings = {}
    ref_urls = {}
    for age, fpath in edited_paths.items():
        try:
            ref_embeddings[age] = embedder.embed(fpath)
            ref_urls[age] = raw_urls.get(str(age), raw_urls.get(age, f"/outputs/jobs/{job_id}/{os.path.basename(fpath)}"))
        except ValueError:
            continue
    if not ref_embeddings:
        raise HTTPException(status_code=400, detail="Không trích được embedding từ ảnh FADING đã sinh.")
    return ref_embeddings, ref_urls


def _match_records(records: list[dict], ref_embeddings: dict, ref_urls: dict) -> list[VideoFaceMatch]:
    """Đối chiếu các crop đã có embedding với tập ref (giữ nguyên thứ tự điểm)."""
    matches: list[VideoFaceMatch] = []
    for rec in records:
        emb = rec.get("embedding")
        if emb is None:
            continue
        face_emb = np.asarray(emb, dtype=np.float64)
        best_age = max(ref_embeddings.keys(), key=lambda a: cosine_sim(face_emb, ref_embeddings[a]))
        best_score = cosine_sim(face_emb, ref_embeddings[best_age])
        matches.append(
            VideoFaceMatch(
                face_image_url=rec["url"],
                frame_index=int(rec["frame_index"]),
                timestamp_sec=round(float(rec["timestamp_sec"]), 2),
                bbox=[float(v) for v in rec["bbox"]],
                det_score=float(rec["det_score"]),
                best_age=int(best_age),
                best_age_image_url=str(ref_urls[best_age]),
                score=float(best_score),
            )
        )
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:TOP_KEEP]


def _save_search_run(
    *, job_id: str, source_id: str, matches: list[VideoFaceMatch],
    crop_ids: list[str], threshold: float, ref_ages: list[int],
    conditions: dict, truncated: bool,
) -> Optional[str]:
    """T11: lưu lượt đối chiếu (best_age = liên kết ảnh tạo sinh cụ thể)."""
    try:
        from backend.api import repositories as repo
        from backend.api.database import get_pool
        from backend.api.dependencies import get_config as _get_config
        from backend.api.persistence import get_generation_job_detail

        cfg = _get_config()
        model_name = str(cfg.get("embedding", {}).get("model_name", "buffalo_l"))
        variant_by_age: dict[int, str] = {}
        db_job_id = None
        detail = get_generation_job_detail(job_id)
        if detail is not None:
            db_job_id = detail["job_id"]
            for variant in detail.get("variants", []):
                variant_by_age.setdefault(int(variant["target_age"]), variant["id"])
        with get_pool().connection() as conn:
            run_id = repo.create_search_run(
                conn, query_kind="generated_set",
                scope_source_id=source_id, generation_job_id=db_job_id,
                model_name=model_name, model_version=model_name,
                preprocessing_version="video_preprocess_v1",
                index_version="adhoc", threshold=float(threshold),
                parameters={"edited_ages": sorted(ref_ages),
                            "conditions": conditions, "truncated": truncated,
                            "top_keep": TOP_KEEP})
            for rank, (match, crop_id) in enumerate(zip(matches, crop_ids), start=1):
                repo.add_search_result(
                    conn, run_id=run_id, candidate_crop_id=crop_id,
                    best_generated_image_id=variant_by_age.get(match.best_age),
                    score=match.score, rank=rank,
                    accepted_by_threshold=bool(match.score >= threshold))
        return run_id
    except Exception:
        return None


def _load_source_records(source_id: str, embedder) -> tuple[list[dict], list[str]]:
    """Nạp crops của nguồn đã ingest + embed từng crop (không xử lý lại video)."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    from backend.api.storage import get_storage

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM face_media.sources WHERE id = %s", (source_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Không tìm thấy nguồn search.")
        items, _ = repo.list_source_crops(conn, source_id=source_id, limit=500, offset=0)
    storage = get_storage()
    records, crop_ids = [], []
    for item in items:
        checkpoint()
        try:
            with storage.open(item["crop_key"]) as handle:
                data = handle.read()
        except FileNotFoundError:
            continue
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            embedding = np.asarray(embedder.embed(tmp), dtype=np.float64)
        except ValueError:
            continue
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        records.append({
            "crop_id": item["crop_id"],
            "url": storage.get_access_url(item["crop_key"]),
            "frame_index": item["frame_index"],
            "timestamp_sec": item["offset_ms"] / 1000.0,
            "bbox": item["bbox"], "det_score": item["det_score"],
            "embedding": embedding})
        crop_ids.append(item["crop_id"])
    return records, crop_ids


@router.post("/{job_id}/video-verify", response_model=VideoVerifyResponse)
@job_operation
def video_verify(
    job_id: str,
    file: Optional[UploadFile] = File(None),
    source_id: Optional[str] = Form(None),
):
    _, raw_urls = _resolve_reference(job_id)
    if (file is None) == (source_id is None):
        raise HTTPException(
            status_code=400,
            detail="Chỉ định đúng một trong: file video mới hoặc source_id đã nạp.",
        )
    if source_id is not None and not db_ping():
        raise HTTPException(
            status_code=503,
            detail="Database face_media không sẵn sàng — không thể đối chiếu source đã nạp.",
        )
    embedder = get_video_embedder()
    ref_embeddings, ref_urls = _embed_references(job_id, raw_urls, embedder)
    threshold = float(get_config().get("search", {}).get("rejection_threshold", 0.6))

    if source_id is not None or db_ping():
        return _verify_persisted(
            job_id, raw_urls, ref_embeddings, ref_urls, embedder, threshold,
            file=file, source_id=source_id)
    assert file is not None
    return _verify_legacy(
        job_id, raw_urls, ref_embeddings, ref_urls, embedder, file)


def _verify_persisted(
    job_id: str, raw_urls: dict, ref_embeddings: dict, ref_urls: dict,
    embedder, threshold: float,
    *, file: Optional[UploadFile], source_id: Optional[str],
) -> VideoVerifyResponse:
    """Đối chiếu có lưu face_media (nguồn mới hoặc nguồn đã nạp)."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool

    conditions: dict = {}
    truncated = False
    frames_total = None
    duration_sec = 0.0
    crop_ids: list[str] = []
    if source_id is not None:
        records, crop_ids = _load_source_records(source_id, embedder)
        faces_found = len(records)
        frames_sampled = len({r["frame_index"] for r in records})
    else:
        assert file is not None
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Định dạng video không hỗ trợ ({ext or 'trống'}). Hỗ trợ: {', '.join(VIDEO_EXTENSIONS)}.",
            )
        video_bytes = file.file.read()
        if not video_bytes:
            raise HTTPException(status_code=400, detail="File video tải lên không có dữ liệu.")
        if len(video_bytes) > MAX_VIDEO_BYTES:
            raise HTTPException(status_code=400, detail="Video vượt quá 200MB.")
        mime = (file.content_type or "").split(";")[0].strip().lower() or "video/mp4"
        saved = persist_video_source(video_bytes=video_bytes, ext=ext, mime_type=mime)
        if saved is None:
            raise HTTPException(status_code=500, detail="Không lưu được video vào face_media.")
        source_id = saved["source_id"]
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(video_bytes)
            preprocess_on = bool(get_config().get("video_preprocess", {}).get("enabled", True))
            frames, meta = sample_video_frames(tmp_path)
            frames_total = meta["frames_total"]
            duration_sec = meta["duration_sec"]
            truncated = meta["truncated"]
            with get_pool().connection() as conn:
                run_id = repo.create_ingestion_run(conn, source_id=source_id)
                repo.update_ingestion_run(conn, run_id, status="running")
            try:
                ingested = ingest_sampled_frames(
                    source_id=source_id, frames=frames, embedder=embedder,
                    preprocess_on=preprocess_on, run_id=run_id)
                ingest_status = "done"
            except TaskCancelled:
                with get_pool().connection() as conn:
                    repo.update_ingestion_run(conn, run_id, status="canceled")
                raise
            except Exception as exc:
                ingest_status = f"error: {type(exc).__name__}: {exc}"
                ingested = {"faces_found": 0, "crops": [], "conditions": {}}
            with get_pool().connection() as conn:
                if ingest_status == "done":
                    repo.update_ingestion_run(
                        conn, run_id, status="done",
                        frames_sampled=len(frames),
                        faces_found=ingested["faces_found"],
                        frames_total=frames_total, duration_sec=duration_sec,
                        truncated=truncated)
                else:
                    repo.update_ingestion_run(
                        conn, run_id, status="error",
                        frames_sampled=len(frames),
                        error_message=ingest_status[:2000])
            conditions = ingested["conditions"]
            records = ingested["crops"]
            crop_ids = [r["crop_id"] for r in records]
            faces_found = ingested["faces_found"]
            frames_sampled = len(frames)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    assert source_id is not None
    checkpoint()
    matches = _match_records(records, ref_embeddings, ref_urls)
    ordered_crop_ids = []
    if crop_ids and matches:
        by_url = {r["url"]: r["crop_id"] for r in records}
        ordered_crop_ids = [by_url.get(m.face_image_url, "") for m in matches]
    search_run_id = _save_search_run(
        job_id=job_id, source_id=source_id, matches=matches,
        crop_ids=ordered_crop_ids, threshold=threshold,
        ref_ages=list(ref_embeddings.keys()), conditions=conditions,
        truncated=truncated)
    return VideoVerifyResponse(
        job_id=job_id,
        frames_sampled=frames_sampled,
        faces_found=faces_found,
        best_match=matches[0] if matches else None,
        matches=matches,
        conditions=conditions,
        source_id=source_id,
        search_run_id=search_run_id,
        processing={"frames_total": frames_total, "duration_sec": duration_sec,
                    "truncated": truncated},
    )


def _verify_legacy(
    job_id: str, raw_urls: dict, ref_embeddings: dict, ref_urls: dict,
    embedder, file: UploadFile,
) -> VideoVerifyResponse:
    """Luồng cũ khi DB không sẵn sàng: file tạm + thư mục job (giữ nguyên)."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng video không hỗ trợ ({ext or 'trống'}). Hỗ trợ: {', '.join(VIDEO_EXTENSIONS)}.",
        )

    video_bytes = file.file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="File video tải lên không có dữ liệu.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=400, detail="Video vượt quá 200MB.")

    job_dir = os.path.join("outputs", "jobs", job_id)
    video_dir = os.path.join(job_dir, "video_uploads")
    faces_dir = os.path.join(job_dir, "video_faces")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)

    video_path = os.path.join(video_dir, f"{uuid.uuid4().hex}{ext}")
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    # Trích frame 3 fps, tối đa MAX_FRAMES (~30s video).
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Không đọc được video. Hãy thử file .mp4 (H.264) khác.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or fps > 120:
            fps = 25.0
        step = max(1, int(round(fps / FPS_TARGET)))
        preprocess_on = bool(get_config().get("video_preprocess", {}).get("enabled", True))
        frame_idx = 0
        sampled = 0
        faces_found = 0
        cond_counter: Counter = Counter()
        matches: list[VideoFaceMatch] = []
        while sampled < MAX_FRAMES:
            checkpoint()
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    frame_idx += 1
                    continue
                sampled += 1
                timestamp = frame_idx / fps
                # Tiền xử lý theo điều kiện (mưa/tối/chói/mù/...) trước khi detect.
                if preprocess_on:
                    frame, frame_report = preprocess_video_frame(frame)
                    for c in frame_report.get("conditions", []):
                        cond_counter[str(c)] += 1
                detect_frame, scale = _downscale_for_detect(frame)
                try:
                    faces = embedder.detect_faces(detect_frame)
                except ValueError:
                    faces = []
                for k, face in enumerate(faces):
                    checkpoint()
                    det = float(face.det_score)
                    if det < MIN_DET_SCORE:
                        continue
                    faces_found += 1
                    face_emb = np.asarray(face.normed_embedding, dtype=np.float64)
                    best_age = max(ref_embeddings.keys(), key=lambda a: cosine_sim(face_emb, ref_embeddings[a]))
                    best_score = cosine_sim(face_emb, ref_embeddings[best_age])
                    # bbox detect trên ảnh thu nhỏ -> quy về frame gốc để crop nét
                    bbox_orig = [float(v) / scale for v in face.bbox]
                    crop = _crop_bbox(frame, bbox_orig)
                    crop_name = f"frame_{frame_idx}_face_{k}.jpg"
                    crop_path = os.path.join(faces_dir, crop_name)
                    cv2.imwrite(crop_path, crop)
                    matches.append(
                        VideoFaceMatch(
                            face_image_url=f"/outputs/jobs/{job_id}/video_faces/{crop_name}",
                            frame_index=int(frame_idx),
                            timestamp_sec=round(float(timestamp), 2),
                            bbox=[float(v) for v in bbox_orig],
                            det_score=det,
                            best_age=int(best_age),
                            best_age_image_url=str(ref_urls[best_age]),
                            score=float(best_score),
                        )
                    )
            frame_idx += 1
    finally:
        cap.release()

    matches.sort(key=lambda m: m.score, reverse=True)
    top_matches = matches[:TOP_KEEP]
    conditions = dict(cond_counter)

    if not top_matches:
        return VideoVerifyResponse(
            job_id=job_id,
            frames_sampled=sampled,
            faces_found=faces_found,
            best_match=None,
            matches=[],
            conditions=conditions,
        )
    return VideoVerifyResponse(
        job_id=job_id,
        frames_sampled=sampled,
        faces_found=faces_found,
        best_match=top_matches[0],
        matches=top_matches,
        conditions=conditions,
    )
