"""8 camera thay thế bằng video (chỉ dùng sau khi job đã có ảnh tạo sinh).

Mỗi ô camera (1–8) nhận một file video, chạy ĐÚNG pipeline camera thật ở chế độ
phát lại (tracker, chọn crop chất lượng, K/q_min, lưu face_media crop-only), rồi
đối soát crop của các nguồn với ảnh FADING của job; mỗi lượt đối soát lưu
search_runs/search_results như video-verify.

- POST /api/replay/start         multipart: job_id, slot, file, start_offset_sec, ...
- GET  /api/replay/status/{sid}  tiến độ phát lại + số tracklet/crop đã lưu
- POST /api/replay/match         đối soát các nguồn với ảnh tạo sinh, trả theo camera
                                 + dòng thời gian xuất hiện qua các camera
"""

from __future__ import annotations

import os
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.api import cameras as cam_service
from backend.api.persistence import db_ping

router = APIRouter(prefix="/api/replay", tags=["replay"])

SLOTS = 8
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
MAX_VIDEO_BYTES = 500 * 1024 * 1024
REPLAY_DIR = os.path.join("outputs", "replay")
DEFAULT_FPS = 5.0
# Ngưỡng đối soát mặc định cho màn hình này: 0.6 (config) bỏ sót ~40% cặp cùng người
# gần tuổi trên FG-NET; 0.45 giữ ~90% với FMR ~7e-5 (docs/quy-trinh-toi-uu-camera.md).
DEFAULT_THRESHOLD = 0.45
CAMERA_NAME = "Camera {slot} (video thay thế)"


def _require_db() -> None:
    if not db_ping():
        raise HTTPException(status_code=503,
                            detail="Database face_media không sẵn sàng — màn hình 8 camera "
                                   "cần DB để lưu crop và kết quả đối soát.")


def _require_generated(job_id: str) -> dict:
    """Chỉ cho chạy khi job đã có ảnh tạo sinh (kiểm cả phía server)."""
    from backend.api.routers.video_verify import _resolve_reference
    _, raw_urls = _resolve_reference(job_id)
    return raw_urls


def _camera_for_slot(slot: int) -> str:
    """Camera DB ổn định cho mỗi ô (tạo một lần, tìm lại theo tên)."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    name = CAMERA_NAME.format(slot=slot)
    with get_pool().connection() as conn:
        items, _ = repo.list_cameras(conn, limit=500, offset=0)
        for cam in items:
            if cam.get("name") == name:
                return str(cam.get("camera_id") or cam.get("id"))
        return repo.create_camera(conn, name=name, location=f"replay-slot-{slot}")


@router.post("/start", status_code=202)
def start_replay(
    job_id: str = Form(...),
    slot: int = Form(..., ge=1, le=SLOTS),
    file: UploadFile = File(...),
    start_offset_sec: float = Form(0.0, ge=0, le=86400),
    base_epoch: Optional[float] = Form(None),
    sample_fps: float = Form(DEFAULT_FPS, gt=0, le=10),
    max_candidates: int = Form(3, ge=1, le=5),
    min_quality: float = Form(0.0, ge=0, lt=1),
):
    _require_db()
    _require_generated(job_id)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400,
                            detail=f"Định dạng không hỗ trợ ({ext or 'trống'}). "
                                   f"Hỗ trợ: {', '.join(VIDEO_EXTENSIONS)}.")
    os.makedirs(REPLAY_DIR, exist_ok=True)
    path = os.path.join(REPLAY_DIR, f"slot{slot}_{uuid.uuid4().hex}{ext}")
    size = 0
    with open(path, "wb") as out:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                out.close()
                os.remove(path)
                raise HTTPException(status_code=400, detail="Video vượt quá 500MB.")
            out.write(chunk)
    if size == 0:
        os.remove(path)
        raise HTTPException(status_code=400, detail="File video rỗng.")
    camera_id = _camera_for_slot(slot)
    try:
        started = cam_service.start_capture(
            camera_id=camera_id, sample_fps=sample_fps, max_frames=100000,
            preprocess_on=True, max_candidates=max_candidates, min_quality=min_quality,
            config_source="replay", replay_path=path, replay_offset_sec=start_offset_sec,
            replay_base_epoch=base_epoch if base_epoch is not None else time.time())
    except (ValueError, ConnectionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"slot": slot, "camera_id": camera_id, "source_id": started["source_id"],
            "run_id": started["run_id"], "file_name": file.filename, "bytes": size,
            "sample_fps": sample_fps, "max_candidates": max_candidates,
            "min_quality": min_quality, "start_offset_sec": start_offset_sec}


@router.get("/status/{source_id}")
def replay_status(source_id: str):
    try:
        return cam_service.capture_status(source_id)
    except LookupError:
        raise HTTPException(status_code=404,
                            detail="Phiên phát lại không còn trong bộ nhớ (đã chốt hoặc backend "
                                   "khởi động lại).")


class ReplaySource(BaseModel):
    slot: int = Field(..., ge=1, le=SLOTS)
    source_id: str
    start_offset_sec: float = Field(0.0, ge=0)


class ReplayMatchRequest(BaseModel):
    job_id: str
    sources: List[ReplaySource] = Field(..., min_length=1, max_length=SLOTS)
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0, le=1)
    top_k: int = Field(5, ge=1, le=20)


@router.post("/match")
def match_replay(body: ReplayMatchRequest):
    """Đối soát crop của từng nguồn với ảnh tạo sinh; lưu một search_run mỗi nguồn."""
    from backend.api.dependencies import get_video_embedder
    from backend.api.routers import video_verify as vv

    _require_db()
    raw_urls = _require_generated(body.job_id)
    embedder = get_video_embedder()
    ref_embeddings, ref_urls = vv._embed_references(body.job_id, raw_urls, embedder)
    cameras, timeline = [], []
    for src in sorted(body.sources, key=lambda s: s.slot):
        summary = None
        try:
            st = cam_service.capture_status(src.source_id)
            if not (st.get("replay") or {}).get("finished"):
                raise HTTPException(status_code=409,
                                    detail=f"Camera {src.slot} chưa xử lý xong video.")
            summary = cam_service.stop_capture(src.source_id)   # chốt run + ended_at
        except LookupError:
            pass                                                  # đã chốt từ trước
        records, crop_ids = vv._load_source_records(src.source_id, embedder)
        matches = vv._match_records(records, ref_embeddings, ref_urls)
        by_url = {r["url"]: r["crop_id"] for r in records}
        ordered = [by_url.get(m.face_image_url, "") for m in matches]
        run_id = vv._save_search_run(
            job_id=body.job_id, source_id=src.source_id, matches=matches, crop_ids=ordered,
            threshold=body.threshold, ref_ages=list(ref_embeddings.keys()),
            conditions={"replay_slot": src.slot}, truncated=False)
        top = [m.model_dump() | {"accepted": m.score >= body.threshold,
                                 "camera_time_sec": round(src.start_offset_sec + m.timestamp_sec, 2)}
               for m in matches[:body.top_k]]
        accepted = [t for t in top if t["accepted"]]
        for t in accepted:
            timeline.append({"slot": src.slot, **t})
        cameras.append({
            "slot": src.slot, "source_id": src.source_id, "search_run_id": run_id,
            "crops": len(records), "best_match": top[0] if top else None,
            "matches": top, "accepted": len(accepted),
            "tracks_closed": (summary or {}).get("tracks_closed"),
            "crops_persisted": (summary or {}).get("crops_persisted"),
        })
    # Một mục mỗi lần xuất hiện (các crop cùng tracklet chung mốc thời gian): giữ điểm cao nhất.
    best_per_seen: dict = {}
    for t in timeline:
        key = (t["slot"], t["camera_time_sec"])
        if key not in best_per_seen or t["score"] > best_per_seen[key]["score"]:
            best_per_seen[key] = t
    timeline = sorted(best_per_seen.values(), key=lambda t: t["camera_time_sec"])
    return {"job_id": body.job_id, "threshold": body.threshold,
            "reference_ages": sorted(ref_embeddings.keys()),
            "cameras": cameras, "timeline": timeline,
            "found_on_cameras": sorted({t["slot"] for t in timeline})}
