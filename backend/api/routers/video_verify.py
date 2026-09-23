"""Đối soát bổ sung bằng video (sau bước Kết quả FADING).

Luồng: user upload 1 video sau khi job pipeline đã done ->
backend trích frame (3 fps, tối đa 90 frame ~30s) -> InsightFace buffalo_l
trên GPU (get_video_embedder, tự rơi về CPU nếu lỗi) detect_faces() trên
từng frame đã thu nhỏ (cạnh dài 960px) -> crop mặt nét từ frame gốc theo
bbox quy đổi -> embedding mặt video có sẵn trong Face.normed_embedding ->
cosine similarity với embedding của từng ảnh FADING đã sinh (edited_images
của job) -> trả về mặt có độ tương đồng cao nhất + top matches.
"""

import glob
import os
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.dependencies import get_video_embedder
from backend.api.schemas import VideoFaceMatch, VideoVerifyResponse
from backend.api.session_store import get_job

router = APIRouter(prefix="/api/jobs", tags=["video-verify"])

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
MAX_VIDEO_BYTES = 200 * 1024 * 1024
FPS_TARGET = 3  # mỗi giây video chỉ lấy 3 frame để detect/crop/so khớp
MAX_FRAMES = 90  # tối đa ~30s video ở 3 fps
DETECT_MAX_DIM = 960  # frame lớn hơn được thu nhỏ trước khi detect (detect nhanh hơn
                      # nhiều mà embedding vẫn chuẩn vì recognition chỉ crop mặt 112x112)
MIN_DET_SCORE = 0.3
TOP_KEEP = 20


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity cho 2 vector đã L2-normalize (dot product)."""
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)))


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


def _crop_bbox(frame_bgr: np.ndarray, bbox) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2]


def _downscale_for_detect(frame_bgr: np.ndarray):
    """Thu nhỏ frame lớn về cạnh dài = DETECT_MAX_DIM trước khi detect.

    Trả về (frame_detect, scale): frame_detect đưa vào detect_faces(), bbox kết quả
    nhân với 1/scale để crop trên frame gốc nét. Frame đã nhỏ thì giữ nguyên."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= DETECT_MAX_DIM:
        return frame_bgr, 1.0
    scale = DETECT_MAX_DIM / float(longest)
    small = cv2.resize(
        frame_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return small, scale


@router.post("/{job_id}/video-verify", response_model=VideoVerifyResponse)
async def video_verify(job_id: str, file: UploadFile = File(...)):
    job = get_job(job_id)
    raw_urls: dict = {}
    if job and job.status == "done" and job.result and job.result.get("edited_images"):
        raw_urls = dict(job.result["edited_images"])
    if not raw_urls:
        # Fallback đĩa: store job chỉ lưu in-memory nên mất khi backend restart,
        # hoặc khi xem lại job cũ từ lịch sử — dựng lại từ age_*.png trên đĩa.
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

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng video không hỗ trợ ({ext or 'trống'}). Hỗ trợ: {', '.join(VIDEO_EXTENSIONS)}.",
        )

    video_bytes = await file.read()
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

    # Embed ảnh FADING đã sinh (reference) — mỗi mốc tuổi 1 vector.
    edited_paths = _resolve_edited_paths(job_id, raw_urls)
    if not edited_paths:
        raise HTTPException(status_code=400, detail="Không tìm thấy file ảnh FADING đã sinh của job.")
    embedder = get_video_embedder()  # GPU (tự rơi về CPU nếu lỗi), vì pipeline đã xong
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

    # Trích frame 3 fps, tối đa MAX_FRAMES (~30s video).
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Không đọc được video. Hãy thử file .mp4 (H.264) khác.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or fps > 120:
            fps = 25.0
        step = max(1, int(round(fps / FPS_TARGET)))
        frame_idx = 0
        sampled = 0
        faces_found = 0
        matches: list[VideoFaceMatch] = []
        while sampled < MAX_FRAMES:
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
                detect_frame, scale = _downscale_for_detect(frame)
                try:
                    faces = embedder.detect_faces(detect_frame)
                except ValueError:
                    faces = []
                for k, face in enumerate(faces):
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

    if not top_matches:
        return VideoVerifyResponse(
            job_id=job_id,
            frames_sampled=sampled,
            faces_found=faces_found,
            best_match=None,
            matches=[],
        )
    return VideoVerifyResponse(
        job_id=job_id,
        frames_sampled=sampled,
        faces_found=faces_found,
        best_match=top_matches[0],
        matches=top_matches,
    )
