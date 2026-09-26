"""T07/T08/T09 — Engine nạp quan sát search dùng chung (ảnh / video / camera).

- Ảnh: source search/image → frame tĩnh → detections → crops (bbox, giữ pixel gốc).
- Video: source search/video độc lập (verify sau nhận source_id, không xử lý lại);
  lưu video gốc + frame gốc trước preprocessing + detections + crops.
- Timestamp thực từ decoder (CAP_PROP_POS_MSEC), không suy từ FPS (đúng với
  video FPS biến đổi). Bbox quy về frame gốc. Công bố phạm vi đã xử lý và cờ
  truncated khi chạm giới hạn max_frames.
- Crop key dùng UUID (không bao giờ trùng giữa các video/run).
"""

from __future__ import annotations

from src.utils.cancellation import checkpoint
import logging
import os
import tempfile
import threading
import uuid
from collections import Counter
from typing import Any, BinaryIO, Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FPS_TARGET = 3
MAX_FRAMES = 90
DETECT_MAX_DIM = 960
MIN_DET_SCORE = 0.3
TOP_KEEP = 20

_EMBEDDER_LOCK = threading.Lock()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity cho 2 vector đã L2-normalize (dot product)."""
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)))


def _crop_bbox(frame_bgr: np.ndarray, bbox) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2]


def _downscale_for_detect(frame_bgr: np.ndarray):
    """Thu nhỏ frame lớn về cạnh dài = DETECT_MAX_DIM trước khi detect."""
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


def get_ingest_embedder():  # type: ignore[no-untyped-def]
    """Embedder video (GPU khi rảnh) — khởi tạo 1 lần, có lock cho thread nền."""
    from backend.api.dependencies import get_video_embedder

    with _EMBEDDER_LOCK:
        return get_video_embedder()


def sample_video_frames(
    video_path: str, *, fps_target: float = FPS_TARGET, max_frames: int = MAX_FRAMES,
    cancel: Optional[threading.Event] = None,
) -> tuple[list[dict], dict]:
    """Trích frame để ingest: timestamp thực từ decoder.

    Trả (frames, meta) với frames=[{frame_index, timestamp_sec, frame_bgr}],
    meta={fps, frames_total, duration_sec, frames_processed, truncated}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không đọc được video.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or fps > 120:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / fps_target)))
        frames: list[dict] = []
        frame_idx = 0
        while len(frames) < max_frames:
            checkpoint()
            if cancel is not None and cancel.is_set():
                break
            ok = cap.grab()
            if not ok:
                break
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    frame_idx += 1
                    continue
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = (msec / 1000.0) if msec > 0 else (frame_idx / fps)
                frames.append(
                    {
                        "frame_index": int(frame_idx),
                        "timestamp_sec": round(float(timestamp), 3),
                        "frame_bgr": frame,
                    }
                )
            frame_idx += 1
        truncated = (
            len(frames) >= max_frames and total > 0 and frame_idx < total
        ) or (len(frames) >= max_frames and total <= 0)
        meta = {
            "fps": float(fps),
            "frames_total": total or None,
            "duration_sec": round(total / fps, 3) if total > 0 else round(frame_idx / fps, 3),
            "frames_processed": len(frames),
            "truncated": bool(truncated),
        }
        return frames, meta
    finally:
        cap.release()


def detect_faces_in_frame(
    embedder, frame_bgr: np.ndarray, *, preprocess_on: bool,
) -> tuple[list[dict], list[str], np.ndarray]:
    """Detect 1 frame: trả (faces, conditions, frame_gốc).

    faces=[{bbox_orig, det_score, embedding, kps}]; bbox đã quy về frame gốc.
    Preprocessing chỉ dùng cho detect; crop/persist dùng frame gốc.
    """
    from src.preprocessing import preprocess_video_frame

    original = frame_bgr
    conditions: list[str] = []
    detect_input = frame_bgr
    if preprocess_on:
        detect_input, report = preprocess_video_frame(frame_bgr)
        conditions = [str(c) for c in report.get("conditions", [])]
    detect_frame, scale = _downscale_for_detect(detect_input)
    try:
        raw_faces = embedder.detect_faces(detect_frame)
    except ValueError:
        raw_faces = []
    faces = []
    for face in raw_faces or []:
        det = float(face.det_score)
        if det < MIN_DET_SCORE:
            continue
        emb = getattr(face, "normed_embedding", None)
        faces.append(
            {
                "bbox_orig": [float(v) / scale for v in face.bbox],
                "det_score": det,
                "embedding": np.asarray(emb, dtype=np.float64) if emb is not None else None,
                "kps": getattr(face, "kps", None),
            }
        )
    return faces, conditions, original


def _encode_jpg(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise IOError("Encode frame/crop thất bại.")
    return bytes(buf)


def persist_camera_tracklet(
    conn, storage, *, source_id: str, camera_id: Optional[str],
    track_local_id: str, frame_index: int, offset_ms: int,
    captured_at: Optional[str], received_at: Optional[str],
    source_timestamp: Optional[str], timestamp_uncertainty_ms: Optional[int],
    frame_width: int, frame_height: int,
    candidates: list[dict],
    detector_name: str, detector_version: str, run_id: str,
    preprocessing_base: Optional[dict] = None,
    # Checkpoint sớm (§4.1): slot có revision, KHÔNG đóng tracklet.
    close: bool = True, slot_revision: int = 0,
    observation_count: int = 0,
) -> dict:
    """P0/P1 — Persist 1 tracklet camera theo chính sách crop-only.

    - KHÔNG ghi file frame (search/frames); frame là metadata-only
      (asset_id NULL, kèm source_width/height + timestamp provenance).
    - Chỉ lưu 1–3 crop đã chọn từ tracker (candidates), mỗi crop encode JPEG
      cạnh dài ≤256 Q85 qua camera_tracks.encode_crop_jpeg.
    - Tạo tracklets row (idempotent theo source + local_track_id) rồi detections
      gắn tracklet_id + crops + embeddings. Transaction ngắn do caller giữ.
    - close=True (mặc định): chốt tracklet khi đóng. close=False: checkpoint
      sớm cùng tracklet — touch hàng tracklet status open, ghi slot_revision
      vào preprocessing để truy vết thay slot.
    - Trả {tracklet_id, frame_id, slot_revision,
      detections:[{detection_id, crop_id, ...}]}.

    candidates: [{crop_bgr (copy), bbox, det_score, quality, blur, exposure,
      frontal, face_area, embedding, kps}].
    """
    import uuid as _uuid

    from backend.api import camera_tracks as ct
    from backend.api import repositories as repo
    from backend.api.storage import (
        PREFIX_SEARCH_CROPS,
        build_dated_key,
        build_key,
        sha256_bytes,
    )

    # Shard prefix camera/ngày cho crop camera (§4.3); tắt bằng
    # SEARCH_SHARD_BY_CAMERA_DAY=false. Chỉ ảnh hưởng key mới.
    import os as _os
    shard_on = _os.environ.get(
        "SEARCH_SHARD_BY_CAMERA_DAY", "true").strip().lower() == "true"

    if not candidates:
        raise ValueError("Tracklet không có candidate để persist.")
    kept = candidates[: ct.MAX_CANDIDATES_PER_TRACKLET]

    frame_id = str(_uuid.uuid4())
    tracklet_id = str(_uuid.uuid5(_uuid.NAMESPACE_URL,
                                  f"{source_id}/{track_local_id}"))
    stored_keys: list[str] = []
    try:
        # Idempotent tracklet: ON CONFLICT không tạo trùng khi retry/checkpoint.
        try:
            repo.create_tracklet(
                conn, source_id=source_id, camera_id=camera_id,
                local_track_id=track_local_id, status="open",
                started_at=captured_at, last_seen_at=captured_at,
                observation_count=0,
                quality_summary={"policy": "crop_only"},
                tracklet_id=tracklet_id)
        except repo.ConflictError:
            pass
        except RuntimeError:
            # DB chưa migrate 005: bỏ qua tracklet row, vẫn lưu detection
            # với track_id text để không mất dữ liệu.
            tracklet_id = None  # type: ignore[assignment]
        repo.create_frame(
            conn, purpose="search", source_id=source_id, asset_id=None,
            frame_index=int(frame_index), offset_ms=int(offset_ms),
            captured_at=captured_at, frame_id=frame_id,
            source_width=int(frame_width), source_height=int(frame_height),
            received_at=received_at, source_timestamp=source_timestamp,
            timestamp_uncertainty_ms=timestamp_uncertainty_ms)
        detections = []
        for face_index, cand in enumerate(kept):
            checkpoint()
            x1, y1, x2, y2 = (float(v) for v in cand["bbox"])
            x1c = max(0.0, min(x1, float(frame_width)))
            y1c = max(0.0, min(y1, float(frame_height)))
            x2c = max(0.0, min(x2, float(frame_width)))
            y2c = max(0.0, min(y2, float(frame_height)))
            if x2c <= x1c or y2c <= y1c:
                continue
            kps = cand.get("kps")
            landmarks = None
            if kps is not None:
                try:
                    # Landmark quy về tọa độ crop để căn chỉnh lại (doc §4.2).
                    import numpy as _np
                    pts = _np.asarray(kps, dtype=float).reshape(-1, 2)
                    crop_bgr = cand["crop_bgr"]
                    ch0, cw0 = crop_bgr.shape[:2]
                    # Ước lượng gốc crop từ bbox + pad 0.15 (đồng bộ copy_crop).
                    bw, bh = max(1.0, x2c - x1c), max(1.0, y2c - y1c)
                    ox, oy = x1c - bw * 0.15, y1c - bh * 0.15
                    landmarks = [[float(x - ox), float(y - oy)] for x, y in pts]
                    _ = (ch0, cw0)
                except Exception:
                    landmarks = None
            quality = {"det_score": float(cand.get("det_score", 0.0)),
                       "track_quality": float(cand.get("quality", 0.0)),
                       "blur": float(cand.get("blur", 0.0)),
                       "exposure": float(cand.get("exposure", 0.0)),
                       "frontal": float(cand.get("frontal", 0.5)),
                       "face_area": float(cand.get("face_area", 0.0)),
                       "policy": "crop_only"}
            try:
                detection_id = repo.create_detection(
                    conn, purpose="search", frame_id=frame_id,
                    detector_name=detector_name, detector_version=detector_version,
                    run_id=run_id, face_index=face_index,
                    bbox=(x1c, y1c, x2c, y2c), confidence=float(cand.get("det_score", 0.0)),
                    frame_width=int(frame_width), frame_height=int(frame_height),
                    landmarks=landmarks, track_id=track_local_id,
                    quality=quality, tracklet_id=tracklet_id)
            except TypeError:
                detection_id = repo.create_detection(
                    conn, purpose="search", frame_id=frame_id,
                    detector_name=detector_name, detector_version=detector_version,
                    run_id=run_id, face_index=face_index,
                    bbox=(x1c, y1c, x2c, y2c), confidence=float(cand.get("det_score", 0.0)),
                    frame_width=int(frame_width), frame_height=int(frame_height),
                    landmarks=landmarks, track_id=track_local_id, quality=quality)
            crop_data, prov = ct.encode_crop_jpeg(cand["crop_bgr"])
            ch, cw = cand["crop_bgr"].shape[:2]
            # Provenance resize thực tế (doc §4.2: ghi transform/codec/preprocessing).
            dh, dw = prov["size_dst"]
            crop_id = str(_uuid.uuid4())
            crop_asset_id = str(_uuid.uuid4())
            if shard_on:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    day = (_dt.fromisoformat(captured_at).date().isoformat()
                           if captured_at else _dt.now(_tz.utc).date().isoformat())
                except Exception:
                    from datetime import datetime as _dt2, timezone as _tz2
                    day = _dt2.now(_tz2.utc).date().isoformat()
                crop_key = build_dated_key(
                    PREFIX_SEARCH_CROPS, crop_asset_id, "image/jpeg",
                    shard=str(camera_id or ""), date=day)
            else:
                crop_key = build_key(PREFIX_SEARCH_CROPS, crop_asset_id, "image/jpeg")
            storage.put_bytes(crop_data, crop_key, mime_type="image/jpeg")
            stored_keys.append(crop_key)
            repo.create_asset(
                conn, storage_key=crop_key, media_type="image",
                mime_type="image/jpeg", sha256=sha256_bytes(crop_data),
                byte_size=len(crop_data), width=int(dw), height=int(dh),
                role="crop", asset_id=crop_asset_id)
            preproc = dict(preprocessing_base or {})
            preproc.update({"crop_method": "camera_tracklet",
                            "codec": prov["codec"], "quality": prov["quality"],
                            "size_src": prov["size_src"], "size_dst": prov["size_dst"],
                            "bbox_original": [x1c, y1c, x2c, y2c],
                            "slot_revision": int(slot_revision),
                            "preprocessing_version": prov["preprocessing_version"]})
            repo.create_crop(
                conn, purpose="search", detection_id=detection_id,
                asset_id=crop_asset_id, method="bbox",
                preprocessing=preproc,
                transform_to_source={"bbox_original": [x1c, y1c, x2c, y2c]},
                crop_id=crop_id)
            embedding = cand.get("embedding")
            if embedding is not None:
                try:
                    from backend.api.gallery import current_embed_triple, emit_outbox
                    m, mv, pv = current_embed_triple()
                    emb_id = repo.create_embedding(
                        conn, model_name=m, model_version=mv,
                        preprocessing_version=pv, dimensions=len(embedding),
                        values=[float(v) for v in embedding],
                        normalized=True, crop_id=crop_id)
                    # P3: commit vector + metadata + outbox cùng transaction.
                    emit_outbox(conn, entity="embedding", entity_id=emb_id,
                                op="upsert", triple=(m, mv, pv),
                                payload={"crop_id": crop_id})
                except Exception as exc:
                    logger.debug("bỏ qua embedding crop %s: %s", crop_id, exc)
            detections.append({"detection_id": detection_id, "crop_id": crop_id,
                               "crop_key": crop_key, "bbox": [x1c, y1c, x2c, y2c],
                               "det_score": float(cand.get("det_score", 0.0)),
                               "quality": float(cand.get("quality", 0.0)),
                               "embedding": embedding})
        if tracklet_id is not None:
            try:
                if close:
                    repo.close_tracklet(
                        conn, tracklet_id, ended_at=captured_at,
                        observation_count=observation_count or len(detections),
                        quality_summary={"candidates": len(detections),
                                         "slot_revision": int(slot_revision),
                                         "policy": "crop_only"})
                else:
                    # Checkpoint sớm: giữ tracklet mở, cập nhật observation.
                    repo.touch_tracklet(
                        conn, tracklet_id, source_id=source_id,
                        camera_id=camera_id, local_track_id=track_local_id,
                        status="open", last_seen_at=captured_at,
                        observation_count=observation_count or len(detections),
                        quality_summary={"candidates": len(detections),
                                         "slot_revision": int(slot_revision),
                                         "policy": "crop_only"})
            except Exception:
                pass
    except Exception:
        for key in stored_keys:
            storage.delete_quiet(key)
        raise
    return {"tracklet_id": tracklet_id, "frame_id": frame_id,
            "frame_available": False, "frame_url": None,
            "slot_revision": int(slot_revision),
            "detections": detections}


def persist_observation_frame(
    conn, storage, *, source_id: str, purpose: str,
    frame_bgr: np.ndarray, frame_index: int, offset_ms: int,
    captured_at: Optional[str], faces: list[dict],
    detector_name: str, detector_version: str, run_id: str,
    preprocessing_base: Optional[dict] = None,
    min_det_score: float = MIN_DET_SCORE,
) -> dict:
    """Lưu 1 frame quan sát + detections + crops (bbox, method='bbox').

    Trả {frame_id, detections:[{detection_id, crop_id, crop_key, embedding...}]}.
    Mọi crop đạt ngưỡng thu nhận đều được lưu TRƯỚC khi lọc top matches.
    """
    import uuid as _uuid

    from backend.api import repositories as repo
    from backend.api.storage import (
        PREFIX_REFERENCE_CROPS,
        PREFIX_SEARCH_CROPS,
        build_key,
        sha256_bytes,
    )

    height, width = frame_bgr.shape[:2]
    data = _encode_jpg(frame_bgr)
    frame_id = str(_uuid.uuid4())
    frame_asset_id = str(_uuid.uuid4())
    prefix_frames = (
        "reference/frames" if purpose == "reference" else "search/frames"
    )
    prefix_crops = PREFIX_REFERENCE_CROPS if purpose == "reference" else PREFIX_SEARCH_CROPS
    frame_key = build_key(prefix_frames, frame_asset_id, "image/jpeg")
    storage.put_bytes(data, frame_key, mime_type="image/jpeg")
    stored_keys = [frame_key]
    try:
        repo.create_asset(
            conn, storage_key=frame_key, media_type="image",
            mime_type="image/jpeg", sha256=sha256_bytes(data),
            byte_size=len(data), width=int(width), height=int(height),
            role="frame", asset_id=frame_asset_id,
        )
        repo.create_frame(
            conn, purpose=purpose, source_id=source_id, asset_id=frame_asset_id,
            frame_index=int(frame_index), offset_ms=int(offset_ms),
            captured_at=captured_at, frame_id=frame_id,
        )
        detections = []
        for face_index, face in enumerate(faces):
            checkpoint()
            if face["det_score"] < min_det_score:
                continue
            x1, y1, x2, y2 = face["bbox_orig"]
            x1c = max(0.0, min(float(x1), float(width)))
            y1c = max(0.0, min(float(y1), float(height)))
            x2c = max(0.0, min(float(x2), float(width)))
            y2c = max(0.0, min(float(y2), float(height)))
            if x2c <= x1c or y2c <= y1c:
                continue
            kps = face.get("kps")
            landmarks = (
                [[float(x), float(y)] for x, y in kps] if kps is not None else None
            )
            detection_id = repo.create_detection(
                conn, purpose=purpose, frame_id=frame_id,
                detector_name=detector_name, detector_version=detector_version,
                run_id=run_id, face_index=face_index,
                bbox=(x1c, y1c, x2c, y2c), confidence=face["det_score"],
                frame_width=int(width), frame_height=int(height),
                landmarks=landmarks, track_id=face.get("track_id"),
                quality=face.get("quality") or {"det_score": face["det_score"]},
            )
            crop_bgr = _crop_bbox(frame_bgr, (x1c, y1c, x2c, y2c))
            crop_data = _encode_jpg(crop_bgr)
            ch, cw = crop_bgr.shape[:2]
            crop_id = str(_uuid.uuid4())
            crop_asset_id = str(_uuid.uuid4())
            crop_key = build_key(prefix_crops, crop_asset_id, "image/jpeg")
            storage.put_bytes(crop_data, crop_key, mime_type="image/jpeg")
            stored_keys.append(crop_key)
            repo.create_asset(
                conn, storage_key=crop_key, media_type="image",
                mime_type="image/jpeg", sha256=sha256_bytes(crop_data),
                byte_size=len(crop_data), width=int(cw), height=int(ch),
                role="crop", asset_id=crop_asset_id,
            )
            preproc = dict(preprocessing_base or {})
            preproc.setdefault("crop_method", "bbox_from_observation")
            repo.create_crop(
                conn, purpose=purpose, detection_id=detection_id,
                asset_id=crop_asset_id, method="bbox",
                preprocessing=preproc,
                transform_to_source={"bbox_original": [x1c, y1c, x2c, y2c]},
                crop_id=crop_id,
            )
            # T10: giữ vector quan sát (miễn phí từ detect) — idempotent.
            embedding = face.get("embedding")
            if embedding is not None:
                try:
                    from backend.api.gallery import current_embed_triple, emit_outbox
                    m, mv, pv = current_embed_triple()
                    emb_id = repo.create_embedding(
                        conn, model_name=m, model_version=mv,
                        preprocessing_version=pv, dimensions=len(embedding),
                        values=[float(v) for v in embedding],
                        normalized=True, crop_id=crop_id,
                    )
                    # P3: commit vector + metadata + outbox cùng transaction.
                    emit_outbox(conn, entity="embedding", entity_id=emb_id,
                                op="upsert", triple=(m, mv, pv),
                                payload={"crop_id": crop_id})
                except Exception as exc:
                    logger.debug("bỏ qua embedding crop %s: %s", crop_id, exc)
            detections.append(
                {
                    "detection_id": detection_id,
                    "crop_id": crop_id,
                    "crop_key": crop_key,
                    "bbox": [x1c, y1c, x2c, y2c],
                    "det_score": face["det_score"],
                    "embedding": face.get("embedding"),
                }
            )
    except Exception:
        for key in stored_keys:
            storage.delete_quiet(key)
        raise
    return {"frame_id": frame_id, "detections": detections}


def ingest_image_bytes(
    *, data: bytes, filename: str, content_type: Optional[str],
    embedder, preprocess_on: bool = True,
    detector_version: Optional[str] = None,
) -> Optional[dict]:
    """T07: nạp 1 ảnh search (không cần job tạo sinh). None khi DB down."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    from backend.api.persistence import db_ping
    from backend.api.storage import (
        PREFIX_SEARCH_ORIGINALS,
        build_key,
        check_image_decodable,
        get_storage,
        sha256_bytes,
        sniff_mime,
    )

    if not db_ping():
        return None
    try:
        width, height = check_image_decodable(data)
        mime = (content_type or "").split(";")[0].strip().lower()
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            mime = sniff_mime(data, "image/png")
        storage = get_storage()
        source_id = str(uuid.uuid4())
        asset_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        key = build_key(PREFIX_SEARCH_ORIGINALS, asset_id, mime)
        storage.put_bytes(data, key, mime_type=mime)
        try:
            with get_pool().connection() as conn:
                repo.create_asset(
                    conn, storage_key=key, media_type="image", mime_type=mime,
                    sha256=sha256_bytes(data), byte_size=len(data),
                    width=width, height=height, role="original_image",
                    asset_id=asset_id,
                )
                repo.create_source(
                    conn, purpose="search", kind="image",
                    original_asset_id=asset_id, source_id=source_id,
                )
                image_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                faces, conditions, original = detect_faces_in_frame(
                    embedder, image_bgr, preprocess_on=preprocess_on
                )
                det_version = detector_version or getattr(embedder, "model_name", "buffalo_l")
                frame = persist_observation_frame(
                    conn, storage, source_id=source_id, purpose="search",
                    frame_bgr=original, frame_index=0, offset_ms=0,
                    captured_at=None, faces=faces,
                    detector_name="insightface", detector_version=det_version,
                    run_id=run_id,
                    preprocessing_base={"source": "search_image_upload",
                                        "conditions": conditions},
                )
        except Exception:
            storage.delete_quiet(key)
            raise
        return {
            "source_id": source_id, "asset_id": asset_id,
            "faces_found": len(frame["detections"]),
            "crops": [
                {"crop_id": d["crop_id"], "crop_key": d["crop_key"],
                 "bbox": d["bbox"], "det_score": d["det_score"],
                 "url": storage.get_access_url(d["crop_key"])}
                for d in frame["detections"]
            ],
            "conditions": conditions,
        }
    except Exception as exc:
        logger.warning("ingest_image_bytes bỏ qua: %s: %s", type(exc).__name__, exc)
        return None


def _probe_video_dims(video_bytes: bytes, ext: str) -> tuple[int, int]:
    """Đọc kích thước từ frame đầu (qua file tạm, có cleanup)."""
    fd, tmp = tempfile.mkstemp(suffix=ext or ".mp4")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(video_bytes)
        cap = cv2.VideoCapture(tmp)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise ValueError("Không đọc được video.")
        height, width = frame.shape[:2]
        return int(width), int(height)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def persist_video_source(
    *, video_bytes: bytes, ext: str, mime_type: str,
) -> Optional[dict]:
    """Lưu video gốc + source search/video. Trả {source_id, video_key} hoặc None."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    from backend.api.persistence import db_ping
    from backend.api.storage import (
        PREFIX_SEARCH_ORIGINALS,
        build_key,
        get_storage,
        sha256_bytes,
    )

    if not db_ping():
        return None
    try:
        storage = get_storage()
        source_id = str(uuid.uuid4())
        asset_id = str(uuid.uuid4())
        key = build_key(PREFIX_SEARCH_ORIGINALS, asset_id, mime_type)
        width, height = _probe_video_dims(video_bytes, ext)
        storage.put_bytes(video_bytes, key, mime_type=mime_type)
        try:
            with get_pool().connection() as conn:
                repo.create_asset(
                    conn, storage_key=key, media_type="video",
                    mime_type=mime_type, sha256=sha256_bytes(video_bytes),
                    byte_size=len(video_bytes), width=width, height=height,
                    role="original_video", asset_id=asset_id,
                )
        except Exception:
            storage.delete_quiet(key)
            raise
        with get_pool().connection() as conn:
            try:
                repo.create_source(
                    conn, purpose="search", kind="video",
                    original_asset_id=asset_id, source_id=source_id,
                )
            except Exception:
                storage.delete_quiet(key)
                raise
        return {"source_id": source_id, "asset_id": asset_id, "video_key": key}
    except Exception as exc:
        logger.warning("persist_video_source bỏ qua: %s: %s", type(exc).__name__, exc)
        return None


def ingest_sampled_frames(
    *, source_id: str, frames: list[dict], embedder,
    preprocess_on: bool, run_id: str,
    cancel: Optional[threading.Event] = None,
    on_progress=None,
) -> dict:
    """Nạp các frame đã sample vào source (detections + crops + conditions)."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    from backend.api.storage import get_storage

    storage = get_storage()
    det_version = getattr(embedder, "model_name", "buffalo_l")
    cond_counter: Counter = Counter()
    faces_found = 0
    crop_records: list[dict] = []
    with get_pool().connection() as conn:
        for item in frames:
            checkpoint()
            if cancel is not None and cancel.is_set():
                break
            faces, conditions, original = detect_faces_in_frame(
                embedder, item["frame_bgr"], preprocess_on=preprocess_on
            )
            for cond in conditions:
                cond_counter[cond] += 1
            frame = persist_observation_frame(
                conn, storage, source_id=source_id, purpose="search",
                frame_bgr=original, frame_index=item["frame_index"],
                offset_ms=int(item["timestamp_sec"] * 1000),
                captured_at=None, faces=faces,
                detector_name="insightface", detector_version=det_version,
                run_id=run_id,
                preprocessing_base={"source": "search_video_ingest",
                                    "conditions": conditions},
            )
            faces_found += len(frame["detections"])
            for det in frame["detections"]:
                crop_records.append(
                    {
                        **det,
                        "frame_index": item["frame_index"],
                        "timestamp_sec": item["timestamp_sec"],
                        "url": storage.get_access_url(det["crop_key"]),
                    }
                )
            if on_progress is not None:
                on_progress(len(crop_records))
    return {
        "faces_found": faces_found,
        "crops": crop_records,
        "conditions": dict(cond_counter),
    }
