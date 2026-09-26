"""T09 + P0/P1 — Thu nhận camera crop-only (đa camera tiết kiệm bộ nhớ).

Theo docs/kien-truc-da-camera-tiet-kiem-bo-nho.md:
- P0: luồng camera CHỈ lưu crop khuôn mặt (JPEG 256px) + vector + metadata.
  Không ghi video hoặc ảnh toàn khung của luồng này. Không tạo recorder phiên
  (CAMERA_SAVE_SESSION_VIDEO bị bỏ qua trên đường crop-only).
- P1: tracking trong từng camera (CameraTracker IoU), gom thành tracklet, giữ
  1–3 crop đại diện cho mỗi lượt xuất hiện (selector trong RAM, bounded).
- Queue 1–2 frame, bỏ frame cũ khi đầy, cap tổng byte; producer drain stream
  liên tục, chọn frame mới nhất, đo độ trễ; consumer transaction ngắn theo
  từng tracklet (không giữ connection xuyên vòng lặp).
- Mỗi phiên start tạo MỘT source search/camera với storage_policy='crop_only'.
- URL/credential camera KHÔNG nằm trong DB (chỉ connection_secret_ref).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import cv2

logger = logging.getLogger(__name__)

_SESSIONS: dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()

# P0: queue ngắn để RAM không tăng theo camera × độ phân giải.
QUEUE_MAXSIZE = 2
# Cap tổng byte payload queue: 2 frame 1080p BGR ≈ 12 MiB → cap 16 MiB.
QUEUE_MAX_BYTES = 16 * 1024 * 1024
RECONNECT_MAX = 5
RECONNECT_DELAY_SEC = 2.0
# Drain tối đa N grab cũ trước khi retrieve frame mới nhất.
DRAIN_GRABS = 3


def redact_url(url: str) -> str:
    """Ẩn credential trong URL để log/status (không lộ mật khẩu)."""
    try:
        parts = urlsplit(url)
        if "@" in parts.netloc:
            host = parts.netloc.split("@", 1)[1]
            clean = parts._replace(netloc=f"***@{host}")
            return urlunsplit(clean)
        return url
    except Exception:
        return "***"


def resolve_camera_target(connection_secret_ref: Optional[str]) -> str:
    """Đọc URL/index camera từ secret store (biến môi trường backend).

    - Ref là tên biến môi trường, vd CAMERA_PORCH_URL="rtsp://user:pass@host/...".
    - Giá trị số ("0") → webcam device index. Không ref → webcam 0 mặc định.
    """
    if not connection_secret_ref:
        return "0"
    value = os.environ.get(connection_secret_ref, "").strip()
    if not value:
        raise ValueError(
            f"Secret {connection_secret_ref!r} chưa đặt trong môi trường backend."
        )
    return value


def _open_capture(target: str):  # type: ignore[no-untyped-def]
    if target.isdigit():
        cap = cv2.VideoCapture(int(target))
    else:
        cap = cv2.VideoCapture(target)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        raise ConnectionError("Không mở được camera.")
    return cap


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame_bytes(frame) -> int:
    try:
        return int(frame.nbytes)
    except Exception:
        h, w = frame.shape[:2]
        return int(h * w * 3)


def _queue_put_newest(session: dict, item: dict) -> None:
    """Đẩy frame mới nhất; khi đầy bỏ frame cũ nhất (không chặn producer).

    Cap tổng byte: bỏ frame cũ cho đến khi đủ chỗ hoặc queue rỗng.
    """
    q: queue.Queue = session["queue"]
    size = _frame_bytes(item["frame_bgr"])
    item["bytes"] = size
    while True:
        queued_bytes = int(session.get("queue_bytes", 0))
        if q.qsize() < q.maxsize and queued_bytes + size <= QUEUE_MAX_BYTES:
            break
        try:
            old = q.get_nowait()
            session["queue_bytes"] = max(0, queued_bytes - int(old.get("bytes", 0)))
            session["dropped"] = int(session.get("dropped", 0)) + 1
        except queue.Empty:
            break
        if q.empty():
            if size > QUEUE_MAX_BYTES:
                # Frame đơn vượt cap (độ phân giải rất cao): vẫn nhận 1 frame
                # mới nhất để không đứng hình, nhưng đếm cảnh báo.
                session["oversize_frames"] = int(session.get("oversize_frames", 0)) + 1
            break
    try:
        q.put_nowait(item)
        session["queue_bytes"] = int(session.get("queue_bytes", 0)) + size
    except queue.Full:
        session["dropped"] = int(session.get("dropped", 0)) + 1


def _adapt_interval(session: dict, window: int = 30) -> float:
    """Điều chỉnh nhịp sample theo tỷ lệ rớt frame (§9). Trả scale mới."""
    window_drops = int(session.get("dropped", 0)) - int(session.get("_drop_mark", 0))
    session["_drop_mark"] = int(session.get("dropped", 0))
    scale = float(session.get("interval_scale", 1.0))
    if window_drops / float(window) > 0.20:
        scale = min(4.0, scale * 1.5)
    elif window_drops / float(window) < 0.05:
        scale = max(1.0, scale / 1.25)
    session["interval_scale"] = scale
    if scale > 1.0:
        session["degraded"] = (
            f"overload: giảm fps phân tích còn {1.0 / scale:.0%} "
            f"(rớt {window_drops}/{window} frame)")
    elif (session.get("degraded") or "").startswith("overload:"):
        session.pop("degraded", None)
    return scale


def _producer(session: dict) -> None:
    """Đọc/drain stream liên tục, chọn frame mới nhất, đo độ trễ."""
    q: queue.Queue = session["queue"]
    cancel: threading.Event = session["cancel"]
    base_interval = 1.0 / max(float(session["sample_fps"]), 0.01)
    session.setdefault("interval_scale", 1.0)
    deadline = session.get("deadline")
    reconnects = 0
    cap = None
    next_tick = time.monotonic()
    processed_ticks = 0
    start_mono = time.monotonic()
    try:
        cap = _open_capture(session["target"])
        guard_hits = 0
        while not cancel.is_set():
            if deadline is not None and time.time() >= deadline:
                break
            if session["frame_count"] >= session["max_frames"]:
                break
            # Doc §9: kiểm tra cap đĩa trước khi nhận thêm (cache 60s).
            if session["frame_count"] % 10 == 0:
                try:
                    from backend.api.retention import storage_guard
                    g = storage_guard()
                    if g["level"] == "stop":
                        session["error"] = (
                            "Dung lượng media chạm cap cứng "
                            f"({g.get('used_bytes')}/{g.get('cap_bytes')} byte) — "
                            "từ chối nhận thêm, chạy GC rồi start phiên mới.")
                        break
                    if g["level"] == "warn":
                        guard_hits += 1
                        session["degraded"] = (
                            f"media {g.get('ratio', 0):.0%} cap — nên chạy GC")
                        if guard_hits == 1:
                            logger.warning(
                                "phiên %s: media vượt 80%% cap (%s/%s byte)",
                                session["source_id"], g.get("used_bytes"),
                                g.get("cap_bytes"))
                except Exception as exc:
                    logger.debug("storage guard bỏ qua: %s", exc)
            # Drain bộ đệm decoder: grab bỏ frame cũ, retrieve frame mới nhất.
            # Giảm nguy cơ frame trễ do bộ đệm (doc §2).
            grabbed = False
            try:
                for _ in range(DRAIN_GRABS):
                    if cap is not None and cap.grab():
                        grabbed = True
                    else:
                        break
                if grabbed:
                    ok, frame = cap.retrieve()
                else:
                    ok, frame = cap.read() if cap is not None else (False, None)
            except Exception:
                ok, frame = False, None
            t_recv_mono = time.monotonic()
            if not ok or frame is None:
                reconnects += 1
                session["reconnects"] = reconnects
                if reconnects > RECONNECT_MAX:
                    session["error"] = "Mất kết nối camera (quá số lần reconnect)."
                    break
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                time.sleep(RECONNECT_DELAY_SEC)
                if cancel.is_set():
                    break
                try:
                    cap = _open_capture(session["target"])
                    if reconnects > 0:
                        session["_recovered"] = True
                except Exception:
                    continue
                continue
            reconnects = 0
            if session.pop("_recovered", False):
                # Stream vừa reconnect thành công: đóng track mở để timeline
                # thể hiện khoảng trống (§5), tăng độ bất định timestamp.
                session["stream_gap"] = True
                session["stream_gaps"] = int(session.get("stream_gaps", 0)) + 1
            session["frame_count"] += 1
            # Độ trễ đo từ lúc retrieve xong tới lúc consumer xử lý (consumer
            # ghi nhận thêm ingest_latency_ms). timestamp nguồn hiếm khi có
            # trên RTSP/OpenCV nên dùng giờ host đọc + độ bất định.
            item = {"frame_index": session["frame_count"] - 1,
                    "captured_at": _utcnow_iso(),
                    "received_monotonic": t_recv_mono,
                    "frame_bgr": frame}
            _queue_put_newest(session, item)
            # Giữ nhịp sample_fps nhưng không sleep mù: bù thời gian đọc.
            processed_ticks += 1
            elapsed = t_recv_mono - start_mono
            session["effective_fps"] = (processed_ticks / elapsed) if elapsed > 0 else 0.0
            # Doc §9 CPU/GPU quá tải: giảm fps phân tích khi rớt frame nhiều,
            # hồi dần khi hết tải; luôn báo degraded + effective fps.
            if processed_ticks % 30 == 0:
                _adapt_interval(session)
            interval = base_interval * float(session.get("interval_scale", 1.0))
            next_tick += interval
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Quá tải: bỏ nhịp cũ, giữ nhịp công bằng giữa camera.
                next_tick = time.monotonic()
    except Exception as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        session["producer_done"] = True


def _auto_link_enabled() -> bool:
    return os.environ.get("CAMERA_AUTO_LINK", "true").strip().lower() == "true"


def _auto_link_tracklet(session: dict, tracklet_db_id, detections: list):
    """P2 — Liên kết tracklet vừa persist → global ID (best-effort).

    Transaction ngắn riêng, không giữ connection của persist. Thất bại (DB chưa
    migrate 006, xung đột writer, thiếu ứng viên) chỉ đếm stat, không crash
    consumer; tracklet vẫn liên kết được sau qua POST /api/tracklets/.../link.
    Trả action ('assign'/'unresolved'/'conflict'/'error'/'skipped').
    """
    if not _auto_link_enabled() or not detections:
        return "skipped"
    try:
        from datetime import datetime, timezone as _tz

        import numpy as _np

        from backend.api import identity_link as _link
        from backend.api.database import get_pool

        members = []
        for d in detections:
            emb = d.get("embedding")
            if emb is None:
                continue
            try:
                members.append({
                    "embedding": _np.asarray(emb, dtype=_np.float64),
                    "embedding_id": None, "crop_id": d.get("crop_id"),
                    "quality": float(d.get("quality", 0.5)),
                })
            except Exception:
                continue
        if not members:
            session["link_unresolved"] = int(session.get("link_unresolved", 0)) + 1
            return "unresolved"
        descriptor = _link.tracklet_descriptor(members)
        with get_pool().connection() as conn:
            result = _link.link_tracklet(
                conn, tracklet_id=str(tracklet_db_id),
                descriptor=descriptor, camera_id=session.get("camera_id"),
                at_time=datetime.now(_tz.utc),
                config=_link.LinkConfig(), site="default")
        action = result.get("action")
        if action == "assign":
            session["linked"] = int(session.get("linked", 0)) + 1
        elif action == "conflict":
            session["link_conflicts"] = int(session.get("link_conflicts", 0)) + 1
        else:
            session["link_unresolved"] = int(session.get("link_unresolved", 0)) + 1
        return action or "unresolved"
    except Exception as exc:
        session["link_errors"] = int(session.get("link_errors", 0)) + 1
        session["last_link_error"] = f"{type(exc).__name__}: {exc}"[:300]
        logger.debug("auto-link tracklet bỏ qua: %s", exc)
        return "error"


def _persist_ready_tracklets(session: dict, tracks: list, *, force_note: str = "",
                             is_checkpoint: bool = False,
                             tracker=None) -> None:
    """Persist từng tracklet đóng (hoặc checkpoint sớm) bằng transaction ngắn."""
    if not tracks:
        return
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    from backend.api.ingest import persist_camera_tracklet
    from backend.api.storage import get_storage

    storage = get_storage()
    # Độ bất định sau reconnect (§5): áp dụng một lần cho đợt flush sau gap.
    gap_ms = session.pop("gap_uncertainty_ms", None)
    for tr in tracks:
        if not tr.candidates:
            if is_checkpoint and tracker is not None:
                try:
                    tracker.release_checkpoint_candidates(tr)
                except Exception:
                    pass
            continue
        # Chuyển Candidate RAM -> dict persist (crop_bgr đã là bản copy).
        cands = []
        for c in sorted(tr.candidates, key=lambda x: x.quality, reverse=True):
            cands.append({
                "crop_bgr": c.crop_bgr, "bbox": list(c.bbox),
                "det_score": float(c.det_score), "quality": float(c.quality),
                "blur": float(c.blur), "exposure": float(c.exposure),
                "frontal": float(c.frontal), "face_area": float(c.face_area),
                "embedding": c.embedding, "kps": None,
            })
        # Frame đại diện: dùng bbox tốt nhất để suy kích thước nguồn; offset 0
        # vì tracklet gom nhiều frame (timeline thể hiện qua tracklet times).
        best = cands[0]
        try:
            with get_pool().connection() as conn:
                # Transaction ngắn: 1 frame metadata + N detection/crop/vector.
                res = persist_camera_tracklet(
                    conn, storage, source_id=session["source_id"],
                    camera_id=session["camera_id"],
                    track_local_id=tr.local_id,
                    frame_index=session.get("persisted_frames", 0),
                    offset_ms=0,
                    captured_at=_utcnow_iso(), received_at=_utcnow_iso(),
                    source_timestamp=None,
                    timestamp_uncertainty_ms=gap_ms,
                    frame_width=int(session.get("last_width", 1280)),
                    frame_height=int(session.get("last_height", 720)),
                    candidates=cands,
                    detector_name="insightface",
                    detector_version=session.get("det_version", "buffalo_l"),
                    run_id=session["run_key"],
                    close=(not is_checkpoint),
                    slot_revision=int(getattr(tr, "slot_revision", 0)),
                    observation_count=int(getattr(tr, "observation_count", 0)),
                    preprocessing_base={"source": "camera_capture",
                                        "camera_id": session["camera_id"],
                                        "track_local_id": tr.local_id,
                                        "observations": tr.observation_count,
                                        **({"note": force_note} if force_note else {})})
                _ = best
                session["persisted_frames"] = int(session.get("persisted_frames", 0)) + 1
                session["faces_found"] = int(session.get("faces_found", 0)) + len(res["detections"])
                session["crops_persisted"] = int(session.get("crops_persisted", 0)) + len(res["detections"])
                if is_checkpoint:
                    session["checkpoints"] = int(session.get("checkpoints", 0)) + 1
                    if tracker is not None:
                        try:
                            tracker.release_checkpoint_candidates(tr)
                        except Exception:
                            pass
                else:
                    session["tracks_closed"] = int(session.get("tracks_closed", 0)) + 1
                # P2: auto-link best-effort sang global ID (transaction riêng).
                # Tracklet đã link ở checkpoint thì giữ quyết định sớm, không
                # ghi đè (một writer/tracklet, §5.6).
                tid = res.get("tracklet_id") or tr.local_id
                linked_set = session.setdefault("linked_tracklets", set())
                if tid in linked_set and not is_checkpoint:
                    pass
                else:
                    action = _auto_link_tracklet(session, tid, res["detections"])
                    if action == "assign":
                        linked_set.add(tid)
        except Exception as exc:
            # DB/storage lỗi (§9): spool đĩa chỉ crop đã chọn + metadata (cap +
            # TTL, không spool frame/video). Spool đầy → báo mất, persisted≠true.
            try:
                from backend.api import spool as _spool
                out = _spool.write_spool(
                    cands=cands,
                    meta={"source_id": session["source_id"],
                          "camera_id": session["camera_id"],
                          "track_local_id": tr.local_id,
                          "captured_at": _utcnow_iso(),
                          "received_at": _utcnow_iso(),
                          "timestamp_uncertainty_ms": gap_ms,
                          "frame_width": int(session.get("last_width", 1280)),
                          "frame_height": int(session.get("last_height", 720)),
                          "detector_name": "insightface",
                          "detector_version": session.get("det_version", "buffalo_l"),
                          "run_id": session.get("run_key", ""),
                          "close": (not is_checkpoint),
                          "slot_revision": int(getattr(tr, "slot_revision", 0)),
                          "observation_count": int(getattr(tr, "observation_count", 0)),
                          "preprocessing_base": {
                              "source": "camera_capture",
                              "camera_id": session["camera_id"],
                              "track_local_id": tr.local_id,
                              "observations": tr.observation_count,
                              **({"note": force_note} if force_note else {})}})
                if out.get("spooled"):
                    session["spooled"] = int(session.get("spooled", 0)) + 1
                else:
                    session["spool_dropped"] = int(
                        session.get("spool_dropped", 0)) + int(out.get("dropped", 0))
                    session["last_spool_error"] = str(out.get("reason", ""))[:300]
            except Exception as exc2:
                logger.debug("spool bỏ qua: %s", exc2)
            session["persist_errors"] = int(session.get("persist_errors", 0)) + 1
            session["last_persist_error"] = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning("persist tracklet %s bỏ qua: %s", tr.local_id, exc)


def _drain_tracker(session: dict, tracker) -> None:
    """Flush track đóng + checkpoint sớm (transaction ngắn từng tracklet)."""
    # Stream gap sau reconnect (§5): đóng track mở để timeline thể hiện khoảng
    # trống thay vì nối tracklet qua đoạn mất hình; tăng độ bất định timestamp.
    if session.pop("stream_gap", False):
        tracker.close_all()
        gaps = int(session.get("stream_gaps", 1))
        session["gap_uncertainty_ms"] = min(30000, 2000 * max(1, gaps))
    # Redrive spool tồn đọng trước (§9), bounded mỗi nhịp.
    try:
        from backend.api import spool as _spool
        _spool.redrive(session)
    except Exception as exc:
        logger.debug("redrive bỏ qua: %s", exc)
    ready = tracker.pop_closed_ready()
    _persist_ready_tracklets(session, ready)
    cps = tracker.pop_checkpoints()
    if cps:
        _persist_ready_tracklets(session, cps, is_checkpoint=True, tracker=tracker)


def _consumer(session: dict) -> None:
    """Detect + tracking trong RAM; persist khi tracklet đóng/checkpoint (bounded)."""
    from backend.api import repositories as repo
    from backend.api.camera_tracks import CameraTracker
    from backend.api.database import get_pool
    from backend.api.ingest import detect_faces_in_frame

    q: queue.Queue = session["queue"]
    cancel: threading.Event = session["cancel"]
    embedder = session["embedder"]
    session["det_version"] = getattr(embedder, "model_name", "buffalo_l")
    session["run_key"] = str(uuid.uuid4())
    tracker = CameraTracker()
    session["tracker"] = tracker
    processed = 0
    try:
        while True:
            if cancel.is_set() and q.empty():
                break
            if session.get("producer_done") and q.empty():
                break
            try:
                item = q.get(timeout=0.5)
                session["queue_bytes"] = max(
                    0, int(session.get("queue_bytes", 0)) - int(item.get("bytes", 0)))
            except queue.Empty:
                # Nhịp flush tracklet timeout ngay cả khi không có frame mới.
                _drain_tracker(session, tracker)
                if session.get("error") or session.get("producer_done"):
                    break
                continue
            try:
                h, w = item["frame_bgr"].shape[:2]
                session["last_width"], session["last_height"] = int(w), int(h)
            except Exception:
                pass
            faces, conditions, _original = detect_faces_in_frame(
                embedder, item["frame_bgr"],
                preprocess_on=session["preprocess_on"])
            # Ghi nhận độ trễ host-đọc → xử lý (doc §5: đo độ trễ).
            try:
                latency_ms = (time.monotonic() - float(item["received_monotonic"])) * 1000.0
                session["ingest_latency_ms"] = round(latency_ms, 1)
            except Exception:
                pass
            for cond in conditions:
                session.setdefault("conditions", {}).setdefault(cond, 0)
                session["conditions"][cond] += 1
            tracker.update(item["frame_bgr"], faces)
            # Giải phóng frame RAM ngay sau inference/crop (P0): xóa tham chiếu.
            try:
                del item["frame_bgr"]
            except KeyError:
                pass
            del item
            processed += 1
            session["frames_sampled"] = processed
            session["active_tracks"] = len(tracker.open_tracks())
            # Persist tracklet vừa đóng/checkpoint — transaction ngắn, trả
            # connection trước khi chờ frame tiếp theo (doc §2).
            _drain_tracker(session, tracker)
            if processed % 5 == 0:
                try:
                    with get_pool().connection() as progress_conn:
                        repo.update_ingestion_run(
                            progress_conn, session["run_id"],
                            frames_sampled=processed,
                            faces_found=int(session.get("faces_found", 0)))
                except Exception:
                    pass
        # Flush cuối: đóng mọi track mở còn ứng viên (kết thúc phiên).
        remaining = []
        for tr in tracker.open_tracks():
            tr.closed = True
            if tr.candidates:
                remaining.append(tr)
        _persist_ready_tracklets(session, remaining, force_note="session_flush")
        # Đóng track rỗng để giải phóng RAM.
        tracker.pop_closed_ready()
    except Exception as exc:
        session["error"] = session.get("error") or f"{type(exc).__name__}: {exc}"
    finally:
        session["consumer_done"] = True


def start_capture(
    *, camera_id: str, sample_fps: float = 1.0, max_frames: int = 900,
    capture_seconds: Optional[float] = None, preprocess_on: bool = True,
) -> dict:
    """Mở phiên capture crop-only mới. Trả {source_id, run_id, camera_id}."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    from backend.api.ingest import get_ingest_embedder

    with get_pool().connection() as conn:
        camera = repo.get_camera(conn, camera_id)
    target = resolve_camera_target(
        _camera_secret_ref(camera_id))
    embedder = get_ingest_embedder()
    source_id = str(uuid.uuid4())
    started = _utcnow_iso()
    with get_pool().connection() as conn:
        try:
            repo.create_source(
                conn, purpose="search", kind="camera", camera_id=camera_id,
                started_at=started, source_id=source_id, storage_policy="crop_only")
        except TypeError:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO face_media.sources
                        (id, purpose, kind, camera_id, started_at)
                    VALUES (%s, 'search', 'camera', %s, %s)
                    """,
                    (source_id, camera_id, started))
        run_id = repo.create_ingestion_run(
            conn, source_id=source_id, fps_target=sample_fps,
            max_frames=max_frames)
        repo.update_ingestion_run(conn, run_id, status="running")
    # P0: đường camera crop-only KHÔNG tạo recorder video toàn phiên.
    if os.environ.get("CAMERA_SAVE_SESSION_VIDEO", "false").strip().lower() == "true":
        logger.warning(
            "CAMERA_SAVE_SESSION_VIDEO=true bị bỏ qua: phiên %s crop-only "
            "(không ghi video toàn khung).", source_id)
    session = {
        "source_id": source_id, "camera_id": camera_id, "run_id": run_id,
        "target": target, "target_redacted": redact_url(target)
        if not target.isdigit() else f"webcam:{target}",
        "sample_fps": sample_fps, "max_frames": max_frames,
        "deadline": (time.time() + capture_seconds) if capture_seconds else None,
        "preprocess_on": preprocess_on, "embedder": embedder,
        "queue": queue.Queue(maxsize=QUEUE_MAXSIZE),
        "cancel": threading.Event(), "frame_count": 0,
        "frames_sampled": 0, "faces_found": 0, "dropped": 0,
        "queue_bytes": 0, "oversize_frames": 0,
        "persisted_frames": 0, "crops_persisted": 0, "tracks_closed": 0,
        "checkpoints": 0, "spooled": 0, "spool_dropped": 0,
        "spooled_redriven": 0, "spool_errors": 0, "last_spool_error": None,
        "persist_errors": 0, "last_persist_error": None,
        "linked": 0, "link_unresolved": 0, "link_conflicts": 0,
        "link_errors": 0, "last_link_error": None,
        "stream_gap": False, "stream_gaps": 0,
        "active_tracks": 0, "effective_fps": 0.0, "ingest_latency_ms": None,
        "conditions": {},
        "reconnects": 0, "error": None,
        "producer_done": False, "consumer_done": False,
        # P0: không recorder; giữ key record_path=None để API cũ không vỡ.
        "recorder": None, "record_path": None,
        "storage_policy": "crop_only", "frame_available": False,
        "camera_name": camera["name"],
    }
    producer = threading.Thread(target=_producer, args=(session,), daemon=True)
    consumer = threading.Thread(target=_consumer, args=(session,), daemon=True)
    session["threads"] = (producer, consumer)
    with _SESSIONS_LOCK:
        _SESSIONS[source_id] = session
    producer.start()
    consumer.start()
    return {"source_id": source_id, "run_id": run_id, "camera_id": camera_id}


def _camera_secret_ref(camera_id: str) -> Optional[str]:
    from backend.api import repositories as repo
    from backend.api.database import get_pool

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT connection_secret_ref FROM face_media.cameras WHERE id = %s",
                (camera_id,))
            row = cur.fetchone()
    return row[0] if row else None


def stop_capture(source_id: str) -> dict:
    """Dừng phiên: đóng capture, ghi ended_at, chốt run. Luôn đóng thiết bị."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool

    with _SESSIONS_LOCK:
        session = _SESSIONS.pop(source_id, None)
    if session is None:
        # Phiên không còn trong RAM (restart?) — chốt ended_at best-effort.
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE face_media.sources SET ended_at = now()
                    WHERE id = %s AND ended_at IS NULL
                    """,
                    (source_id,))
        return {"source_id": source_id, "status": "stopped"}
    session["cancel"].set()
    for thread in session.get("threads", ()):
        thread.join(timeout=10)
    recorder = session.get("recorder")
    if recorder is not None:
        try:
            recorder.release()
        except Exception:
            pass
    error = session.get("error") or session.get("last_persist_error")
    status = "error" if session.get("error") else "done"
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE face_media.sources SET ended_at = now()
                WHERE id = %s AND ended_at IS NULL
                """,
                (source_id,))
        try:
            repo.update_ingestion_run(
                conn, session["run_id"], status=status,
                frames_sampled=session["frames_sampled"],
                faces_found=session["faces_found"],
                error_message=(error[:2000] if error else None))
        except Exception:
            pass
    return {
        "source_id": source_id, "status": "stopped",
        "run_status": status,
        "frames_sampled": session["frames_sampled"],
        "faces_found": session["faces_found"],
        "dropped_frames": session["dropped"],
        "reconnects": session["reconnects"],
        # P0: crop-only — không có file recording phiên.
        "record_path": None,
        "storage_policy": "crop_only",
        "frame_available": False,
        "crops_persisted": session.get("crops_persisted", 0),
        "tracks_closed": session.get("tracks_closed", 0),
        "checkpoints": session.get("checkpoints", 0),
        "spooled": session.get("spooled", 0),
        "spool_dropped": session.get("spool_dropped", 0),
        "spooled_redriven": session.get("spooled_redriven", 0),
        "spool_errors": session.get("spool_errors", 0),
        "stream_gaps": session.get("stream_gaps", 0),
        "persist_errors": session.get("persist_errors", 0),
        "linked": session.get("linked", 0),
        "link_unresolved": session.get("link_unresolved", 0),
        "link_conflicts": session.get("link_conflicts", 0),
        "link_errors": session.get("link_errors", 0),
        "degraded": session.get("degraded"),
        "effective_fps": round(float(session.get("effective_fps", 0.0)), 2),
        "interval_scale": round(float(session.get("interval_scale", 1.0)), 2),
        "ingest_latency_ms": session.get("ingest_latency_ms"),
        "error_message": error,
    }


def capture_status(source_id: str) -> dict:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(source_id)
        if session is None:
            raise LookupError(f"Phiên capture không còn hoạt động: {source_id}")
        live = not (session.get("producer_done") and session.get("consumer_done"))
        return {
            "source_id": source_id, "camera_id": session["camera_id"],
            "run_id": session["run_id"],
            "status": "error" if session.get("error")
            else ("running" if live else "finishing"),
            "frames_sampled": session["frames_sampled"],
            "faces_found": session["faces_found"],
            "dropped_frames": session["dropped"],
            "reconnects": session["reconnects"],
            "error_message": session.get("error"),
        }
