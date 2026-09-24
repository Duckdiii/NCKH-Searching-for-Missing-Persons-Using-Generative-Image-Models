"""T09 — Thu nhận camera: quản lý cameras + phiên capture (source search/camera).

- Mỗi phiên start tạo MỘT source search/camera (camera_id + started_at UTC).
- URL/credential camera KHÔNG nằm trong DB: bảng chỉ giữ connection_secret_ref
  (tên biến môi trường chứa URL, đọc phía backend). Log/T09 status không bao giờ
  in URL gốc (redact user:pass).
- Frame index tăng trong phiên; captured_at UTC từng frame. Tái dùng ingest
  frame của T08. Hàng đợi có chặn (maxsize) + đếm frame rớt (backpressure) để
  RAM không tăng vô hạn. Mất kết nối → reconnect giới hạn lần, ghi trạng thái.
- Chính sách lưu video toàn phiên: mặc định TẮT (CAMERA_SAVE_SESSION_VIDEO=false);
  khi bật, ghi file recording local ngoài face_media (đường dẫn trả trong status).
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

QUEUE_MAXSIZE = 8
RECONNECT_MAX = 5
RECONNECT_DELAY_SEC = 2.0


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


def _producer(session: dict) -> None:
    """Đọc frame theo nhịp sample_fps, đẩy vào queue (rớt khi đầy)."""
    q: queue.Queue = session["queue"]
    cancel: threading.Event = session["cancel"]
    interval = 1.0 / max(float(session["sample_fps"]), 0.01)
    deadline = session.get("deadline")
    reconnects = 0
    cap = None
    try:
        cap = _open_capture(session["target"])
        while not cancel.is_set():
            if deadline is not None and time.time() >= deadline:
                break
            if session["frame_count"] >= session["max_frames"]:
                break
            ok, frame = cap.read() if cap is not None else (False, None)
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
                except Exception:
                    continue
                continue
            reconnects = 0
            session["frame_count"] += 1
            item = {"frame_index": session["frame_count"] - 1,
                    "captured_at": _utcnow_iso(), "frame_bgr": frame}
            try:
                q.put(item, timeout=interval)
            except queue.Full:
                session["dropped"] += 1
            if session.get("recorder") is not None:
                try:
                    session["recorder"].write(frame)
                except Exception:
                    pass
            time.sleep(interval)
    except Exception as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        session["producer_done"] = True


def _consumer(session: dict) -> None:
    """Lấy frame từ queue, detect + persist (bounded, không tăng RAM vô hạn)."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    from backend.api.ingest import detect_faces_in_frame, persist_observation_frame
    from backend.api.storage import get_storage

    q: queue.Queue = session["queue"]
    cancel: threading.Event = session["cancel"]
    storage = get_storage()
    embedder = session["embedder"]
    det_version = getattr(embedder, "model_name", "buffalo_l")
    run_key = str(uuid.uuid4())
    processed = 0
    try:
        with get_pool().connection() as conn:
            while True:
                if cancel.is_set() and q.empty():
                    break
                if session.get("producer_done") and q.empty():
                    break
                try:
                    item = q.get(timeout=0.5)
                except queue.Empty:
                    if session.get("error") or session.get("producer_done"):
                        break
                    continue
                faces, conditions, original = detect_faces_in_frame(
                    embedder, item["frame_bgr"],
                    preprocess_on=session["preprocess_on"])
                frame = persist_observation_frame(
                    conn, storage, source_id=session["source_id"],
                    purpose="search", frame_bgr=original,
                    frame_index=item["frame_index"], offset_ms=0,
                    captured_at=item["captured_at"], faces=faces,
                    detector_name="insightface", detector_version=det_version,
                    run_id=run_key,
                    preprocessing_base={"source": "camera_capture",
                                        "camera_id": session["camera_id"],
                                        "conditions": conditions})
                processed += 1
                session["frames_sampled"] = processed
                session["faces_found"] += len(frame["detections"])
                if processed % 5 == 0:
                    with get_pool().connection() as progress_conn:
                        repo.update_ingestion_run(
                            progress_conn, session["run_id"],
                            frames_sampled=processed,
                            faces_found=session["faces_found"])
    except Exception as exc:
        session["error"] = session.get("error") or f"{type(exc).__name__}: {exc}"
    finally:
        session["consumer_done"] = True


def start_capture(
    *, camera_id: str, sample_fps: float = 1.0, max_frames: int = 900,
    capture_seconds: Optional[float] = None, preprocess_on: bool = True,
) -> dict:
    """Mở phiên capture mới. Trả {source_id, run_id, camera_id}."""
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
    recorder = None
    record_path = None
    if os.environ.get("CAMERA_SAVE_SESSION_VIDEO", "false").strip().lower() == "true":
        os.makedirs(os.path.join("outputs", "camera_sessions"), exist_ok=True)
        record_path = os.path.join("outputs", "camera_sessions", f"{source_id}.mp4")
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
        "reconnects": 0, "error": None,
        "producer_done": False, "consumer_done": False,
        "recorder": recorder, "record_path": record_path,
        "camera_name": camera["name"],
    }
    if record_path is not None:
        session["recorder"] = cv2.VideoWriter(
            record_path, cv2.VideoWriter_fourcc(*"mp4v"),
            max(sample_fps, 1.0), (640, 480))
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
    error = session.get("error")
    status = "error" if error else "done"
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
        "record_path": session.get("record_path"),
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
