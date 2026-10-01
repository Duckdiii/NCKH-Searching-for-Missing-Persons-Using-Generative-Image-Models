"""Spool đĩa khi DB/storage lỗi (doc §9).

Hành vi yêu cầu: spool đĩa CHỈ crop đã chọn + metadata (không spool
frame/video), có cap byte + TTL; spool đầy → dừng ghi/bỏ mẫu theo ưu tiên,
báo mất dữ liệu, không trả persisted=true.

- write_spool(): encode crop_bgr → JPEG (dùng đúng camera_tracks.encode_crop_jpeg
  để đồng nhất Q85/256px), ghi {id}.json + {id}_{i}.jpg. Đo trước khi ghi
  (reserve): nếu vượt cap → evict file cũ nhất cho đến khi đủ chỗ hoặc hết;
  không đủ → bỏ mẫu kém nhất (quality thấp nhất) và đếm spool_dropped.
- TTL: file quá SPOOL_TTL_SEC bị coi hết hạn — redrive bỏ qua + xóa.
- redrive(): đọc tối đa N spool cũ nhất, dựng lại candidates, gọi
  persist_camera_tracklet bằng connection mới; thành công → xóa file.
  Tất cả best-effort, không raise ra consumer.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)

SPOOL_DIR = os.environ.get("CAMERA_SPOOL_DIR", os.path.join("outputs", "spool"))
SPOOL_MAX_BYTES = int(os.environ.get("CAMERA_SPOOL_MAX_BYTES", str(512 * 1024 * 1024)))
SPOOL_TTL_SEC = float(os.environ.get("CAMERA_SPOOL_TTL_SEC", "3600"))
REDRIVE_PER_CYCLE = int(os.environ.get("CAMERA_SPOOL_REDRIVE", "3"))


def _root() -> str:
    root = SPOOL_DIR
    os.makedirs(root, exist_ok=True)
    return root


def _size_of(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def spool_usage() -> dict:
    """{bytes, files} hiện tại (bỏ qua file đang ghi .tmp)."""
    total, n = 0, 0
    try:
        for f in os.listdir(_root()):
            if f.endswith(".tmp"):
                continue
            p = os.path.join(_root(), f)
            if os.path.isfile(p):
                total += _size_of(p)
                n += 1
    except OSError:
        pass
    return {"bytes": total, "files": n}


def _evict_oldest(need: int) -> int:
    """Xóa file cũ nhất đến khi đủ chỗ cho `need` byte. Trả byte đã giải phóng."""
    freed = 0
    try:
        entries = [(os.path.getmtime(os.path.join(_root(), f)), f)
                   for f in os.listdir(_root())]
    except OSError:
        return 0
    for _, f in sorted(entries):
        if spool_usage()["bytes"] + need <= SPOOL_MAX_BYTES:
            break
        p = os.path.join(_root(), f)
        freed += _size_of(p)
        try:
            os.remove(p)
        except OSError:
            pass
    return freed


def _expired(path: str, now: float) -> bool:
    try:
        return now - os.path.getmtime(path) > SPOOL_TTL_SEC
    except OSError:
        return True


def write_spool(*, cands: list[dict], meta: dict) -> dict:
    """Spool 1 tracklet. Trả {spooled, dropped, reason}.

    cands: dict persist (crop_bgr, bbox, det_score, quality, ..., embedding
    ndarray hoặc None). meta: source/camera/track/frame/detector/run/slot...
    """
    from backend.api import camera_tracks as ct

    sid = uuid.uuid4().hex
    jpgs: list[tuple[str, bytes]] = []
    total = 0
    # Bỏ mẫu kém nhất trước nếu cần: sắp quality giảm dần, cắt từ cuối khi đầy.
    ordered = sorted(cands, key=lambda c: float(c.get("quality", 0)), reverse=True)
    try:
        for i, c in enumerate(ordered):
            data, _prov = ct.encode_crop_jpeg(c["crop_bgr"])
            jpgs.append((f"{sid}_{i}.jpg", data))
            total += len(data)
    except Exception as exc:
        return {"spooled": False, "dropped": len(cands),
                "reason": f"encode: {type(exc).__name__}"}
    meta_doc = {
        "sid": sid, "created_at": time.time(),
        "meta": meta,
        "cands": [
            {k: (v.tolist() if k == "embedding" and v is not None
                 and hasattr(v, "tolist") else v)
             for k, v in c.items() if k != "crop_bgr"}
            for c in ordered[:len(jpgs)]
        ],
    }
    doc = json.dumps(meta_doc).encode("utf-8")
    total += len(doc)
    usage = spool_usage()
    if usage["bytes"] + total > SPOOL_MAX_BYTES:
        _evict_oldest(total)
        usage = spool_usage()
    if usage["bytes"] + total > SPOOL_MAX_BYTES:
        return {"spooled": False, "dropped": len(cands),
                "reason": "spool_full"}
    try:
        root = _root()
        with open(os.path.join(root, f"{sid}.json.tmp"), "wb") as fh:
            fh.write(doc)
        for name, data in jpgs:
            with open(os.path.join(root, name + ".tmp"), "wb") as fh:
                fh.write(data)
        # Publish nguyên tử: đổi tên sau khi ghi xong hết.
        os.replace(os.path.join(root, f"{sid}.json.tmp"),
                   os.path.join(root, f"{sid}.json"))
        for name, _ in jpgs:
            os.replace(os.path.join(root, name + ".tmp"), os.path.join(root, name))
    except Exception as exc:
        for f in [f"{sid}.json.tmp"] + [n + ".tmp" for n, _ in jpgs] + \
                 [f"{sid}.json"] + [n for n, _ in jpgs]:
            try:
                os.remove(os.path.join(_root(), f))
            except OSError:
                pass
        return {"spooled": False, "dropped": len(cands),
                "reason": f"write: {type(exc).__name__}"}
    return {"spooled": True, "sid": sid, "dropped": 0,
            "bytes": total, "crops": len(jpgs)}


def _load_spool(sid: str):
    import numpy as _np
    root = _root()
    with open(os.path.join(root, f"{sid}.json"), "rb") as fh:
        doc = json.loads(fh.read().decode("utf-8"))
    cands = []
    for i, c in enumerate(doc["cands"]):
        p = os.path.join(root, f"{sid}_{i}.jpg")
        with open(p, "rb") as fh:
            data = fh.read()
        import cv2 as _cv2
        img = _cv2.imdecode(_np.frombuffer(data, _np.uint8), _cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"spool crop hỏng: {sid}_{i}.jpg")
        emb = c.get("embedding")
        cands.append({**c, "crop_bgr": img,
                      "embedding": _np.asarray(emb, dtype=_np.float64)
                      if emb is not None else None})
    return doc["meta"], cands


def _remove_spool(sid: str, n_crops: int) -> None:
    root = _root()
    for f in [f"{sid}.json"] + [f"{sid}_{i}.jpg" for i in range(n_crops)]:
        try:
            os.remove(os.path.join(root, f))
        except OSError:
            pass


def redrive(session: dict, *, max_files: int = REDRIVE_PER_CYCLE) -> dict:
    """Thử persist lại spool tồn đọng (gọi mỗi nhịp flush, bounded).

    Cập nhật session: spooled_redriven / spool_errors. Không raise.
    """
    from backend.api.database import get_pool
    from backend.api.ingest import persist_camera_tracklet
    from backend.api.storage import get_storage

    done, failed, expired = 0, 0, 0
    try:
        sids = sorted(f[:-5] for f in os.listdir(_root())
                      if f.endswith(".json"))
    except OSError:
        return {"redriven": 0, "failed": 0, "expired": 0}
    now = time.time()
    for sid in sids[:max_files]:
        jp = os.path.join(_root(), f"{sid}.json")
        if _expired(jp, now):
            try:
                import json as _json
                with open(jp, encoding="utf-8") as fh:
                    n = len(_json.load(fh).get("cands", []))
            except Exception:
                n = 0
            _remove_spool(sid, n)
            expired += 1
            continue
        try:
            meta, cands = _load_spool(sid)
            if meta.get("source_id") != session.get("source_id"):
                continue  # spool của phiên khác — worker phiên đó xử lý
            storage = get_storage()
            with get_pool().connection() as conn:
                res = persist_camera_tracklet(
                    conn, storage, source_id=meta["source_id"],
                    camera_id=meta.get("camera_id"),
                    track_local_id=meta["track_local_id"],
                    frame_index=int(session.get("persisted_frames", 0)),
                    offset_ms=0, captured_at=meta.get("captured_at"),
                    received_at=meta.get("received_at"),
                    source_timestamp=None,
                    timestamp_uncertainty_ms=meta.get("timestamp_uncertainty_ms"),
                    frame_width=int(meta.get("frame_width", 1280)),
                    frame_height=int(meta.get("frame_height", 720)),
                    candidates=cands, detector_name=meta.get("detector_name", "insightface"),
                    detector_version=meta.get("detector_version", "buffalo_l"),
                    run_id=session.get("run_key") or meta.get("run_id", ""),
                    close=bool(meta.get("close", True)),
                    slot_revision=int(meta.get("slot_revision", 0)),
                    observation_count=int(meta.get("observation_count", 0)),
                    preprocessing_base=meta.get("preprocessing_base"))
            _remove_spool(sid, len(cands))
            done += 1
            session["persisted_frames"] = int(session.get("persisted_frames", 0)) + 1
            session["faces_found"] = int(session.get("faces_found", 0)) + len(res["detections"])
            session["crops_persisted"] = int(session.get("crops_persisted", 0)) + len(res["detections"])
        except Exception as exc:
            failed += 1
            logger.debug("redrive %s bỏ qua: %s", sid, exc)
    session["spooled_redriven"] = int(session.get("spooled_redriven", 0)) + done
    if failed:
        session["spool_errors"] = int(session.get("spool_errors", 0)) + failed
    return {"redriven": done, "failed": failed, "expired": expired}
