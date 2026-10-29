"""Thumbnail tạo khi đọc, cache RAM giới hạn (doc §4.2).

Không lưu thêm một bản thumbnail mặc định trên storage — chỉ tạo theo yêu
cầu và giữ trong cache bounded (LRU + TTL + cap byte).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

THUMB_CACHE_BYTES = int(os.environ.get("THUMB_CACHE_BYTES", str(32 * 1024 * 1024)))
THUMB_TTL_SEC = float(os.environ.get("THUMB_TTL_SEC", "300"))
THUMB_QUALITY = 75

_CACHE: OrderedDict = OrderedDict()
_BYTES = 0
_LOCK = threading.Lock()


def _evict_locked(need: int) -> None:
    global _BYTES
    now = time.time()
    # Hết TTL trước.
    for key in [k for k, (_, _, ts) in _CACHE.items() if now - ts > THUMB_TTL_SEC]:
        data, _, _ = _CACHE.pop(key)
        _BYTES -= len(data)
    # LRU tới khi đủ chỗ.
    while _CACHE and _BYTES + need > THUMB_CACHE_BYTES:
        _, (data, _, _) = _CACHE.popitem(last=False)
        _BYTES -= len(data)


def get_thumb(storage, crop_id: str, storage_key: str, edge: int = 128) -> bytes:
    """Trả JPEG thumbnail (cache hit hoặc tạo mới). Raise FileNotFoundError."""
    import cv2
    import numpy as _np

    edge = max(32, min(int(edge), 512))
    key = (str(crop_id), edge)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            data, _, ts = hit
            if time.time() - ts <= THUMB_TTL_SEC:
                _CACHE.move_to_end(key)
                _CACHE[key] = (data, len(data), time.time())
                return data
            _CACHE.pop(key, None)
    with storage.open(storage_key) as handle:
        raw = handle.read()
    img = cv2.imdecode(_np.frombuffer(raw, _np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không giải mã được crop: {crop_id}")
    h, w = img.shape[:2]
    if max(h, w) > edge:
        s = edge / float(max(h, w))
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), THUMB_QUALITY])
    if not ok:
        raise IOError("Encode thumbnail thất bại.")
    data = bytes(buf)
    global _BYTES
    with _LOCK:
        if len(data) <= THUMB_CACHE_BYTES:
            _evict_locked(len(data))
            _CACHE[key] = (data, len(data), time.time())
            _CACHE.move_to_end(key)
            _BYTES += len(data)
    return data


def invalidate(crop_id: str) -> None:
    """Xóa cache của 1 crop (gọi sau GC xóa crop)."""
    global _BYTES
    with _LOCK:
        for key in [k for k in _CACHE if k[0] == str(crop_id)]:
            data, _, _ = _CACHE.pop(key)
            _BYTES -= len(data)


def stats() -> dict:
    with _LOCK:
        return {"entries": len(_CACHE), "bytes": _BYTES,
                "cap_bytes": THUMB_CACHE_BYTES, "ttl_sec": THUMB_TTL_SEC}
