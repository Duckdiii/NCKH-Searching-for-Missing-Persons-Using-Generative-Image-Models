"""Admission control GPU chung cho camera inference + diffusion (doc §3).

Vấn đề: mutex diffusion (PIPELINE_LOCK) chỉ điều phối diffusion với nhau,
không thấy tải inference camera — chung GPU ít VRAM có thể OOM khi diffusion
chạy cùng lúc nhiều phiên camera đang detect/embed.

Chính sách:
- Diffusion xin slot qua acquire_diffusion_slot(): kiểm tra lock diffusion
  (giữ nguyên ngữ nghĩa 409 hiện tại), rồi kiểm tra tải camera + VRAM headroom.
- Mặc định KHÔNG đổi hành vi API: bận → 409 ngay (DIFFUSION_QUEUE_ON_BUSY=false).
  Đặt DIFFUSION_QUEUE_ON_BUSY=true để xếp diffusion chờ tới
  DIFFUSION_QUEUE_TIMEOUT_SEC thay vì 409 ngay.
- VRAM đo best-effort (torch → nvidia-smi → unknown-cho qua); không bao giờ
  crash request vì đo VRAM thất bại.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

# Ngưỡng thử nghiệm (doc: cần điều chỉnh theo GPU triển khai).
VRAM_MIN_FREE_MB = float(os.environ.get("GPU_MIN_FREE_MB", "1536"))
VRAM_MIN_FREE_RATIO = float(os.environ.get("GPU_MIN_FREE_RATIO", "0.15"))
QUEUE_ON_BUSY = os.environ.get("DIFFUSION_QUEUE_ON_BUSY", "false").strip().lower() == "true"
QUEUE_TIMEOUT_SEC = float(os.environ.get("DIFFUSION_QUEUE_TIMEOUT_SEC", "300"))
QUEUE_POLL_SEC = 0.5

_waiters = 0
_waiters_lock = threading.Lock()
_vram_warned = False


def diffusion_busy() -> bool:
    try:
        from backend.api.job_runner import PIPELINE_LOCK
        return bool(PIPELINE_LOCK.locked())
    except Exception:
        return False


def camera_inference_load() -> dict:
    """Tải inference camera hiện tại (số phiên live + queue bytes)."""
    try:
        from backend.api import cameras as cam
        with cam._SESSIONS_LOCK:
            live = [s for s in cam._SESSIONS.values()
                    if not (s.get("producer_done") and s.get("consumer_done"))]
            return {"sessions": len(live),
                    "queue_bytes": sum(int(s.get("queue_bytes", 0)) for s in live),
                    "active_tracks": sum(int(s.get("active_tracks", 0)) for s in live)}
    except Exception:
        return {"sessions": 0, "queue_bytes": 0, "active_tracks": 0}


def vram_info() -> dict:
    """VRAM còn trống best-effort: {free_mb, total_mb, source} hoặc {unknown}."""
    global _vram_warned
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {"free_mb": round(free / 1024 / 1024, 1),
                    "total_mb": round(total / 1024 / 1024, 1),
                    "source": "torch.cuda"}
    except Exception as exc:
        logger.debug("đo VRAM qua torch thất bại: %s", exc)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            free_s, total_s = out.stdout.strip().splitlines()[0].split(",")
            return {"free_mb": float(free_s.strip()),
                    "total_mb": float(total_s.strip()),
                    "source": "nvidia-smi"}
    except Exception as exc:
        logger.debug("đo VRAM qua nvidia-smi thất bại: %s", exc)
    if not _vram_warned:
        _vram_warned = True
        logger.warning("Không đo được VRAM (không CUDA/nvidia-smi) — "
                       "admission bỏ qua kiểm tra VRAM.")
    return {"unknown": True}


def check_diffusion_admission() -> dict:
    """Quyết định có cho diffusion chạy ngay. Không chạm lock."""
    if diffusion_busy():
        return {"allowed": False, "reason": "diffusion_busy",
                "detail": "Đang có job diffusion chạy (giữ PIPELINE_LOCK)."}
    load = camera_inference_load()
    vram = vram_info()
    if load["sessions"] > 0 and "free_mb" in vram:
        low_abs = vram["free_mb"] < VRAM_MIN_FREE_MB
        low_ratio = (vram["free_mb"] / max(1.0, vram["total_mb"])) < VRAM_MIN_FREE_RATIO
        if low_abs or low_ratio:
            return {"allowed": False, "reason": "camera_priority_low_vram",
                    "detail": f"VRAM còn {vram['free_mb']}MB với "
                    f"{load['sessions']} phiên camera live — diffusion chờ "
                    f"để ưu tiên inference camera.",
                    "load": load, "vram": vram}
    return {"allowed": True, "reason": "ok", "load": load, "vram": vram}


def acquire_diffusion_slot() -> tuple[bool, dict]:
    """Xin slot diffusion. Trả (acquired, info).

    Giữ nguyên ngữ nghĩa cũ: không blocking khi DIFFUSION_QUEUE_ON_BUSY=false.
    Khi bật queue: chờ lock tới timeout (poll), giữa chừng vẫn tôn trọng cancel
    của caller? Không — caller (jobs.py) quyết định 409 sau timeout.
    """
    from backend.api.job_runner import PIPELINE_LOCK
    decision = check_diffusion_admission()
    if decision["allowed"]:
        acquired = PIPELINE_LOCK.acquire(blocking=False)
        if acquired:
            return True, {**decision, "queued": False}
        decision = {"allowed": False, "reason": "diffusion_busy",
                    "detail": "Lock vừa bị job khác giữ (race)."}
    if not QUEUE_ON_BUSY or decision["reason"] == "camera_priority_low_vram":
        # VRAM thấp không xếp chờ vô hạn — trả 409 để operator giảm tải camera
        # hoặc chuyển diffusion sang GPU khác (doc §3).
        return False, decision
    with _waiters_lock:
        global _waiters
        _waiters += 1
        position = _waiters
    try:
        deadline = time.monotonic() + QUEUE_TIMEOUT_SEC
        while time.monotonic() < deadline:
            time.sleep(QUEUE_POLL_SEC)
            retry = check_diffusion_admission()
            if retry["allowed"] and PIPELINE_LOCK.acquire(blocking=False):
                return True, {**retry, "queued": True, "position": position}
        return False, {"allowed": False, "reason": "queue_timeout",
                       "detail": f"Chờ diffusion slot quá {QUEUE_TIMEOUT_SEC}s "
                       f"(vị trí {position}).", "position": position}
    finally:
        with _waiters_lock:
            _waiters = max(0, _waiters - 1)


def status() -> dict:
    with _waiters_lock:
        q = _waiters
    return {"diffusion_busy": diffusion_busy(),
            "camera": camera_inference_load(),
            "vram": vram_info(),
            "queue": {"enabled": QUEUE_ON_BUSY, "timeout_sec": QUEUE_TIMEOUT_SEC,
                      "waiters": q},
            "thresholds": {"min_free_mb": VRAM_MIN_FREE_MB,
                           "min_free_ratio": VRAM_MIN_FREE_RATIO}}
