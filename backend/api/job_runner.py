import asyncio
import copy
import os
import shutil
import threading
import traceback
from typing import Any, Dict, List, Optional

from src.utils.cancellation import TaskCancelled, cancellation_scope, checkpoint
from backend.api.session_lifecycle import begin_task, finish_task

import main as pipeline
from backend.api.session_store import SessionState, get_job, save_job, JobState
from backend.api.persistence import (
    persist_generated_outputs,
    update_job_record,
)

# Global Mutex to prevent multiple heavy diffusion pipeline jobs from running concurrently on the GPU
PIPELINE_LOCK = threading.Lock()

# Cooperative cancellation: một Event cho mỗi job đang chạy. Worker kiểm tra
# giữa các stage (specialization/inversion/editing/search); endpoint cancel
# chỉ đặt cờ — không kill thread giữa chừng để tránh rò VRAM/khóa GPU.
_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CANCEL_LOCK = threading.Lock()

CANCELLED_MESSAGE = "Tiến trình đã bị người dùng dừng lại."


def request_cancel(job_id: str) -> bool:
    """Đặt cờ hủy cho job đang chạy. Trả False khi job không còn chạy."""
    job = get_job(job_id)
    if job is None or job.status != "running":
        return False
    with _CANCEL_LOCK:
        event = _CANCEL_EVENTS.get(job_id)
        if event is None:
            event = threading.Event()
            _CANCEL_EVENTS[job_id] = event
        event.set()
    return True


def prepare_job_task(job_id: str, session_id: str) -> threading.Event:
    with _CANCEL_LOCK:
        event = _CANCEL_EVENTS.setdefault(job_id, threading.Event())
    return begin_task(session_id, event)


def is_cancelled(job_id: str) -> bool:
    with _CANCEL_LOCK:
        event = _CANCEL_EVENTS.get(job_id)
    return event is not None and event.is_set()


def _clear_cancel(job_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCEL_EVENTS.pop(job_id, None)


def _check_cancelled(job_id: str) -> bool:
    return is_cancelled(job_id)

# Thread-safe WebSocket subscriber management
_SUBSCRIBERS: Dict[str, List[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
_SUBSCRIBERS_LOCK = threading.Lock()


def subscribe_job(job_id: str, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
    with _SUBSCRIBERS_LOCK:
        if job_id not in _SUBSCRIBERS:
            _SUBSCRIBERS[job_id] = []
        _SUBSCRIBERS[job_id].append((loop, queue))


def unsubscribe_job(job_id: str, queue: asyncio.Queue) -> None:
    with _SUBSCRIBERS_LOCK:
        if job_id in _SUBSCRIBERS:
            _SUBSCRIBERS[job_id] = [item for item in _SUBSCRIBERS[job_id] if item[1] is not queue]
            if not _SUBSCRIBERS[job_id]:
                del _SUBSCRIBERS[job_id]


def _notify_subscribers(job_id: str, message: dict) -> None:
    with _SUBSCRIBERS_LOCK:
        subs = list(_SUBSCRIBERS.get(job_id, []))
    for loop, queue in subs:
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, message)
        except Exception:
            pass


def update_stage(job_id: str, stage: str) -> None:
    job = get_job(job_id)
    if job:
        job.stage = stage
        save_job(job)
    _notify_subscribers(job_id, {
        "job_id": job_id,
        "status": "running",
        "stage": stage,
        "error_message": None
    })


def set_done(
    job_id: str,
    edited_images: Dict[int, str],
    final_scores: Dict[str, float],
    accepted: bool,
    top_identity: str,
    top_score: float,
    best_age: Optional[int] = None,
    matched_gallery_image: Optional[str] = None,
    pipeline_params: Optional[Dict[str, Any]] = None,
    cropped_image: Optional[str] = None,
    age_scores: Optional[Dict[int, float]] = None,
    variants: Optional[list] = None,
) -> None:
    # Convert file paths to HTTP-accessible relative URLs
    # e.g. "outputs/jobs/123/age_30.png" -> "/outputs/jobs/123/age_30.png"
    web_images = {}
    for age, fpath in edited_images.items():
        norm_path = fpath.replace("\\", "/")
        if "outputs/" in norm_path:
            rel_path = "/" + norm_path[norm_path.index("outputs/"):]
        else:
            rel_path = f"/outputs/{os.path.basename(fpath)}"
        web_images[int(age)] = rel_path

    if not cropped_image and os.path.exists(os.path.join("outputs", "jobs", job_id, "input_crop.png")):
        cropped_image = f"/outputs/jobs/{job_id}/input_crop.png"

    result = {
        "job_id": job_id,
        "status": "done",
        "edited_images": web_images,
        "variants": variants or [],
        "age_scores": age_scores or {},
        "final_scores": final_scores,
        "accepted": accepted,
        "top_identity": top_identity,
        "top_score": top_score,
        "best_age": best_age,
        "matched_gallery_image": matched_gallery_image,
        "cropped_image": cropped_image,
        "pipeline_params": pipeline_params or {},
        "error_message": None
    }
    job = get_job(job_id)
    if job:
        job.status = "done"
        job.result = result
        save_job(job)

    _notify_subscribers(job_id, {
        "job_id": job_id,
        "status": "done",
        "stage": "complete",
        "result": result,
        "error_message": None
    })


def set_error(job_id: str, error_message: str, db_job_id: Optional[str] = None) -> None:
    job = get_job(job_id)
    if job:
        job.status = "error"
        job.error_message = error_message
        save_job(job)

    # Output đã lưu thành công được giữ; job lỗi ghi trạng thái error vào DB.
    # Chỉ chạm DB khi job này thực sự có bản ghi (luồng legacy: db_job_id None).
    if db_job_id:
        update_job_record(db_job_id, "error", error_message[:2000])

    _notify_subscribers(job_id, {
        "job_id": job_id,
        "status": "error",
        "stage": "failed",
        "error_message": error_message
    })


def run_pipeline_job(
    job_id: str,
    session: SessionState,
    base_config: dict,
    gallery_dir: Optional[str] = None,
    db_job_id: Optional[str] = None,
    input_crop_id: Optional[str] = None,
    task_event: Optional[threading.Event] = None,
) -> None:
    """Runs the 4-stage pipeline in a background worker thread.
    Always releases PIPELINE_LOCK in finally block.

    T06: db_job_id/input_crop_id chốt lineage lúc chạy — job cũ tiếp tục tham
    chiếu đúng crop cũ dù người dùng đổi crop sau đó. Tương thích ngược: các
    caller cũ không truyền 2 tham số mới vẫn chạy luồng legacy (RAM + đĩa).
    """
    # Chốt input lúc chạy (snapshot, không đọc lại session sau này).
    frozen_crop_id = input_crop_id or getattr(session, "current_crop_id", None)
    event = task_event
    try:
        if event is None:
            event = prepare_job_task(job_id, session.session_id)
        with cancellation_scope(event):
            update_job_record(db_job_id, "running")
            if _check_cancelled(job_id):
                set_error(job_id, CANCELLED_MESSAGE, db_job_id=db_job_id)
                return
            # Deepcopy config to isolate output directory per job and prevent mutating base config
            job_config = copy.deepcopy(base_config)
            job_output_dir = os.path.join("outputs", "jobs", job_id)
            os.makedirs(job_output_dir, exist_ok=True)
            job_config["paths"]["output_dir"] = job_output_dir

            if session.cropped_path and os.path.exists(session.cropped_path):
                shutil.copy2(session.cropped_path, os.path.join(job_output_dir, "input_crop.png"))

            if gallery_dir:
                job_config["paths"]["gallery_test_dir"] = gallery_dir

            # Stage 1: Specialization
            update_stage(job_id, "specialization")
            ckpt_dir = pipeline.run_specialization(job_config)

            # Stage 2: Null-text Inversion
            if _check_cancelled(job_id):
                set_error(job_id, CANCELLED_MESSAGE, db_job_id=db_job_id)
                return
            update_stage(job_id, "inversion")
            z_T, null_emb, attn = pipeline.run_inversion(
                job_config,
                ckpt_dir,
                session.cropped_path,
                session.initial_age,
                session.gender_word
            )

            # Stage 3: Editing
            if _check_cancelled(job_id):
                set_error(job_id, CANCELLED_MESSAGE, db_job_id=db_job_id)
                return
            update_stage(job_id, "editing")
            photo_year = session.photo_year if session.photo_year is not None else 2010
            target_ages = pipeline.compute_target_ages(session.initial_age, photo_year)
            edited_images = pipeline.run_editing(
                job_config,
                ckpt_dir,
                z_T,
                null_emb,
                attn,
                session.gender_word,
                initial_age=session.initial_age,
                target_ages=target_ages,
            )

            # Stage 4: Embedding + FAISS Search
            if _check_cancelled(job_id):
                set_error(job_id, CANCELLED_MESSAGE, db_job_id=db_job_id)
                return
            update_stage(job_id, "search")
            if db_job_id:
                search_res = ({}, False, "", 0.0, {})
            else:
                search_res = pipeline.run_embedding_and_search(
                    job_config,
                    edited_images,
                    return_age_scores=True
                )
            checkpoint()
            if len(search_res) == 5:
                final_scores, accepted, top_identity, top_score, age_scores = search_res
            else:
                final_scores, accepted, top_identity, top_score = search_res
                age_scores = {}

            # Determine best matching age and copy gallery match image
            matched_gallery_image = None
            best_age = None
            if age_scores:
                best_age = max(age_scores.items(), key=lambda kv: kv[1])[0]
            elif edited_images:
                sorted_ages = sorted([int(a) for a in edited_images.keys()])
                best_age = sorted_ages[len(sorted_ages) // 2] if sorted_ages else None

            active_gallery_dir = job_config.get("paths", {}).get("gallery_test_dir", "data/test_gallery")
            if top_identity and os.path.exists(active_gallery_dir):
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    cand = os.path.join(active_gallery_dir, f"{top_identity}{ext}")
                    if os.path.exists(cand):
                        dest = os.path.join(job_output_dir, f"gallery_match_{top_identity}{ext}")
                        try:
                            shutil.copyfile(cand, dest)
                            matched_gallery_image = f"/outputs/jobs/{job_id}/gallery_match_{top_identity}{ext}"
                        except Exception:
                            norm = cand.replace("\\", "/")
                            matched_gallery_image = "/" + norm
                        break

            pipeline_params = {
                "num_inference_steps": job_config.get("inversion", {}).get("num_inference_steps", 50),
                "guidance_scale": job_config.get("editing", {}).get("guidance_scale", 4.0),
                "inversion_guidance_scale": job_config.get("inversion", {}).get("guidance_scale", 1.0),
                "attention_control_ratio": job_config.get("editing", {}).get("attention_control_ratio", 0.8),
                "image_size": job_config.get("editing", {}).get("image_size", 512),
                "checkpoint_name": "specialized_unet (stable-diffusion-v1-5)",
                "embedding_model": job_config.get("embedding", {}).get("model_name", "buffalo_l"),
                "rejection_threshold": job_config.get("search", {}).get("rejection_threshold", 0.6),
            }
            # T06: lưu từng ảnh tạo sinh vào face_media trước khi báo done.
            # Chỉ done trong DB khi output đã lưu; file thiếu trên đĩa thì bỏ qua
            # (không tạo bản ghi giả). WebSocket (set_done) bắn sau transaction.
            variants: list = []
            if db_job_id:
                variants = persist_generated_outputs(
                    job_id=db_job_id,
                    edited_images=edited_images,
                    parameters=pipeline_params,
                )
                if not edited_images or len(variants) != len(edited_images):
                    raise RuntimeError("Không lưu đủ ảnh tạo sinh; job không được đánh dấu done.")
                from backend.api.job_search import search_generated_job
                persisted_search = search_generated_job(
                    db_job_id, variants, edited_images,
                    top_k=job_config.get("search", {}).get("top_k", 5),
                    threshold=job_config.get("search", {}).get("rejection_threshold", 0.6))
                final_scores = persisted_search["final_scores"]
                accepted = persisted_search["accepted"]
                top_identity = persisted_search["top_identity"]
                top_score = persisted_search["top_score"]
                best_age = persisted_search["best_age"]
                age_scores = persisted_search["age_scores"]
                matched_gallery_image = persisted_search["matched_gallery_image"]
                pipeline_params["search_run_id"] = persisted_search["search_run_id"]
                update_job_record(db_job_id, "done")
            checkpoint()
            set_done(
                job_id,
                edited_images,
                final_scores,
                accepted,
                top_identity,
                top_score,
                best_age=best_age,
                matched_gallery_image=matched_gallery_image,
                pipeline_params=pipeline_params,
                age_scores=age_scores,
                variants=variants,
            )

    except TaskCancelled:
        set_error(job_id, CANCELLED_MESSAGE, db_job_id=db_job_id)
    except Exception as e:
        traceback.print_exc()
        set_error(job_id, str(e), db_job_id=db_job_id)
    finally:
        _clear_cancel(job_id)
        if PIPELINE_LOCK.locked():
            try:
                PIPELINE_LOCK.release()
            except RuntimeError:
                pass

        if event is not None:
            finish_task(session.session_id, event)
