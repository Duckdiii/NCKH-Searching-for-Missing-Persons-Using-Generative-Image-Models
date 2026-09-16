import asyncio
import copy
import os
import shutil
import threading
import traceback
from typing import Any, Dict, List, Optional

import main as pipeline
from backend.api.session_store import SessionState, get_job, save_job, JobState

# Global Mutex to prevent multiple heavy diffusion pipeline jobs from running concurrently on the GPU
PIPELINE_LOCK = threading.Lock()

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
    age_scores: Optional[Dict[int, float]] = None
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


def set_error(job_id: str, error_message: str) -> None:
    job = get_job(job_id)
    if job:
        job.status = "error"
        job.error_message = error_message
        save_job(job)

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
    gallery_dir: Optional[str] = None
) -> None:
    """Runs the 4-stage pipeline in a background worker thread.
    Always releases PIPELINE_LOCK in finally block.
    """
    try:
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
        update_stage(job_id, "inversion")
        z_T, null_emb, attn = pipeline.run_inversion(
            job_config,
            ckpt_dir,
            session.cropped_path,
            session.initial_age,
            session.gender_word
        )

        # Stage 3: Editing
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
        update_stage(job_id, "search")
        search_res = pipeline.run_embedding_and_search(
            job_config,
            edited_images,
            return_age_scores=True
        )
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
            age_scores=age_scores
        )

    except Exception as e:
        traceback.print_exc()
        set_error(job_id, str(e))
    finally:
        if PIPELINE_LOCK.locked():
            try:
                PIPELINE_LOCK.release()
            except RuntimeError:
                pass
