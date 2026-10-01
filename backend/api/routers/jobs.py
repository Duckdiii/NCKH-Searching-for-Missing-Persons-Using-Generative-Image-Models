import asyncio
import os
import threading
import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from backend.api.session_lifecycle import finish_task, session_operation
from backend.api.dependencies import get_config
from backend.api.job_runner import (
    PIPELINE_LOCK,
    request_cancel,
    prepare_job_task,
    set_error,
    _clear_cancel,
    run_pipeline_job,
    subscribe_job,
    unsubscribe_job,
)
from backend.api.persistence import (
    create_generation_job_record,
    db_ping,
    get_generation_job_detail,
)
from backend.api.schemas import GenerationJobDetail, JobResult, JobStatus, RunPipelineRequest
from backend.api.session_store import JobState, get_job, get_session, save_job, jobs

router = APIRouter(prefix="/api", tags=["jobs"])


@router.post("/sessions/{session_id}/run", status_code=status.HTTP_202_ACCEPTED)
@session_operation
def run_pipeline(session_id: str, req: RunPipelineRequest = RunPipelineRequest()):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")

    if not session.cropped_path:
        raise HTTPException(status_code=400, detail="Chưa chọn và căn chỉnh khuôn mặt.")

    if session.initial_age is None:
        raise HTTPException(status_code=400, detail="Chưa xác định tuổi ban đầu (initial_age).")

    if req.photo_year is not None:
        session.photo_year = req.photo_year

    if session.photo_year is None:
        raise HTTPException(status_code=400, detail="Chưa xác định năm chụp ảnh (photo_year).")

    # Admission GPU chung camera + diffusion (doc §3): ưu tiên inference
    # camera khi VRAM thấp; giữ ngữ nghĩa 409 khi bận (có thể xếp chờ khi
    # DIFFUSION_QUEUE_ON_BUSY=true).
    from backend.api import gpu_admission
    acquired, admission = gpu_admission.acquire_diffusion_slot()
    if not acquired:
        reason = admission.get("reason", "busy")
        # Giữ message 409 cũ cho tương thích; thêm hậu tố nguyên nhân mới.
        detail = ("Hệ thống đang bận xử lý tác vụ khác, "
                  "vui lòng đợi job hiện tại hoàn tất.")
        if reason == "camera_priority_low_vram":
            detail += (" (VRAM thấp trong khi camera đang inference — "
                       "giảm phiên camera hoặc chuyển diffusion sang GPU khác.)")
        elif reason == "queue_timeout":
            detail = (f"Chờ diffusion slot quá {admission.get('position', '?')} lượt. "
                      f"Thử lại sau.")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)

    task_event = None
    job_id = None
    db_job_id = None
    try:
        job_id = str(uuid.uuid4())
        task_event = prepare_job_task(job_id, session_id)
        job_state = JobState(
            job_id=job_id,
            session_id=session_id,
            status="running",
            stage="specialization"
        )

        # T06: chốt input_crop_id tại lúc chạy và tạo bản ghi generation_jobs
        # (pending). Không có crop bền vững (luồng legacy) → chạy RAM như cũ.
        input_crop_id = session.current_crop_id
        if input_crop_id is not None:
            db_job_id = create_generation_job_record(
                job_id=job_id,
                input_crop_id=input_crop_id,
                session_id=session_id,
                initial_age=session.initial_age,
                parameters={"photo_year": session.photo_year},
            )
            if db_job_id is None:
                raise HTTPException(503, "Không tạo được job trong database; chưa chạy pipeline.")
            job_state.db_job_id = db_job_id
        else:
            db_job_id = None
        save_job(job_state)

        config = get_config()

        # Chạy pipeline trong thread nền độc lập
        worker_thread = threading.Thread(
            target=run_pipeline_job,
            args=(job_id, session, config, req.gallery_dir, db_job_id, input_crop_id, task_event),
            daemon=True
        )
        worker_thread.start()

        return {"job_id": job_id, "status": "running"}

    except Exception:
        try:
            if job_id is not None and get_job(job_id) is not None:
                set_error(job_id, "Không thể khởi động tác vụ.", db_job_id=db_job_id)
        finally:
            if job_id is not None:
                _clear_cancel(job_id)
            if task_event is not None:
                finish_task(session_id, task_event)
            PIPELINE_LOCK.release()
        raise


@router.get("/jobs", response_model=List[Dict[str, Any]])
def list_jobs():
    """Trả về danh sách tất cả các job đã chạy từ session store in-memory."""
    job_list = []
    for job_id, job in list(jobs.items()):
        if job.result and not job.result.get("cropped_image") and os.path.exists(os.path.join("outputs", "jobs", job_id, "input_crop.png")):
            job.result["cropped_image"] = f"/outputs/jobs/{job_id}/input_crop.png"
        item = {
            "job_id": job.job_id,
            "session_id": job.session_id,
            "status": job.status,
            "stage": job.stage,
            "top_identity": job.result.get("top_identity") if job.result else None,
            "top_score": job.result.get("top_score") if job.result else None,
            "accepted": job.result.get("accepted") if job.result else None,
            "result": job.result,
            "error_message": job.error_message
        }
        job_list.append(item)
    return job_list


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy job.")

    if job.status == "done" and job.result:
        if not job.result.get("cropped_image") and os.path.exists(os.path.join("outputs", "jobs", job_id, "input_crop.png")):
            job.result["cropped_image"] = f"/outputs/jobs/{job_id}/input_crop.png"
        return job.result

    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "error_message": job.error_message
    }


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Dừng job pipeline đang chạy (hợp tác: worker dừng giữa các stage,
    giải phóng GPU mutex; trạng thái cuối là error với thông báo đã hủy)."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy job.")
    if job.status != "running":
        raise HTTPException(
            status_code=409,
            detail=f"Job đã ở trạng thái {job.status}, không thể dừng.",
        )
    if not request_cancel(job_id):
        raise HTTPException(
            status_code=409,
            detail="Job vừa kết thúc, không thể dừng.",
        )
    return {"job_id": job_id, "status": "cancel_requested"}


@router.get("/generation-jobs/{job_id}", response_model=GenerationJobDetail)
def get_generation_job(job_id: str):
    """T06: xem lại job tạo sinh từ DB sau restart (không phụ thuộc RAM)."""
    if not db_ping():
        raise HTTPException(
            status_code=503,
            detail="Database face_media không sẵn sàng — không thể xem lịch sử job.",
        )
    detail = get_generation_job_detail(job_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy generation job.")
    return detail


@router.get("/generation-jobs", response_model=Dict[str, Any])
def list_generation_jobs_view(
    status: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """T12: lịch sử job tạo sinh từ DB (phân trang, lọc status)."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool

    if not db_ping():
        raise HTTPException(
            status_code=503,
            detail="Database face_media không sẵn sàng — không thể xem lịch sử job.",
        )
    try:
        with get_pool().connection() as conn:
            items, total = repo.list_generation_jobs(
                conn, status=status,
                limit=max(1, min(limit, 200)), offset=max(0, offset))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.websocket("/jobs/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    subscribe_job(job_id, loop, queue)

    try:
        # Gửi trạng thái hiện tại ngay khi client kết nối
        job = get_job(job_id)
        if job:
            if job.status == "done" and job.result:
                await websocket.send_json(job.result)
                return
            elif job.status == "error":
                await websocket.send_json({
                    "job_id": job_id,
                    "status": "error",
                    "stage": "failed",
                    "error_message": job.error_message
                })
                return
            else:
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job.status,
                    "stage": job.stage,
                    "error_message": None
                })

        # Lắng nghe cập nhật từ job_runner
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("status") in ["done", "error"]:
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        unsubscribe_job(job_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
