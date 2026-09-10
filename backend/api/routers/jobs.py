import asyncio
import threading
import uuid
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from backend.api.dependencies import get_config
from backend.api.job_runner import (
    PIPELINE_LOCK,
    run_pipeline_job,
    subscribe_job,
    unsubscribe_job,
)
from backend.api.schemas import JobResult, JobStatus, RunPipelineRequest
from backend.api.session_store import JobState, get_job, get_session, save_job

router = APIRouter(prefix="/api", tags=["jobs"])


@router.post("/sessions/{session_id}/run", status_code=status.HTTP_202_ACCEPTED)
def run_pipeline(session_id: str, req: RunPipelineRequest = RunPipelineRequest()):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")

    if not session.cropped_path:
        raise HTTPException(status_code=400, detail="Chưa chọn và căn chỉnh khuôn mặt.")

    if session.initial_age is None:
        raise HTTPException(status_code=400, detail="Chưa xác định tuổi ban đầu (initial_age).")

    # Kiểm tra Mutex GPU: nếu đang có job chạy, trả 409 ngay lập tức
    acquired = PIPELINE_LOCK.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Hệ thống đang bận xử lý tác vụ khác, vui lòng đợi job hiện tại hoàn tất."
        )

    job_id = str(uuid.uuid4())
    job_state = JobState(
        job_id=job_id,
        session_id=session_id,
        status="running",
        stage="specialization"
    )
    save_job(job_state)

    config = get_config()

    # Chạy pipeline trong thread nền độc lập
    worker_thread = threading.Thread(
        target=run_pipeline_job,
        args=(job_id, session, config, req.gallery_dir),
        daemon=True
    )
    worker_thread.start()

    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy job.")

    if job.status == "done" and job.result:
        return job.result

    return {
        "job_id": job.job_id,
        "status": job.status,
        "stage": job.stage,
        "error_message": job.error_message
    }


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
