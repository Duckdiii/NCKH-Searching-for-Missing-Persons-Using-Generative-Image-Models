"""T09 — API camera: đăng ký camera, start/stop capture, trạng thái phiên."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api import cameras as cam_service
from backend.api import repositories as repo
from backend.api.database import get_pool
from backend.api.persistence import db_ping
from backend.api.schemas import (
    CameraCreate,
    CameraInfo,
    CameraListResponse,
    CaptureSessionStatus,
    CaptureStartRequest,
    SamplingPlanRequest,
)
from backend.api.sampling_policy import plan_from_dict

router = APIRouter(prefix="/api", tags=["cameras"])


def resolve_capture_config(body: CaptureStartRequest) -> dict:
    """Chọn fps, K (crop/tracklet), q_min cho phiên camera. Giá trị nhập tay thắng."""
    from backend.api.camera_tracks import MAX_CANDIDATES_PER_TRACKLET

    fps, k, q_min = body.sample_fps, body.max_candidates, body.min_quality
    plan = optimizer = None
    source = "manual"
    if body.geometry is not None:
        geo = body.geometry.model_dump()
        geo.pop("include_grid", None)
        plan_keys = SamplingPlanRequest.model_fields.keys()
        plan = plan_from_dict({key: v for key, v in geo.items() if key in plan_keys})
        source = "sampling_plan"
        if body.auto_optimize:
            from backend.api import pipeline_optimizer as po
            res = po.optimize_from_dict(geo)
            rec = res["recommended"]
            optimizer = {"recommended": rec, "best_p_found": res["best_p_found"],
                         "quality_model": res["quality_model"],
                         "warnings": res["warnings"]}
            source = "optimizer"
            fps = rec["fps"] if fps is None else fps
            # Bị tách tracklet (k=None): mỗi tracklet 1 khung → K không đổi kết quả.
            k = (rec["k"] or MAX_CANDIDATES_PER_TRACKLET) if k is None else k
            q_min = rec["q_min"] if q_min is None else q_min
        elif fps is None:
            fps = plan["fps"]
    if body.sample_fps is not None and source != "manual":
        source += "+manual"
    return {"sample_fps": float(min(fps or 1.0, 10.0)),
            "max_candidates": int(k or MAX_CANDIDATES_PER_TRACKLET),
            "min_quality": float(q_min or 0.0), "config_source": source,
            "plan": plan, "optimizer": optimizer}


@router.post("/cameras/sampling-plan")
def sampling_plan_view(body: SamplingPlanRequest):
    """Tính fps lấy mẫu từ hình học camera (không cần DB, không bật camera)."""
    try:
        return plan_from_dict(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _require_db() -> None:
    if not db_ping():
        raise HTTPException(
            status_code=503,
            detail="Database face_media không sẵn sàng — không thể quản lý camera.",
        )


@router.post("/cameras", response_model=CameraInfo)
def register_camera(body: CameraCreate):
    """Đăng ký camera. Chỉ lưu connection_secret_ref (tên biến môi trường),
    không nhận/lưu URL hay mật khẩu qua API."""
    _require_db()
    if body.connection_secret_ref and len(body.connection_secret_ref) > 128:
        raise HTTPException(status_code=400, detail="connection_secret_ref quá dài.")
    try:
        with get_pool().connection() as conn:
            camera_id = repo.create_camera(
                conn, name=body.name, location=body.location,
                connection_secret_ref=body.connection_secret_ref)
            return CameraInfo(**repo.get_camera(conn, camera_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/cameras", response_model=CameraListResponse)
def list_cameras_view(limit: int = 50, offset: int = 0):
    _require_db()
    with get_pool().connection() as conn:
        items, total = repo.list_cameras(conn, limit=limit, offset=offset)
    return CameraListResponse(
        items=[CameraInfo(**item) for item in items], total=total)


@router.post("/cameras/{camera_id}/start", response_model=CaptureSessionStatus,
             status_code=202)
def start_capture_view(camera_id: str, body: CaptureStartRequest):
    """Start phiên capture mới (mỗi phiên một source search/camera)."""
    _require_db()
    try:
        from backend.api.dependencies import get_config

        preprocess_on = bool(
            get_config().get("video_preprocess", {}).get("enabled", True))
        cfg = resolve_capture_config(body)
        started = cam_service.start_capture(
            camera_id=camera_id, sample_fps=cfg["sample_fps"],
            max_frames=body.max_frames, capture_seconds=body.capture_seconds,
            preprocess_on=preprocess_on, sampling_plan=cfg["plan"],
            max_candidates=cfg["max_candidates"], min_quality=cfg["min_quality"],
            config_source=cfg["config_source"])
    except repo.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, ConnectionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CaptureSessionStatus(
        source_id=started["source_id"], camera_id=camera_id,
        run_id=started["run_id"], status="running",
        sample_fps=cfg["sample_fps"], max_candidates=cfg["max_candidates"],
        min_quality=cfg["min_quality"], config_source=cfg["config_source"],
        optimizer=cfg["optimizer"])


@router.get("/capture-sessions/{source_id}", response_model=CaptureSessionStatus)
def capture_status_view(source_id: str):
    try:
        return CaptureSessionStatus(**cam_service.capture_status(source_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/capture-sessions/{source_id}/stop")
def stop_capture_view(source_id: str):
    """Stop phiên: đóng capture, chốt ended_at + run."""
    _require_db()
    try:
        return cam_service.stop_capture(source_id)
    except repo.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
