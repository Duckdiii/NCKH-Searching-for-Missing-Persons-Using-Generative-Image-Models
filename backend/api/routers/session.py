import os
import re
from fastapi.responses import JSONResponse
from backend.api.session_lifecycle import (session_operation, stop_tasks, lifecycle_status, delete_when_idle)
from src.utils.cancellation import checkpoint
import tempfile
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.api.dependencies import get_age_estimator, get_embedder
from backend.api.schemas import (
    FaceBox,
    ResolveAgeRequest,
    ResolveAgeResponse,
    SelectFaceRequest,
    SelectFaceResponse,
    RestorePreviewRequest,
    RestorePreviewResponse,
    ApplyRestoreRequest,
    UploadResponse,
)
from backend.api.session_store import (
    SessionState,
    delete_session,
    get_session,
    save_session,
)
from backend.api.persistence import (
    persist_crop_revision,
    persist_reference_upload,
    restore_session,
    save_session_record,
)
from src.utils.age_estimator import resolve_initial_age
from src.utils.face_enhancement import preprocess_face_image, apply_white_balance_from_point
from src.utils.ffhq_align import align_to_ffhq
from src.utils.head_pose import check_image_quality

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File tải lên không có dữ liệu.")

    image_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Không thể giải mã định dạng ảnh.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    embedder = get_embedder()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        faces = embedder.detect_faces(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="Không phát hiện khuôn mặt nào trong ảnh. Vui lòng chọn ảnh khác."
        )

    session_id = str(uuid.uuid4())
    session_state = SessionState(
        session_id=session_id,
        image_bgr=image_bgr,
        image_rgb=image_rgb,
        faces=faces,
        file_name=file.filename or "upload.png"
    )

    # T04: lưu ảnh gốc bền vững (asset → source reference/image → frame tĩnh
    # → detections). Best-effort: DB không sẵn sàng thì giữ luồng RAM cũ.
    persisted = persist_reference_upload(
        image_bytes=file_bytes,
        filename=file.filename or "upload.png",
        content_type=file.content_type,
        faces=faces,
        image_width=int(image_bgr.shape[1]),
        image_height=int(image_bgr.shape[0]),
        detector_name="insightface",
        detector_version=getattr(embedder, "model_name", "buffalo_l"),
    )
    if persisted is not None:
        session_state.source_id = persisted["source_id"]
        session_state.frame_id = persisted["frame_id"]
        session_state.detection_ids = persisted["detection_ids"]
    save_session(session_state)
    # T12: session metadata bền vững (RAM chỉ là cache).
    save_session_record(session_state)

    face_boxes = [
        FaceBox(
            index=i,
            bbox=[float(v) for v in face.bbox],
            det_score=float(face.det_score)
        )
        for i, face in enumerate(faces)
    ]

    return UploadResponse(
        session_id=session_id,
        faces=face_boxes,
        source_id=session_state.source_id,
        detection_ids=session_state.detection_ids,
    )


@router.post("/{session_id}/select-face", response_model=SelectFaceResponse)
@session_operation
def select_face(session_id: str, req: SelectFaceRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")

    if req.selected_idx < 0 or req.selected_idx >= len(session.faces):
        raise HTTPException(
            status_code=400,
            detail=f"Chỉ mục khuôn mặt không hợp lệ. (0 đến {len(session.faces)-1})"
        )

    chosen_face = session.faces[req.selected_idx]
    session.chosen_face = chosen_face

    # Kiểm tra chất lượng ảnh & góc nghiêng
    warnings = check_image_quality(
        session.image_bgr,
        chosen_face.kps,
        float(chosen_face.det_score)
    )

    # Tiền xử lý theo chuẩn Kaggle 3 & FG-NET batch: Adaptive Padding + Shades of Gray WB + CodeFormer
    preprocessed_bgr, updated_kps = preprocess_face_image(
        session.image_bgr, kps=chosen_face.kps, bbox=chosen_face.bbox
    )

    # Căn chỉnh FFHQ chuẩn
    cropped = align_to_ffhq(preprocessed_bgr, updated_kps, output_size=512)
    checkpoint()
    os.makedirs("outputs/app_uploads", exist_ok=True)
    cropped_filename = f"{session_id}_crop.png"
    cropped_path = os.path.join("outputs", "app_uploads", cropped_filename)
    if not cv2.imwrite(cropped_path, cropped):
        raise HTTPException(status_code=500, detail="Ghi ảnh crop thất bại.")

    session.cropped_path = cropped_path
    # T05: lưu revision crop mới thay vì chỉ ghi đè file. Mỗi lần chọn lại mặt
    # tạo một face_crops mới; job chốt input_crop_id lúc chạy nên job cũ giữ
    # nguyên lineage dù người dùng đổi crop sau đó.
    if req.selected_idx < len(session.detection_ids):
        session.chosen_detection_id = session.detection_ids[req.selected_idx]
    revision = (
        persist_crop_revision(
            purpose="reference",
            detection_id=session.chosen_detection_id,
            cropped_bgr=cropped,
            method="restored",
            preprocessing={
                "pipeline": "preprocess_face_image+align_to_ffhq",
                "mode": "auto",
                "padding_enabled": True,
                "white_balance_enabled": True,
                "output_size": 512,
            },
            transform_to_source={
                "bbox_original": [float(v) for v in chosen_face.bbox],
                "output_size": 512,
            },
        )
        if session.chosen_detection_id
        else None
    )
    if revision is not None:
        session.current_crop_id = revision["crop_id"]
    save_session(session)
    save_session_record(
        session, chosen_face_index=req.selected_idx,
        chosen_detection_id=session.chosen_detection_id,
        current_crop_id=session.current_crop_id)

    cropped_preview_url = f"/outputs/app_uploads/{cropped_filename}"
    return SelectFaceResponse(
        warnings=warnings,
        cropped_preview_url=cropped_preview_url,
        crop_id=session.current_crop_id,
    )


@router.post("/{session_id}/restore-preview", response_model=RestorePreviewResponse)
@session_operation
def restore_preview(session_id: str, req: RestorePreviewRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")
    if not session.chosen_face:
        raise HTTPException(status_code=400, detail="Chưa chọn khuôn mặt.")

    wb_point = (req.click_x, req.click_y) if (req.click_x is not None and req.click_y is not None) else None
    
    wb_info = None
    if req.white_balance_enabled and wb_point is not None:
        _, wb_info = apply_white_balance_from_point(session.image_rgb, wb_point[0], wb_point[1], return_info=True)

    preprocessed_bgr, updated_kps = preprocess_face_image(
        session.image_bgr,
        kps=session.chosen_face.kps,
        bbox=session.chosen_face.bbox,
        mode=req.mode,
        padding_enabled=req.padding_enabled,
        white_balance_enabled=req.white_balance_enabled,
        wb_point=wb_point,
        fidelity_weight=req.fidelity_weight,
    )
    cropped = align_to_ffhq(preprocessed_bgr, updated_kps, output_size=512)
    checkpoint()
    os.makedirs("outputs/app_uploads", exist_ok=True)
    preview_filename = f"{session_id}_preview.png"
    preview_path = os.path.join("outputs", "app_uploads", preview_filename)
    cv2.imwrite(preview_path, cropped)

    preview_url = f"/outputs/app_uploads/{preview_filename}"
    return RestorePreviewResponse(preview_url=preview_url, wb_info=wb_info)


@router.post("/{session_id}/apply-restore")
@session_operation
def apply_restore(session_id: str, req: ApplyRestoreRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")
    if not session.chosen_face:
        raise HTTPException(status_code=400, detail="Chưa chọn khuôn mặt.")

    if not req.use_restored:
        cropped = align_to_ffhq(session.image_bgr, session.chosen_face.kps, output_size=512)
    else:
        wb_point = (req.click_x, req.click_y) if (req.click_x is not None and req.click_y is not None) else None
        preprocessed_bgr, updated_kps = preprocess_face_image(
            session.image_bgr,
            kps=session.chosen_face.kps,
            bbox=session.chosen_face.bbox,
            mode=req.mode,
            padding_enabled=req.padding_enabled,
            white_balance_enabled=req.white_balance_enabled,
            wb_point=wb_point,
            fidelity_weight=req.fidelity_weight,
        )
        cropped = align_to_ffhq(preprocessed_bgr, updated_kps, output_size=512)

    checkpoint()
    os.makedirs("outputs/app_uploads", exist_ok=True)
    cropped_filename = f"{session_id}_crop.png"
    cropped_path = os.path.join("outputs", "app_uploads", cropped_filename)
    if not cv2.imwrite(cropped_path, cropped):
        raise HTTPException(status_code=500, detail="Ghi ảnh crop thất bại.")

    session.cropped_path = cropped_path
    # T05: apply-restore cũng tạo revision mới (preview tạm ở restore-preview
    # không tự thay input; chỉ endpoint này thay crop đã áp dụng).
    revision = (
        persist_crop_revision(
            purpose="reference",
            detection_id=session.chosen_detection_id,
            cropped_bgr=cropped,
            method="restored" if req.use_restored else "ffhq",
            preprocessing={
                "pipeline": "preprocess_face_image+align_to_ffhq"
                if req.use_restored
                else "align_to_ffhq",
                "mode": req.mode,
                "padding_enabled": req.padding_enabled,
                "white_balance_enabled": req.white_balance_enabled,
                "fidelity_weight": req.fidelity_weight,
                "output_size": 512,
            },
            transform_to_source={
                "bbox_original": [float(v) for v in session.chosen_face.bbox],
                "output_size": 512,
            },
        )
        if session.chosen_detection_id
        else None
    )
    if revision is not None:
        session.current_crop_id = revision["crop_id"]
    save_session(session)
    save_session_record(session, current_crop_id=session.current_crop_id)

    cropped_preview_url = f"/outputs/app_uploads/{cropped_filename}"
    return {
        "status": "ok",
        "cropped_preview_url": cropped_preview_url,
        "crop_id": session.current_crop_id,
    }


@router.post("/{session_id}/resolve-age", response_model=ResolveAgeResponse)
@session_operation
def resolve_age(session_id: str, req: ResolveAgeRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")

    if not session.cropped_path or not os.path.exists(session.cropped_path):
        raise HTTPException(status_code=400, detail="Vui lòng chọn khuôn mặt trước khi xác định tuổi.")

    session.gender_word = req.gender_word
    if req.photo_year is not None:
        session.photo_year = req.photo_year

    if req.mode == "manual":
        if req.manual_age is None:
            raise HTTPException(status_code=422, detail="Cần nhập số tuổi (manual_age) khi chọn mode 'manual'.")
        initial_age = resolve_initial_age(mode="manual", manual_age=req.manual_age)
        warning_text = None
    elif req.mode == "mivolo":
        estimator = get_age_estimator()
        initial_age = resolve_initial_age(
            mode="mivolo",
            image_path=session.cropped_path,
            age_estimator=estimator
        )
        warning_text = (
            f"Hệ thống ước tính khoảng {initial_age} tuổi. "
            f"Lưu ý: có thể sai lệch ±4-5 năm so với thực tế (sai số vốn có của MiVOLO, đã đo thật)."
        )
    else:
        raise HTTPException(status_code=400, detail="Mode không hợp lệ.")

    session.initial_age = initial_age
    save_session(session)
    save_session_record(
        session, initial_age=initial_age, gender_word=session.gender_word,
        photo_year=session.photo_year)

    return ResolveAgeResponse(
        initial_age=initial_age,
        gender_word=session.gender_word,
        warning_text=warning_text
    )


@router.get("/{session_id}")
@session_operation
def get_session_state(session_id: str):
    """T12: mở lại session sau restart (RAM trước, dựng lại từ DB + storage)."""
    session = get_session(session_id) or restore_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")
    faces = [
        {"index": i, "bbox": [float(v) for v in face.bbox],
         "det_score": float(face.det_score)}
        for i, face in enumerate(session.faces)
    ]
    cropped_url = None
    if session.cropped_path:
        norm = session.cropped_path.replace("\\", "/")
        if "outputs/" in norm:
            cropped_url = "/" + norm[norm.index("outputs/"):]
    return {
        "session_id": session.session_id,
        "file_name": session.file_name,
        "faces": faces,
        "source_id": session.source_id,
        "detection_ids": session.detection_ids,
        "chosen_detection_id": session.chosen_detection_id,
        "crop_id": session.current_crop_id,
        "cropped_preview_url": cropped_url,
        "gender_word": session.gender_word,
        "initial_age": session.initial_age,
        "photo_year": session.photo_year,
    }


def _delete_session_records(session_id: str):
    """Xóa phiên: gỡ RAM, bản ghi sessions trong DB và file crop/preview legacy.

    Dữ liệu lineage face_media (source/crop/job) được giữ lại để bảo toàn audit;
    file gốc trong storage không bị xóa ở đây.
    """
    removed: dict = {"ram": False, "db_record": False, "files": []}
    from backend.api.database import get_pool, DatabaseConfigError
    try:

        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM face_media.sessions WHERE id = %s",
                    (session_id,))
                if cur.rowcount:
                    removed["db_record"] = True
    except DatabaseConfigError:
        # Legacy mode has no configured database.
        pass
    except Exception:
        raise HTTPException(503, "Không thể xóa bản ghi phiên trong database. Hãy thử lại.") from None
    if get_session(session_id) is not None:
        delete_session(session_id)
        removed["ram"] = True
    for name in (f"{session_id}_crop.png", f"{session_id}_preview.png"):
        path = os.path.join("outputs", "app_uploads", name)
        if os.path.exists(path):
            try:
                os.remove(path)
                removed["files"].append(name)
            except OSError:
                pass
    if not removed["ram"] and not removed["db_record"] and not removed["files"]:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")
    from backend.api.session_store import jobs
    for job_id, job in list(jobs.items()):
        if job.session_id == session_id:
            jobs.pop(job_id, None)
    return {"session_id": session_id, "deleted": removed}


def _validate_session_id(session_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        raise HTTPException(400, "Mã phiên không hợp lệ.")


@router.post("/{session_id}/stop")
def stop_session(session_id: str):
    _validate_session_id(session_id)
    from backend.api.job_runner import request_cancel
    from backend.api.session_store import jobs
    state = lifecycle_status(session_id)
    related = [j for j in list(jobs.values()) if j.session_id == session_id]
    if get_session(session_id) is None and not related and not state['active_tasks']:
        if restore_session(session_id) is None:
            raise HTTPException(404, "Phiên làm việc không tồn tại.")
    state = stop_tasks(session_id)
    for job in related:
        request_cancel(job.job_id)
    return state


@router.get("/{session_id}/lifecycle")
def get_session_lifecycle(session_id: str):
    _validate_session_id(session_id)
    return lifecycle_status(session_id)


@router.delete("/{session_id}")
def delete_session_state(session_id: str):
    _validate_session_id(session_id)
    if lifecycle_status(session_id)['status'] == 'deleted':
        raise HTTPException(404, "Phiên làm việc không tồn tại.")
    result = delete_when_idle(session_id, _delete_session_records)
    return JSONResponse(result, status_code=200 if result['status'] == 'deleted' else 202)
