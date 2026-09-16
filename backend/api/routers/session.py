import os
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
from backend.api.session_store import SessionState, get_session, save_session
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
    save_session(session_state)

    face_boxes = [
        FaceBox(
            index=i,
            bbox=[float(v) for v in face.bbox],
            det_score=float(face.det_score)
        )
        for i, face in enumerate(faces)
    ]

    return UploadResponse(session_id=session_id, faces=face_boxes)


@router.post("/{session_id}/select-face", response_model=SelectFaceResponse)
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
    os.makedirs("outputs/app_uploads", exist_ok=True)
    cropped_filename = f"{session_id}_crop.png"
    cropped_path = os.path.join("outputs", "app_uploads", cropped_filename)
    cv2.imwrite(cropped_path, cropped)

    session.cropped_path = cropped_path
    save_session(session)

    cropped_preview_url = f"/outputs/app_uploads/{cropped_filename}"
    return SelectFaceResponse(warnings=warnings, cropped_preview_url=cropped_preview_url)


@router.post("/{session_id}/restore-preview", response_model=RestorePreviewResponse)
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
    os.makedirs("outputs/app_uploads", exist_ok=True)
    preview_filename = f"{session_id}_preview.png"
    preview_path = os.path.join("outputs", "app_uploads", preview_filename)
    cv2.imwrite(preview_path, cropped)

    preview_url = f"/outputs/app_uploads/{preview_filename}"
    return RestorePreviewResponse(preview_url=preview_url, wb_info=wb_info)


@router.post("/{session_id}/apply-restore")
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

    os.makedirs("outputs/app_uploads", exist_ok=True)
    cropped_filename = f"{session_id}_crop.png"
    cropped_path = os.path.join("outputs", "app_uploads", cropped_filename)
    cv2.imwrite(cropped_path, cropped)

    session.cropped_path = cropped_path
    save_session(session)

    cropped_preview_url = f"/outputs/app_uploads/{cropped_filename}"
    return {"status": "ok", "cropped_preview_url": cropped_preview_url}


@router.post("/{session_id}/resolve-age", response_model=ResolveAgeResponse)
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

    return ResolveAgeResponse(
        initial_age=initial_age,
        gender_word=session.gender_word,
        warning_text=warning_text
    )
