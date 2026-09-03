"""
app.py - Streamlit UI cho pipeline Missing Person Search via FADING.

Đây là lớp add-on 3 (UI chọn khuôn mặt) + lớp add-on 4 (cảnh báo chất lượng ảnh), bọc quanh
pipeline chính - KHÔNG viết lại logic Module 1-5: tái sử dụng TRỰC TIẾP các hàm
run_specialization/run_inversion/run_editing/run_embedding_and_search/get_initial_age đã có
trong main.py (import, không copy code), main.py vẫn là nguồn sự thật duy nhất của pipeline.

Luồng: upload ảnh -> detect_faces() (1 lần detect duy nhất, dùng chung cho cả bước chọn mặt
lẫn cảnh báo chất lượng) -> 0 mặt: báo lỗi dừng; 1 mặt: tự chọn; nhiều mặt: chọn qua UI ->
check_image_quality() hiện cảnh báo (không chặn) -> chọn giới tính (MiVOLO chỉ đoán tuổi,
không đoán giới tính) -> nút "Chạy pipeline" gọi tuần tự đúng các hàm run_* của main.py ->
hiện ảnh sinh ra + bảng final_scores (đã ensemble) + kết quả accept/reject.

FIX v1 (phát hiện qua chạy thử thật app.py với ảnh trẻ em - ảnh sinh ra bị méo/lệch rõ rệt so
với kết quả main.py trên cùng identity): bản đầu chỉ crop_face() theo bounding box thô, KHÔNG
align gì cả - đã sửa bằng insightface.utils.face_align.norm_crop() (template ArcFace).

FIX v2 (phát hiện qua kiểm chứng lại bằng số liệu thật: align ArcFace làm mặt "to" hơn 90.4%
khung so với đúng 76.3% mà 140 ảnh FFHQ dùng train Module 1 có - lệch ~15%, vẫn làm giảm điểm
search dù đã align "đúng hướng"): ArcFace là 1 chuẩn align KHÁC, không phải chuẩn FFHQ dùng để
tạo ra chính bộ ảnh Module 1 đã học. Đã thay bằng src/utils/ffhq_align.align_to_ffhq() - dùng
ĐÚNG NGUYÊN VĂN công thức gốc công bố bởi NVlabs/ffhq-dataset (download_ffhq.py hàm
recreate_aligned_images, xem chi tiết trích dẫn trong ffhq_align.py), không phải xấp xỉ đo đạc.
Đã verify: align lại 1 ảnh FFHQ đã align sẵn cho kết quả gần như y hệt bản gốc (chiều cao mặt
74.9% so với gốc 76.3%, sai lệch pixel trung bình ~3.4%) - đúng tính "bất biến" kỳ vọng.

FIX v3 (vấn đề #2 - phát hiện qua kiểm chứng thật: MiVOLO đoán 17 tuổi cho ảnh có ground-truth
12 tuổi, chênh 5 năm, nằm trong MAE đã biết của MiVOLO ~4.2-4.3 năm - không phải bug, là hạn
chế cố hữu của MỌI age estimator): KHÔNG "sửa" MiVOLO (không khả thi) - tách 3 nguồn lấy
Initial Age riêng biệt (xem src/utils/age_estimator.resolve_initial_age), ưu tiên tuổi gia
đình tự nhập (đáng tin ngang ground truth) hơn hẳn để MiVOLO tự đoán mặc định như bản trước.

Chạy: streamlit run app.py
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import main as pipeline
from src.search.embedding import FaceEmbedder
from src.utils.age_estimator import AgeEstimator, resolve_initial_age
from src.utils.ffhq_align import align_to_ffhq
from src.utils.head_pose import check_image_quality

st.set_page_config(page_title="Missing Person Search via FADING", layout="wide")


@st.cache_resource
def get_embedder(_config: dict) -> FaceEmbedder:
    """Cache FaceEmbedder (insightface buffalo_l) qua các lần rerun của Streamlit - tránh
    load lại model mỗi lần người dùng tương tác với UI (chọn mặt, đổi giới tính...). Tham số
    đặt tên `_config` (gạch dưới đầu) theo đúng quy ước của st.cache_resource: tham số bắt
    đầu bằng `_` không bị đưa vào tính hash (dict config không hashable)."""
    return FaceEmbedder(
        model_name=_config["embedding"]["model_name"],
        ctx_id=_config["embedding"]["ctx_id"],
        det_size=tuple(_config["embedding"]["det_size"]),
    )


@st.cache_resource
def get_age_estimator(_config: dict) -> AgeEstimator:
    """Cache AgeEstimator (MiVOLO) qua các lần rerun của Streamlit - tránh load lại model
    YOLO detector + MiVOLO age/gender (nặng) mỗi lần người dùng tương tác UI. Tham số `_config`
    - xem quy ước dấu gạch dưới ở get_embedder()."""
    return AgeEstimator(
        detector_checkpoint=_config["age_estimator"]["detector_checkpoint"],
        age_checkpoint=_config["age_estimator"]["age_checkpoint"],
        device=_config["age_estimator"].get("device", "cuda"),
    )


def crop_face(image_bgr: np.ndarray, kps: np.ndarray, out_size: int = 256) -> np.ndarray:
    """Align + crop mặt về out_size x out_size (mặc định 256, khớp kích thước ảnh FADING đã
    tinh chỉnh), dùng align_to_ffhq() - ĐÚNG công thức gốc NVlabs/ffhq-dataset, khớp tỷ lệ
    khung hình mà 140 ảnh Module 1 đã học (xem FIX v2 ở đầu file), KHÔNG phải template
    ArcFace (khác tỷ lệ) hay crop bbox thô (bản đầu tiên).

    kps: 5 điểm mốc insightface (Face.kps) - mắt trái, mắt phải, mũi, khoé miệng trái, khoé
    miệng phải (điểm mũi không dùng đến, xem ffhq_align.py)."""
    return align_to_ffhq(image_bgr, kps, output_size=out_size)


def draw_numbered_boxes(image_rgb: np.ndarray, faces: list) -> Image.Image:
    """Vẽ box đỏ đánh số quanh từng khuôn mặt phát hiện được - dùng khi ảnh có nhiều người,
    để người dùng chọn đúng người cần tìm."""
    pil_image = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 20)), f"#{i + 1}", fill="red")
    return pil_image


def main_ui() -> None:
    st.title("Missing Person Search via FADING")
    st.caption("Upload ảnh → chọn mặt → cảnh báo chất lượng → chạy pipeline thật → kết quả")

    config = pipeline.load_config()
    embedder = get_embedder(config)

    uploaded_file = st.file_uploader("Chọn ảnh (có khuôn mặt)", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Vui lòng upload 1 ảnh để bắt đầu.")
        return

    file_bytes = uploaded_file.getvalue()
    file_key = f"{uploaded_file.name}_{len(file_bytes)}"

    # File MỚI (khác lần upload trước) -> xoá sạch session_state, tránh giữ lại kết quả/lựa
    # chọn của ảnh cũ (đúng rủi ro #2 đã lường trước: session_state phải phản ánh đúng ảnh
    # đang xử lý, không lẫn giữa các lần upload).
    if st.session_state.get("file_key") != file_key:
        st.session_state.clear()
        st.session_state["file_key"] = file_key

    image_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if "faces" not in st.session_state:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            st.session_state["faces"] = embedder.detect_faces(tmp_path)
        finally:
            os.remove(tmp_path)

    faces = st.session_state["faces"]

    if len(faces) == 0:
        st.error("Không phát hiện khuôn mặt nào trong ảnh. Vui lòng chọn ảnh khác.")
        st.stop()
    elif len(faces) == 1:
        selected_idx = 0
        st.image(image_rgb, caption="Ảnh đã upload (1 khuôn mặt)", width=300)
    else:
        st.image(draw_numbered_boxes(image_rgb, faces), caption="Đã phát hiện nhiều khuôn mặt")
        selected_idx = st.selectbox(
            "Chọn khuôn mặt của người cần tìm:",
            options=range(len(faces)),
            format_func=lambda i: f"Khuôn mặt #{i + 1}",
        )

    chosen_face = faces[selected_idx]

    warnings = check_image_quality(image_bgr, chosen_face.kps, float(chosen_face.det_score))
    for w in warnings:
        st.warning(f"⚠️ {w}")
    if warnings:
        st.info("Vẫn có thể tiếp tục, nhưng nên chọn ảnh khác nếu có.")

    # Align + lưu ảnh mặt NGAY tại đây (không đợi đến khi bấm "Chạy pipeline") - vì bước
    # đoán tuổi MiVOLO (nếu người dùng chọn nhánh đó) cần ảnh này để chạy trước, cho người
    # dùng xem trước kết quả đoán tuổi mà chưa cần bấm nút.
    cropped = crop_face(image_bgr, chosen_face.kps)
    os.makedirs("outputs/app_uploads", exist_ok=True)
    cropped_path = os.path.join("outputs/app_uploads", f"{file_key}.png")
    cv2.imwrite(cropped_path, cropped)

    gender_word = st.selectbox("Giới tính của người trong ảnh:", options=["man", "woman"])

    # FIX v3 (vấn đề #2): 3 nguồn IA tách biệt, KHÔNG mặc định MiVOLO như bản trước - ưu tiên
    # tuổi gia đình tự biết (đáng tin ngang ground truth), chỉ dùng MiVOLO khi không có lựa
    # chọn nào đáng tin cậy hơn, và LUÔN cảnh báo rõ sai số vốn có của nó.
    age_input_mode = st.radio(
        "Bạn có biết chính xác tuổi của người trong ảnh lúc chụp không?",
        ["Có, tôi biết chính xác tuổi", "Không, nhờ hệ thống ước tính"],
    )

    if age_input_mode == "Có, tôi biết chính xác tuổi":
        manual_age = st.number_input("Nhập tuổi lúc chụp ảnh:", min_value=0, max_value=120, value=10)
        initial_age = resolve_initial_age(mode="manual", manual_age=manual_age)
    else:
        age_estimator = get_age_estimator(config)
        with st.spinner("Đang ước tính tuổi (MiVOLO)..."):
            initial_age = resolve_initial_age(mode="mivolo", image_path=cropped_path, age_estimator=age_estimator)
        st.info(
            f"Hệ thống ước tính khoảng {initial_age} tuổi. "
            f"Lưu ý: có thể sai lệch ±4-5 năm so với thực tế (sai số vốn có của MiVOLO, đã đo thật)."
        )

    if st.button("Chạy pipeline"):
        with st.spinner("Đang chuẩn bị checkpoint (Module 1 - bỏ qua nếu đã có sẵn)..."):
            ckpt_dir = pipeline.run_specialization(config)

        with st.spinner("Đang chạy Module 2 (Null-text Inversion) - có thể mất vài phút..."):
            z_T, null_embeddings, attention_maps = pipeline.run_inversion(
                config, ckpt_dir, cropped_path, initial_age, gender_word
            )

        with st.spinner("Đang chạy Module 3 (Editing)..."):
            edited_images = pipeline.run_editing(
                config, ckpt_dir, z_T, null_embeddings, attention_maps, gender_word
            )

        with st.spinner("Đang chạy Module 4+5 (Embedding + FAISS Search)..."):
            final_scores, accepted, top_identity, top_score = pipeline.run_embedding_and_search(
                config, edited_images
            )

        # Lưu vào session_state (không chỉ hiện trong nhánh if này) - để kết quả KHÔNG biến
        # mất khi Streamlit rerun lại toàn bộ script do 1 tương tác UI khác sau đó (rủi ro #2
        # đã lường trước: st.button() chỉ True đúng lần rerun người dùng bấm nút).
        st.session_state["edited_images"] = edited_images
        st.session_state["final_scores"] = final_scores
        st.session_state["accepted"] = accepted
        st.session_state["top_identity"] = top_identity
        st.session_state["top_score"] = top_score

    if "final_scores" in st.session_state:
        edited_images = st.session_state["edited_images"]
        st.subheader("Ảnh sinh ra theo từng độ tuổi")
        cols = st.columns(len(edited_images))
        for col, (age, path) in zip(cols, edited_images.items()):
            col.image(path, caption=f"{age} tuổi")

        st.subheader("Kết quả tìm kiếm (đã qua ensemble)")
        final_scores = st.session_state["final_scores"]
        st.table({"identity": list(final_scores.keys()), "score": list(final_scores.values())})

        if st.session_state["accepted"]:
            st.success(
                f"Tìm thấy: {st.session_state['top_identity']} "
                f"(độ tin cậy: {st.session_state['top_score']:.2%})"
            )
        else:
            st.error(
                f"Không tìm thấy kết quả đủ tin cậy (điểm cao nhất: "
                f"{st.session_state['top_score']:.2%} < ngưỡng "
                f"{config['search']['rejection_threshold']:.0%})"
            )


if __name__ == "__main__":
    main_ui()
