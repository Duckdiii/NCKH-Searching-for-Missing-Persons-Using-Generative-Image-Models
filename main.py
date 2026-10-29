"""
Pipeline chính: Missing Person Search via FADING (happy path)

Gọi tuần tự 6 bước:
  1.5. Align (FFHQ-exact) -> căn chỉnh ảnh input về đúng bố cục 140 ảnh FFHQ đã train Module 1
       - BẮT BUỘC cho mọi ảnh (kể cả ảnh FFHQ đã align sẵn vẫn đi qua, gần như bất biến qua
       align lần 2). Thiếu bước này, ảnh "trong tự nhiên" (chưa align, vd FG-NET) làm Module 2/3
       sinh sai hoàn toàn cấu trúc khuôn mặt (đã kiểm chứng thực nghiệm).
  1. Specialization  -> checkpoint UNet (tự bỏ qua nếu checkpoint đã có sẵn)
  2. Null-text Inversion -> (z_T, {null_t}, M_t_alpha) từ 1 ảnh input test + Initial Age
  3. Editing -> ảnh PNG cho từng target_age
  4. InsightFace Embedding -> gallery embedding + embedding từng ảnh vừa sinh
  5. FAISS Search -> top-K identity cho từng target_age (search riêng, không ensemble)

Đã verify end-to-end: ảnh sinh ra ở cả 3 target_age [30, 50, 70] đều tìm đúng Top-1 = "01366"
(chính ảnh gốc) trong gallery test.
"""

from src.utils.cancellation import checkpoint
import datetime
import os
import sys

# Ep stdout/stderr sang UTF-8 - tren Windows, khi output bi redirect ra file (khong phai
# console tuong tac), Python mac dinh dung code page cua he thong (thuong la cp1252), gay
# UnicodeEncodeError ngay khi print() chuoi co dau tieng Viet. Phat hien qua chay thu that.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import pandas as pd
import torch
import yaml
from typing import Optional, List, Dict, Tuple

from src.fading.editing import Editor
from src.fading.inversion import NullTextInverter
from src.fading.specialization import FFHQAgingDataset, Specializer
from src.search.embedding import FaceEmbedder
from src.search.ensemble import ensemble_search_results
from src.search.faiss_index import build_index, search
from src.search.rejection import apply_rejection_threshold
from src.utils.age_estimator import AgeEstimator
from src.utils.face_enhancement import preprocess_face_image
from src.utils.ffhq_align import align_to_ffhq
from src.utils.prompts import age_group_to_age, gender_to_word, compute_target_ages

CONFIG_PATH = "configs/config.yaml"

# ===== Chọn ảnh test (happy path - chỉ tên file là nhập tay, tuổi/giới tính TRA CUU tu CSV) =====
TEST_IMAGE_NAME = "01366.png"
PHOTO_YEAR = 2010  # Năm chụp ảnh
TARGET_AGES = compute_target_ages(34, PHOTO_YEAR)  # Mốc tuổi tính theo thời điểm hiện tại



def load_config() -> dict:
    """Đọc configs/config.yaml, dùng chung cho toàn bộ pipeline."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _lookup_csv_age_gender(labels_csv: str, image_name: str):
    """Tra cứu (age_group, gender) của 1 ảnh trong sampled_labels.csv theo image_number.
    CHỈ dùng được với ảnh có sẵn trong CSV (140 ảnh FFHQ đã gán nhãn). Trả về (row, initial_age,
    gender_word) - initial_age suy từ age_group_to_age(), gender_word từ gender_to_word()."""
    image_number = int(os.path.splitext(image_name)[0])

    df = pd.read_csv(labels_csv)
    matched = df[df["image_number"] == image_number]
    if matched.empty:
        raise ValueError(f"Không tìm thấy image_number={image_number} trong {labels_csv}")
    row = matched.iloc[0]

    initial_age = age_group_to_age(row["age_group"])
    gender_word = gender_to_word(row["gender"], initial_age)
    return row, initial_age, gender_word


def get_initial_age(
    image_path: str,
    method: str = "mivolo",
    labels_csv: str = None,
    age_estimator_config: dict = None,
) -> int:
    """Lấy INITIAL_AGE cho 1 ảnh, theo 2 nguồn phục vụ 2 mục đích khác nhau (giữ CẢ HAI):

      method="csv":    tra cứu labels_csv theo image_number của ảnh. Chỉ chạy được với ảnh
                       đã có nhãn sẵn - dùng để test/so sánh độ chính xác MiVOLO, hoặc debug
                       nhanh không cần load model.
      method="mivolo": gọi AgeEstimator.estimate() (MiVOLO) - dùng được với ẢNH BẤT KỲ, kể cả
                       ảnh thật của người mất tích không có CSV. Đây là đường dùng chính.

    Raise ValueError nếu method không hợp lệ, hoặc thiếu tham số bắt buộc cho method đã chọn.
    """
    if method == "csv":
        if labels_csv is None:
            raise ValueError("Cần labels_csv khi method='csv'")
        _, initial_age, _ = _lookup_csv_age_gender(labels_csv, os.path.basename(image_path))
        return initial_age
    elif method == "mivolo":
        if age_estimator_config is None:
            raise ValueError("Cần age_estimator_config (từ config.yaml) khi method='mivolo'")
        estimator = AgeEstimator(
            detector_checkpoint=age_estimator_config["detector_checkpoint"],
            age_checkpoint=age_estimator_config["age_checkpoint"],
            device=age_estimator_config.get("device", "cuda"),
        )
        return estimator.estimate(image_path)
    else:
        raise ValueError(f"method không hợp lệ: {method} (chỉ nhận 'csv' hoặc 'mivolo')")


def resolve_test_person(config: dict, image_name: str):
    """Trả về (test_image_path, initial_age, gender_word) cho ảnh test.

    - initial_age: theo config["initial_age"]["method"] ("csv" hoặc "mivolo").
    - gender_word: nếu method="csv" thì lấy luôn từ CSV; nếu "mivolo" thì lấy từ
      config["initial_age"]["gender_word"] (MiVOLO tập trung đoán tuổi; giới tính với ảnh
      không nhãn hiện nhập tay qua config, mặc định "man")."""
    ffhq_dir = config["paths"]["ffhq_dir"]
    labels_csv = config["paths"]["labels_csv"]
    method = config["initial_age"]["method"]
    image_path = os.path.join(ffhq_dir, image_name)

    if method == "csv":
        row, initial_age, gender_word = _lookup_csv_age_gender(labels_csv, image_name)
        print(
            f"[main] Ảnh: {image_name} | nguồn tuổi: CSV | age_group thật: {row['age_group']} | "
            f"gender thật: {row['gender']} => INITIAL_AGE={initial_age}, GENDER_WORD='{gender_word}'"
        )
    else:
        initial_age = get_initial_age(
            image_path, method="mivolo", age_estimator_config=config["age_estimator"]
        )
        gender_word = config["initial_age"].get("gender_word", "man")
        print(
            f"[main] Ảnh: {image_name} | nguồn tuổi: MiVOLO => INITIAL_AGE={initial_age}, "
            f"GENDER_WORD='{gender_word}' (nhập tay qua config)"
        )

    return image_path, initial_age, gender_word


def run_specialization(config: dict) -> str:
    """Module 1: chỉ train nếu checkpoint CHƯA tồn tại - tránh train lại 150 step (tốn thời
    gian) mỗi lần chạy main.py để debug. LUÔN in rõ ràng khi bỏ qua (kèm thời gian sửa đổi lần
    cuối của checkpoint), KHÔNG skip trong im lặng - để người chạy biết rõ đang dùng checkpoint
    cũ, tránh vô tình dùng nhầm model đã lỗi thời nếu đã sửa dữ liệu/code Module 1 sau đó."""
    ckpt_dir = config["paths"]["specialized_unet_ckpt"]

    if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
        mtime = max(
            os.path.getmtime(os.path.join(ckpt_dir, f)) for f in os.listdir(ckpt_dir)
        )
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[main] Dùng lại checkpoint có sẵn tại {ckpt_dir} (sửa đổi lần cuối: {mtime_str}). "
            f"Xoá thư mục này nếu muốn train lại Module 1."
        )
        return ckpt_dir

    print("[main] Chạy Module 1 (Specialization)...")
    dataset = FFHQAgingDataset(config["paths"]["ffhq_dir"], config["paths"]["labels_csv"])
    specializer = Specializer(
        pretrained_model_name_or_path=config["base_model"]["pretrained_model_name_or_path"],
        lr=float(config["specialization"]["lr"]),
        betas=tuple(config["specialization"]["betas"]),
        train_steps=config["specialization"]["train_steps"],
        batch_size=config["specialization"]["batch_size"],
    )
    specializer.train(dataset)
    specializer.save_checkpoint(ckpt_dir)

    del specializer
    torch.cuda.empty_cache()
    return ckpt_dir


def run_alignment(config: dict, embedder: FaceEmbedder, image_path: str) -> str:
    """Module 1.5 (MOI, sau Van de #1) - align_to_ffhq() BAT BUOC ap dung cho MOI anh input
    truoc khi vao Module 2 (_load_image_latent() trong inversion.py chi resize thang, khong
    tu align gi ca).

    FIX (phat hien qua chan doan thuc te, xem outputs/diag_fgnet_no_align_log.txt): dua thang
    anh CHUA align (vd FG-NET - anh "trong tu nhien", khong nhu FFHQ da duoc NVIDIA align san)
    qua main.py truoc day cho ID Score gan 0 hoac AM (vd -3.10%) - te hon ca truong hop
    align-2-lan tren anh FFHQ da align san (11.3%, xem Van de #1). Ly do: SD1.5 UNet fine-tune
    (Module 1) chi hoc tren anh co bo cuc dung FFHQ-align (mat chiem ti le/vi tri co dinh trong
    khung 256x256) - anh lech bo cuc lam Module 2/3 sinh sai hoan toan cau truc khuon mat.

    Dung lai DUNG code that cua app.py: embedder.detect_faces() (chay 1 lan, lay Face dau tien -
    happy path, khong co UI chon mat o day) -> align_to_ffhq() voi 4/5 diem mat/mieng. Neu
    khong detect duoc mat nao, raise ValueError ro rang (khong am tham dung anh goc chua align).

    Luu anh da align ra config["paths"]["output_dir"]/aligned_input.png (ghi de moi lan goi -
    chi la file trung gian, khong phai output can giu lau dai) va tra ve duong dan nay."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    faces = embedder.detect_faces(image_bgr)
    if len(faces) == 0:
        raise ValueError(f"Align that bai: khong phat hien duoc khuon mat nao trong {image_path}")

    # Tiền xử lý theo chuẩn Kaggle 3 & FG-NET batch: Adaptive Padding + Grayscale Check + Shades of Gray WB + CodeFormer
    image_bgr, updated_kps = preprocess_face_image(
        image_bgr, kps=faces[0].kps, bbox=faces[0].bbox, embedder=embedder
    )

    img_size = config.get("editing", {}).get("image_size", 512)
    aligned = align_to_ffhq(image_bgr, updated_kps, output_size=img_size)

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    aligned_path = os.path.join(output_dir, "aligned_input.png")
    cv2.imwrite(aligned_path, aligned)
    return aligned_path


def run_inversion(config: dict, ckpt_dir: str, test_image_path: str, initial_age: int, gender_word: str):
    """Module 2: invert test_image_path + initial_age thành (z_T, {null_t}, M_t_alpha)."""
    print("[main] Chạy Module 2 (Null-text Inversion)...")
    inverter = NullTextInverter(
        pretrained_model_name_or_path=config["base_model"]["pretrained_model_name_or_path"],
        unet_checkpoint_dir=ckpt_dir,
        num_inference_steps=config["inversion"]["num_inference_steps"],
        guidance_scale=config["inversion"]["guidance_scale"],
        num_inner_steps=config["inversion"]["num_inner_steps"],
        early_stop_epsilon=float(config["inversion"]["early_stop_epsilon"]),
        image_size=config.get("inversion", {}).get("image_size", 512),
        debug_check_nan=config["debug"]["check_nan"],
    )
    z_T, null_embeddings, attention_maps = inverter.invert(test_image_path, initial_age, gender_word)

    del inverter
    torch.cuda.empty_cache()
    return z_T, null_embeddings, attention_maps


def run_editing(
    config: dict,
    ckpt_dir: str,
    z_T,
    null_embeddings,
    attention_maps,
    gender_word: str,
    initial_age: Optional[int] = None,
    target_ages: Optional[List[int]] = None,
) -> dict:
    """Module 3: sinh ảnh PNG cho từng target_ages. num_inference_steps LUÔN lấy từ
    config["inversion"] (không phải config["editing"]) để đảm bảo khớp tuyệt đối với Module 2 -
    xem lý do trong comment của configs/config.yaml."""
    if target_ages is None:
        target_ages = TARGET_AGES

    print(f"[main] Chạy Module 3 (Editing) cho các mốc tuổi: {target_ages}...")
    editor = Editor(
        pretrained_model_name_or_path=config["base_model"]["pretrained_model_name_or_path"],
        unet_checkpoint_dir=ckpt_dir,
        num_inference_steps=config["inversion"]["num_inference_steps"],
        guidance_scale=config["editing"]["guidance_scale"],
        attention_control_ratio=config["editing"]["attention_control_ratio"],
        image_size=config.get("editing", {}).get("image_size", 512),
        use_local_blend=config["editing"].get("use_local_blend", True),
        local_blend_threshold=config["editing"].get("local_blend_threshold", 0.3),
        debug_check_nan=config["debug"]["check_nan"],
    )
    results = editor.edit(
        z_T,
        null_embeddings,
        attention_maps,
        target_ages,
        gender_word,
        config["paths"]["output_dir"],
        initial_age=initial_age,
    )

    del editor
    torch.cuda.empty_cache()
    return results  # {target_age: đường_dẫn_ảnh_PNG}


def run_embedding_and_search(config: dict, edited_images: dict, return_age_scores: bool = False):
    """Module 4 + Module 5: build gallery 1 lần, embed từng ảnh target_age sinh ra, search
    RIÊNG cho từng target_age (không ensemble ở Module 5 - đúng yêu cầu happy path gốc).

    Add-on (bọc bên ngoài Module 4/5, không đụng vào build_index()/search()): gộp kết quả
    tất cả target_age qua ensemble_search_results() (dùng XẾP HẠNG candidate) rồi áp rejection
    threshold TRÊN ĐIỂM THÔ CAO NHẤT của candidate đó (KHÔNG phải điểm ensemble đã pha loãng -
    xem FIX trong ensemble.py/rejection.py) - để có 1 quyết định CHẤP NHẬN/TỪ CHỐI cuối cùng
    thay vì phải tự đọc 3 bảng kết quả rời rạc.

    FIX (phát hiện qua debug thực tế): bọc try/except quanh embedder.embed() cho TỪNG
    target_age - nếu 1 ảnh sinh ra không detect được mặt (có thể xảy ra do FADING sinh ảnh
    chưa hoàn hảo), chỉ cảnh báo và bỏ qua đúng target_age đó, KHÔNG để crash cả vòng lặp làm
    mất kết quả của các target_age còn lại đã chạy thành công.

    Trả về (final_scores, accepted, top_identity, top_score[, age_scores]) để người gọi khác (vd app.py)
    dùng lại được mà không cần parse lại stdout."""
    print("[main] Chạy Module 4 (Embedding) + Module 5 (FAISS Search)...")
    embedder = FaceEmbedder(
        model_name=config["embedding"]["model_name"],
        ctx_id=config["embedding"]["ctx_id"],
        det_size=tuple(config["embedding"]["det_size"]),
    )

    gallery_embeddings, gallery_labels, failed_files = embedder.build_gallery(
        config["paths"]["gallery_test_dir"]
    )
    if failed_files:
        print(f"[main] CẢNH BÁO: {len(failed_files)} ảnh trong gallery bị lỗi (xem chi tiết ở trên).")

    index = build_index(gallery_embeddings)

    search_results_per_age = {}
    for target_age, image_path in edited_images.items():
        checkpoint()
        try:
            query_embedding = embedder.embed(image_path)
        except ValueError as e:
            print(f"[main] BỎ QUA target_age={target_age}: {e}")
            continue

        top_k = search(index, query_embedding, gallery_labels, k=config["search"]["top_k"])
        search_results_per_age[target_age] = top_k

        print(f"[main] target_age={target_age} ({image_path}):")
        for identity, score in top_k:
            print(f"    {identity}: {score:.4f}")

    # FIX (phát hiện qua chạy thật): ensemble_search_results() giờ trả thêm scores_per_identity
    # (điểm thô, chưa pha loãng) - apply_rejection_threshold() dùng final_scores để CHỌN top
    # candidate (xếp hạng), nhưng xét threshold trên điểm THÔ CAO NHẤT của candidate đó, không
    # phải điểm ensemble đã bị pha loãng bởi các target_age xa tuổi gốc (xem chi tiết trong
    # docstring ensemble.py/rejection.py).
    final_scores, scores_per_identity = ensemble_search_results(search_results_per_age)
    accepted, top_identity, top_score = apply_rejection_threshold(
        scores_per_identity, final_scores, threshold=config["search"]["rejection_threshold"]
    )

    # Trích xuất điểm id_score thô (đầy đủ số thập phân) cho từng mốc tuổi với top_identity
    age_scores: Dict[int, float] = {}
    for target_age, top_k in search_results_per_age.items():
        score_for_top = next((float(score) for identity, score in top_k if identity == top_identity), None)
        if score_for_top is not None:
            age_scores[int(target_age)] = score_for_top
            print(f"[ID_EVAL] Mốc tuổi {target_age}: id_score thô với {top_identity} = {score_for_top:.6f}")

    print(f"[main] --- Kết quả ensemble (gộp {len(search_results_per_age)} target_age, dùng để xếp hạng) ---")
    for identity, score in final_scores.items():
        print(f"    {identity}: {score:.4f}")

    if accepted:
        print(f"[main] Tìm thấy: {top_identity} (độ tin cậy - điểm thô cao nhất: {top_score:.2%})")
    else:
        print(
            f"[main] Không tìm thấy kết quả đủ tin cậy "
            f"(điểm thô cao nhất: {top_score:.2%} < ngưỡng {config['search']['rejection_threshold']:.0%})"
        )

    if return_age_scores:
        return final_scores, accepted, top_identity, top_score, age_scores

    return final_scores, accepted, top_identity, top_score


def main() -> None:
    config = load_config()
    test_image_path, initial_age, gender_word = resolve_test_person(config, TEST_IMAGE_NAME)

    ckpt_dir = run_specialization(config)

    # Module 1.5 (align) - BAT BUOC tu sau Van de #1: FADING UNet chi hoc tren bo cuc FFHQ-align,
    # anh lech bo cuc (vd anh "trong tu nhien") lam sinh sai hoan toan cau truc khuon mat. Tao
    # rieng 1 FaceEmbedder o day (Module 4 trong run_embedding_and_search tu tao instance rieng
    # cua no - khong dung chung, chap nhan load buffalo_l 2 lan de giu 2 ham doc lap nhau).
    print("[main] Chạy Module 1.5 (Align FFHQ-exact)...")
    embedder = FaceEmbedder(
        model_name=config["embedding"]["model_name"],
        ctx_id=config["embedding"]["ctx_id"],
        det_size=tuple(config["embedding"]["det_size"]),
    )
    aligned_image_path = run_alignment(config, embedder, test_image_path)
    print(f"[main] Đã align, ảnh dùng cho Module 2: {aligned_image_path}")

    z_T, null_embeddings, attention_maps = run_inversion(
        config, ckpt_dir, aligned_image_path, initial_age, gender_word
    )
    edited_images = run_editing(
        config, ckpt_dir, z_T, null_embeddings, attention_maps, gender_word, initial_age=initial_age
    )
    run_embedding_and_search(config, edited_images)


if __name__ == "__main__":
    main()
