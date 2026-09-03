"""
Script chan doan TAM THOI - phep thu con thieu cuoi cung cho "Van de #1" (Alignment):
chay dung pipeline app.py se lam (detect_faces() -> align_to_ffhq()) NHUNG voi IA=12 (dung,
nhap tay - khong dung MiVOLO) va gender="boy" (dung, khop voi baseline moi ~57%) - de tach
bach xem RIENG bien "align FFHQ-exact" con anh huong bao nhieu so voi baseline moi (khong
align) ~57%, sau khi da loai bo het cac bien nhieu khac (IA sai, gender sai) o cac phep thu
truoc.

Chi dung de chan doan 1 lan, khong nhet vao app.py chinh.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import cv2

import main as pipeline
from src.search.embedding import FaceEmbedder
from src.utils.ffhq_align import align_to_ffhq

RAW_IMAGE_PATH = "D:/Data/project/nckh/ffhq_aging_150_samples/01366.png"
ALIGNED_IMAGE_PATH = "./outputs/_diag_01366_ffhq_aligned.png"
INITIAL_AGE = 12
GENDER_WORD = "boy"

config = pipeline.load_config()

# Buoc align - dung DUNG code app.py dang dung: detect_faces() -> lay face dau tien -> align_to_ffhq()
print("=== Buoc 1: align FFHQ-exact (dung code that cua embedding.py + ffhq_align.py) ===")
embedder = FaceEmbedder(
    model_name=config["embedding"]["model_name"],
    ctx_id=config["embedding"]["ctx_id"],
    det_size=tuple(config["embedding"]["det_size"]),
)
faces = embedder.detect_faces(RAW_IMAGE_PATH)
if len(faces) == 0:
    raise SystemExit(f"Khong phat hien duoc mat nao trong {RAW_IMAGE_PATH}")
face = faces[0]
print(f"Phat hien {len(faces)} mat, dung face[0]: det_score={face.det_score:.4f}, kps={face.kps.tolist()}")

image_bgr = cv2.imread(RAW_IMAGE_PATH)
aligned = align_to_ffhq(image_bgr, face.kps, output_size=256)
cv2.imwrite(ALIGNED_IMAGE_PATH, aligned)
print(f"Da luu anh align -> {ALIGNED_IMAGE_PATH}")

ckpt_dir = pipeline.run_specialization(config)

print()
print(f"=== FFHQ-EXACT ALIGN: dung {ALIGNED_IMAGE_PATH}, IA={INITIAL_AGE}, gender={GENDER_WORD} ===")
print(
    "[DEBUG-DOICHIEU][_diag_ffhq_align_boy.py] "
    f"test_image_path={ALIGNED_IMAGE_PATH}, initial_age={INITIAL_AGE}, gender_word={GENDER_WORD}, "
    f"target_ages={pipeline.TARGET_AGES}, "
    f"inversion.num_inference_steps={config['inversion']['num_inference_steps']}, "
    f"inversion.guidance_scale={config['inversion']['guidance_scale']}, "
    f"editing.guidance_scale={config['editing']['guidance_scale']}, "
    f"editing.attention_control_ratio={config['editing']['attention_control_ratio']}"
)

z_T, null_embeddings, attention_maps = pipeline.run_inversion(
    config, ckpt_dir, ALIGNED_IMAGE_PATH, INITIAL_AGE, GENDER_WORD
)
edited_images = pipeline.run_editing(config, ckpt_dir, z_T, null_embeddings, attention_maps, GENDER_WORD)
final_scores, accepted, top_identity, top_score = pipeline.run_embedding_and_search(config, edited_images)

print()
print("=== KET QUA THAT (FFHQ-EXACT ALIGN + IA=12 dung + gender=boy dung) ===")
print("final_scores day du:")
for identity, score in final_scores.items():
    print(f"    {identity}: {score:.4f}")
print()
print(f"accepted={accepted}, top_identity={top_identity}, top_score={top_score}")
print(f"Diem cua chinh identity 01366: {final_scores.get('01366', 'KHONG XUAT HIEN trong bang')}")
