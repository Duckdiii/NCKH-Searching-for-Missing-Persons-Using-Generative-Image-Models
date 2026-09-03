"""
Script chan doan TAM THOI - CHAY LAI dung 3 anh FG-NET da test truoc (001A05, 002A12,
001A33) SAU KHI da them Module 1.5 (align_to_ffhq()) vao main.py - de so sanh TRUOC/SAU
voi ID Score cu (1.70%, -1.08%, -3.10%) da do o outputs/diag_fgnet_no_align_log.txt.

Khac voi _diag_fgnet_no_align.py CHI o 1 diem: goi them pipeline.run_alignment() truoc
run_inversion() - dung DUNG ham that vua them vao main.py (khong viet lai logic align rieng
o day).
"""

import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

import main as pipeline
from src.search.embedding import FaceEmbedder

FGNET_DIR = "D:/Data/project/nckh/data/FGNET (1)/FGNET/images"
IMAGE_NAMES = ["001A05.JPG", "002A12.JPG", "001A33.JPG"]
GENDER_WORD = "person"
FILENAME_PATTERN = re.compile(r"^(\d{3})A(\d{2})")

config = pipeline.load_config()
ckpt_dir = pipeline.run_specialization(config)

embedder = FaceEmbedder(
    model_name=config["embedding"]["model_name"],
    ctx_id=config["embedding"]["ctx_id"],
    det_size=tuple(config["embedding"]["det_size"]),
)

results = []

for image_name in IMAGE_NAMES:
    image_path = os.path.join(FGNET_DIR, image_name)
    match = FILENAME_PATTERN.match(image_name)
    if not match:
        raise ValueError(f"Ten file khong khop pattern: {image_name}")
    person_id, initial_age = match.group(1), int(match.group(2))
    target_age = initial_age + 20

    print()
    print(f"===== Anh: {image_name} | person_id={person_id} | IA={initial_age} | target_age={target_age} =====")

    # Buoc 1: embed ANH GOC (chua align) - lam ID Score doi chieu, giong het script cu.
    try:
        original_embedding = embedder.embed(image_path)
        print("[embed goc] Phat hien duoc mat, da lay embedding.")
    except ValueError as e:
        original_embedding = None
        print(f"[embed goc] KHONG phat hien duoc mat trong anh goc: {e}")

    # Buoc 2 (MOI): align_to_ffhq() - dung DUNG ham vua them vao main.py.
    aligned_image_path = pipeline.run_alignment(config, embedder, image_path)
    print(f"[align] Da align -> {aligned_image_path}")

    # Buoc 3+4: Module 2 (Inversion) + Module 3 (Editing, CHI 1 target_age) - tren ANH DA ALIGN.
    pipeline.TARGET_AGES = [target_age]
    z_T, null_embeddings, attention_maps = pipeline.run_inversion(
        config, ckpt_dir, aligned_image_path, initial_age, GENDER_WORD
    )
    edited_images = pipeline.run_editing(config, ckpt_dir, z_T, null_embeddings, attention_maps, GENDER_WORD)
    generated_path = edited_images[target_age]
    print(f"[Module 3] Anh sinh ra: {generated_path}")

    # Buoc 5: embed ANH SINH RA, tinh ID Score = cosine similarity voi anh GOC (chua align,
    # y het cach tinh trong script cu, de so sanh cong bang).
    try:
        generated_embedding = embedder.embed(generated_path)
        if original_embedding is not None:
            id_score = float(np.dot(original_embedding, generated_embedding))
            print(f"[ID Score] cosine(anh_goc, anh_sinh_ra) = {id_score:.4f} ({id_score:.2%})")
        else:
            id_score = None
            print("[ID Score] KHONG tinh duoc - anh goc khong detect duoc mat.")
    except ValueError as e:
        generated_embedding = None
        id_score = None
        print(f"[embed anh sinh ra] KHONG phat hien duoc mat trong anh sinh ra: {e}")

    results.append(
        {
            "image_name": image_name,
            "initial_age": initial_age,
            "target_age": target_age,
            "generated_path": generated_path,
            "original_face_detected": original_embedding is not None,
            "generated_face_detected": generated_embedding is not None,
            "id_score": id_score,
        }
    )

print()
print("===== TOM TAT KET QUA THAT (FG-NET, CO ALIGN, qua main.py sau khi sua) =====")
for r in results:
    print(
        f"{r['image_name']} (IA={r['initial_age']} -> target={r['target_age']}): "
        f"goc_detect={r['original_face_detected']}, sinh_ra_detect={r['generated_face_detected']}, "
        f"ID_Score={r['id_score']}"
    )
    print(f"    Anh sinh ra luu tai: {r['generated_path']}")
