"""
Script chan doan TAM THOI - kiem tra dua thang anh FG-NET (CHUA align) qua main.py
HIEN TAI (khong sua code gi ca, khong them buoc align nao) - de quan sat pipeline
phan ung the nao voi anh "trong tu nhien" (chua duoc can chinh nhu FFHQ), lam co so
quyet dinh co can them align vao main.py hay khong.

3 anh FG-NET chon: 001A05.JPG (5 tuoi), 002A12.JPG (12 tuoi), 001A33.JPG (33 tuoi) -
parse tuoi tu ten file theo pattern (\\d{3})A(\\d{2}).

Gender_word = "person" (trung tinh, FG-NET khong co nhan gioi tinh san).
Chi chay 1 target_age = initial_age + 20 cho moi anh (khong ensemble/nhieu target_age).

ID Score = cosine similarity (dot product, vi normed_embedding da L2-normalize san)
giua embedding cua CHINH anh FG-NET goc (dung lam input) va embedding cua anh sinh ra -
dung lai FaceEmbedder.embed() that, KHONG dung gallery/FAISS search.
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

    # Buoc 1: embed ANH GOC (FG-NET, chua align) - de co ID Score doi chieu sau nay.
    try:
        original_embedding = embedder.embed(image_path)
        print("[embed goc] Phat hien duoc mat, da lay embedding.")
    except ValueError as e:
        original_embedding = None
        print(f"[embed goc] KHONG phat hien duoc mat trong anh goc: {e}")

    # Buoc 2+3: Module 2 (Inversion) + Module 3 (Editing, CHI 1 target_age).
    pipeline.TARGET_AGES = [target_age]  # ghi de tam thoi bien global cua main.py cho phep thu nay
    z_T, null_embeddings, attention_maps = pipeline.run_inversion(
        config, ckpt_dir, image_path, initial_age, GENDER_WORD
    )
    edited_images = pipeline.run_editing(config, ckpt_dir, z_T, null_embeddings, attention_maps, GENDER_WORD)
    generated_path = edited_images[target_age]
    print(f"[Module 3] Anh sinh ra: {generated_path}")

    # Buoc 4: embed ANH SINH RA, tinh ID Score = cosine similarity voi anh goc.
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
print("===== TOM TAT KET QUA THAT (FG-NET, KHONG ALIGN, qua main.py hien tai) =====")
for r in results:
    print(
        f"{r['image_name']} (IA={r['initial_age']} -> target={r['target_age']}): "
        f"goc_detect={r['original_face_detected']}, sinh_ra_detect={r['generated_face_detected']}, "
        f"ID_Score={r['id_score']}"
    )
    print(f"    Anh sinh ra luu tai: {r['generated_path']}")
