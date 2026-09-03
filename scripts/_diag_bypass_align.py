"""
Script chan doan TAM THOI (khong phai code san pham) - chay thang cac ham cua main.py voi
anh GOC 01366.png, KHONG qua buoc crop_face()/align_to_ffhq() nao ca - de tach bach xem
viec align 2 lan lien tiep (dung cong thuc) co phai nguyen nhan gay sut diem hay khong.

Chi dung de chan doan 1 lan, khong nhet vao app.py chinh (anh that luon can align that su).
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import main as pipeline

IMAGE_PATH = "D:/Data/project/nckh/ffhq_aging_150_samples/01366.png"  # ANH GOC, KHONG align
INITIAL_AGE = 12
GENDER_WORD = "boy"  # FIX: dung dung gender_to_word(male, 12) = "boy", KHONG PHAI "man"

config = pipeline.load_config()
ckpt_dir = pipeline.run_specialization(config)  # se tu bo qua vi checkpoint da co san

print()
print(f"=== BYPASS ALIGN: dung thang {IMAGE_PATH}, IA={INITIAL_AGE}, gender={GENDER_WORD} ===")

# DEBUG DOI CHIEU (tam thoi): in het gia tri runtime THAT truoc khi goi Module 2/3, de
# doi chieu truc tiep voi main.py - khong suy luan qua doc code.
print(
    "[DEBUG-DOICHIEU][_diag_bypass_align.py] "
    f"test_image_path={IMAGE_PATH}, initial_age={INITIAL_AGE}, gender_word={GENDER_WORD}, "
    f"target_ages={pipeline.TARGET_AGES}, "
    f"inversion.num_inference_steps={config['inversion']['num_inference_steps']}, "
    f"inversion.guidance_scale={config['inversion']['guidance_scale']}, "
    f"editing.guidance_scale={config['editing']['guidance_scale']}, "
    f"editing.attention_control_ratio={config['editing']['attention_control_ratio']}"
)

z_T, null_embeddings, attention_maps = pipeline.run_inversion(
    config, ckpt_dir, IMAGE_PATH, INITIAL_AGE, GENDER_WORD
)
edited_images = pipeline.run_editing(config, ckpt_dir, z_T, null_embeddings, attention_maps, GENDER_WORD)
final_scores, accepted, top_identity, top_score = pipeline.run_embedding_and_search(config, edited_images)

print()
print("=== KET QUA THAT (BYPASS ALIGN) ===")
print("final_scores day du:")
for identity, score in final_scores.items():
    print(f"    {identity}: {score:.4f}")
print()
print(f"accepted={accepted}, top_identity={top_identity}, top_score={top_score}")
print(f"Diem cua chinh identity 01366: {final_scores.get('01366', 'KHONG XUAT HIEN trong bang')}")
