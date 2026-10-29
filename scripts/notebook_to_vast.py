"""Chuyển notebook Kaggle -> bản chạy trên vast.ai (papermill, không tương tác).

    python scripts/notebook_to_vast.py notebooks/FADING_eval_fgnet.ipynb eval_vast.ipynb eval
    python scripts/notebook_to_vast.py notebooks/batch_preprocess_fgnet.ipynb prep_vast.ipynb prep

Bố cục trên máy vast: /workspace/nckh/{data/FGNET, checkpoints/specialized_unet, old/ (CSV + aligned cũ), work/}.
"""
import json
import sys

src, dst, kind = sys.argv[1], sys.argv[2], sys.argv[3]
nb = json.load(open(src, encoding="utf-8"))

W = "/workspace/nckh"
REPL = [
    ("/kaggle/working", f"{W}/work"),
    ("runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
]
if kind == "eval":
    REPL += [
        ('DATA_DIR = "/kaggle/input/datasets/menonkk/nckh-2025-2026"', f'DATA_DIR = "{W}/data"'),
        ('"/kaggle/input/fgnet-preprocessed-full/FGNET_preprocessed"', f'"{W}/work/FGNET_preprocessed"'),
        ('SHOW_INLINE_IMAGES = True', 'SHOW_INLINE_IMAGES = False'),
    ]
    # các cell demo ảnh đơn (FFHQ) — không thuộc đánh giá FG-NET
    DROP_MARKERS = ["TEST_IMAGE_NAME = \"01366.png\"\nTEST_IMAGE_PATH","inverter.invert(\n    ALIGNED_TEST_IMAGE_PATH",
                    "RECONSTRUCTION SANITY-CHECK", "editor.edit(\n    z_T", "Cosine Similarity (ID Score) cho các mốc tuổi"]
else:
    REPL += [
        ('"/kaggle/input/datasets/menonkk/nckh-2025-2026/FGNET (1)/FGNET/images",',
         f'"{W}/data/FGNET/images",'),
    ]
    DROP_MARKERS = ["KNOWN_BUG_FILES", "plt.savefig", "test_sanity_files"]

kept, dropped = [], 0
for c in nb["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "code" and any(m in s for m in DROP_MARKERS):
        dropped += 1
        continue
    for a, b in REPL:
        s = s.replace(a, b)
    # [vast] thư viện đã cài sẵn (onnxruntime-gpu 1.22 cho CUDA 12) — không để !pip nâng cấp lại
    s = "\n".join(("# [vast] đã cài sẵn: " + l) if l.lstrip().startswith("!pip install") else l
                  for l in s.split("\n"))
    if kind == "eval" and '"ctx_id": -1,' in s and '"model_name": "buffalo_l"' in s:
        # Host 128 lõi nhưng container ~6.5 lõi: ORT CPU tạo hàng trăm luồng, rất chậm.
        s = s.replace('"ctx_id": -1,', '"ctx_id": 0,  # [vast] GPU')
    if kind == "eval" and "CKPT_DIR = DATA_DIR" in s:
        s = s.replace('CKPT_DIR = DATA_DIR  #', f'CKPT_DIR = "{W}/checkpoints/specialized_unet"  #')
    if kind == "eval" and "MAX_PAIRS_PER_PERSON = 2" in s:
        # Dùng ĐÚNG các cặp của lần chạy cũ để so sánh được với kết quả trước.
        s += f'''

# [vast] Ghi đè all_pairs = các cặp trong CSV cũ (so sánh ghép cặp với lần chạy trước)
import csv as _csv
_OLD_CSV = "{W}/old/fgnet_eval_results.csv"
if os.path.isfile(_OLD_CSV):
    with open(_OLD_CSV, encoding="utf-8") as _f:
        all_pairs = [{{
            "person_id": r["person_id"],
            "source_img": os.path.join(fgnet_images_dir, os.path.basename(r["source_img"])),
            "source_age": int(r["source_age"]),
            "target_img": os.path.join(fgnet_images_dir, os.path.basename(r["target_img"])),
            "target_age": int(r["target_age"]),
        }} for r in _csv.DictReader(_f)]
    _missing = [p for p in all_pairs if not (os.path.isfile(p["source_img"]) and os.path.isfile(p["target_img"]))]
    assert not _missing, f"Thiếu ảnh cho {{len(_missing)}} cặp: {{_missing[:3]}}"
    _labels = set(gallery_labels_fgnet)
    assert all(p["person_id"] in _labels for p in all_pairs), "person_id CSV cũ không khớp nhãn gallery"
    with open(os.path.join(OUTPUT_DIR, "fgnet_total_pairs.txt"), "w") as _f:
        _f.write(str(len(all_pairs)))
    print(f"[vast] Dùng {{len(all_pairs)}} cặp từ CSV cũ.")
'''
    if kind == "eval" and "aligned_source_path = align_image_for_pipeline(" in s and "fgnet_eval_results_v2" in s:
        old = '''                aligned_source_path = align_image_for_pipeline(
                    preprocessed_source_path, embedder, FGNET_ALIGNED_DIR, unique_tag
                )'''
        assert old in s
        s = s.replace(old, f'''                _reuse_aligned = os.path.join("{W}/old/fgnet_aligned_inputs", f"aligned_{{unique_tag}}.png")
                if os.path.isfile(_reuse_aligned):
                    # [vast] Ảnh nguồn đã tiền xử lý + căn chỉnh từ lần chạy cũ (đầu vào y hệt)
                    aligned_source_path = _reuse_aligned
                    preprocessing_fallback = False
                else:
                    aligned_source_path = align_image_for_pipeline(
                        preprocessed_source_path, embedder, FGNET_ALIGNED_DIR, unique_tag
                    )''')
    c["source"] = s.splitlines(keepends=True)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    kept.append(c)
nb["cells"] = kept
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
left = [i for i, c in enumerate(kept) if "/kaggle/" in "".join(c["source"])]
print(f"{dst}: giữ {len(kept)} cell, bỏ {dropped}; cell còn '/kaggle/': {left}")
