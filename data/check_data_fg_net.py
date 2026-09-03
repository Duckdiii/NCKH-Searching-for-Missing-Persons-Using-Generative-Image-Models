import os

BASE = r"D:\Data\project\nckh"
DATA_DIR = os.path.join(BASE, "data")   # THEM: cap thu muc "data" con thieu

checks = {
    "1. Repo FFHQ-Aging-Dataset (code + labels)": {
        "path": os.path.join(BASE, "FFHQ-Aging-Dataset"),
        "expect_files": ["ffhq_aging_labels.csv"],
    },
    "2. 140 ảnh đã sample cho Specialization": {
        "path": os.path.join(BASE, "ffhq_aging_150_samples"),
        "expect_files": ["sampled_labels.csv"],
        "expect_min_files": 141,
    },
    "3. Gallery test": {
        "path": os.path.join(BASE, "test_gallery"),
        "expect_min_files": 26,
    },
    "4. Checkpoint Module 1": {
        "path": os.path.join(BASE, "checkpoints", "specialized_unet"),
        "expect_files": ["config.json", "diffusion_pytorch_model.safetensors"],
    },
    "5. FG-NET (MỚI — cần cho Nhánh A)": {
        "path": os.path.join(DATA_DIR, "FGNET (1)"),   # SUA: dung dung ten + cap "data"
        "expect_min_files": 1,
    },
}

print(f"{'Mục':<50} {'Tồn tại?':<10} {'Số file':<10} {'Ghi chú'}")
print("-" * 100)

for name, cfg in checks.items():
    path = cfg["path"]
    exists = os.path.isdir(path)
    if not exists:
        print(f"{name:<50} {'❌ KHÔNG':<10} {'-':<10} Chưa có thư mục: {path}")
        continue

    files = os.listdir(path)
    n_files = len(files)

    note = ""
    if "expect_min_files" in cfg and n_files < cfg["expect_min_files"]:
        note = f"⚠ Thiếu — cần ≥{cfg['expect_min_files']}"
    if "expect_files" in cfg:
        missing = [f for f in cfg["expect_files"] if f not in files]
        if missing:
            note += f" ⚠ Thiếu file: {missing}"

    status = "✅ CÓ" if not note else "⚠ CÓ (lỗi)"
    print(f"{name:<50} {status:<10} {n_files:<10} {note}")

# ----- Rieng FG-NET: kiem tra sau hon -----
fgnet_dir = checks["5. FG-NET (MỚI — cần cho Nhánh A)"]["path"]
if os.path.isdir(fgnet_dir):
    print("\n--- Kiểm tra sâu FG-NET ---")
    jpg_files = []
    for root, dirs, files in os.walk(fgnet_dir):
        jpg_files += [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
    print(f"Số ảnh .jpg tìm thấy (đệ quy trong mọi thư mục con): {len(jpg_files)}")

    import re
    pattern = re.compile(r"(\d{3})A(\d{2})", re.IGNORECASE)
    matched = [f for f in jpg_files if pattern.match(f)]
    print(f"Số ảnh khớp đúng pattern tên file (VD 001A05.JPG): {len(matched)}")
    if len(jpg_files) > 0 and len(matched) == 0:
        print("⚠ Có ảnh nhưng KHÔNG khớp pattern — kiểm tra lại tên file thật")