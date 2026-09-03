import os
import shutil
import random

# Đã sửa: thêm \ffhq256 vào cuối đường dẫn ảnh gốc
FFHQ_ALL_IMAGES = r"D:\Data\project\nckh\ffhq256_images\ffhq256"     # <-- ĐÃ SỬA
SAMPLED_140_DIR = r"D:\Data\project\nckh\ffhq_aging_150_samples"
TEST_IMAGE_PATH = r"D:\Data\project\nckh\ffhq_aging_150_samples\01366.png"  # chọn đúng ảnh bạn test
GALLERY_DIR = r"D:\Data\project\nckh\test_gallery"

os.makedirs(GALLERY_DIR, exist_ok=True)

used_filenames = set(os.listdir(SAMPLED_140_DIR))
all_filenames = os.listdir(FFHQ_ALL_IMAGES)
candidates = [f for f in all_filenames if f not in used_filenames and f.endswith(".png")]

print(f"Tổng ảnh FFHQ gốc tìm thấy: {len(all_filenames)}")  # kỳ vọng ~70000
print(f"Số ảnh khả dụng (loại trừ 140 ảnh đã dùng): {len(candidates)}")

noise_samples = random.sample(candidates, 25)
for fname in noise_samples:
    shutil.copy(os.path.join(FFHQ_ALL_IMAGES, fname), os.path.join(GALLERY_DIR, fname))

shutil.copy(TEST_IMAGE_PATH, os.path.join(GALLERY_DIR, os.path.basename(TEST_IMAGE_PATH)))

print(f"Gallery hoàn tất: {len(os.listdir(GALLERY_DIR))} ảnh trong {GALLERY_DIR}")