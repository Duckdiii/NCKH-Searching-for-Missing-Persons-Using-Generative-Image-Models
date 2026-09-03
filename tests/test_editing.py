"""
Test nhỏ cho Module 3 (Editing) - chạy độc lập, không cần Module 1 chạy xong trước
(unet_checkpoint_dir=None ở cả 2 phía). Tự tạo dữ liệu đầu vào bằng cách gọi trước 1 lần
NullTextInverter rút gọn (Module 2), rồi truyền kết quả vào Editor - vì Editor không thể
test có ý nghĩa nếu không có (z_T, {null_t}, M_t_alpha) thật.

Lưu ý: đây là test tích hợp nhẹ (load thật Stable Diffusion, chạy thật vài bước DDIM cả 2
chiều). Chạy tay bằng: pytest tests/test_editing.py -v
"""

import glob
import os
import shutil

import pytest
from PIL import Image

from src.fading.editing import Editor
from src.fading.inversion import NullTextInverter

FFHQ_DIR = "D:/Data/project/nckh/ffhq_aging_150_samples"
OUTPUT_DIR = "./outputs/_test_edited_images"


def _sample_image_path() -> str:
    """Lấy đại diện 1 ảnh bất kỳ trong dataset FFHQ để làm ảnh input test."""
    candidates = sorted(glob.glob(os.path.join(FFHQ_DIR, "*.png")))
    assert candidates, f"Không tìm thấy ảnh .png nào trong {FFHQ_DIR}"
    return candidates[0]


def test_edit_produces_png_for_each_target_age():
    """Chạy Module 2 rút gọn (5 bước DDIM) lấy z_T/{null_t}/M_t_alpha, rồi Module 3 CÙNG 5
    bước, kiểm tra: mỗi target_age có 1 file PNG được tạo, kích thước đúng 256x256."""
    inverter = NullTextInverter(unet_checkpoint_dir=None, num_inference_steps=5, num_inner_steps=2)
    z_T, null_embeddings, attention_maps = inverter.invert(
        image_path=_sample_image_path(), initial_age=30, gender_word="woman"
    )

    editor = Editor(unet_checkpoint_dir=None, num_inference_steps=5, attention_control_ratio=0.8)
    results = editor.edit(
        z_T,
        null_embeddings,
        attention_maps,
        target_ages=[50, 70],
        gender_word="woman",
        output_dir=OUTPUT_DIR,
    )

    assert set(results.keys()) == {50, 70}
    for path in results.values():
        assert os.path.isfile(path)
        with Image.open(path) as img:
            assert img.size == (256, 256)

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def test_edit_raises_on_mismatched_num_inference_steps():
    """Kiểm tra hành vi CHẶN CỨNG (không fallback im lặng) khi num_inference_steps của Editor
    khác với lúc chạy NullTextInverter - phải raise ValueError ngay, không được chạy tiếp và
    âm thầm tắt injection."""
    inverter = NullTextInverter(unet_checkpoint_dir=None, num_inference_steps=5, num_inner_steps=2)
    z_T, null_embeddings, attention_maps = inverter.invert(
        image_path=_sample_image_path(), initial_age=30, gender_word="woman"
    )

    editor = Editor(unet_checkpoint_dir=None, num_inference_steps=10)  # cố tình để KHÁC 5

    with pytest.raises(ValueError):
        editor.edit(
            z_T,
            null_embeddings,
            attention_maps,
            target_ages=[50],
            gender_word="woman",
            output_dir=OUTPUT_DIR,
        )
