"""
Test nhỏ cho Module 2 (Null-text Inversion) - chạy độc lập, KHÔNG cần Module 1 chạy xong trước
(unet_checkpoint_dir=None -> tự động fallback về UNet gốc chưa fine-tune).

Lưu ý: đây là test tích hợp nhẹ (load thật Stable Diffusion + chạy thật vài bước DDIM), cần
GPU và mạng để tải model lần đầu. Dùng số bước nhỏ (num_inference_steps, num_inner_steps) để
chỉ xác nhận pipeline không lỗi cú pháp/logic, không cần chất lượng reconstruct tốt. Chạy tay
bằng: pytest tests/test_inversion.py -v
"""

import glob
import os

from src.fading.inversion import NullTextInverter

FFHQ_DIR = "D:/Data/project/nckh/ffhq_aging_150_samples"


def _sample_image_path() -> str:
    """Lấy đại diện 1 ảnh bất kỳ trong dataset FFHQ để làm ảnh input test."""
    candidates = sorted(glob.glob(os.path.join(FFHQ_DIR, "*.png")))
    assert candidates, f"Không tìm thấy ảnh .png nào trong {FFHQ_DIR}"
    return candidates[0]


def test_invert_returns_expected_shapes_and_lengths():
    """Chạy NullTextInverter với số bước rút gọn (5 bước DDIM, 2 vòng tối ưu/bước), kiểm tra:
    - z_T có shape latent đúng (1,4,32,32) ứng với ảnh 256x256
    - số lượng null_t == num_inference_steps
    - mỗi timestep có attention map (dict M_t_alpha không rỗng)."""
    inverter = NullTextInverter(
        unet_checkpoint_dir=None,  # test độc lập, chưa cần checkpoint Module 1
        num_inference_steps=5,
        num_inner_steps=2,
    )

    z_T, null_embeddings, attention_maps = inverter.invert(
        image_path=_sample_image_path(), initial_age=30, gender_word="woman"
    )

    assert z_T.shape == (1, 4, 32, 32)
    assert len(null_embeddings) == 5
    assert all(emb.shape[-2:] == (77, 768) for emb in null_embeddings)
    assert len(attention_maps) == 5
    assert all(len(layer_maps) > 0 for layer_maps in attention_maps.values())
