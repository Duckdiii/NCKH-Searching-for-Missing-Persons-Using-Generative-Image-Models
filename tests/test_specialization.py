"""
Test nhỏ cho Module 1 (Specialization) - chạy độc lập, KHÔNG phụ thuộc các module khác.

Lưu ý: đây là test tích hợp nhẹ (load thật Stable Diffusion + chạy vài step thật), cần GPU
và mạng để tải model lần đầu. Không chạy trong CI tự động - chạy tay bằng:
    pytest tests/test_specialization.py -v
"""

import shutil
import yaml

from src.fading.specialization import FFHQAgingDataset, Specializer

CONFIG_PATH = "configs/config.yaml"
TEST_CKPT_DIR = "./outputs/_test_specialized_unet"


def _load_config():
    """Đọc configs/config.yaml dùng chung cho các test case bên dưới."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_dataset_loads_samples():
    """Kiểm tra FFHQAgingDataset đọc đúng CSV + ảnh, trả về 140 sample, mỗi sample có
    P_alpha/P_neutral hợp lệ (không rỗng, P_alpha có chứa chữ số tuổi)."""
    config = _load_config()
    dataset = FFHQAgingDataset(config["paths"]["ffhq_dir"], config["paths"]["labels_csv"])

    assert len(dataset) == 140

    image, p_alpha, p_neutral = dataset[0]
    assert image.shape == (3, 256, 256)
    assert p_alpha.startswith("photo of a")
    assert any(ch.isdigit() for ch in p_alpha)
    assert not any(ch.isdigit() for ch in p_neutral)


def test_specializer_runs_two_steps_and_saves_checkpoint():
    """Chạy Specializer với 2 step, batch_size=2 trên dataset thật (chỉ để kiểm tra pipeline
    chạy không lỗi + loss hữu hạn + checkpoint được lưu ra đĩa), không đòi hỏi hội tụ."""
    config = _load_config()
    dataset = FFHQAgingDataset(config["paths"]["ffhq_dir"], config["paths"]["labels_csv"])

    specializer = Specializer(
        pretrained_model_name_or_path=config["base_model"]["pretrained_model_name_or_path"],
        train_steps=2,
        batch_size=2,
    )

    losses = specializer.train(dataset)

    assert len(losses) == 2
    assert all(loss == loss for loss in losses)  # không có NaN (NaN != NaN)

    specializer.save_checkpoint(TEST_CKPT_DIR)

    import os

    assert os.path.isdir(TEST_CKPT_DIR)
    assert any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(TEST_CKPT_DIR))

    shutil.rmtree(TEST_CKPT_DIR, ignore_errors=True)
