"""
Module 1 - Specialization (Double-Prompt fine-tuning)

Fine-tune UNet của Stable Diffusion trên 140 ảnh FFHQ
(D:/Data/project/nckh/ffhq_aging_150_samples) theo đúng FADING (BMVC 2023): mỗi ảnh đồng
thời học qua 2 prompt (P_alpha - có tuổi, P_neutral - không tuổi) bằng 2 nhánh nhiễu độc lập
chia sẻ 1 timestep. Output là checkpoint UNet đã specialize, dùng làm đầu vào cho Module 2
(Null-text Inversion).

Tham khảo: https://github.com/MunchkinChen/FADING (specialize.py)
"""

import os
import random
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.prompts import (
    age_group_to_age,
    gender_to_word,
    build_prompt_alpha,
    build_prompt_neutral,
)


class FFHQAgingDataset(Dataset):
    """Dataset đọc 140 ảnh FFHQ + sampled_labels.csv, trả về (ảnh tensor, P_alpha, P_neutral)
    đã build sẵn cho từng sample, dùng trực tiếp cho vòng lặp training của Specializer."""

    def __init__(self, ffhq_dir: str, labels_csv: str, image_size: int = 256):
        """Đọc CSV, suy tuổi đại diện + gender_word + P_alpha/P_neutral cho từng ảnh; chỉ giữ
        lại những dòng mà file ảnh tương ứng thực sự tồn tại trong ffhq_dir."""
        self.ffhq_dir = ffhq_dir
        df = pd.read_csv(labels_csv)

        self.samples: List[Tuple[str, str, str]] = []
        for _, row in df.iterrows():
            image_path = os.path.join(ffhq_dir, f"{int(row['image_number']):05d}.png")
            if not os.path.isfile(image_path):
                continue
            age = age_group_to_age(row["age_group"])
            gender_word = gender_to_word(row["gender"], age)
            self.samples.append(
                (image_path, build_prompt_alpha(age, gender_word), build_prompt_neutral(gender_word))
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # đưa về [-1, 1] cho VAE
            ]
        )

    def __len__(self) -> int:
        """Số lượng sample hợp lệ (ảnh + label khớp nhau) trong dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Trả về 1 sample: (ảnh tensor [3,H,W] trong [-1,1], P_alpha, P_neutral)."""
        image_path, p_alpha, p_neutral = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), p_alpha, p_neutral


class Specializer:
    """Bọc toàn bộ Module 1: load Stable Diffusion, chạy double-prompt fine-tuning trên UNet
    theo đúng loss của FADING, và lưu checkpoint UNet đã specialize."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        lr: float = 5e-6,
        betas: Tuple[float, float] = (0.9, 0.999),
        train_steps: int = 150,
        batch_size: int = 2,
    ):
        """Lưu lại hyperparameters (lr, betas, số step, batch size). Model CHƯA được load ở đây
        - sẽ load lazily trong _load_models() khi train() được gọi, tránh chiếm VRAM sớm."""
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        self.lr = lr
        self.betas = betas
        self.train_steps = train_steps
        self.batch_size = batch_size

        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.noise_scheduler = None
        self.optimizer = None

    def _load_models(self) -> None:
        """Load VAE / UNet / CLIP text encoder / tokenizer / noise scheduler từ checkpoint SD
        gốc. Freeze VAE + text encoder (chỉ UNet được fine-tune), bật gradient checkpointing và
        chuyển toàn bộ sang fp16 để vừa VRAM GPU nhỏ (~6GB)."""
        model_id = self.pretrained_model_name_or_path

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float16
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.vae.to(self.device).eval().requires_grad_(False)
        self.text_encoder.to(self.device).eval().requires_grad_(False)
        self.unet.to(self.device).train()
        self.unet.enable_gradient_checkpointing()

        # 8-bit Adam để giảm bộ nhớ optimizer state (~4 lần) - cần thiết vì UNet SD1.5 (~860M
        # tham số) + optimizer state fp16 thường (~6.9GB) vượt quá VRAM 6GB của máy test. Về
        # công thức update vẫn là Adam(lr=5e-6, betas=(0.9,0.999)) đúng như spec, chỉ khác cách
        # lưu trữ state (quantized 8-bit) để tiết kiệm bộ nhớ. Nếu không có bitsandbytes thì
        # fallback về Adam thường (có thể OOM trên GPU VRAM thấp).
        try:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.Adam8bit(self.unet.parameters(), lr=self.lr, betas=self.betas)
        except ImportError:
            print(
                "[Specializer] bitsandbytes không có sẵn, dùng torch.optim.Adam fp16 thường "
                "(có thể OOM trên GPU VRAM thấp)."
            )
            self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr, betas=self.betas)

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Tokenize 1 list prompt text và encode qua CLIP text encoder (không tính gradient,
        vì text encoder đã freeze), trả về tensor embedding [batch, seq_len, hidden_dim] dùng
        làm điều kiện (encoder_hidden_states) cho UNet."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            return self.text_encoder(tokens.input_ids)[0]

    def train(self, dataset: FFHQAgingDataset) -> List[float]:
        """Vòng lặp training chính của Module 1.

        Với mỗi step (mặc định 150 step, batch_size=2):
          1. Lấy ngẫu nhiên `batch_size` ảnh từ dataset (có lặp lại vì dataset chỉ 140 ảnh).
          2. Encode ảnh qua VAE -> z0 = vae.encode(ảnh).latent_dist.sample() * scaling_factor.
          3. Sample chung 1 timestep t cho từng ảnh (dùng chung giữa 2 nhánh neutral/alpha).
          4. Sample 2 nhiễu độc lập eps, eps' cùng shape z0.
          5. z_t = add_noise(z0, eps, t) ghép với P_neutral; z_t' = add_noise(z0, eps', t) ghép
             với P_alpha - gộp cả 2 nhánh thành 1 batch để chỉ forward UNet 1 lần/step.
          6. loss = MSE(eps_theta(z_t, t, P_neutral), eps) + MSE(eps_theta(z_t', t, P_alpha), eps').
          7. Backprop + Adam step, cập nhật UNet.

        Trả về list loss theo từng step (phục vụ debug/test, không bắt buộc dùng trong pipeline
        chính).
        """
        if self.unet is None:
            self._load_models()

        losses: List[float] = []
        n = len(dataset)

        for step in range(self.train_steps):
            idxs = [random.randrange(n) for _ in range(self.batch_size)]
            images, p_alphas, p_neutrals = [], [], []
            for idx in idxs:
                image, p_alpha, p_neutral = dataset[idx]
                images.append(image)
                p_alphas.append(p_alpha)
                p_neutrals.append(p_neutral)

            images_t = torch.stack(images).to(self.device, dtype=torch.float16)

            with torch.no_grad():
                z0 = self.vae.encode(images_t).latent_dist.sample()
                z0 = z0 * self.vae.config.scaling_factor

            t = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (self.batch_size,), device=self.device
            ).long()

            eps = torch.randn_like(z0)
            eps_prime = torch.randn_like(z0)

            z_t = self.noise_scheduler.add_noise(z0, eps, t)
            z_t_prime = self.noise_scheduler.add_noise(z0, eps_prime, t)

            neutral_emb = self._encode_prompts(p_neutrals)
            alpha_emb = self._encode_prompts(p_alphas)

            # Gộp 2 nhánh (neutral + alpha) thành 1 batch để chỉ gọi UNet 1 lần / step.
            latents_in = torch.cat([z_t, z_t_prime], dim=0)
            timesteps_in = torch.cat([t, t], dim=0)
            emb_in = torch.cat([neutral_emb, alpha_emb], dim=0)

            noise_pred = self.unet(latents_in, timesteps_in, encoder_hidden_states=emb_in).sample
            noise_pred_neutral, noise_pred_alpha = noise_pred.chunk(2, dim=0)

            loss = F.mse_loss(noise_pred_neutral.float(), eps.float()) + F.mse_loss(
                noise_pred_alpha.float(), eps_prime.float()
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            print(f"[Specializer] step {step + 1}/{self.train_steps} loss={loss.item():.4f}")

        return losses

    def save_checkpoint(self, output_dir: str) -> None:
        """Lưu UNet đã fine-tune vào output_dir bằng unet.save_pretrained (định dạng chuẩn
        diffusers), để Module 2/3 load lại bằng UNet2DConditionModel.from_pretrained(output_dir)."""
        os.makedirs(output_dir, exist_ok=True)
        self.unet.save_pretrained(output_dir)
        print(f"[Specializer] đã lưu checkpoint UNet vào {output_dir}")


if __name__ == "__main__":
    # Chạy độc lập Module 1: đọc configs/config.yaml, train, lưu checkpoint.
    import yaml

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_dataset = FFHQAgingDataset(config["paths"]["ffhq_dir"], config["paths"]["labels_csv"])
    specializer = Specializer(
        pretrained_model_name_or_path=config["base_model"]["pretrained_model_name_or_path"],
        lr=float(config["specialization"]["lr"]),
        betas=tuple(config["specialization"]["betas"]),
        train_steps=config["specialization"]["train_steps"],
        batch_size=config["specialization"]["batch_size"],
    )
    specializer.train(train_dataset)
    specializer.save_checkpoint(config["paths"]["specialized_unet_ckpt"])
