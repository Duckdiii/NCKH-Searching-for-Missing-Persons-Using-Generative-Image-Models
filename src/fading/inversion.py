"""
Module 2 - Null-text Inversion (Mokady et al., CVPR 2023)

Từ 1 ảnh thật + Initial Age (IA, nhập tay) và model UNet đã specialize (Module 1), suy ra:
  - z_T: nhiễu ban đầu (dùng làm điểm xuất phát cho Module 3 - Editing)
  - {null_t}: danh sách null-text embedding tối ưu RIÊNG cho từng timestep, thay cho 1 vector
    null cố định trong classifier-free guidance (CFG) - để khi denoise xuôi lại bằng CFG,
    kết quả khớp đúng quỹ đạo DDIM inversion (tái tạo đúng ảnh gốc).
  - M_t_alpha: attention map (nhánh P_alpha) chụp lại ở mỗi cross-attention layer trong quá
    trình reconstruct, dùng làm "vùng ảnh hưởng" tham chiếu cho Module 3.

Tham khảo: https://github.com/MunchkinChen/FADING (null_inversion.py, p2p.py)
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.debug import check_nan
from src.utils.prompts import build_prompt_alpha

NUM_DDIM_STEPS_DEFAULT = 50
GUIDANCE_SCALE_DEFAULT = 7.5
# Chỉ lưu attention map của layer có độ phân giải (HxW) <= 32x32, bỏ layer 64x64 - giảm VRAM,
# dùng theo đúng tối ưu của Prompt-to-Prompt gốc (layer độ phân giải thấp mang nhiều ngữ nghĩa
# hơn, layer 64x64 gần như không cần cho attention injection).
MAX_ATTN_RESOLUTION = 32 * 32


class DualAttentionCapture:
    """Bọc AttnProcessor tùy chỉnh, gắn vào TẤT CẢ layer attention của UNet để
    bắt đồng thời cả Self-Attention (attn1) và Cross-Attention (attn2).
    Lọc bỏ layer có độ phân giải không gian cao nhất (tránh khóa kết cấu bề mặt vi mô)."""

    def __init__(self, unet: UNet2DConditionModel, max_attn_resolution: int = 16 * 16):
        self.unet = unet
        self._orig_processors = unet.attn_processors
        self.max_attn_resolution = max_attn_resolution
        self.captured_self: Dict[str, torch.Tensor] = {}
        self.captured_cross: Dict[str, torch.Tensor] = {}
        self.enabled = False

    def _build_processor(self, name: str):
        capture = self
        is_cross = name.endswith("attn2.processor")

        class _CapturingProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                query = attn.to_q(hidden_states)
                context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = attn.to_k(context)
                value = attn.to_v(context)

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)

                if capture.enabled and attention_probs.shape[1] <= capture.max_attn_resolution:
                    if is_cross:
                        capture.captured_cross[name] = attention_probs.detach().cpu()
                    else:
                        capture.captured_self[name] = attention_probs.detach().cpu()

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states

        return _CapturingProcessor()

    def register(self) -> None:
        new_processors = {name: self._build_processor(name) for name in self.unet.attn_processors.keys()}
        self.unet.set_attn_processor(new_processors)

    def restore(self) -> None:
        self.unet.set_attn_processor(self._orig_processors)

    def capture_step(self, forward_fn):
        self.captured_self = {}
        self.captured_cross = {}
        self.enabled = True
        with torch.no_grad():
            result = forward_fn()
        self.enabled = False
        return dict(self.captured_self), dict(self.captured_cross), result


class NullTextInverter:
    """Bọc Null-text Inversion (Mokady et al., CVPR 2023): từ 1 ảnh thật + Initial Age (IA),
    suy ra z_T (nhiễu ban đầu), danh sách null-text embedding {null_t} tối ưu riêng từng
    timestep, và attention map tham chiếu M_t_alpha (nhánh P_alpha) - dùng làm đầu vào cho
    Module 3 (Editing).
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        unet_checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        num_inference_steps: int = NUM_DDIM_STEPS_DEFAULT,
        guidance_scale: float = GUIDANCE_SCALE_DEFAULT,
        num_inner_steps: int = 10,
        early_stop_epsilon: float = 1e-5,
        image_size: int = 256,
        debug_check_nan: bool = False,
    ):
        """Lưu hyperparameters (số bước DDIM, guidance scale, số vòng tối ưu null_t mỗi bước,
        ngưỡng early-stop). unet_checkpoint_dir=None -> dùng UNet gốc chưa fine-tune (cho phép
        test Module 2 độc lập mà không cần Module 1 chạy xong trước); nếu truyền vào 1 đường
        dẫn hợp lệ thì dùng checkpoint đã specialize. Model CHƯA được load ở đây, sẽ load lazily
        trong _load_models(). debug_check_nan=True bật kiểm tra NaN tùy chọn (xem
        src/utils/debug.py) - công cụ đã dùng để tìm ra bug Adam eps underflow ở fp16."""
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.unet_checkpoint_dir = unet_checkpoint_dir
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_inner_steps = num_inner_steps
        self.early_stop_epsilon = early_stop_epsilon
        self.image_size = image_size
        self.debug_check_nan = debug_check_nan

        # Ngưỡng bỏ layer chi tiết nhất: 16x16 khi image_size=256, 32x32 khi image_size=512
        self.max_attn_resolution = (self.image_size // 16) ** 2

        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def _load_models(self) -> None:
        """Load VAE/tokenizer/text_encoder từ pretrained_model_name_or_path; UNet từ
        unet_checkpoint_dir nếu tồn tại trên đĩa (checkpoint đã specialize ở Module 1), ngược
        lại fallback về UNet gốc chưa fine-tune. Dùng DDIMScheduler (deterministic) thay vì
        DDPMScheduler của Module 1. Module 2 KHÔNG train UNet (chỉ tối ưu null_t - 1 vector
        nhỏ ~77x768) nên áp lực VRAM nhẹ hơn nhiều so với Module 1, không cần 8-bit optimizer /
        gradient checkpointing."""
        model_id = self.pretrained_model_name_or_path

        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float16
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

        if self.unet_checkpoint_dir and os.path.isdir(self.unet_checkpoint_dir):
            self.unet = UNet2DConditionModel.from_pretrained(
                self.unet_checkpoint_dir, torch_dtype=torch.float16
            )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet", torch_dtype=torch.float16
            )

        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(self.num_inference_steps)

        self.vae.to(self.device).eval().requires_grad_(False)
        self.text_encoder.to(self.device).eval().requires_grad_(False)
        self.unet.to(self.device).eval().requires_grad_(False)

    def _load_image_latent(self, image_path: str) -> torch.Tensor:
        """Đọc ảnh, resize/normalize về [-1,1], encode qua VAE bằng .mean (KHÔNG sample từ
        phân phối, để đảm bảo deterministic - cần thiết cho DDIM inversion)."""
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        image_t = transform(image).unsqueeze(0).to(self.device, dtype=torch.float16)
        with torch.no_grad():
            latent = self.vae.encode(image_t).latent_dist.mean * self.vae.config.scaling_factor
        return latent

    def _encode_text(self, prompt: str) -> torch.Tensor:
        """Tokenize + encode 1 prompt (có thể là chuỗi rỗng "" cho nhánh null/uncond) qua CLIP
        text encoder, trả về embedding [1, 77, 768]."""
        tokens = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            return self.text_encoder(tokens.input_ids)[0]

    def _predict_noise(self, latent: torch.Tensor, t, embedding: torch.Tensor) -> torch.Tensor:
        """Gọi UNet 1 lần với 1 nhánh embedding duy nhất (KHÔNG CFG), trả về tensor noise_pred."""
        return self.unet(latent, t, encoder_hidden_states=embedding).sample

    def _ddim_next_step(self, noise_pred: torch.Tensor, t: int, sample: torch.Tensor) -> torch.Tensor:
        """DDIM bước XUÔI (t hiện tại -> t lớn hơn), dùng trong DDIM inversion - Bước A.
        Công thức lấy đúng từ null_inversion.py (hàm next_step) của FADING gốc."""
        step = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        timestep, next_timestep = min(t - step, 999), t
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
        return alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction

    def _ddim_prev_step(self, noise_pred: torch.Tensor, t: int, sample: torch.Tensor) -> torch.Tensor:
        """DDIM bước NGƯỢC (t hiện tại -> t nhỏ hơn), dùng trong reconstruction CFG - Bước B.
        Công thức lấy đúng từ null_inversion.py (hàm prev_step) của FADING gốc."""
        step = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        prev_timestep = t - step
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred
        return alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

    def _ddim_inversion(self, z0: torch.Tensor, cond_embedding: torch.Tensor) -> List[torch.Tensor]:
        """Bước A: chạy num_inference_steps bước forward (guidance=1, chỉ dùng nhánh P_alpha,
        KHÔNG CFG), trả về list quỹ đạo pivot [z_0*, z_1*, ..., z_T*] CỐ ĐỊNH - không sửa ở
        Bước B, dùng làm tham chiếu để tối ưu null_t."""
        latent = z0.clone().detach()
        pivot_latents = [latent]
        timesteps = self.scheduler.timesteps
        for i in range(self.num_inference_steps):
            t = timesteps[len(timesteps) - i - 1]
            with torch.no_grad():
                noise_pred = self._predict_noise(latent, t, cond_embedding)
                latent = self._ddim_next_step(noise_pred, t, latent)
            pivot_latents.append(latent)
        return pivot_latents

    def _null_text_optimization(
        self,
        pivot_latents: List[torch.Tensor],
        uncond_embedding: torch.Tensor,
        cond_embedding: torch.Tensor,
        attn_capture: DualAttentionCapture,
    ) -> Tuple[List[torch.Tensor], Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]]]:
        """Bước B: đi từ t=T về t=1, tối ưu null_t và bắt đồng thời cả self và cross attention maps."""
        uncond_embeddings = uncond_embedding.clone()
        null_embeddings_list: List[torch.Tensor] = []
        self_attention_maps: Dict[int, Dict[str, torch.Tensor]] = {}
        cross_attention_maps: Dict[int, Dict[str, torch.Tensor]] = {}

        latent_cur = pivot_latents[-1]  # z_T*
        timesteps = self.scheduler.timesteps

        for i in range(self.num_inference_steps):
            # Giữ uncond_embeddings ở FP32 trong lúc Adam tối ưu
            uncond_embeddings = uncond_embeddings.clone().detach().float().requires_grad_(True)
            # Lịch trình LR chuẩn kaggle_3: giữ 1e-2 cho 25 bước đầu, giảm dần về sau
            lr_scale = 1.0 if i < 25 else max(0.4, 1.0 - (i - 25) / 35.0)
            optimizer = Adam([uncond_embeddings], lr=1e-2 * lr_scale)
            latent_prev = pivot_latents[len(pivot_latents) - i - 2]  # z_{t-1}*
            t = timesteps[i]

            with torch.no_grad():
                noise_pred_cond = self._predict_noise(latent_cur, t, cond_embedding)

            for _ in range(self.num_inner_steps):
                noise_pred_uncond = self._predict_noise(latent_cur, t, uncond_embeddings.half())
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_prev_rec = self._ddim_prev_step(noise_pred, t, latent_cur)
                loss = F.mse_loss(latent_prev_rec, latent_prev)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < self.early_stop_epsilon:
                    break

            # Lưu lại dạng fp16
            null_embeddings_list.append(uncond_embeddings[:1].detach().half())
            check_nan(null_embeddings_list[-1], f"null_embeddings[t={int(t)}]", self.debug_check_nan)

            with torch.no_grad():
                noise_pred_uncond_final = self._predict_noise(latent_cur, t, uncond_embeddings.half())
            s_maps, c_maps, noise_pred_cond_final = attn_capture.capture_step(
                lambda: self._predict_noise(latent_cur, t, cond_embedding)
            )
            self_attention_maps[int(t)] = s_maps
            cross_attention_maps[int(t)] = c_maps
            if self.debug_check_nan:
                for layer_name, amap in c_maps.items():
                    check_nan(amap, f"cross_maps[t={int(t)}][{layer_name}]", self.debug_check_nan)

            with torch.no_grad():
                noise_pred = noise_pred_uncond_final + self.guidance_scale * (
                    noise_pred_cond_final - noise_pred_uncond_final
                )
                latent_cur = self._ddim_prev_step(noise_pred, t, latent_cur)

            print(f"[NullTextInverter] t={int(t)} ({i + 1}/{self.num_inference_steps}) loss={loss.item():.6f}")

        return null_embeddings_list, (self_attention_maps, cross_attention_maps)

    def invert(
        self, image_path: str, initial_age: int, gender_word: str
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]]]:
        """Hàm chính Module 2: build P_alpha từ (initial_age, gender_word), chạy DDIM inversion
        (Bước A) rồi Null-text optimization (Bước B). Trả về (z_T, list null_t, (self_maps, cross_maps))
        - dùng làm đầu vào trực tiếp cho Module 3 (Editing)."""
        if self.unet is None:
            self._load_models()

        original_inner_steps = self.num_inner_steps
        if initial_age < 10:
            self.num_inner_steps = max(self.num_inner_steps, 20)
            print(f"[NullTextInverter] initial_age={initial_age} < 10 -> child-adaptive num_inner_steps={self.num_inner_steps}")

        try:
            p_alpha = build_prompt_alpha(initial_age, gender_word)
            uncond_embedding = self._encode_text("")
            cond_embedding = self._encode_text(p_alpha)

            z0 = self._load_image_latent(image_path)
            check_nan(z0, "z0", self.debug_check_nan)

            pivot_latents = self._ddim_inversion(z0, cond_embedding)
            if self.debug_check_nan:
                for i, lat in enumerate(pivot_latents):
                    if check_nan(lat, f"pivot_latents[{i}]", True):
                        break

            attn_capture = DualAttentionCapture(self.unet, max_attn_resolution=self.max_attn_resolution)
            attn_capture.register()
            try:
                null_embeddings_list, (self_maps, cross_maps) = self._null_text_optimization(
                    pivot_latents, uncond_embedding, cond_embedding, attn_capture
                )
            finally:
                attn_capture.restore()

            z_T = pivot_latents[-1]
            return z_T, null_embeddings_list, (self_maps, cross_maps)
        finally:
            self.num_inner_steps = original_inner_steps

