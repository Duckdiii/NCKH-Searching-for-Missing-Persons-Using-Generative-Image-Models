"""
Module 3 - Editing (Cross-Attention Control)

Từ (z_T, {null_t}, M_t_alpha) do Module 2 tạo ra + danh sách target_age, sinh ảnh PNG cho
từng target_age bằng cách denoise DDIM từ z_T, dùng {null_t} thay cho vector null cố định
trong classifier-free guidance (CFG). Ở mỗi cross-attention layer, trong t_M/T (mặc định 0.8)
bước đầu tiên: GHI ĐÈ attention_probs bằng M_t_alpha đã lưu (giữ nguyên "vùng ảnh hưởng"),
nhưng attention VALUES vẫn tính từ P_tau (không phải P_alpha). Ngoài khoảng đó, attention tính
tự do theo P_tau như bình thường. CHỈ can thiệp cross-attention, KHÔNG self-attention.

Tham khảo: https://github.com/MunchkinChen/FADING (age_editing.py, p2p.py)
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.debug import check_nan
from src.utils.prompts import build_prompt_tau

NUM_DDIM_STEPS_DEFAULT = 50
GUIDANCE_SCALE_DEFAULT = 7.5
ATTENTION_CONTROL_RATIO_DEFAULT = 0.8  # t_M / T


class AttentionInjector:
    """Bọc AttnProcessor tùy chỉnh, gắn vào TẤT CẢ layer cross-attention (attn2) của UNet để
    GHI ĐÈ attention_probs bằng M_t_alpha (từ Module 2) trong t_M/T bước đầu tiên, nhưng attention
    VALUES (nhân với V) vẫn luôn tính từ P_tau (không phải P_alpha).

    QUAN TRỌNG: CHỈ can thiệp cross-attention (tên layer kết thúc "attn2.processor") - giống
    hệt bộ lọc của CrossAttentionCapture ở Module 2. Self-attention (attn1) LUÔN giữ nguyên
    processor gốc của diffusers, KHÔNG bị đụng vào trong bất kỳ nhánh nào bên dưới.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        reference_maps: Dict[int, Dict[str, torch.Tensor]],
        attention_control_ratio: float,
        num_inference_steps: int,
    ):
        """Lưu tham chiếu UNet, dict M_t_alpha (từ Module 2), tỷ lệ t_M/T và tổng số bước -
        dùng để tính bước nào được phép ghi đè attention."""
        self.unet = unet
        self._orig_processors = unet.attn_processors
        self.reference_maps = reference_maps
        self.attention_control_ratio = attention_control_ratio
        self.num_inference_steps = num_inference_steps
        self.enabled = False
        self.current_t: Optional[int] = None

    def _build_processor(self, name: str):
        """Tạo 1 AttnProcessor thay thế cho layer cross-attention `name`: Q/K/V vẫn tính từ
        P_tau như bình thường; nếu đang trong 80% bước đầu (self.enabled=True) VÀ layer này có
        map tham chiếu tại timestep hiện tại thì ghi đè attention_probs bằng map đó (fallback
        tính tự do nếu layer không có map - vd layer >32x32 đã bị lọc bỏ ở Module 2, đây là
        hành vi CHỦ ĐÍCH, không phải lỗi)."""
        injector = self

        class _InjectingProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                query = attn.to_q(hidden_states)
                context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = attn.to_k(context)
                value = attn.to_v(context)  # LUON tinh tu P_tau (nhanh cond dang duoc goi)

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                reference = None
                if injector.enabled:
                    reference = injector.reference_maps.get(injector.current_t, {}).get(name)

                if reference is not None:
                    attention_probs = reference.to(device=value.device, dtype=value.dtype)
                else:
                    attention_probs = attn.get_attention_scores(query, key, attention_mask)

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states

        return _InjectingProcessor()

    def register(self) -> None:
        """Gắn processor tùy chỉnh CHỈ vào layer có tên kết thúc 'attn2.processor'
        (cross-attention). Self-attention (attn1) giữ nguyên processor gốc, không thay đổi."""
        new_processors = {}
        for name in self.unet.attn_processors.keys():
            if name.endswith("attn2.processor"):
                new_processors[name] = self._build_processor(name)
            else:
                new_processors[name] = self._orig_processors[name]
        self.unet.set_attn_processor(new_processors)

    def restore(self) -> None:
        """Khôi phục lại AttnProcessor gốc của UNet (gọi trong finally, sau khi dùng xong)."""
        self.unet.set_attn_processor(self._orig_processors)

    def inject_step(self, step_index: int, t, forward_fn):
        """Bật cờ enabled CHỈ trong đúng 1 lần gọi UNet(P_tau) - chỉ cho phép ghi đè nếu
        step_index nằm trong t_M/T bước đầu tiên (step_index < ratio * num_inference_steps).
        Tắt cờ enabled ngay sau đó, để không ảnh hưởng sang nhánh null_t (uncond)."""
        self.current_t = int(t)
        self.enabled = step_index < self.attention_control_ratio * self.num_inference_steps
        with torch.no_grad():
            result = forward_fn()
        self.enabled = False
        return result


class Editor:
    """Bọc toàn bộ Module 3: load Stable Diffusion (độc lập với Module 2 - xem ghi chú ở
    _load_models), denoise DDIM từ z_T bằng {null_t} + attention injection, sinh ảnh PNG cho
    từng target_age."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        unet_checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        num_inference_steps: int = NUM_DDIM_STEPS_DEFAULT,
        guidance_scale: float = GUIDANCE_SCALE_DEFAULT,
        attention_control_ratio: float = ATTENTION_CONTROL_RATIO_DEFAULT,
        image_size: int = 256,
        debug_check_nan: bool = False,
    ):
        """Lưu hyperparameters. QUAN TRỌNG: num_inference_steps PHẢI khớp với giá trị đã dùng
        khi chạy NullTextInverter.invert() (Module 2) để sinh ra attention_maps truyền vào
        edit() - nếu khác, _validate_timesteps_available() sẽ raise ValueError ngay (xem
        docstring của hàm đó để biết lý do không được fallback im lặng ở đây). debug_check_nan=
        True bật kiểm tra NaN tùy chọn trên latent mỗi bước denoise (xem src/utils/debug.py)."""
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.unet_checkpoint_dir = unet_checkpoint_dir
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.attention_control_ratio = attention_control_ratio
        self.image_size = image_size
        self.debug_check_nan = debug_check_nan

        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def _load_models(self) -> None:
        """Load VAE/UNet/text encoder/tokenizer RIÊNG cho Editor - KHÔNG dùng chung instance
        với NullTextInverter của Module 2, để 2 module độc lập, test riêng được từng module mà
        không phụ thuộc lẫn nhau (đúng yêu cầu "test từng module chạy độc lập"). Đánh đổi: tốn
        thêm VRAM/thời gian nếu chạy nối tiếp cả 2 module trong cùng 1 process - main.py nên
        gọi `del inverter; torch.cuda.empty_cache()` giữa Module 2 và Module 3 để tránh cộng
        dồn VRAM trên GPU nhỏ (~6GB). Đây là điểm có thể tối ưu sau này (chạy hàng loạt nhiều
        ảnh), chưa cần làm ngay ở happy path."""
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

    def _encode_text(self, prompt: str) -> torch.Tensor:
        """Tokenize + encode 1 prompt qua CLIP text encoder, trả về embedding [1, 77, 768]."""
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
        """Gọi UNet 1 lần với 1 nhánh embedding duy nhất, trả về tensor noise_pred."""
        return self.unet(latent, t, encoder_hidden_states=embedding).sample

    def _ddim_prev_step(self, noise_pred: torch.Tensor, t: int, sample: torch.Tensor) -> torch.Tensor:
        """DDIM bước NGƯỢC (t hiện tại -> t nhỏ hơn) - công thức giống hệt Module 2
        (_ddim_prev_step trong inversion.py), viết riêng ở đây để Module 3 không phụ thuộc
        import Module 2 (giữ 2 module độc lập)."""
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

    def _decode_latent_to_image(self, latent: torch.Tensor) -> Image.Image:
        """Giải mã latent [1,4,32,32] qua VAE decoder về ảnh RGB [0,255], trả về PIL.Image.

        FIX (phát hiện qua debug thực tế trên Colab T4): VAE decode ở fp16 gây lỗi NaN/ảnh đen
        nổi tiếng của SD1.5 VAE trên một số GPU. Ép VAE + latent lên fp32 trước khi decode, trả
        VAE về fp16 ngay sau đó để tiết kiệm VRAM cho các bước khác."""
        with torch.no_grad():
            latent = latent / self.vae.config.scaling_factor
            self.vae.to(dtype=torch.float32)
            image = self.vae.decode(latent.float()).sample
            self.vae.to(dtype=torch.float16)
        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(image_np)

    def _validate_timesteps_available(
        self, attention_maps: Dict[int, Dict[str, torch.Tensor]], timesteps, num_injection_steps: int
    ) -> None:
        """Kiểm tra CỨNG: mọi timestep cần injection (t_M/T bước đầu tiên) phải có mặt trong
        attention_maps. Nếu thiếu -> raise ValueError NGAY, KHÔNG fallback im lặng.

        Lý do bắt buộc raise (khác hẳn cách xử lý layer bị lọc bỏ - xem AttentionInjector):
        đây là lỗi CẤU HÌNH (num_inference_steps của Editor không khớp lúc chạy Module 2),
        không phải hành vi chủ đích. Timestep của DDIM không phải tập con của nhau khi đổi số
        bước (vd 50 bước và 30 bước cho ra 2 tập giá trị t gần như khác hoàn toàn) - nếu chỉ
        fallback nhẹ nhàng, gần như toàn bộ injection sẽ bị tắt âm thầm, ảnh vẫn sinh ra bình
        thường (không crash) nhưng identity KHÔNG được giữ - lỗi im lặng nguy hiểm cho bài toán
        tìm người thất lạc.
        """
        missing = [int(t) for t in timesteps[:num_injection_steps] if int(t) not in attention_maps]
        if missing:
            raise ValueError(
                f"attention_maps thiếu timestep {missing}. num_inference_steps của Editor "
                f"({self.num_inference_steps}) có thể không khớp với giá trị đã dùng khi chạy "
                f"NullTextInverter.invert() (Module 2) để sinh ra attention_maps này - 2 module "
                f"BẮT BUỘC phải dùng cùng 1 giá trị num_inference_steps."
            )

    def edit(
        self,
        z_T: torch.Tensor,
        null_embeddings: List[torch.Tensor],
        attention_maps: Dict[int, Dict[str, torch.Tensor]],
        target_ages: List[int],
        gender_word: str,
        output_dir: str,
    ) -> Dict[int, str]:
        """Hàm chính Module 3: với mỗi target_age, denoise DDIM từ z_T bằng {null_t} + attention
        injection (M_t_alpha trong t_M/T bước đầu), lưu ảnh PNG. Trả về dict
        {target_age: đường_dẫn_ảnh_PNG}."""
        if self.unet is None:
            self._load_models()

        os.makedirs(output_dir, exist_ok=True)
        timesteps = self.scheduler.timesteps
        num_injection_steps = int(self.attention_control_ratio * self.num_inference_steps)
        self._validate_timesteps_available(attention_maps, timesteps, num_injection_steps)

        results: Dict[int, str] = {}

        for target_age in target_ages:
            p_tau = build_prompt_tau(target_age, gender_word)
            cond_embedding = self._encode_text(p_tau)
            latent = z_T.clone()

            injector = AttentionInjector(
                self.unet, attention_maps, self.attention_control_ratio, self.num_inference_steps
            )
            injector.register()
            try:
                for i in range(self.num_inference_steps):
                    t = timesteps[i]
                    null_t = null_embeddings[i]

                    with torch.no_grad():
                        noise_uncond = self._predict_noise(latent, t, null_t)

                    noise_cond = injector.inject_step(
                        i, t, lambda: self._predict_noise(latent, t, cond_embedding)
                    )

                    with torch.no_grad():
                        noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
                        latent = self._ddim_prev_step(noise_pred, t, latent)
                        if self.debug_check_nan and check_nan(
                            latent, f"editor_latent[target_age={target_age}, step={i}]", True
                        ):
                            raise RuntimeError(
                                f"[Editor] NaN xuất hiện tại step i={i}, t={int(t)}, "
                                f"target_age={target_age}. Dừng ở đây để debug tiếp."
                            )
            finally:
                injector.restore()

            image = self._decode_latent_to_image(latent)
            path = os.path.join(output_dir, f"age_{target_age}.png")
            image.save(path)
            results[target_age] = path
            print(f"[Editor] target_age={target_age} -> {path}")

        return results
