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


class CrossAttentionCapture:
    """Bọc AttnProcessor tùy chỉnh, gắn vào TẤT CẢ layer cross-attention (attn2) của UNet để
    có thể chụp lại attention map (softmax(QK^T)) trong 1 lần forward chỉ định, lưu về CPU.
    Chỉ can thiệp cross-attention (attn2), giữ nguyên self-attention (attn1) - đúng yêu cầu
    spec "chỉ can thiệp cross-attention".

    Lưu ý: mô phỏng lại đúng logic của AttnProcessor mặc định trong diffusers (to_q/to_k/to_v +
    get_attention_scores), có thể cần chỉnh lại nếu version diffusers cài đặt khác API.
    """

    def __init__(self, unet: UNet2DConditionModel):
        """Lưu tham chiếu UNet và các AttnProcessor gốc (để khôi phục lại sau khi dùng xong)."""
        self.unet = unet
        self._orig_processors = unet.attn_processors
        self.captured: Dict[str, torch.Tensor] = {}
        self.enabled = False

    def _build_processor(self, name: str):
        """Tạo 1 AttnProcessor thay thế cho layer `name`: tính attention y hệt AttnProcessor
        mặc định của diffusers, chèn thêm bước chụp attention_probs khi self.enabled=True và
        độ phân giải <= 32x32."""
        capture = self

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

                if capture.enabled and attention_probs.shape[1] <= MAX_ATTN_RESOLUTION:
                    capture.captured[name] = attention_probs.detach().cpu()

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states

        return _CapturingProcessor()

    def register(self) -> None:
        """Gắn processor tùy chỉnh vào riêng các layer cross-attention (tên kết thúc bằng
        'attn2.processor'), giữ nguyên processor gốc cho self-attention (attn1)."""
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

    def capture_step(self, forward_fn):
        """Bật cờ enabled, chạy forward_fn() (không tính gradient - thường là 1 lần gọi UNet
        với nhánh P_alpha), tắt cờ enabled, trả về (dict attention map chụp được trong lần
        forward này, kết quả trả về của forward_fn)."""
        self.captured = {}
        self.enabled = True
        with torch.no_grad():
            result = forward_fn()
        self.enabled = False
        return dict(self.captured), result


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
                transforms.Resize((self.image_size, self.image_size)),
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
        attn_capture: CrossAttentionCapture,
    ) -> Tuple[List[torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
        """Bước B: đi từ t=T về t=1. Với từng timestep:
          1. Warm-start null_t từ giá trị null_{t+1} đã tối ưu ở bước trước (không reset về
             vector rỗng mỗi lần).
          2. Tối ưu Adam tối đa num_inner_steps vòng (lr giảm dần 1e-2*(1-i/100)), early-stop
             khi loss < epsilon + i*2e-5. Loss = MSE giữa latent CFG dự đoán và latent pivot
             (Bước A) ở t-1.
          3. Sau khi null_t hội tụ, chạy lại 1 bước CFG THẬT (không gradient) để cập nhật
             latent_cur cho bước kế tiếp, ĐỒNG THỜI chụp attention map của nhánh P_alpha qua
             attn_capture -> M_t_alpha[t].

        Trả về (list null_t theo đúng thứ tự duyệt, dict M_t_alpha theo timestep)."""
        uncond_embeddings = uncond_embedding.clone()
        null_embeddings_list: List[torch.Tensor] = []
        attention_maps: Dict[int, Dict[str, torch.Tensor]] = {}

        latent_cur = pivot_latents[-1]  # z_T*
        timesteps = self.scheduler.timesteps

        for i in range(self.num_inference_steps):
            # FIX (phat hien qua debug thuc te tren Colab T4): giu uncond_embeddings o FP32
            # trong luc Adam toi uu - Adam eps=1e-8 mac dinh bi lam tron ve 0 trong fp16
            # (sqrt(v)+eps underflow), gay NaN ngay tu buoc dau tien optimizer.step(). Chi cast
            # ve fp16 (.half()) dung luc dua vao UNet forward (UNet van chay fp16 binh thuong).
            uncond_embeddings = uncond_embeddings.clone().detach().float().requires_grad_(True)
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
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

                if loss.item() < self.early_stop_epsilon + i * 2e-5:
                    break

            # Luu lai dang fp16 - dung dinh dang UNet mong doi khi Module 3 dung lai null_t nay.
            null_embeddings_list.append(uncond_embeddings[:1].detach().half())
            check_nan(null_embeddings_list[-1], f"null_embeddings[t={int(t)}]", self.debug_check_nan)

            with torch.no_grad():
                noise_pred_uncond_final = self._predict_noise(latent_cur, t, uncond_embeddings.half())
            attn_maps_t, noise_pred_cond_final = attn_capture.capture_step(
                lambda: self._predict_noise(latent_cur, t, cond_embedding)
            )
            attention_maps[int(t)] = attn_maps_t
            if self.debug_check_nan:
                for layer_name, amap in attn_maps_t.items():
                    check_nan(amap, f"attention_maps[t={int(t)}][{layer_name}]", self.debug_check_nan)

            with torch.no_grad():
                noise_pred = noise_pred_uncond_final + self.guidance_scale * (
                    noise_pred_cond_final - noise_pred_uncond_final
                )
                latent_cur = self._ddim_prev_step(noise_pred, t, latent_cur)

            print(f"[NullTextInverter] t={int(t)} ({i + 1}/{self.num_inference_steps}) loss={loss.item():.6f}")

        return null_embeddings_list, attention_maps

    def invert(
        self, image_path: str, initial_age: int, gender_word: str
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
        """Hàm chính Module 2: build P_alpha từ (initial_age, gender_word), chạy DDIM inversion
        (Bước A) rồi Null-text optimization (Bước B). Trả về (z_T, list null_t, dict M_t_alpha)
        - dùng làm đầu vào trực tiếp cho Module 3 (Editing)."""
        if self.unet is None:
            self._load_models()

        p_alpha = build_prompt_alpha(initial_age, gender_word)
        uncond_embedding = self._encode_text("")
        cond_embedding = self._encode_text(p_alpha)

        z0 = self._load_image_latent(image_path)
        check_nan(z0, "z0", self.debug_check_nan)

        pivot_latents = self._ddim_inversion(z0, cond_embedding)
        if self.debug_check_nan:
            for i, lat in enumerate(pivot_latents):
                if check_nan(lat, f"pivot_latents[{i}]", True):
                    break  # chi bao NaN dau tien, tranh spam log cho cac buoc sau do

        attn_capture = CrossAttentionCapture(self.unet)
        attn_capture.register()
        try:
            null_embeddings_list, attention_maps = self._null_text_optimization(
                pivot_latents, uncond_embedding, cond_embedding, attn_capture
            )
        finally:
            attn_capture.restore()

        z_T = pivot_latents[-1]
        return z_T, null_embeddings_list, attention_maps
