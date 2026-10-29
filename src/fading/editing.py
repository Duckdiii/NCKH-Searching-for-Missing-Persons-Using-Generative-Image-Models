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

from src.utils.cancellation import checkpoint
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms.functional as TF

from src.utils.debug import check_nan
from src.utils.prompts import build_prompt_alpha, build_prompt_tau

NUM_DDIM_STEPS_DEFAULT = 50
GUIDANCE_SCALE_DEFAULT = 4.0  # Mức cân bằng chuẩn của FADING để sinh nếp nhăn già hóa (photorealism)
ATTENTION_CONTROL_RATIO_DEFAULT = 0.8  # t_M / T
USE_LOCAL_BLEND_DEFAULT = True
LOCAL_BLEND_THRESHOLD_DEFAULT = 0.3


def get_word_inds(prompt: str, word: str, tokenizer) -> np.ndarray:
    """Tìm vị trí token của từ `word` trong `prompt` đã tokenize bởi CLIPTokenizer.
    Dùng để định vị token chủ thể chung (như "person", "woman", "man") giữa prompt gốc và prompt đích.
    Trả về mảng 1D các index token."""
    word_clean = word.strip().lower()
    tokens = tokenizer.encode(prompt)
    inds = []
    for idx, token_id in enumerate(tokens):
        tok_str = tokenizer.decode([token_id]).strip().lower().replace("</w>", "").strip(",.!?\"'")
        if tok_str and (tok_str == word_clean or word_clean in tok_str):
            inds.append(idx)
    if not inds:
        for idx, token_id in enumerate(tokens):
            tok_str = tokenizer.decode([token_id]).strip().lower()
            if word_clean in tok_str or (len(tok_str) > 2 and tok_str in word_clean):
                inds.append(idx)
    if not inds:
        inds = [4]
    return np.array(inds, dtype=int)


def local_blend(
    recon_latent: torch.Tensor,
    edit_latent: torch.Tensor,
    cross_attn_maps_recon: List[torch.Tensor],
    cross_attn_maps_edit: List[torch.Tensor],
    word_inds_recon: np.ndarray,
    word_inds_edit: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cơ chế LocalBlend (Hertz et al. - Prompt-to-Prompt):
    Chỉ cho phép thay đổi ở ĐÚNG vùng chứa chủ thể (face/person), GIỮ NGUYÊN latent gốc
    ở mọi vùng khác (nền, tóc, quần áo) tại MỖI bước denoising."""
    if len(cross_attn_maps_recon) == 0 or len(cross_attn_maps_edit) == 0:
        return edit_latent, torch.ones_like(edit_latent[:, :1])

    if len(word_inds_recon) == 0:
        word_inds_recon = np.array([4])
    if len(word_inds_edit) == 0:
        word_inds_edit = np.array([4])

    recon_maps = []
    for m in cross_attn_maps_recon:
        spatial_dim = int(round(m.shape[1] ** 0.5))
        sub_m = m[:, :, word_inds_recon].mean(dim=-1).reshape(-1, spatial_dim, spatial_dim)
        recon_maps.append(sub_m)

    edit_maps = []
    for m in cross_attn_maps_edit:
        spatial_dim = int(round(m.shape[1] ** 0.5))
        sub_m = m[:, :, word_inds_edit].mean(dim=-1).reshape(-1, spatial_dim, spatial_dim)
        edit_maps.append(sub_m)

    recon_avg = torch.cat(recon_maps, dim=0).mean(dim=0, keepdim=True).unsqueeze(0)
    edit_avg = torch.cat(edit_maps, dim=0).mean(dim=0, keepdim=True).unsqueeze(0)

    maps = torch.cat([recon_avg, edit_avg], dim=0)
    maps = F.max_pool2d(maps, kernel_size=3, stride=1, padding=1)
    maps = F.interpolate(maps, size=recon_latent.shape[2:], mode="bilinear", align_corners=False)

    max_val = maps.flatten(2).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
    norm_maps = maps / max_val

    mask = norm_maps.gt(threshold)
    mask = (mask[:1] | mask[1:]).to(dtype=edit_latent.dtype)

    # Làm mờ biên mặt nạ (Gaussian feathering) ở latent-space để khử đường nối cứng và vệt sáng viền
    mask = TF.gaussian_blur(mask, kernel_size=[3, 3], sigma=[1.0, 1.0]).clamp(0.0, 1.0)

    blended = recon_latent + mask * (edit_latent - recon_latent)
    return blended, mask


class DualAttentionInjector:
    """Tiêm Self-Attention (attn1) để khóa hình học khuôn mặt,
    và tiêm Cross-Attention (attn2) có chọn lọc (Token-level Filtering) để hòa trộn tuổi tác tự nhiên.
    Bắt live cross-attention maps phục vụ LocalBlend."""

    def __init__(
        self,
        unet: UNet2DConditionModel,
        self_maps: Dict[int, Dict[str, torch.Tensor]],
        cross_maps: Dict[int, Dict[str, torch.Tensor]],
        attention_control_ratio: float,
        num_inference_steps: int,
        live_capture_resolution: int = 16 * 16,
    ):
        self.unet = unet
        self._orig_processors = unet.attn_processors
        self.self_maps = self_maps
        self.cross_maps = cross_maps
        self.attention_control_ratio = attention_control_ratio
        self.num_inference_steps = num_inference_steps
        self.live_capture_resolution = live_capture_resolution
        self.enabled = False
        self.current_t: Optional[int] = None
        self.capture_mode: Optional[str] = None
        self.captured_cross: Dict[str, List[torch.Tensor]] = {"recon": [], "edit": []}

    def _build_processor(self, name: str):
        injector = self
        is_cross = name.endswith("attn2.processor")

        class _InjectingProcessor:
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                query = attn.to_q(hidden_states)
                context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = attn.to_k(context)
                value = attn.to_v(context)

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)

                # Bắt live cross-attention map TRƯỚC khi bị ghi đè, phục vụ LocalBlend
                if (
                    injector.capture_mode is not None
                    and is_cross
                    and attention_probs.shape[1] == injector.live_capture_resolution
                ):
                    injector.captured_cross[injector.capture_mode].append(attention_probs.detach())

                if injector.enabled:
                    if not is_cross:
                        # 1. Khóa hình học khuôn mặt bằng Self-Attention (attn1)
                        ref_self = injector.self_maps.get(injector.current_t, {}).get(name)
                        if ref_self is not None:
                            attention_probs = ref_self.to(device=value.device, dtype=value.dtype)
                    else:
                        # 2. Tiêm Cross-Attention (attn2) có chọn lọc (Token-level Filtering):
                        # Giữ các token chung: "<start>", "photo", "of", "a" (index 0..3)
                        # và các token đệm/EOS phía sau (index >= 7).
                        # Thả tự do các token tuổi (index 4..6: "{age}", "year", "old") để nếp nhăn già hóa sinh tự nhiên.
                        ref_cross = injector.cross_maps.get(injector.current_t, {}).get(name)
                        if ref_cross is not None:
                            ref_cross = ref_cross.to(device=value.device, dtype=value.dtype)
                            if attention_probs.shape[-1] == ref_cross.shape[-1]:
                                attention_probs[:, :, :4] = ref_cross[:, :, :4]
                                attention_probs[:, :, 7:] = ref_cross[:, :, 7:]

                if attention_probs.dtype != value.dtype:
                    attention_probs = attention_probs.to(value.dtype)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states

        return _InjectingProcessor()

    def register(self) -> None:
        # Gắn processor vào toàn bộ các layer attention (cả attn1 và attn2)
        new_processors = {name: self._build_processor(name) for name in self.unet.attn_processors.keys()}
        self.unet.set_attn_processor(new_processors)

    def restore(self) -> None:
        self.unet.set_attn_processor(self._orig_processors)

    def inject_step(self, step_index: int, t, forward_fn):
        self.current_t = int(t)
        self.enabled = step_index < (self.attention_control_ratio * self.num_inference_steps)
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
        image_size: int = 512,
        use_local_blend: bool = USE_LOCAL_BLEND_DEFAULT,
        local_blend_threshold: float = LOCAL_BLEND_THRESHOLD_DEFAULT,
        debug_check_nan: bool = False,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.unet_checkpoint_dir = unet_checkpoint_dir
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.attention_control_ratio = attention_control_ratio
        self.image_size = image_size
        self.use_local_blend = use_local_blend
        self.local_blend_threshold = local_blend_threshold
        self.debug_check_nan = debug_check_nan

        self.live_capture_resolution = (min(self.image_size // 16, 16)) ** 2
        self.last_local_blend_mask = None

        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def _load_models(self) -> None:
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
        return self.unet(latent, t, encoder_hidden_states=embedding).sample

    def _ddim_prev_step(self, noise_pred: torch.Tensor, t: int, sample: torch.Tensor) -> torch.Tensor:
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
        missing = [int(t) for t in timesteps[:num_injection_steps] if int(t) not in attention_maps]
        if missing:
            raise ValueError(
                f"attention_maps thiếu timestep {missing}. num_inference_steps của Editor "
                f"({self.num_inference_steps}) có thể không khớp với giá trị đã dùng khi chạy "
                f"NullTextInverter.invert() (Module 2) để sinh ra attention_maps này - 2 module "
                f"BẮT BUỘC phải dùng cùng 1 giá trị num_inference_steps."
            )

    def reconstruct(
        self,
        z_T: torch.Tensor,
        null_embeddings: List[torch.Tensor],
        initial_age: int,
        gender_word: str,
        guidance_scale: Optional[float] = 1.0,
    ) -> Image.Image:
        """
        Tái tạo lại ảnh ban đầu từ z_T và null_embeddings.
        Để kiểm tra độ trung thực (Sanity-Check) mà không bị CFG bóp méo thành tranh vẽ,
        sử dụng chính trajectory đảo ngược chuẩn xác của DDIM (mặc định guidance_scale=1.0 theo ODE gốc).
        Đồng bộ từ Cell 15 của FADING_pipeline_kaggle_3.ipynb.
        """
        if self.unet is None or self.vae is None:
            self._load_models()
        g_scale = 1.0 if guidance_scale is None else guidance_scale
        p_alpha = build_prompt_alpha(initial_age, gender_word)
        cond_embedding = self._encode_text(p_alpha)
        latent = z_T.clone()
        timesteps = self.scheduler.timesteps

        with torch.no_grad():
            for i in range(self.num_inference_steps):
                checkpoint()
                t = timesteps[i]
                null_t = null_embeddings[i]

                # Dự đoán nhiễu với null-text optimization
                noise_uncond = self._predict_noise(latent, t, null_t)
                noise_cond = self._predict_noise(latent, t, cond_embedding)

                # Áp dụng guidance_scale đồng bộ với quá trình Inversion (1.0 theo chuẩn ODE vi phân)
                noise_pred = noise_uncond + g_scale * (noise_cond - noise_uncond)
                latent = self._ddim_prev_step(noise_pred, t, latent)

        return self._decode_latent_to_image(latent)

    def edit(
        self,
        z_T: torch.Tensor,
        null_embeddings: List[torch.Tensor],
        attention_maps: Union[Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]], Dict[int, Dict[str, torch.Tensor]]],
        target_ages: List[int],
        gender_word: str,
        output_dir: str,
        initial_age: Optional[int] = None,
        use_local_blend: Optional[bool] = None,
        local_blend_threshold: Optional[float] = None,
        original_image_path: Optional[str] = None,
        embedder: Optional[object] = None,
        apply_mask_blending_flag: bool = False,
    ) -> Dict[int, str]:
        """Hàm chính Module 3: với mỗi target_age, denoise DDIM từ z_T bằng {null_t} + attention
        injection (Dual Attention: Self-Attn khóa hình học + Cross-Attn Token Filtering), kết hợp LocalBlend."""
        if self.unet is None:
            self._load_models()

        if isinstance(attention_maps, tuple):
            self_maps, cross_maps = attention_maps
        else:
            self_maps, cross_maps = {}, attention_maps

        os.makedirs(output_dir, exist_ok=True)
        timesteps = self.scheduler.timesteps
        num_injection_steps = int(self.attention_control_ratio * self.num_inference_steps)
        self._validate_timesteps_available(cross_maps, timesteps, num_injection_steps)

        do_local_blend = self.use_local_blend if use_local_blend is None else use_local_blend
        lb_threshold = self.local_blend_threshold if local_blend_threshold is None else local_blend_threshold

        results: Dict[int, str] = {}

        for target_age in target_ages:
            checkpoint()
            p_tau = build_prompt_tau(target_age, gender_word)
            cond_embedding = self._encode_text(p_tau)
            latent = z_T.clone()

            # Chuẩn bị latent reconstruction và prompt neo cho LocalBlend
            if do_local_blend:
                recon_latent = z_T.clone()
                p_alpha = (
                    build_prompt_alpha(initial_age, gender_word)
                    if initial_age is not None
                    else f"photo of a {gender_word}"
                )
                cond_embedding_recon = self._encode_text(p_alpha)
                word_inds_recon = get_word_inds(p_alpha, gender_word, self.tokenizer)
                word_inds_edit = get_word_inds(p_tau, gender_word, self.tokenizer)
                print(
                    f"[Editor LocalBlend] Khởi chạy song song recon_latent | "
                    f"p_alpha='{p_alpha}' | p_tau='{p_tau}' | threshold={lb_threshold}"
                )

            # CHÚ Ý: Bước 2.3 KHÔNG dùng dynamic_ratio, giữ cố định self.attention_control_ratio (0.8)
            ratio = self.attention_control_ratio
            print(f"[Editor] target_age={target_age} -> attention_control_ratio={ratio}")

            injector = DualAttentionInjector(
                self.unet,
                self_maps,
                cross_maps,
                ratio,
                self.num_inference_steps,
                live_capture_resolution=self.live_capture_resolution,
            )
            injector.register()
            try:
                for i in range(self.num_inference_steps):
                    checkpoint()
                    t = timesteps[i]
                    null_t = null_embeddings[i]

                    # 1. Nếu bật LocalBlend: Denoise bước tái tạo (recon_latent) song song
                    if do_local_blend:
                        injector.captured_cross["recon"].clear()
                        injector.captured_cross["edit"].clear()

                        with torch.no_grad():
                            noise_uncond_recon = self._predict_noise(recon_latent, t, null_t)

                        # Bật capture_mode="recon" cho conditional forward pass
                        injector.capture_mode = "recon"
                        injector.enabled = False  # Không tiêm attention vào recon pass
                        with torch.no_grad():
                            noise_cond_recon = self._predict_noise(recon_latent, t, cond_embedding_recon)
                        injector.capture_mode = None

                        with torch.no_grad():
                            noise_pred_recon = noise_uncond_recon + self.guidance_scale * (
                                noise_cond_recon - noise_uncond_recon
                            )
                            recon_latent = self._ddim_prev_step(noise_pred_recon, t, recon_latent)

                    # 2. Denoise edit latent (tiêm attention bình thường)
                    with torch.no_grad():
                        noise_uncond = self._predict_noise(latent, t, null_t)

                    if do_local_blend:
                        injector.capture_mode = "edit"

                    noise_cond = injector.inject_step(
                        i, t, lambda: self._predict_noise(latent, t, cond_embedding)
                    )

                    if do_local_blend:
                        injector.capture_mode = None

                    with torch.no_grad():
                        noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
                        latent = self._ddim_prev_step(noise_pred, t, latent)

                    # 3. Trộn LocalBlend giữa recon và edit latent ngay sau mỗi bước ddim
                    if do_local_blend:
                        if len(injector.captured_cross["recon"]) > 0 and len(injector.captured_cross["edit"]) > 0:
                            latent, mask = local_blend(
                                recon_latent=recon_latent,
                                edit_latent=latent,
                                cross_attn_maps_recon=injector.captured_cross["recon"],
                                cross_attn_maps_edit=injector.captured_cross["edit"],
                                word_inds_recon=word_inds_recon,
                                word_inds_edit=word_inds_edit,
                                threshold=lb_threshold,
                            )
                            self.last_local_blend_mask = mask.detach().cpu()
                        injector.captured_cross["recon"].clear()
                        injector.captured_cross["edit"].clear()

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

            if apply_mask_blending_flag and original_image_path and os.path.exists(original_image_path) and embedder is not None:
                from src.utils.face_enhancement import apply_mask_blending
                image = apply_mask_blending(original_image_path, image, embedder)

            path = os.path.join(output_dir, f"age_{target_age}.png")
            image.save(path)
            results[target_age] = path
            print(f"[Editor] target_age={target_age} -> {path}")

        return results
