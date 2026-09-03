# Module 3 — Editing (Cross-Attention Control): giải thích kèm code

File thật: [`src/fading/editing.py`](../src/fading/editing.py)

## Đầu vào Module 3 nhận

```
z_T            ← từ Module 2 (nhiễu ban đầu)
{null_t}       ← từ Module 2 (danh sách 50 vector null đã tối ưu, mỗi timestep 1 cái)
M_t_alpha      ← từ Module 2 (attention map chụp sẵn, theo từng timestep × layer)
target_ages    ← nhập tay, vd [30, 50, 70]
gender_word    ← vd "woman"
```

---

## Việc 1 — Kiểm tra an toàn trước khi chạy gì cả

Xem `M_t_alpha` có đủ dữ liệu cho 80% bước đầu không. Nếu Module 2 chạy với số bước khác
Module 3 → **dừng ngay, báo lỗi rõ ràng** — không chạy tiếp rồi ra ảnh sai âm thầm.

```python
def _validate_timesteps_available(
    self, attention_maps, timesteps, num_injection_steps: int
) -> None:
    missing = [int(t) for t in timesteps[:num_injection_steps] if int(t) not in attention_maps]
    if missing:
        raise ValueError(
            f"attention_maps thiếu timestep {missing}. num_inference_steps của Editor "
            f"({self.num_inference_steps}) có thể không khớp với giá trị đã dùng khi chạy "
            f"NullTextInverter.invert() (Module 2)..."
        )
```

Gọi ngay đầu `edit()`, **trước khi** vào vòng lặp target_age:

```python
timesteps = self.scheduler.timesteps
num_injection_steps = int(self.attention_control_ratio * self.num_inference_steps)
self._validate_timesteps_available(attention_maps, timesteps, num_injection_steps)
```

---

## Việc 2 — Với MỖI target_age (vd chạy 3 lần cho [30, 50, 70])

```python
for target_age in target_ages:
    ...
```

### 2a. Tạo prompt mới

`P_tau = "photo of a 50 year old woman"` (ví dụ target_age=50), encode qua CLIP → 1 embedding.

```python
p_tau = build_prompt_tau(target_age, gender_word)
cond_embedding = self._encode_text(p_tau)
latent = z_T.clone()
```

### 2b. Gắn "bộ ghi đè attention" vào UNet

```python
injector = AttentionInjector(
    self.unet, attention_maps, self.attention_control_ratio, self.num_inference_steps
)
injector.register()
```

`register()` thay processor của các lớp **cross-attention** (chỉ `attn2`, không đụng `attn1`
self-attention) bằng phiên bản có thể ghi đè:

```python
def register(self) -> None:
    new_processors = {}
    for name in self.unet.attn_processors.keys():
        if name.endswith("attn2.processor"):
            new_processors[name] = self._build_processor(name)
        else:
            new_processors[name] = self._orig_processors[name]
    self.unet.set_attn_processor(new_processors)
```

### 2c. Chạy vòng lặp khử nhiễu 50 bước, đi từ z_T về ảnh sạch

```python
for i in range(self.num_inference_steps):
    t = timesteps[i]
    null_t = null_embeddings[i]
```

**Gọi UNet lần 1** với `null_t` → dự đoán nhiễu "không điều kiện". Bước này **không** ghi đè
attention gì cả (vì `injector.enabled` mặc định `False`):

```python
    with torch.no_grad():
        noise_uncond = self._predict_noise(latent, t, null_t)
```

**Gọi UNet lần 2** với `P_tau` — đúng lúc này `injector.inject_step()` mới bật cờ `enabled`:

```python
    noise_cond = injector.inject_step(
        i, t, lambda: self._predict_noise(latent, t, cond_embedding)
    )
```

```python
def inject_step(self, step_index: int, t, forward_fn):
    self.current_t = int(t)
    self.enabled = step_index < self.attention_control_ratio * self.num_inference_steps
    with torch.no_grad():
        result = forward_fn()
    self.enabled = False
    return result
```

Bên trong lần forward đó, ở **mỗi cross-attention layer**, `_InjectingProcessor` quyết định
ghi đè hay không:

```python
query = attn.to_q(hidden_states)
context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
key = attn.to_k(context)
value = attn.to_v(context)          # LUÔN tính từ P_tau (nhánh cond đang được gọi)

reference = None
if injector.enabled:
    reference = injector.reference_maps.get(injector.current_t, {}).get(name)

if reference is not None:
    # Trong 80% bước đầu, layer NÀY có map lưu sẵn -> ghi đè
    attention_probs = reference.to(device=value.device, dtype=value.dtype)
else:
    # Ngoài 80% đầu, HOẶC layer bị lọc bỏ ở Module 2 (>32x32) -> tính tự do (chủ đích)
    attention_probs = attn.get_attention_scores(query, key, attention_mask)

hidden_states = torch.bmm(attention_probs, value)   # value luôn từ P_tau
```

- Nếu `i` nằm trong 80% bước đầu → thay `attention_probs` bằng `M_t_alpha[t][layer]` (nếu
  layer đó có lưu), nhưng vẫn nhân với `value` tính từ `P_tau`.
- Nếu ngoài 80% đó, hoặc layer không có map lưu sẵn → tính attention hoàn toàn tự do theo
  `P_tau`.

**Trộn CFG và lùi 1 bước latent**:

```python
    with torch.no_grad():
        noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
        latent = self._ddim_prev_step(noise_pred, t, latent)
```

Lặp lại đúng 50 lần như vậy → latent cuối cùng gần như hết nhiễu.

### 2d. Gỡ bộ ghi đè ra

```python
finally:
    injector.restore()
```

Trả UNet về trạng thái bình thường, để lần lặp target_age tiếp theo không bị ảnh hưởng.

### 2e. Giải mã ra ảnh

```python
image = self._decode_latent_to_image(latent)
path = os.path.join(output_dir, f"age_{target_age}.png")
image.save(path)
results[target_age] = path
```

```python
def _decode_latent_to_image(self, latent: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(image_np)
```

---

## Việc 3 — Trả kết quả

Sau khi lặp xong hết `target_ages`:

```python
return results
```

```python
{30: "outputs/edited_images/age_30.png",
 50: "outputs/edited_images/age_50.png",
 70: "outputs/edited_images/age_70.png"}
```

---

## Tóm 1 câu

Module 3 lấy "khung xương" (`z_T`) và "keo dán giữ mặt không đổi" (`null_t` + `M_t_alpha`) từ
Module 2, rồi với mỗi độ tuổi mong muốn, chạy lại quá trình sinh ảnh của Stable Diffusion —
nhưng ép nó "nhìn" vào đúng những vùng ảnh cũ (mắt, mũi, khuôn mặt...) trong phần lớn quá
trình, chỉ đổi câu prompt tuổi, để ra ảnh người đó lúc già/trẻ hơn mà vẫn giữ được đặc điểm
nhận dạng.
