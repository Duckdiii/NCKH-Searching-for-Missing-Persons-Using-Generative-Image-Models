"""
Hàm debug tùy chọn: kiểm tra NaN trong tensor, bật/tắt qua config debug.check_nan.

Bổ sung sau khi debug thực tế trên Colab T4 phát hiện 2 nguồn NaN:
  1. VAE decode ở fp16 (lỗi NaN/ảnh đen nổi tiếng của SD1.5 VAE trên một số GPU).
  2. Adam optimizer eps=1e-8 mặc định bị underflow về 0 trong fp16 (sqrt(v)+eps),
     gây NaN ngay từ bước đầu tiên optimizer.step().

Cả 2 đã được fix trực tiếp trong code (xem Editor._decode_latent_to_image và
NullTextInverter._null_text_optimization). Hàm ở đây giữ lại làm công cụ kiểm tra nhanh nếu
cần debug lại trong tương lai (vd đổi GPU khác, đổi phiên bản thư viện) - không xoá hẳn vì đã
chứng minh hữu ích khi khoanh vùng lỗi thực tế.
"""

import torch


def check_nan(tensor: torch.Tensor, name: str, enabled: bool = True) -> bool:
    """Nếu enabled=True và tensor có giá trị NaN, in cảnh báo rõ ràng kèm tên biến + shape,
    trả về True. Không làm gì (trả về False) nếu enabled=False hoặc tensor sạch."""
    if not enabled:
        return False
    has_nan = bool(torch.isnan(tensor).any().item())
    if has_nan:
        print(f"[DEBUG] CẢNH BÁO: phát hiện NaN trong '{name}' (shape={tuple(tensor.shape)})")
    return has_nan
