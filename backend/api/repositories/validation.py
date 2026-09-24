"""T02 — Validation service-side (chặn sớm, message rõ) trước khi chạm DB.

DDL đã có CHECK/FK (purpose ghép, bbox x2>x1, vector NOT NULL...). Các hàm
này chặn sớm ở service để trả lỗi 4xx rõ ràng; DB vẫn là chốt cuối cùng.
Quyết định điều kiện nào ở đâu:
- DB: purpose ghép (id, purpose), reference chỉ là image, input_purpose
  cố định reference, unique idempotency/variant/embedding.
- Service (file này): media_type theo vai trò, bbox trong frame, frame tĩnh
  index=0/offset=0, vector hữu hạn + đúng chiều + chuẩn hóa.
"""

from __future__ import annotations

import math
import re

from .errors import PurposeError

PURPOSES = ("reference", "search")
ASSET_ROLES: dict[str, str] = {
    "original_image": "image",
    "original_video": "video",
    "frame": "image",
    "crop": "image",
    "generated": "image",
}
CROP_METHODS = ("bbox", "ffhq", "restored")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def require_purpose(purpose: str, allowed: tuple[str, ...]) -> str:
    """Purpose phải do endpoint truyền rõ, không tin frontend (T02)."""
    if purpose not in allowed:
        raise PurposeError(
            f"purpose={purpose!r} không hợp lệ; chỉ chấp nhận {list(allowed)}."
        )
    return purpose


def check_asset_role(media_type: str, role: str) -> None:
    expected = ASSET_ROLES.get(role)
    if expected is None:
        raise ValueError(f"Vai trò asset không xác định: {role!r}.")
    if media_type != expected:
        raise ValueError(
            f"Asset vai trò {role!r} phải là {expected!r}, nhận {media_type!r}."
        )


def check_static_frame(frame_index: int, offset_ms: int) -> None:
    if frame_index != 0 or offset_ms != 0:
        raise ValueError(
            "Frame ảnh tĩnh phải có frame_index=0 và offset_ms=0."
        )


def check_bbox(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> None:
    for name, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if not math.isfinite(value):
            raise ValueError(f"Bbox {name} phải là số hữu hạn.")
    if not (x2 > x1 and y2 > y1):
        raise ValueError("Bbox phải thỏa x2 > x1 và y2 > y1.")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"Bbox [{x1},{y1},{x2},{y2}] nằm ngoài frame {width}x{height}."
        )
    if min(x2 - x1, y2 - y1) < 1.0:
        raise ValueError("Bbox quá nhỏ (< 1px).")


def check_sha256(value: str) -> None:
    if not _SHA256_RE.match(value or ""):
        raise ValueError("sha256 phải là 64 ký tự hex thường.")


def check_vector(
    values: list[float], dimensions: int, normalized: bool
) -> None:
    if dimensions <= 0:
        raise ValueError("dimensions phải > 0.")
    if len(values) != dimensions:
        raise ValueError(
            f"Vector có {len(values)} chiều, kỳ vọng {dimensions}."
        )
    for value in values:
        if not math.isfinite(value):
            raise ValueError("Vector chứa NaN/Infinity — từ chối ghi DB.")
    if normalized:
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:
            raise ValueError("Vector zero không thể chuẩn hóa L2.")
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(
                f"Vector khai normalized nhưng ||v||={norm:.4f} (kỳ vọng ~1.0)."
            )


def check_model_meta(model_name: str, model_version: str) -> None:
    if not model_name.strip() or not model_version.strip():
        raise ValueError("Cần đủ model_name/model_version để tái hiện xử lý.")
