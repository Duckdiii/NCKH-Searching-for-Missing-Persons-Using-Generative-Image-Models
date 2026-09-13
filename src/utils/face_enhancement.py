"""
Các tiện ích tiền xử lý và hậu xử lý khuôn mặt (đồng bộ từ Cell 19 của FADING_pipeline_kaggle_3.ipynb):
1. Adaptive Padding (BORDER_REPLICATE) - tránh mất cằm/tai khi align.
2. Shades of Gray White Balance (Minkowski p=6) - triệt tiêu ám xanh lá/sepia trên ảnh cũ.
3. CodeFormer Face Restoration (với fallback an toàn khi không cài đặt CLI).
4. Mask-based Blending - ghép khuôn mặt đã lão hóa vào phông nền và trang phục gốc.
"""

import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def is_effectively_grayscale(image_rgb: np.ndarray, threshold: float = 6.0) -> bool:
    """Bước 2a (từ batch_preprocess_fgnet.ipynb): Kiểm tra ảnh có phải là ảnh đen-trắng/xám thực tế không.
    Tính độ chênh lệch màu trung bình giữa các kênh R-G và G-B.
    Nếu chênh lệch < threshold (mặc định 6.0) -> Ảnh xám -> Bỏ qua White Balance để bảo toàn tông xám gốc."""
    img_float = image_rgb.astype(np.float64)
    diff_rg = np.abs(img_float[:, :, 0] - img_float[:, :, 1]).mean()
    diff_gb = np.abs(img_float[:, :, 1] - img_float[:, :, 2]).mean()
    return float((diff_rg + diff_gb) / 2.0) < threshold


def apply_adaptive_padding(
    image_rgb: np.ndarray,
    face_occupancy_thresh: float = 0.85,
    pad_ratio: float = 0.20,
    border_mode: str = "replicate",
    embedder=None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    return_offset: bool = False
) -> Tuple:
    """Bước 1: Adaptive Padding với cv2.BORDER_REPLICATE.
    Nếu khuôn mặt chiếm >= 85% chiều dài/rộng ảnh gốc hoặc sát mép (< 5% biên),
    thêm padding lặp mép 20% mỗi cạnh để tránh cắt lẹm cằm/tai khi align."""
    H, W = image_rgb.shape[:2]
    faces = []
    if bbox is not None:
        faces.append({"bbox": bbox})
    elif embedder is not None:
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if hasattr(embedder, "get"):
            try:
                faces = embedder.get(img_bgr)
            except Exception:
                pass
        elif hasattr(embedder, "detect_faces"):
            try:
                faces = embedder.detect_faces(img_bgr)
            except Exception:
                pass

    if len(faces) == 0:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            h_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            for (x, y, w, h) in h_faces:
                faces.append({"bbox": [x, y, x + w, y + h]})
        except Exception:
            pass

    need_padding = False
    if len(faces) > 0:
        face = max(faces, key=lambda f: (
            (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
            if isinstance(f, dict)
            else (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))
        bbox_val = face["bbox"] if isinstance(face, dict) else face.bbox
        x1, y1, x2, y2 = bbox_val
        w_face, h_face = x2 - x1, y2 - y1
        occ_w = w_face / W
        occ_h = h_face / H
        if (
            occ_w >= face_occupancy_thresh
            or occ_h >= face_occupancy_thresh
            or x1 < 0.05 * W
            or x2 > 0.95 * W
            or y1 < 0.05 * H
            or y2 > 0.95 * H
        ):
            need_padding = True
    else:
        if min(H, W) < 300:
            need_padding = True

    if need_padding:
        pad_h = int(H * pad_ratio)
        pad_w = int(W * pad_ratio)
        cv_border = cv2.BORDER_REPLICATE if border_mode == "replicate" else cv2.BORDER_REFLECT_101
        padded = cv2.copyMakeBorder(image_rgb, pad_h, pad_h, pad_w, pad_w, cv_border)
        if return_offset:
            return padded, True, (pad_w, pad_h)
        return padded, True

    if return_offset:
        return image_rgb, False, (0, 0)
    return image_rgb, False


def preprocess_face_image(
    image_bgr: np.ndarray,
    kps: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    embedder=None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Pipeline tiền xử lý khuôn mặt (Đồng bộ từ batch_preprocess_fgnet.ipynb & Kaggle 3):
    1. Adaptive Padding (replicate) nếu mặt chiếm >= 85% hoặc sát viền
    2. Kiểm tra Grayscale (is_effectively_grayscale) -> Nếu ảnh xám thì bỏ qua White Balance
    3. Shades of Gray White Balance (Minkowski p=6, kẹp gain [0.75, 1.30])
    4. CodeFormer Face Restoration (w=0.7, fallback an toàn nếu không cài đặt CLI)
    Trả về: (preprocessed_bgr, updated_kps)
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    padded_rgb, was_padded, (pad_w, pad_h) = apply_adaptive_padding(
        image_rgb, border_mode="replicate", bbox=bbox, embedder=embedder, return_offset=True
    )
    new_kps = kps.copy() if kps is not None else None
    if was_padded and new_kps is not None:
        new_kps = new_kps + np.array([pad_w, pad_h], dtype=new_kps.dtype)

    # Bước 2: Kiểm tra Grayscale trước White Balance (chuẩn batch_preprocess_fgnet.ipynb)
    was_grayscale = is_effectively_grayscale(padded_rgb, threshold=6.0)
    if was_grayscale:
        wb_rgb = padded_rgb
    else:
        wb_rgb = apply_white_balance(padded_rgb, p=6, max_shift_thresh=35.0)

    # Bước 3: Phục hồi CodeFormer (w=0.7)
    cf_rgb = run_codeformer(wb_rgb, fidelity_weight=0.7)
    preprocessed_bgr = cv2.cvtColor(cf_rgb, cv2.COLOR_RGB2BGR)
    return preprocessed_bgr, new_kps


def apply_white_balance(
    image_rgb: np.ndarray,
    p: int = 6,
    max_shift_thresh: float = 35.0,
    gain_min: float = 0.75,
    gain_max: float = 1.30,
    return_info: bool = False
) -> np.ndarray:
    """Bước 2: Shades of Gray White Balance (Minkowski p-norm, p=6, Finlayson & Trezzi 2004)
    kết hợp giới hạn gain [0.75, 1.30] & Dynamic Alpha Blending.
    Triệt tiêu vệt xanh lá ở trán trên ảnh sepia đậm và bảo toàn sắc da ấm tự nhiên."""
    img_norm = image_rgb.astype(np.float64) / 255.0

    # 1. Minkowski p-norm (p=6) cho từng kênh
    norm_r = np.power(np.mean(np.power(img_norm[:, :, 0], p)), 1.0 / p)
    norm_g = np.power(np.mean(np.power(img_norm[:, :, 1], p)), 1.0 / p)
    norm_b = np.power(np.mean(np.power(img_norm[:, :, 2], p)), 1.0 / p)

    avg_gray = (norm_r + norm_g + norm_b) / 3.0

    # 2. Raw gains
    raw_gain_r = float(avg_gray / (norm_r + 1e-8))
    raw_gain_g = float(avg_gray / (norm_g + 1e-8))
    raw_gain_b = float(avg_gray / (norm_b + 1e-8))

    # 3. Kẹp gains vào khoảng an toàn
    clamped_gain_r = float(np.clip(raw_gain_r, gain_min, gain_max))
    clamped_gain_g = float(np.clip(raw_gain_g, gain_min, gain_max))
    clamped_gain_b = float(np.clip(raw_gain_b, gain_min, gain_max))

    # 4. Áp dụng gains đã kẹp
    img_float = image_rgb.astype(np.float32)
    wb_temp = np.zeros_like(img_float)
    wb_temp[:, :, 0] = np.clip(img_float[:, :, 0] * clamped_gain_r, 0, 255)
    wb_temp[:, :, 1] = np.clip(img_float[:, :, 1] * clamped_gain_g, 0, 255)
    wb_temp[:, :, 2] = np.clip(img_float[:, :, 2] * clamped_gain_b, 0, 255)

    # 5. Đo độ lệch màu thực tế
    avg_r = float(np.mean(img_float[:, :, 0]))
    avg_g = float(np.mean(img_float[:, :, 1]))
    avg_b = float(np.mean(img_float[:, :, 2]))

    shift_r = abs(float(np.mean(wb_temp[:, :, 0])) - avg_r)
    shift_g = abs(float(np.mean(wb_temp[:, :, 1])) - avg_g)
    shift_b = abs(float(np.mean(wb_temp[:, :, 2])) - avg_b)
    max_shift = max(shift_r, shift_g, shift_b)

    # 6. Dynamic Alpha Blending nếu max_shift > max_shift_thresh
    alpha = max_shift_thresh / (max_shift + 1e-6) if max_shift > max_shift_thresh else 1.0

    blended = img_float * (1.0 - alpha) + wb_temp * alpha
    out_rgb = np.clip(blended, 0, 255).astype(np.uint8)

    if return_info:
        info = {
            "p_norm": (round(float(norm_r), 4), round(float(norm_g), 4), round(float(norm_b), 4)),
            "raw_gains": (round(raw_gain_r, 3), round(raw_gain_g, 3), round(raw_gain_b, 3)),
            "clamped_gains": (round(clamped_gain_r, 3), round(clamped_gain_g, 3), round(clamped_gain_b, 3)),
            "max_shift": round(max_shift, 1),
            "alpha": round(alpha, 2),
        }
        return out_rgb, info
    return out_rgb


def run_codeformer(
    image_rgb: np.ndarray,
    fidelity_weight: float = 0.7,
    unique_tag: str = "temp",
    temp_dir: str = "./outputs/temp_codeformer"
) -> np.ndarray:
    """Bước 3: Chạy CodeFormer inference trên 1 ảnh RGB (mặc định w=0.7).
    Nếu không tìm thấy script CodeFormer CLI, fallback an toàn giữ nguyên ảnh gốc."""
    codeformer_script = None
    for cand_d in ["./CodeFormer", "../CodeFormer", "/kaggle/working/CodeFormer"]:
        cand = os.path.join(cand_d, "inference_codeformer.py")
        if os.path.exists(cand):
            codeformer_script = cand
            break

    if not codeformer_script or not os.path.exists(codeformer_script):
        return image_rgb

    in_dir = os.path.join(temp_dir, f"in_{unique_tag}")
    out_dir = os.path.join(temp_dir, f"out_{unique_tag}")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        in_file = os.path.join(in_dir, "input.png")
        cv2.imwrite(in_file, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        cmd = [
            sys.executable, codeformer_script,
            "-w", str(fidelity_weight),
            "--input_path", in_file,
            "-o", out_dir,
            "--face_upsample"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode != 0:
            return image_rgb

        res_path = os.path.join(out_dir, "final_results", "input.png")
        if not os.path.exists(res_path):
            fin_dir = os.path.join(out_dir, "final_results")
            if os.path.exists(fin_dir) and os.listdir(fin_dir):
                res_path = os.path.join(fin_dir, os.listdir(fin_dir)[0])
            else:
                return image_rgb

        res_bgr = cv2.imread(res_path)
        if res_bgr is None:
            return image_rgb
        return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return image_rgb
    finally:
        shutil.rmtree(in_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def apply_mask_blending(original_image_path: str, generated_image: Image.Image, embedder) -> Image.Image:
    """Hậu xử lý Mask-based Blending: Sử dụng landmark của InsightFace để tạo mặt nạ mềm (soft mask)
    cho vùng khuôn mặt, sau đó pha trộn ảnh đã edit vào ảnh gốc để giữ trọn phông nền, tóc và áo quần."""
    orig_img = cv2.imread(original_image_path)
    if orig_img is None:
        return generated_image
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    gen_np = np.array(generated_image.resize((orig_img.shape[1], orig_img.shape[0]), Image.BICUBIC))

    faces = embedder.detect_faces(original_image_path)
    if len(faces) == 0:
        return generated_image

    face = faces[0]
    kps = face["kps"] if isinstance(face, dict) else face.kps

    mask = np.zeros(orig_img.shape[:2], dtype=np.float32)

    eye_center = (kps[0] + kps[1]) / 2.0
    mouth_center = (kps[3] + kps[4]) / 2.0
    face_center = (eye_center + mouth_center) / 2.0

    eye_dist = np.linalg.norm(kps[0] - kps[1])
    face_radius_x = int(eye_dist * 1.5)
    face_radius_y = int(eye_dist * 1.8)

    cv2.ellipse(
        mask,
        (int(face_center[0]), int(face_center[1])),
        (face_radius_x, face_radius_y),
        0, 0, 360, 1.0, -1
    )

    blur_sigma = max(3.0, eye_dist * 0.15)
    mask = gaussian_filter(mask, sigma=blur_sigma)
    mask = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis]

    blended = (gen_np.astype(np.float32) * mask + orig_img.astype(np.float32) * (1.0 - mask))
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
