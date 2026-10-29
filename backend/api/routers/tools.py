"""Công cụ cho giao diện: chọn ảnh mặt tốt nhất trong một tập ảnh (crop_quality).

POST /api/tools/rank-faces  (multipart, nhiều file): mỗi ảnh lấy mặt lớn nhất, chấm
chất lượng 7 thành phần + nhất quán danh tính (trừ khi mixed=true), trả xếp hạng kèm
thumbnail để hiển thị. Không lưu gì — dùng để chọn ảnh truy vấn từ ảnh người nhà gửi.
"""

from __future__ import annotations

import base64
import os
import tempfile
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.api import crop_quality as cq

router = APIRouter(prefix="/api/tools", tags=["tools"])

MAX_FILES = 30
MAX_BYTES = 15 * 1024 * 1024
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _thumb(img: np.ndarray, bbox: list[float], edge: int = 160) -> str:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = 0.25 * max(x2 - x1, y2 - y1)
    crop = img[max(0, int(y1 - pad)):min(h, int(y2 + pad)), max(0, int(x1 - pad)):min(w, int(x2 + pad))]
    if crop.size == 0:
        crop = img
    s = edge / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def _largest_face(embedder, path: str):
    faces = embedder.detect_faces(path) or []
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


@router.post("/rank-faces")
def rank_faces_view(files: List[UploadFile] = File(...), mixed: bool = Form(False)):
    from backend.api.dependencies import get_embedder

    if not files:
        raise HTTPException(status_code=400, detail="Chưa chọn ảnh nào.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Tối đa {MAX_FILES} ảnh mỗi lần.")
    embedder = get_embedder()
    items, names, no_face = [], [], []
    for up in files:
        name = up.filename or "anh"
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            no_face.append({"file": name, "reason": "định dạng không hỗ trợ"})
            continue
        data = up.file.read()
        if not data or len(data) > MAX_BYTES:
            no_face.append({"file": name, "reason": "rỗng hoặc vượt 15MB"})
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            no_face.append({"file": name, "reason": "không đọc được ảnh"})
            continue
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(tmp, img)
            face = _largest_face(embedder, tmp)
        except ValueError:
            face = None
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if face is None:
            no_face.append({"file": name, "reason": "không thấy khuôn mặt"})
            continue
        raw = getattr(face, "embedding", None)
        items.append({"frame_bgr": img, "bbox": [float(v) for v in face.bbox],
                      "kps": getattr(face, "kps", None), "det_score": float(face.det_score),
                      "embedding": None if mixed else getattr(face, "normed_embedding", None),
                      "embedding_norm": float(np.linalg.norm(raw)) if raw is not None else None})
        names.append(name)
    ranked = cq.rank_faces(items) if items else []
    out = []
    for pos, r in enumerate(ranked, start=1):
        comps = {k: r.detail[k] for k in cq.WEIGHTS if k in r.detail}
        weakest = min(comps, key=comps.get) if comps else None
        out.append({"rank": pos, "file": names[r.index], "score": round(r.score, 4),
                    "quality": round(r.quality, 4), "consistency": round(r.consistency, 3),
                    "gate": r.gate, "components": comps, "weakest": weakest,
                    "thumbnail": _thumb(items[r.index]["frame_bgr"], items[r.index]["bbox"])})
    median_c = float(np.median([r.consistency for r in ranked])) if ranked else 1.0
    return {"ranked": out, "no_face": no_face, "mixed": mixed,
            "weights": cq.WEIGHTS,
            "warning": ("Phần lớn ảnh lệch danh tính so với số đông — có vẻ nhiều người. "
                        "Bật 'Nhiều người' để chỉ xếp theo chất lượng.")
            if not mixed and len(ranked) >= 3 and median_c < 0.5 else None}
