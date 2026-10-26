#!/usr/bin/env python3
"""Xếp hạng ảnh mặt trong một thư mục, chọn ảnh tốt nhất (cùng thuật toán với camera).

Dùng khi có nhiều ảnh của một người (ảnh gia đình gửi, crop từ camera) và cần chọn
ảnh làm truy vấn. Mỗi ảnh lấy khuôn mặt lớn nhất; ảnh lệch danh tính so với số đông
bị hạ điểm (có thể là người khác lẫn vào thư mục).

    python scripts/rank_crops.py --dir data/test_gallery --top 5
    python scripts/rank_crops.py --dir anh_nguoi_than --copy-best outputs/best.jpg
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.api import crop_quality as cq  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--ctx-id", type=int, default=-1, help="-1 CPU, 0 GPU")
    ap.add_argument("--copy-best", default=None, help="chép ảnh tốt nhất tới đường dẫn này")
    ap.add_argument("--mixed", action="store_true",
                    help="thư mục nhiều người: chỉ xếp theo chất lượng, bỏ kiểm tra cùng người")
    args = ap.parse_args()

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    sizes = ((640, 640), (256, 256))   # ảnh nhỏ (vd 256px) cần det_size nhỏ mới thấy mặt

    paths = sorted(p for p in glob.glob(os.path.join(args.dir, "*")) if p.lower().endswith(EXTS))
    items, names, no_face = [], [], []
    for p in paths:
        img = cv2.imread(p)
        faces = []
        for size in sizes if img is not None else ():
            app.prepare(ctx_id=args.ctx_id, det_size=size)
            faces = app.get(img)
            if faces:
                break
        if not faces:
            no_face.append(os.path.basename(p))
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        items.append({"frame_bgr": img, "bbox": [float(v) for v in f.bbox], "kps": f.kps,
                      "det_score": float(f.det_score),
                      "embedding": None if args.mixed else f.normed_embedding,
                      "embedding_norm": float(np.linalg.norm(f.embedding))})
        names.append(os.path.basename(p))
    if not items:
        print("Không tìm thấy khuôn mặt nào.")
        return 1

    ranked = cq.rank_faces(items)
    print(f"{len(items)} ảnh có mặt, {len(no_face)} ảnh không thấy mặt.")
    consist = [r.consistency for r in ranked]
    if not args.mixed and len(items) >= 3 and float(np.median(consist)) < 0.5:
        print("CẢNH BÁO: phần lớn ảnh lệch danh tính so với số đông — thư mục có vẻ chứa "
              "nhiều người. Dùng --mixed để chỉ xếp theo chất lượng.")
    print(f"{'hạng':>4}  {'điểm':>5}  {'chất lượng':>10}  {'cùng người':>10}  ảnh  [điểm yếu nhất]")
    for i, r in enumerate(ranked[:args.top], start=1):
        comps = {k: r.detail[k] for k in cq.WEIGHTS if k in r.detail}
        weakest = min(comps, key=comps.get) if comps else "-"
        note = f"loại: {r.gate}" if r.gate else f"{weakest}={comps.get(weakest, 0):.2f}"
        print(f"{i:>4}  {r.score:5.3f}  {r.quality:10.3f}  {r.consistency:10.2f}  "
              f"{names[r.index]}  [{note}]")
    if args.copy_best:
        src = os.path.join(args.dir, names[ranked[0].index])
        shutil.copyfile(src, args.copy_best)
        print(f"Đã chép ảnh tốt nhất: {src} -> {args.copy_best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
