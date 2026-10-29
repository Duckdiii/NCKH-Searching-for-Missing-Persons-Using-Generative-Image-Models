#!/usr/bin/env python3
"""Sinh embedding + CSV cặp từ FG-NET cho calibrate_link.py.

FG-NET: 1002 ảnh / 82 người, tên file "078A11.JPG" = người 078, 11 tuổi.
- Cặp cùng người: mọi cặp trong cùng một người (có khoảng cách tuổi → khó).
- Cặp khác người: lấy ngẫu nhiên, mặc định gấp 3 lần số cặp cùng người.
Cột person_a/person_b để calibrate_link chia calibration/test THEO NGƯỜI.

LƯU Ý: FG-NET là ảnh tĩnh xuyên tuổi, KHÔNG phải crop camera — ngưỡng rút ra
chỉ là tham khảo tạm, phải hiệu chuẩn lại trên crop camera thật.

    python scripts/build_fgnet_pairs.py --images data/FGNET/FGNET/images \\
        --out outputs/fgnet_pairs
    python scripts/calibrate_link.py --pairs outputs/fgnet_pairs/pairs.csv \\
        --out outputs/cal_fgnet
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from itertools import combinations

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NAME_RE = re.compile(r"^(\d{3})A(\d{2})", re.IGNORECASE)


DET_SIZES = ((640, 640), (256, 256))  # FG-NET kích thước lẫn lộn: thử 640, lùi 256


def make_app(ctx_id: int):
    from insightface.app import FaceAnalysis

    # Chỉ detection + recognition (buffalo_l đủ 5 model dễ hết RAM arena).
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=ctx_id, det_size=DET_SIZES[0])
    return app


def embed_one(app, ctx_id: int, path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    found = None
    for k, size in enumerate(DET_SIZES):
        if k:
            app.prepare(ctx_id=ctx_id, det_size=size)
        faces = app.get(img)
        if faces:
            # Ảnh FG-NET một người: lấy mặt lớn nhất.
            found = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            break
    if len(DET_SIZES) > 1 and k:
        app.prepare(ctx_id=ctx_id, det_size=DET_SIZES[0])
    return None if found is None else found.normed_embedding


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", default="data/FGNET/FGNET/images")
    ap.add_argument("--out", default="outputs/fgnet_pairs")
    ap.add_argument("--diff-ratio", type=float, default=3.0)
    ap.add_argument("--ctx-id", type=int, default=0, help="0=GPU, -1=CPU")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    emb_dir = os.path.join(args.out, "emb")
    os.makedirs(emb_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.images, "*.[jJ][pP][gG]")))
    app = make_app(args.ctx_id)

    items = []  # (person, age, npy_path)
    no_face = []
    for i, p in enumerate(paths):
        m = NAME_RE.match(os.path.basename(p))
        if not m:
            continue
        npy = os.path.join(emb_dir, os.path.splitext(os.path.basename(p))[0] + ".npy")
        if not os.path.exists(npy):
            v = embed_one(app, args.ctx_id, p)
            if v is None:
                no_face.append(os.path.basename(p))
                continue
            np.save(npy, v.astype(np.float32))
        items.append((m.group(1), int(m.group(2)), npy))
        if (i + 1) % 100 == 0:
            print(f"[fgnet] {i + 1}/{len(paths)}")

    by_person: dict[str, list] = {}
    for it in items:
        by_person.setdefault(it[0], []).append(it)

    rows = []
    for person, its in by_person.items():
        for a, b in combinations(its, 2):
            rows.append((a[2], b[2], 1, person, person, a[1], b[1]))
    n_same = len(rows)
    rng = np.random.default_rng(args.seed)
    seen = set()
    target = int(n_same * args.diff_ratio)
    while len(seen) < target:
        i, j = rng.integers(0, len(items), size=2)
        if items[i][0] == items[j][0] or (i, j) in seen or (j, i) in seen:
            continue
        seen.add((i, j))
        a, b = items[i], items[j]
        rows.append((a[2], b[2], 0, a[0], b[0], a[1], b[1]))

    out_csv = os.path.join(args.out, "pairs.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["emb_a", "emb_b", "same", "person_a", "person_b", "age_a", "age_b"])
        w.writerows(rows)
    with open(os.path.join(args.out, "no_face.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(no_face))
    print(f"[fgnet] ảnh={len(paths)} có mặt={len(items)} không mặt={len(no_face)} "
          f"người={len(by_person)}")
    print(f"[fgnet] cặp cùng người={n_same} khác người={len(rows) - n_same} → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
