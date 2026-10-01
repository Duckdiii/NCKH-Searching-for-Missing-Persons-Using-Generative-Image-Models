#!/usr/bin/env python3
"""Benchmark chính sách crop JPEG (§10 tài liệu kiến trúc đa camera).

So sánh Q80/85/90 × cạnh dài 192/256/320 với baseline chất lượng cao trên
crop khuôn mặt đã gán nhãn của camera thử nghiệm. Đo:
- byte/crop (p50/p95/p99 — 256px KHÔNG bảo đảm kích thước file),
- thời gian encode/decode,
- nhận dạng sau giải mã (cosine vs embedding baseline, nếu embedder sẵn sàng).

Chạy không cần DB/GPU:
    python scripts/bench_crop_jpeg.py --crops data/camera_crops --out outputs/bench_crop

Embedder là tùy chọn: nếu backend InsightFace import được và --embedder đặt,
đo thêm cosine-after-decode; nếu không, chỉ đo byte + tốc độ.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CONFIGS = [(192, 80), (192, 85), (192, 90),
           (256, 80), (256, 85), (256, 90),
           (320, 80), (320, 85), (320, 90)]
BASELINE = (512, 95)
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_crops(root: str, limit: int) -> list[np.ndarray]:
    paths: list[str] = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(EXTS):
                paths.append(os.path.join(dirpath, f))
            if len(paths) >= limit:
                break
        if len(paths) >= limit:
            break
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
    return imgs


def encode(img: np.ndarray, edge: int, q: int) -> tuple[bytes, float, tuple[int, int]]:
    h, w = img.shape[:2]
    longest = max(h, w)
    dst = img
    if longest > edge:
        s = edge / float(longest)
        dst = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    t0 = time.perf_counter()
    ok, buf = cv2.imencode(".jpg", dst,
                           [int(cv2.IMWRITE_JPEG_QUALITY), q,
                            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1])
    dt = (time.perf_counter() - t0) * 1000.0
    if not ok:
        raise IOError("encode thất bại")
    dh, dw = dst.shape[:2]
    return bytes(buf), dt, (dw, dh)


def decode(data: bytes) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    dt = (time.perf_counter() - t0) * 1000.0
    if img is None:
        raise IOError("decode thất bại")
    return img, dt


def try_embedder():
    try:
        from backend.api.dependencies import get_embedder
        emb = get_embedder()
        return emb
    except Exception as exc:
        print(f"[bench] embedder không sẵn sàng ({exc}) — bỏ qua đo nhận dạng.")
        return None


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Q/size crop JPEG (§10).")
    ap.add_argument("--crops", required=True, help="Thư mục crop đã gán nhãn")
    ap.add_argument("--out", default="outputs/bench_crop", help="Thư mục ghi CSV")
    ap.add_argument("--limit", type=int, default=200, help="Số crop tối đa")
    ap.add_argument("--embedder", action="store_true",
                    help="Đo thêm cosine-after-decode (cần model backend)")
    args = ap.parse_args()

    imgs = load_crops(args.crops, args.limit)
    if not imgs:
        print(f"[bench] không tìm thấy ảnh trong {args.crops}")
        return 1
    print(f"[bench] {len(imgs)} crop từ {args.crops}")

    emb = try_embedder() if args.embedder else None
    # Baseline embedding cho mỗi crop (file gốc / Q95-512).
    base_vecs: list[np.ndarray | None] = []
    if emb is not None:
        import tempfile
        for img in imgs:
            fd, tmp = tempfile.mkstemp(suffix=".png")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(cv2.imencode(".png", img)[1].tobytes())
                try:
                    v = np.asarray(emb.embed(tmp), dtype=np.float64)
                    v = v / (np.linalg.norm(v) + 1e-9)
                except Exception:
                    v = None
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            base_vecs.append(v)
    else:
        base_vecs = [None] * len(imgs)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for edge, q in CONFIGS + [BASELINE]:
        tag = f"{edge}px_Q{q}"
        is_base = (edge, q) == BASELINE
        sizes, enc_ms, dec_ms, cos = [], [], [], []
        for i, img in enumerate(imgs):
            data, e_ms, _ = encode(img, edge, q)
            dec, d_ms = decode(data)
            sizes.append(len(data))
            enc_ms.append(e_ms)
            dec_ms.append(d_ms)
            if emb is not None and base_vecs[i] is not None and not is_base:
                import tempfile
                fd, tmp = tempfile.mkstemp(suffix=".jpg")
                try:
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(data)
                    try:
                        v = np.asarray(emb.embed(tmp), dtype=np.float64)
                        v = v / (np.linalg.norm(v) + 1e-9)
                        cos.append(float(np.dot(v, base_vecs[i])))
                    except Exception:
                        pass
                finally:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        rows.append({
            "config": tag + ("_baseline" if is_base else ""),
            "n": len(sizes),
            "bytes_p50": round(pct(sizes, 50), 1),
            "bytes_p95": round(pct(sizes, 95), 1),
            "bytes_p99": round(pct(sizes, 99), 1),
            "enc_ms_p50": round(pct(enc_ms, 50), 3),
            "dec_ms_p50": round(pct(dec_ms, 50), 3),
            "cos_mean": round(float(np.mean(cos)), 4) if cos else "",
            "cos_min": round(float(np.min(cos)), 4) if cos else "",
        })
        print(f"[bench] {rows[-1]['config']}: p50={rows[-1]['bytes_p50']}B "
              f"p95={rows[-1]['bytes_p95']}B cos_mean={rows[-1]['cos_mean']}")
    csv_path = os.path.join(args.out, "bench_crop_jpeg.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[bench] ghi {csv_path}")
    print("[bench] LƯU Ý: 30KiB/crop trong tài liệu là ngân sách giả định — "
          "dùng p50/p95 đo được để lập ngân sách thật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
