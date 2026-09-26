#!/usr/bin/env python3
"""Hiệu chuẩn ngưỡng link ID xuyên camera (§10 + §5 tài liệu kiến trúc).

KHÔNG lấy cố định cosine 0.7/0.9. Script quét lưới (accept_threshold, margin)
trên cặp embedding, báo FAR/FRR/accuracy, đề xuất ngưỡng.

Đầu vào CSV (--pairs), mỗi hàng một cặp:
    emb_a,emb_b,same[,person_a,person_b,time_a,time_b]
- emb_a/emb_b: đường dẫn file .npy (vector), hoặc giá trị cosine tính sẵn
  nếu dùng --cosine-col (tên cột chứa cosine).
- same: 1 = cùng người, 0 = khác người (cặp khác người khó nên chiếm đa số
  để hiệu chuẩn có ý nghĩa).

Chống rò rỉ (§10): chia calibration/test THEO NGƯỜI (person split) — frame gần
nhau của cùng người không bao giờ nằm hai bên. Nếu thiếu cột person, fallback
chia theo hàng với cảnh báo rõ ràng.

Chạy không cần DB/GPU:
    python scripts/calibrate_link.py --pairs data/link_pairs.csv --out outputs/cal
    python scripts/calibrate_link.py --synthetic --out outputs/cal  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_pairs(path: str, cosine_col: str | None) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            same = int(row["same"])
            if cosine_col and row.get(cosine_col):
                cos = float(row[cosine_col])
            else:
                a = np.load(row["emb_a"])
                b = np.load(row["emb_b"])
                cos = cosine(a, b)
            pairs.append({"cosine": cos, "same": same,
                          "person_a": row.get("person_a", ""),
                          "person_b": row.get("person_b", "")})
    return pairs


def synthetic(n_same=400, n_diff=1200, dim=128, seed=7) -> list[dict]:
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_same):
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        for v in (base + rng.normal(scale=0.25, size=dim),
                  base + rng.normal(scale=0.25, size=dim)):
            v /= np.linalg.norm(v)
            pairs.append({"cosine": float(np.dot(base, v)), "same": 1,
                          "person_a": "p", "person_b": "p"})
    for i in range(n_diff):
        a = rng.normal(size=dim)
        a /= np.linalg.norm(a)
        # Cặp khó: một nửa lấy từ phân phối gần (mô phỏng người giống nhau).
        scale = 0.9 if i % 2 == 0 else 2.0
        b = a + rng.normal(scale=scale, size=dim)
        b /= np.linalg.norm(b)
        pairs.append({"cosine": float(np.dot(a, b)), "same": 0,
                      "person_a": f"pa{i}", "person_b": f"pb{i}"})
    return pairs


def split_by_person(pairs: list[dict], test_ratio=0.3, seed=7):
    persons = sorted({p["person_a"] for p in pairs} | {p["person_b"] for p in pairs}
                     - {""})
    if len(persons) < 4:
        print("[cal] CẢNH BÁO: thiếu person để split theo người — "
              "chia theo hàng (có nguy cơ rò rỉ frame gần nhau).")
        rng = random.Random(seed)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - test_ratio))
        cal = [pairs[i] for i in idx[:cut]]
        tst = [pairs[i] for i in idx[cut:]]
        return cal, tst, False
    rng = random.Random(seed)
    rng.shuffle(persons)
    cut = int(len(persons) * (1 - test_ratio))
    test_persons = set(persons[cut:])
    cal = [p for p in pairs if p["person_a"] not in test_persons
           and p["person_b"] not in test_persons]
    tst = [p for p in pairs if p not in cal]
    return cal, tst, True


def evaluate(pairs: list[dict], accept: float):
    tp = sum(1 for p in pairs if p["same"] == 1 and p["cosine"] >= accept)
    fn = sum(1 for p in pairs if p["same"] == 1 and p["cosine"] < accept)
    fp = sum(1 for p in pairs if p["same"] == 0 and p["cosine"] >= accept)
    tn = sum(1 for p in pairs if p["same"] == 0 and p["cosine"] < accept)
    far = fp / max(1, fp + tn)  # false accept
    frr = fn / max(1, tp + fn)  # false reject
    acc = (tp + tn) / max(1, len(pairs))
    return {"accept": accept, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "FAR": round(far, 4), "FRR": round(frr, 4),
            "acc": round(acc, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Hiệu chuẩn ngưỡng link ID (§10).")
    ap.add_argument("--pairs", default=None, help="CSV cặp embedding")
    ap.add_argument("--cosine-col", default=None, help="Cột cosine tính sẵn")
    ap.add_argument("--synthetic", action="store_true", help="Dữ liệu giả smoke test")
    ap.add_argument("--out", default="outputs/cal", help="Thư mục ghi kết quả")
    ap.add_argument("--grid", default="0.30,0.85,0.025",
                    help="start,stop,step lưới accept_threshold")
    ap.add_argument("--target-far", type=float, default=0.01,
                    help="FAR mục tiêu để đề xuất ngưỡng")
    args = ap.parse_args()

    if args.synthetic:
        pairs = synthetic()
    elif args.pairs:
        pairs = load_pairs(args.pairs, args.cosine_col)
    else:
        print("[cal] cần --pairs hoặc --synthetic")
        return 2
    print(f"[cal] {len(pairs)} cặp "
          f"(same={sum(p['same'] for p in pairs)}, diff={len(pairs) - sum(p['same'] for p in pairs)})")

    cal, tst, by_person = split_by_person(pairs)
    print(f"[cal] calibration={len(cal)} test={len(tst)} split_theo_người={by_person}")
    start, stop, step = (float(x) for x in args.grid.split(","))
    grid = np.arange(start, stop + 1e-9, step)
    cal_rows = [evaluate(cal, round(float(a), 4)) for a in grid]
    # Đề xuất: FAR <= target trên calibration với FRR nhỏ nhất.
    feas = [r for r in cal_rows if r["FAR"] <= args.target_far]
    best = min(feas, key=lambda r: (r["FRR"], -r["acc"])) if feas else max(
        cal_rows, key=lambda r: r["acc"])
    test_perf = evaluate(tst, best["accept"])
    print(f"[cal] đề xuất accept={best['accept']} "
          f"(cal FAR={best['FAR']} FRR={best['FRR']} acc={best['acc']})")
    print(f"[cal] trên TEST: FAR={test_perf['FAR']} FRR={test_perf['FRR']} "
          f"acc={test_perf['acc']}")
    print("[cal] Đặt LINK_ACCEPT_THRESHOLD=<giá trị> và LINK_MARGIN sau khi "
          "đo thêm margin top-1/top-2 trên tập triển khai; margin mặc định 0.05 "
          "chưa hiệu chuẩn.")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "calibration_grid.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)
    with open(os.path.join(args.out, "recommendation.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"accept_threshold={best['accept']}\n"
                 f"cal_FAR={best['FAR']} cal_FRR={best['FRR']} cal_acc={best['acc']}\n"
                 f"test_FAR={test_perf['FAR']} test_FRR={test_perf['FRR']} "
                 f"test_acc={test_perf['acc']}\n"
                 f"split_by_person={by_person}\n"
                 f"target_far={args.target_far}\n")
    print(f"[cal] ghi {args.out}/calibration_grid.csv + recommendation.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
