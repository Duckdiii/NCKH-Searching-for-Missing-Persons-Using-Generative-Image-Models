#!/usr/bin/env python3
"""So sánh truy vấn bằng ảnh tạo sinh (FADING) với truy vấn bằng ảnh gốc, trên FG-NET.

Đầu vào:
- --results: fgnet_eval_results.csv (chạy FADING trên Kaggle): mỗi hàng một cặp
  source→target cùng người, id_score = cosine(ảnh sinh, ảnh thật ở tuổi đích).
- --emb: thư mục .npy từ build_fgnet_pairs.py (embedding ảnh thật FG-NET).

Đo trên CÙNG các cặp (so sánh ghép cặp):
- baseline_cos = cosine(ảnh nguồn, ảnh đích)  vs  id_score = cosine(ảnh sinh, ảnh đích).
- Rank-1 baseline (truy vấn = ảnh nguồn) trên gallery toàn FG-NET ĐÃ LOẠI ảnh nguồn.
  Notebook FADING_eval_fgnet giữ ảnh nguồn trong gallery → rank-1 của ảnh sinh
  có rò rỉ (ảnh sinh vốn dựng từ ảnh nguồn). Script in cả hai phiên bản baseline
  để thấy mức rò rỉ; rank-1 công bằng cho ảnh sinh cần chạy lại ở Kaggle với
  gallery loại ảnh nguồn (ảnh sinh không có ở máy này).

    python scripts/eval_generative_vs_source.py --results fgnet_eval_results.csv \\
        --emb outputs/fgnet_pairs/emb --out outputs/eval_generative
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path.replace("\\", "/")))[0].upper()


def load_gallery(emb_dir: str) -> tuple[list[str], list[str], np.ndarray]:
    names, labels, vecs = [], [], []
    for p in sorted(glob.glob(os.path.join(emb_dir, "*.npy"))):
        s = stem(p)
        v = np.load(p).astype(np.float64)
        names.append(s)
        labels.append(s[:3])
        vecs.append(v / np.linalg.norm(v))
    return names, labels, np.stack(vecs)


def summarize(label: str, rows: list[dict]) -> dict:
    if not rows:
        return {"nhóm": label, "n": 0}
    base = np.array([r["baseline_cos"] for r in rows])
    gen = np.array([r["id_score"] for r in rows])
    d = gen - base
    out = {
        "nhóm": label,
        "n": len(rows),
        "baseline_cos_mean": round(float(base.mean()), 4),
        "gen_id_score_mean": round(float(gen.mean()), 4),
        "delta_mean": round(float(d.mean()), 4),
        "gen_thắng_%": round(float((d > 0).mean() * 100), 1),
        "rank1_baseline_công_bằng_%": round(
            float(np.mean([r["rank1_baseline_fair"] for r in rows]) * 100), 1),
        "rank1_gen_notebook_rò_rỉ_%": round(
            float(np.mean([r["rank1_gen_leaky"] for r in rows]) * 100), 1),
    }
    try:
        from scipy.stats import wilcoxon

        out["wilcoxon_p"] = float(f"{wilcoxon(gen, base).pvalue:.3g}")
    except Exception:
        out["wilcoxon_p"] = ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="fgnet_eval_results.csv")
    ap.add_argument("--emb", default="outputs/fgnet_pairs/emb")
    ap.add_argument("--out", default="outputs/eval_generative")
    args = ap.parse_args()

    names, labels, G = load_gallery(args.emb)
    idx = {n: i for i, n in enumerate(names)}
    labels_arr = np.array(labels)

    rows, skipped = [], 0
    with open(args.results, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s, t = stem(r["source_img"]), stem(r["target_img"])
            if s not in idx or t not in idx or not r.get("id_score"):
                skipped += 1
                continue
            q = G[idx[s]]
            sims = G @ q
            sims[idx[s]] = -np.inf  # loại chính ảnh nguồn khỏi gallery
            person = s[:3]
            rows.append({
                "person": person,
                "source": s, "target": t,
                "source_age": int(r["source_age"]), "target_age": int(r["target_age"]),
                "baseline_cos": float(G[idx[t]] @ q),
                "id_score": float(r["id_score"]),
                "rank1_baseline_fair": labels_arr[int(np.argmax(sims))] == person,
                "rank1_gen_leaky": str(r.get("rank1_correct")).strip().lower() == "true",
            })

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pairs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    groups = [
        ("Tất cả", rows),
        ("Nguồn < 15 tuổi", [r for r in rows if r["source_age"] < 15]),
        ("Nguồn ≥ 15 tuổi", [r for r in rows if r["source_age"] >= 15]),
        ("Cách tuổi ≤ 10", [r for r in rows if r["target_age"] - r["source_age"] <= 10]),
        ("Cách tuổi > 10", [r for r in rows if r["target_age"] - r["source_age"] > 10]),
    ]
    summary = [summarize(g, rs) for g, rs in groups]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"[eval-gen] cặp dùng được={len(rows)} bỏ qua={skipped} "
          f"(gallery {len(names)} ảnh / {len(set(labels))} người)")
    for s in summary:
        print("[eval-gen] " + " | ".join(f"{k}={v}" for k, v in s.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
