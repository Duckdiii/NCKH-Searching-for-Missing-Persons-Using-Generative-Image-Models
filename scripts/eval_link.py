#!/usr/bin/env python3
"""Đánh giá association xuyên camera (§10 P2: IDF1, ID switch, false merge/split).

Đầu vào (CSV, không cần DB/GPU):
- --truth: tracklet_id,person_id[,t] — ground truth (t: thứ tự thời gian, tùy chọn)
- --pred:  tracklet_id,global_id   — assignment hệ thống. Lấy từ DB bằng:
    SELECT tracklet_id, global_id FROM face_media.identity_assignments
    WHERE valid_to IS NULL;
  (tracklet chưa link / unresolved: bỏ khỏi pred — tính là miss).

Đo ở mức tracklet (mỗi tracklet = 1 quan sát):
- IDF1/IDP/IDR theo định nghĩa MOT (Ristani et al.): khớp lưỡng phân tham lam
  giữa người-thật và ID-dự đoán theo overlap, IDTP/IDFP/IDFN cộng dồn.
- ID switch (proxy): với mỗi người (sắp theo t nếu có), đếm lần chuyển ID giữa
  2 tracklet liên tiếp — gồm cả camera chồng lấn / mặt khuất / người giống nhau
  nếu các ca đó có trong truth.
- False merge: 1 global_id bao ≥2 người. False split: 1 người bị ≥2 global_id.

Ví dụ:
    python scripts/eval_link.py --truth data/gt.csv --pred outputs/pred.csv
    python scripts/eval_link.py --synthetic --out outputs/eval  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_truth(path: str) -> dict:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["tracklet_id"]] = {"person": row["person_id"],
                                       "t": row.get("t", "")}
    return out


def load_pred(path: str) -> dict:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["tracklet_id"]] = row["global_id"]
    return out


def synthetic(seed: int = 11) -> tuple[dict, dict]:
    """Truth 3 người × 4 tracklet + pred có 1 merge sai và 1 split sai."""
    rng = random.Random(seed)
    truth, pred = {}, {}
    for p in ("P1", "P2", "P3"):
        for i in range(4):
            t = f"t{p}_{i}"
            truth[t] = {"person": p, "t": str(i)}
    # P1: 3 tracklet → G1, 1 tracklet → G2 (false split 1 lần switch).
    for i in range(3):
        pred[f"tP1_{i}"] = "G1"
    pred["tP1_3"] = "G2"
    # P2: toàn bộ → G2 (đúng), P3: 2 tracklet → G2 (false merge với P1_3/P2).
    for i in range(4):
        pred[f"tP2_{i}"] = "G2"
    pred["tP3_0"] = "G3"
    pred["tP3_1"] = "G3"
    pred["tP3_2"] = "G2"
    pred["tP3_3"] = "G2"
    _ = rng
    return truth, pred


def evaluate(truth: dict, pred: dict) -> dict:
    # Chỉ xét tracklet có trong cả hai (pred thiếu = miss → IDFN).
    common = [t for t in truth if t in pred]
    missed = [t for t in truth if t not in pred]
    # Contingency người-thật × ID-dự đoán.
    overlap: dict[tuple[str, str], int] = {}
    true_size: dict[str, int] = {}
    pred_size: dict[str, int] = {}
    for t in common:
        p, g = truth[t]["person"], pred[t]
        overlap[(p, g)] = overlap.get((p, g), 0) + 1
        true_size[p] = true_size.get(p, 0) + 1
        pred_size[g] = pred_size.get(g, 0) + 1
    for t in missed:
        p = truth[t]["person"]
        true_size[p] = true_size.get(p, 0) + 1
    # Khớp lưỡng phân tham lam theo overlap giảm dần.
    matched_p, matched_g = set(), set()
    idtp = 0
    matches = []
    for (p, g), c in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if p in matched_p or g in matched_g:
            continue
        matched_p.add(p)
        matched_g.add(g)
        idtp += c
        matches.append({"person": p, "global_id": g, "overlap": c})
    idfp = sum(pred_size.values()) - idtp
    idfn = sum(true_size.values()) - idtp
    idp = idtp / max(1, idtp + idfp)
    idr = idtp / max(1, idtp + idfn)
    idf1 = 2 * idtp / max(1, 2 * idtp + idfp + idfn)
    # ID switch theo thời gian mỗi người.
    by_person: dict[str, list] = {}
    for t in common:
        by_person.setdefault(truth[t]["person"], []).append(t)
    switches = 0
    switch_detail = []
    for p, ts in by_person.items():
        ts_sorted = sorted(ts, key=lambda t: (truth[t]["t"], t))
        for a, b in zip(ts_sorted, ts_sorted[1:]):
            if pred[a] != pred[b]:
                switches += 1
                switch_detail.append({"person": p, "from": pred[a],
                                      "to": pred[b], "at": truth[b]["t"]})
    # False merge / false split.
    persons_of = {}
    ids_of = {}
    for t in common:
        persons_of.setdefault(pred[t], set()).add(truth[t]["person"])
        ids_of.setdefault(truth[t]["person"], set()).add(pred[t])
    false_merge = sorted([g for g, ps in persons_of.items() if len(ps) > 1])
    false_split = sorted([p for p, gs in ids_of.items() if len(gs) > 1])
    return {
        "tracklets_truth": len(truth), "tracklets_scored": len(common),
        "missed_unlinked": len(missed),
        "IDTP": idtp, "IDFP": idfp, "IDFN": idfn,
        "IDP": round(idp, 4), "IDR": round(idr, 4), "IDF1": round(idf1, 4),
        "id_switches": switches, "switch_detail": switch_detail,
        "false_merge_ids": false_merge, "false_split_persons": false_split,
        "matches": matches,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval link ID xuyên camera (§10 P2).")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="outputs/eval_link")
    ap.add_argument("--min-idf1", type=float, default=0.0,
                    help="Gate CI: exit 1 nếu IDF1 thấp hơn")
    args = ap.parse_args()
    if args.synthetic:
        truth, pred = synthetic()
    elif args.truth and args.pred:
        truth, pred = load_truth(args.truth), load_pred(args.pred)
    else:
        print("[eval] cần --truth + --pred hoặc --synthetic")
        return 2
    res = evaluate(truth, pred)
    print(f"[eval] tracklets truth={res['tracklets_truth']} "
          f"scored={res['tracklets_scored']} missed={res['missed_unlinked']}")
    print(f"[eval] IDF1={res['IDF1']} (IDP={res['IDP']} IDR={res['IDR']}) "
          f"IDTP={res['IDTP']} IDFP={res['IDFP']} IDFN={res['IDFN']}")
    print(f"[eval] id_switches={res['id_switches']} "
          f"false_merge={res['false_merge_ids']} false_split={res['false_split_persons']}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval_link.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"[eval] ghi {args.out}/eval_link.json")
    if res["IDF1"] < args.min_idf1:
        print(f"[eval] IDF1 dưới ngưỡng {args.min_idf1} — CẦN XEM LẠI")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
