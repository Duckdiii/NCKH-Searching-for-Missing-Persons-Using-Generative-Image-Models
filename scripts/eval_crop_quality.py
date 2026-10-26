#!/usr/bin/env python3
"""Đánh giá thuật toán chọn crop (backend/api/crop_quality.py) trên FG-NET.

Hai phép đo chuẩn của FIQA (face image quality assessment):

1. ERC — Error-versus-Reject Curve (Grother & Tabassi): ngưỡng so khớp cố định ở
   FMR = 1% (từ cặp khác người). Loại dần các cặp cùng người có chất lượng thấp nhất
   (chất lượng cặp = min hai ảnh) và đo FNMR còn lại. Bộ chấm tốt → FNMR giảm nhanh.
   So sánh: crop_quality mới, score_candidate cũ, ngẫu nhiên.
2. Làm hỏng có kiểm soát: mờ / tối / giảm phân giải ảnh tốt; điểm chất lượng phải
   giảm cùng chiều với cosine(embedding ảnh hỏng, ảnh gốc) — tương quan Spearman.

Cũng in phân bố norm embedding (ArcFace chưa chuẩn hóa) để hiệu chuẩn NORM_LOW/GOOD.

    python scripts/eval_crop_quality.py --images data/FGNET/FGNET/images --out outputs/eval_crop
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.api import camera_tracks as ct  # noqa: E402
from backend.api import crop_quality as cq  # noqa: E402

NAME_RE = re.compile(r"^(\d{3})A(\d{2})", re.IGNORECASE)
REJECT_STEPS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


class _CachedFace:
    """Kết quả detector lưu cache (bbox, kps, det_score, embedding chưa chuẩn hóa)."""

    def __init__(self, bbox, kps, det_score, embedding):
        self.bbox, self.kps, self.det_score = bbox, kps, det_score
        self.embedding = embedding
        self.normed_embedding = embedding / (np.linalg.norm(embedding) + 1e-12)


def make_app(ctx_id: int):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def detect(app, ctx_id: int, img):
    faces = app.get(img)
    if not faces:
        app.prepare(ctx_id=ctx_id, det_size=(256, 256))
        faces = app.get(img)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def old_score(img, face) -> float:
    bbox = [float(v) for v in face.bbox]
    crop = ct.copy_crop(img, bbox)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return ct.score_candidate(area, ct.estimate_blur(gray), ct.estimate_exposure(gray),
                              ct.estimate_frontal(face.kps, bbox), float(face.det_score))


def new_report(img, face) -> cq.QualityReport:
    return cq.assess_face(img, [float(v) for v in face.bbox], kps=face.kps,
                          det_score=float(face.det_score),
                          embedding_norm=float(np.linalg.norm(face.embedding)))


def erc(emb, person, quality, fmr=0.01, seed=7):
    """FNMR sau khi loại tỷ lệ cặp cùng người chất lượng thấp nhất."""
    n = len(person)
    sims = emb @ emb.T
    iu = np.triu_indices(n, 1)
    same = person[iu[0]] == person[iu[1]]
    gen, imp = sims[iu][same], sims[iu][~same]
    thr = float(np.quantile(imp, 1.0 - fmr))
    pair_q = np.minimum(quality[iu[0]], quality[iu[1]])[same]
    order = np.argsort(pair_q, kind="stable")
    out = []
    for r in REJECT_STEPS:
        kept = order[int(round(r * len(order))):]
        out.append(float((gen[kept] < thr).mean()) if len(kept) else float("nan"))
    return thr, out


def degrade(img, kind, level):
    if kind == "blur":
        k = 2 * level + 1
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "dark":
        return np.clip(img.astype(np.float32) * (1.0 - 0.18 * level), 0, 255).astype(np.uint8)
    if kind == "lowres":
        h, w = img.shape[:2]
        f = 1.0 / (1 + level)
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(kind)


def degrade_camera(img, kind, level, rng):
    """Suy giảm kiểu camera giám sát (level 0 = gốc, 4 = nặng)."""
    if level == 0:
        return img
    if kind == "motion":                              # nhòe chuyển động ngang
        k = 4 * level + 1
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, ker)
    if kind == "noise":                               # nhiễu cảm biến thiếu sáng
        g = rng.normal(0, 8 * level, img.shape)
        return np.clip(img.astype(np.float32) * (1 - 0.15 * level) + g, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), max(5, 45 - 10 * level)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "lowres":                              # mặt ở xa: còn ~1/(2^level) độ phân giải
        h, w = img.shape[:2]
        f = 1.0 / (2 ** level)
        return cv2.resize(cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))),
                                     interpolation=cv2.INTER_AREA), (w, h),
                          interpolation=cv2.INTER_LINEAR)
    return degrade(img, kind, level)


def isotonic(x, y, w=None):
    """Hồi quy đơn điệu tăng (pool adjacent violators) — p(khớp) không giảm theo q."""
    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    w = np.ones_like(y) if w is None else np.asarray(w, float)[order]
    blocks = [[y[i], w[i], x[i], x[i]] for i in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, lo2, hi2 = out.pop()
            v1, w1, lo1, _ = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, lo1, hi2])
    return out


def calibrate(args) -> int:
    """Làm hỏng kiểu camera → đo khớp với ẢNH KHÁC cùng người (±3 tuổi) → p(q)."""
    import pandas as pd
    per = pd.read_csv(os.path.join(args.out, "per_image.csv"))
    per["stem"] = per.file.str.rsplit(".", n=1).str[0].str.upper()
    emb_dir = args.emb
    E = {os.path.splitext(f)[0].upper(): np.load(os.path.join(emb_dir, f))
         for f in os.listdir(emb_dir) if f.endswith(".npy")}
    per = per[per.stem.isin(E)].reset_index(drop=True)
    G = np.stack([E[s] for s in per.stem])
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    P, A = per.person.astype(str).values, per.age.values
    n = len(per)
    S = G @ G.T
    iu = np.triu_indices(n, 1)
    thr = float(np.quantile(S[iu][P[iu[0]] != P[iu[1]]], 0.99))
    # Ảnh có >= 1 ảnh khác cùng người trong ±3 tuổi.
    partners = {i: np.where((P == P[i]) & (np.abs(A - A[i]) <= 3) & (np.arange(n) != i))[0]
                for i in range(n)}
    cand = [i for i in range(n) if len(partners[i]) and not (isinstance(per.gate[i], str) and per.gate[i])]
    cache_path = os.path.join(args.out, "degradation_dets.npz")
    cache = None
    if args.rescore:
        if not os.path.isfile(cache_path):
            print(f"[calibrate] thiếu {cache_path}: chạy --calibrate một lần trước.")
            return 1
        z = np.load(cache_path, allow_pickle=True)
        cache = {str(k): v for k, v in zip(z["keys"], z["dets"])}
        # Tập ảnh + chuỗi suy giảm phải trùng lần chạy detector: lấy lại tham số đã lưu.
        if "degrade_n" in z:
            args.degrade_n, args.seed = int(z["degrade_n"]), int(z["seed"])
        print(f"[calibrate] rescore từ cache: degrade_n={args.degrade_n}, seed={args.seed}")
        app = None
    else:
        app = make_app(args.ctx_id)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(cand, size=min(args.degrade_n, len(cand)), replace=False)
    dets_out = {}
    kinds = ("motion", "noise", "jpeg", "lowres")
    rows = []
    for t, idx in enumerate(chosen):
        img = cv2.imread(os.path.join(args.images, per.file[idx]))
        for kind in kinds:
            for level in range(5):
                if level == 0 and kind != kinds[0]:
                    continue                          # ảnh gốc chỉ đo một lần
                im2 = degrade_camera(img, kind, level, rng)   # luôn tạo: giữ chuỗi rng
                key = f"{per.file[idx]}|{kind}|{level}"
                if cache is not None:
                    if key not in cache:
                        raise KeyError(f"Cache thiếu {key}: tập ảnh khác lần chạy detector.")
                    f2 = cache[key]
                    f2 = None if f2 is None or f2 == () else _CachedFace(*f2)
                else:
                    f2 = detect(app, args.ctx_id, im2)
                    dets_out[key] = () if f2 is None else (
                        np.asarray(f2.bbox, float), np.asarray(f2.kps, float),
                        float(f2.det_score), np.asarray(f2.embedding, float))
                if f2 is None:
                    rows.append({"file": per.file[idx], "kind": kind, "level": level,
                                 "detected": False, "q": 0.0, "match_rate": 0.0,
                                 "cos_orig": 0.0})
                    continue
                rep = new_report(im2, f2)
                cos = G[partners[idx]] @ f2.normed_embedding
                rows.append({"file": per.file[idx], "kind": kind, "level": level,
                             "detected": True, "q": rep.total, "gate": rep.gate or "",
                             "match_rate": float((cos >= thr).mean()),
                             "cos_orig": float(np.dot(f2.normed_embedding, G[idx])),
                             **{f"c_{k}": v for k, v in rep.components.items()}})
        if (t + 1) % 10 == 0:
            print(f"[calibrate] {t + 1}/{len(chosen)} ảnh", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "degradation.csv"), index=False)
    if cache is None:
        keys = list(dets_out)
        arr = np.empty(len(keys), dtype=object)       # 1 chiều: tuple rỗng = không thấy mặt
        arr[:] = [dets_out[k] for k in keys]
        np.savez_compressed(cache_path, keys=np.array(keys), dets=arr,
                            degrade_n=args.degrade_n, seed=args.seed)

    # ERC trên tập suy giảm: một crop "lỗi" khi không khớp đa số ảnh cùng người.
    fail = (df.match_rate < 0.5).values
    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    def erc_single(q):
        o = np.argsort(np.asarray(q, float), kind="stable")
        return [float(fail[o[int(r * len(o)):]].mean()) for r in steps]
    curves = {"q (crop_quality)": erc_single(df.q)}
    for c in [c for c in df.columns if c.startswith("c_")]:
        curves[c] = erc_single(df[c].fillna(0))
    rnd = [np.mean(erc_single(rng.random(len(df)))) for _ in range(20)]
    print(f"[calibrate] {len(df)} mẫu ({len(chosen)} ảnh × suy giảm), ngưỡng FMR1% = {thr:.3f}, "
          f"tỷ lệ lỗi gốc = {fail.mean():.3f}")
    print("   loại x%  " + "  ".join(f"{int(r * 100):>5}" for r in steps) + "   AUC")
    for k, c in curves.items():
        print(f"   {k:<20}" + "  ".join(f"{v:5.3f}" for v in c) + f"   {np.mean(c):.4f}")
    print(f"   {'ngẫu nhiên':<20}" + " " * 41 + f"   {np.mean(rnd):.4f} ± {np.std(rnd):.4f}")
    print("[calibrate] theo kiểu suy giảm (q trung bình / tỷ lệ khớp theo level):")
    for kind in kinds:
        sub = df[(df.kind == kind) | (df.level == 0)]
        g = sub.groupby("level").agg(q=("q", "mean"), m=("match_rate", "mean"))
        print(f"   {kind:<7} q={[round(v, 3) for v in g.q]}  khớp={[round(v, 3) for v in g.m]}  "
              f"Spearman(q, khớp)={spearman(sub.q.values, sub.match_rate.values):.3f}")

    # p(khớp | q): isotonic trên mọi mẫu vừa chấm (gồm ảnh gốc level 0) — cùng
    # định nghĩa q hiện tại; không trộn q cũ trong per_image.csv.
    xq, ym = df.q.values, df.match_rate.values
    clean_q = df.q[df.level == 0].tolist()
    blocks = isotonic(xq, ym)
    grid = np.linspace(0, 1, 21)
    centers = np.array([(b[2] + b[3]) / 2 for b in blocks])
    vals = np.array([b[0] for b in blocks])
    p_curve = np.interp(grid, centers, vals)
    calib = {
        "source": f"FG-NET + suy giảm camera ({len(df)} mẫu, {len(clean_q)} ảnh sạch)",
        "base_samples": [round(float(v), 4) for v in np.quantile(
            df.q[df.detected].values, np.linspace(0.02, 0.98, 49))],
        "p_curve_q": [round(float(v), 3) for v in grid],
        "p_curve_p": [round(float(v), 4) for v in p_curve],
        # Không ghi p_broken: ảnh suy giảm nhân tạo vẫn là chân dung chụp thẳng (~2% hỏng),
        # khác xa camera (quay lưng, che, nghiêng) — giữ giả định của QualityModel.
        "p_detect_fail_synthetic": round(float((~df.detected).mean()), 4),
        "fmr_threshold": round(thr, 4),
    }
    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(calib, fh, ensure_ascii=False, indent=2)
    print(f"[calibrate] p(khớp|q) tại q=0,.25,.5,.75,1: "
          f"{[round(float(np.interp(x, grid, p_curve)), 3) for x in (0, .25, .5, .75, 1)]}")
    calib["analysis"] = analyze_degradation(df, args.seed)
    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(calib, fh, ensure_ascii=False, indent=2)
    print(f"[calibrate] ghi {args.out}/degradation.csv + calibration.json")
    return 0


def analyze_degradation(df, seed: int = 7) -> dict:
    """Phân tích degradation.csv: ERC + phép thử chọn crop. Không cần detector."""
    rng = np.random.default_rng(seed)
    steps = [0.0, 0.05, 0.1, 0.2, 0.3]
    base = df[df.level == 0].set_index("file").match_rate
    intrinsic = sorted(base[base < 0.5].index)          # ảnh gốc vốn khớp sai (do tuổi)
    x = df[~df.file.isin(intrinsic)].reset_index(drop=True)
    fail = ((x.match_rate < 0.5) | (~x.detected.astype(bool))).values

    def erc(q):
        o = np.argsort(np.asarray(q, float), kind="stable")
        return [float(fail[o[int(r * len(o)):]].mean()) for r in steps]

    curves = {"q": erc(x.q)}
    for c in [c for c in x.columns if c.startswith("c_")]:
        curves[c] = erc(x[c].fillna(0))
    rnd = [float(np.mean(erc(rng.random(len(x))))) for _ in range(50)]
    best = x.loc[x.groupby("file").q.idxmax()]
    worst = x.loc[x.groupby("file").q.idxmin()]
    # Kỳ vọng chọn ngẫu nhiên = trung bình mọi phiên bản (không bốc một lần — dễ lệch).
    rand_expected = float(x.groupby("file").match_rate.mean().mean())
    oracle = float(x[x.level == 0].match_rate.mean())
    out = {"intrinsic_fail_images": intrinsic, "n_samples": len(x),
           "degradation_failures": int(fail.sum()), "reject_steps": steps,
           "erc": curves, "erc_auc": {k: round(float(np.mean(v)), 4) for k, v in curves.items()},
           "random_auc": [round(float(np.mean(rnd)), 4), round(float(np.std(rnd)), 4)],
           "selection_match": {"best_q": round(float(best.match_rate.mean()), 4),
                               "random_expected": round(rand_expected, 4),
                               "oracle_original": round(oracle, 4),
                               "worst_q": round(float(worst.match_rate.mean()), 4)},
           "selected_kind": best.kind.where(best.level > 0, "original").value_counts().to_dict()}
    print(f"[analyze] bỏ {len(intrinsic)} ảnh vốn khớp sai {intrinsic}; {len(x)} mẫu, "
          f"{out['degradation_failures']} lỗi do suy giảm")
    print("   loại x%  " + "  ".join(f"{int(r * 100):>5}" for r in steps) + "   AUC")
    for k, c in curves.items():
        print(f"   {k:<20}" + "  ".join(f"{v:5.3f}" for v in c) + f"   {np.mean(c):.4f}")
    print(f"   {'ngẫu nhiên':<20}" + " " * 34 + f"   {out['random_auc'][0]:.4f} ± {out['random_auc'][1]:.4f}")
    print(f"[analyze] chọn crop mỗi người — q cao nhất: {out['selection_match']['best_q']:.3f}, "
          f"ngẫu nhiên (kỳ vọng): {out['selection_match']['random_expected']:.3f}, "
          f"ảnh gốc: {out['selection_match']['oracle_original']:.3f}, "
          f"q thấp nhất: {out['selection_match']['worst_q']:.3f}; crop chọn được: {out['selected_kind']}")
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", default="data/FGNET/FGNET/images")
    ap.add_argument("--out", default="outputs/eval_crop")
    ap.add_argument("--ctx-id", type=int, default=-1, help="-1 CPU, 0 GPU")
    ap.add_argument("--degrade-n", type=int, default=60, help="số ảnh tốt dùng cho phép làm hỏng")
    ap.add_argument("--calibrate", action="store_true",
                    help="chỉ chạy suy giảm kiểu camera + hiệu chuẩn p(khớp|q) (cần per_image.csv)")
    ap.add_argument("--emb", default="outputs/fgnet_pairs/emb",
                    help="embedding FG-NET từ build_fgnet_pairs.py (cho --calibrate)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rescore", action="store_true",
                    help="cùng --calibrate: chấm lại q từ cache detector (không chạy detector)")
    ap.add_argument("--analyze", action="store_true",
                    help="chỉ phân tích degradation.csv đã có (không chạy detector)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.calibrate:
        return calibrate(args)
    if args.analyze:
        import pandas as pd
        analyze_degradation(pd.read_csv(os.path.join(args.out, "degradation.csv")), args.seed)
        return 0
    app = make_app(args.ctx_id)

    paths = sorted(glob.glob(os.path.join(args.images, "*.[jJ][pP][gG]")))
    rows, embs, persons, q_new, q_old, norms = [], [], [], [], [], []
    for i, p in enumerate(paths):
        m = NAME_RE.match(os.path.basename(p))
        img = cv2.imread(p)
        if not m or img is None:
            continue
        face = detect(app, args.ctx_id, img)
        if face is None:
            continue
        rep = new_report(img, face)
        norm = float(np.linalg.norm(face.embedding))
        rows.append({"file": os.path.basename(p), "person": m.group(1), "age": int(m.group(2)),
                     "q_new": rep.total, "q_old": old_score(img, face), "norm": norm,
                     **{f"c_{k}": v for k, v in rep.components.items()}, "gate": rep.gate or ""})
        embs.append(face.normed_embedding)
        persons.append(m.group(1))
        q_new.append(rep.total)
        q_old.append(rows[-1]["q_old"])
        norms.append(norm)
        if (i + 1) % 100 == 0:
            print(f"[crop-eval] {i + 1}/{len(paths)}")
    emb = np.asarray(embs, dtype=np.float64)
    person = np.asarray(persons)
    q_new, q_old, norms = map(np.asarray, (q_new, q_old, norms))

    with open(os.path.join(args.out, "per_image.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rng = np.random.default_rng(7)
    curves = {}
    for name, q in (("crop_quality (mới)", q_new), ("score_candidate (cũ)", q_old),
                    ("norm embedding", norms), ("ngẫu nhiên", rng.random(len(q_new)))):
        thr, c = erc(emb, person, q)
        curves[name] = c
    print(f"[crop-eval] {len(q_new)} ảnh / {len(set(persons))} người; ngưỡng FMR=1%: {thr:.3f}")
    print("[crop-eval] ERC — FNMR khi loại x% cặp cùng người chất lượng thấp nhất:")
    print("   loại x%  " + "  ".join(f"{int(r * 100):>5}" for r in REJECT_STEPS))
    for name, c in curves.items():
        print(f"   {name:<22}" + "  ".join(f"{v:5.3f}" for v in c))

    pct = {f"p{k}": round(float(np.percentile(norms, k)), 2) for k in (5, 10, 25, 50, 75, 90)}
    print(f"[crop-eval] norm embedding: {pct}")
    gates = {g: sum(1 for r in rows if r["gate"] == g) for g in {r["gate"] for r in rows} if g}
    print(f"[crop-eval] bị loại cứng: {gates or 'không'}")

    # Làm hỏng có kiểm soát trên các ảnh tốt nhất.
    best = np.argsort(-q_new)[:args.degrade_n]
    deg = {}
    for kind in ("blur", "dark", "lowres"):
        qs, cs = [], []
        for idx in best:
            img = cv2.imread(os.path.join(args.images, rows[idx]["file"]))
            for level in range(0, 5):
                im2 = degrade(img, kind, level)
                f2 = detect(app, args.ctx_id, im2)
                if f2 is None:
                    qs.append(0.0)
                    cs.append(0.0)
                    continue
                qs.append(new_report(im2, f2).total)
                cs.append(float(np.dot(f2.normed_embedding, emb[idx])))
        deg[kind] = {"spearman_q_vs_cosine": round(spearman(np.array(qs), np.array(cs)), 3),
                     "mean_q_by_level": [round(float(np.mean(qs[l::5])), 3) for l in range(5)],
                     "mean_cos_by_level": [round(float(np.mean(cs[l::5])), 3) for l in range(5)]}
        print(f"[crop-eval] làm hỏng '{kind}': {deg[kind]}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_images": len(q_new), "fmr_threshold": thr, "reject_steps": REJECT_STEPS,
                   "erc_fnmr": curves, "norm_percentiles": pct, "gates": gates,
                   "degradation": deg}, fh, ensure_ascii=False, indent=2)
    print(f"[crop-eval] ghi {args.out}/per_image.csv + summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
