"""Tối ưu đồng thời số khung lấy mẫu (fps) và số crop giữ mỗi tracklet (K).

Ghép ba thuật toán: sampling_policy (hình học → thời gian trong vùng, fps bám vết),
crop_quality (điểm chất lượng q của từng khung) và forgetting (số ngày lưu).
Xem docs/quy-trinh-toi-uu-camera.md.

Mô hình một lượt người đi qua camera (Monte Carlo, cùng số ngẫu nhiên cho mọi cấu hình):

    n      = fps · T                          số khung lấy mẫu trong vùng nhận diện
    q_j    = clip(b + ε_j),  b ~ phân bố nền, ε_j ~ N(0, σ_khung)
             mỗi khung hỏng hẳn (quay lưng, che, nghiêng) với xác suất p_hỏng → q_j = 0
    giữ    = K khung q cao nhất có q ≥ q_min (dưới fps bám vết: mỗi khung một tracklet)
    P(tìm thấy) = 1 − (1 − p(q₁)) · Π_{k≥2} (1 − (1 − ρ)·p(q_k))
             p(q): xác suất một crop chất lượng q khớp đúng (hiệu chuẩn từ dữ liệu);
             ρ: tương quan lỗi giữa các crop cùng lượt (cùng người, cùng ánh sáng)
    lưu    = crop/lượt × t_lược + tracklet/lượt × (t_quên − t_lược)   [crop·ngày]

Mục tiêu: P(tìm thấy) lớn nhất trong ngân sách GPU (fps) và ổ đĩa; khuyến nghị cấu
hình rẻ nhất đạt ≥ ``near_best`` × P tốt nhất (điểm "đầu gối" của đường Pareto).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from backend.api import forgetting as fg
from backend.api import sampling_policy as sp

DEFAULT_CALIBRATION = os.path.join("outputs", "eval_crop", "calibration.json")


@dataclass
class QualityModel:
    """Phân bố chất lượng khung + đường p(khớp | q).

    Mặc định là giả định (logistic quanh q = 0.45); nên nạp từ
    scripts/eval_crop_quality.py (calibration.json) hoặc dữ liệu camera thật.
    """
    base_samples: list = field(default_factory=lambda: list(np.linspace(0.35, 0.85, 21)))
    frame_sigma: float = 0.12
    p_broken: float = 0.4
    p_curve_q: list = field(default_factory=lambda: [round(x, 2) for x in np.linspace(0, 1, 11)])
    p_curve_p: list = field(default_factory=lambda: [
        round(0.97 / (1 + math.exp(-12 * (x - 0.45))), 4) for x in np.linspace(0, 1, 11)])
    source: str = "default-assumption"

    def p_match(self, q: np.ndarray) -> np.ndarray:
        return np.interp(q, self.p_curve_q, self.p_curve_p)

    @classmethod
    def load(cls, path: str = DEFAULT_CALIBRATION) -> "QualityModel":
        if not os.path.isfile(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class Budgets:
    fps_max: Optional[float] = None          # GPU cho camera này
    storage_bytes: Optional[float] = None    # dung lượng ổn định cho camera này
    people_per_hour: float = 60.0
    hours_per_day: float = 12.0


@dataclass(frozen=True)
class SearchSpace:
    fps: tuple = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    k: tuple = (1, 2, 3, 4, 5)
    q_min: tuple = (0.0, 0.3, 0.5)
    rho: float = 0.6
    fmr: float = 1e-4        # khớp nhầm mỗi crop người lạ ở ngưỡng tìm kiếm (cosine 0.6 ArcFace)
    passes: int = 3000
    seed: int = 7
    near_best: float = 0.99


def _simulate(fps: float, k: int, q_min: float, dwell_s: float, fragmented: bool,
              qm: QualityModel, rho: float, rnd: dict) -> dict:
    """Mô phỏng ``passes`` lượt với số ngẫu nhiên dùng chung ``rnd``."""
    n_exp = fps * dwell_s
    n_max = rnd["eps"].shape[1]
    n = np.minimum(np.floor(n_exp + rnd["frac"]).astype(int), n_max)    # làm tròn ngẫu nhiên
    cols = np.arange(n_max)[None, :]
    valid = cols < n[:, None]
    q = np.clip(rnd["base"][:, None] + qm.frame_sigma * rnd["eps"], 0.0, 1.0)
    q = np.where(rnd["broken"] < qm.p_broken, 0.0, q)
    q = np.where(valid & (q >= max(q_min, 1e-9)), q, -1.0)
    kept_q = -np.sort(-q, axis=1)                                   # giảm dần, -1 = không có
    usable = (kept_q >= 0)
    n_usable = usable.sum(axis=1)
    if fragmented:
        # Dưới fps bám vết: mỗi khung là một tracklet 1 crop → giữ mọi khung dùng được.
        keep_n = n_usable
        tracklets = np.maximum(n_usable, (n > 0).astype(int))
    else:
        keep_n = np.minimum(n_usable, k)
        tracklets = (n > 0).astype(int)
    take = cols < keep_n[:, None]
    p = np.where(take, qm.p_match(np.clip(kept_q, 0, 1)), 0.0)
    disc = np.where(cols == 0, 1.0, 1.0 - rho)
    miss = np.prod(1.0 - p * disc, axis=1)
    return {"p_found": float(np.mean(1.0 - miss)),
            "crops_per_pass": float(np.mean(keep_n)),
            "tracklets_per_pass": float(np.mean(tracklets)),
            "samples_per_pass": round(n_exp, 2),
            "p_no_usable": float(np.mean(n_usable == 0))}


def optimize(geom: sp.CameraGeometry, targets: sp.SamplingTargets = sp.SamplingTargets(),
             qm: Optional[QualityModel] = None, budgets: Budgets = Budgets(),
             space: SearchSpace = SearchSpace(),
             forget: Optional[fg.ForgettingParams] = None) -> dict:
    qm = qm or QualityModel.load()
    forget = forget or fg.ForgettingParams()
    base_plan = sp.plan_sampling(geom, targets)
    dwell = base_plan["dwell_s"]
    fps_trk = base_plan["fps_tracking"]
    cap = min(targets.fps_max, geom.camera_fps)
    if budgets.fps_max is not None:
        cap = min(cap, budgets.fps_max)

    rng = np.random.default_rng(space.seed)
    n_max = max(1, int(math.ceil(max(space.fps) * dwell)) + 1)
    rnd = {"base": rng.choice(np.asarray(qm.base_samples, dtype=float), size=space.passes),
           "eps": rng.standard_normal((space.passes, n_max)),
           "broken": rng.random((space.passes, n_max)),
           "frac": rng.random(space.passes)}

    # Ngày lưu theo lịch quên của người lạ thấy 1 lần (trường hợp phổ biến nhất).
    sched = fg.schedule(fg.stability_days(1, float(np.median(qm.base_samples)), None, forget),
                        forget)
    t_thin, t_forget = sched["thin_after_days"], sched["forget_after_days"]
    passes_day = budgets.people_per_hour * budgets.hours_per_day

    rows = []
    for fps in space.fps:
        fragmented = fps < fps_trk
        # Bị tách tracklet: mỗi tracklet 1 khung, K không đổi kết quả → chỉ chạy K nhỏ nhất.
        for k in (space.k[:1] if fragmented else space.k):
            for q_min in space.q_min:
                r = _simulate(fps, k, q_min, dwell, fragmented, qm, space.rho, rnd)
                crop_days = (r["crops_per_pass"] * t_thin
                             + r["tracklets_per_pass"] * (t_forget - t_thin))
                storage = passes_day * crop_days * targets.bytes_per_crop
                # Crop người lạ trong index tại một thời điểm → kỳ vọng số kết quả khớp
                # nhầm mỗi truy vấn (mỗi crop là một cơ hội khớp nhầm với xác suất FMR).
                index_crops = passes_day * crop_days
                rows.append({"fps": fps, "k": None if fragmented else k, "q_min": q_min,
                             "fragmented": fragmented,
                             **{key: round(v, 4) if isinstance(v, float) else v
                                for key, v in r.items()},
                             "storage_bytes": round(storage),
                             "index_crops": round(index_crops),
                             "false_alarms_per_query": round(index_crops * space.fmr, 1),
                             "within_budget": bool(fps <= cap + 1e-9 and (
                                 budgets.storage_bytes is None
                                 or storage <= budgets.storage_bytes))})

    feasible = [r for r in rows if r["within_budget"]] or rows
    best_p = max(r["p_found"] for r in feasible)
    good = [r for r in feasible if r["p_found"] >= space.near_best * best_p]
    # Rẻ nhất: ít crop trong index (ổ đĩa + báo động giả), rồi ít GPU, rồi ít tracklet.
    rec = min(good, key=lambda r: (r["index_crops"], r["fps"], r["tracklets_per_pass"]))
    pareto = _pareto(feasible)
    warnings = list(base_plan["warnings"])
    if qm.source == "default-assumption":
        warnings.append("p(q) và phân bố chất lượng là giả định — chạy "
                        "scripts/eval_crop_quality.py để hiệu chuẩn (calibration.json).")
    if not base_plan["usable"]:
        warnings.append("Vùng camera không nhận diện được mặt: tối ưu fps/K vô nghĩa, "
                        "sửa vị trí/ống kính trước.")
    return {"recommended": rec, "best_p_found": round(best_p, 4), "pareto": pareto,
            "grid": rows, "dwell_s": dwell, "fps_tracking": fps_trk, "fps_cap": cap,
            "retention_days": {"thin": t_thin, "forget": t_forget},
            "quality_model": qm.source, "warnings": warnings,
            "inputs": {"geometry": asdict(geom), "budgets": asdict(budgets),
                       "space": {**asdict(space), "fps": list(space.fps), "k": list(space.k),
                                 "q_min": list(space.q_min)}}}


def _pareto(rows: list[dict]) -> list[dict]:
    """Cấu hình không bị cấu hình khác vừa rẻ hơn (fps, ổ đĩa) vừa tốt hơn."""
    out = []
    for r in rows:
        dominated = any(
            o["fps"] <= r["fps"] and o["storage_bytes"] <= r["storage_bytes"]
            and o["p_found"] >= r["p_found"]
            and (o["fps"] < r["fps"] or o["storage_bytes"] < r["storage_bytes"]
                 or o["p_found"] > r["p_found"])
            for o in rows)
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: (r["fps"], r["storage_bytes"]))


def optimize_from_dict(d: dict) -> dict:
    """Body API phẳng → optimize(). Khóa ngân sách: fps_budget, storage_bytes,
    people_per_hour, hours_per_day; tham số tìm kiếm: rho, near_best."""
    d = {k: v for k, v in d.items() if v is not None}
    budgets = Budgets(fps_max=d.get("fps_budget"), storage_bytes=d.pop("storage_bytes", None),
                      people_per_hour=d.get("people_per_hour", Budgets.people_per_hour),
                      hours_per_day=d.pop("hours_per_day", Budgets.hours_per_day))
    space = SearchSpace(rho=d.pop("rho", SearchSpace.rho),
                        near_best=d.pop("near_best", SearchSpace.near_best))
    geo = {k: v for k, v in d.items() if k in sp.CameraGeometry.__dataclass_fields__}
    tgt = {k: v for k, v in d.items() if k in sp.SamplingTargets.__dataclass_fields__}
    return optimize(sp.CameraGeometry(**geo), sp.SamplingTargets(**tgt), budgets=budgets,
                    space=space)
