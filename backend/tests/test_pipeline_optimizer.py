"""Test bộ tối ưu fps × K × q_min (pipeline_optimizer) — không cần DB/GPU."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("."))

from backend.api import pipeline_optimizer as po  # noqa: E402
from backend.api import sampling_policy as sp  # noqa: E402

CORRIDOR = sp.CameraGeometry(motion="approaching", image_width_px=3840, hfov_deg=60, far_m=15)
QM = po.QualityModel()
FAST = po.SearchSpace(passes=800)


def _row(res, fps, k, q_min=0.0):
    return next(r for r in res["grid"] if r["fps"] == fps and r["k"] == k and r["q_min"] == q_min)


def test_more_frames_and_more_crops_never_hurt():
    res = po.optimize(CORRIDOR, qm=QM, space=FAST)
    ps = [_row(res, f, 3)["p_found"] for f in (2.0, 3.0, 5.0, 10.0)]
    assert ps == sorted(ps)
    pk = [_row(res, 5.0, k)["p_found"] for k in (1, 2, 3, 4, 5)]
    assert pk == sorted(pk)
    crops = [_row(res, 5.0, k)["crops_per_pass"] for k in (1, 2, 3, 4, 5)]
    assert crops == sorted(crops) and crops[0] <= 1.0


def test_extra_crops_have_diminishing_returns():
    res = po.optimize(CORRIDOR, qm=QM, space=FAST)
    p = [_row(res, 5.0, k)["p_found"] for k in (1, 2, 3)]
    assert p[1] - p[0] >= p[2] - p[1] >= 0


def test_fragmented_configs_ignore_k_and_store_more():
    res = po.optimize(sp.CameraGeometry(distance_m=3.0), qm=QM, space=FAST)
    frag = [r for r in res["grid"] if r["fragmented"]]
    assert frag and all(r["k"] is None for r in frag)
    assert max(r["tracklets_per_pass"] for r in frag) > 1


def test_q_min_trades_recall_for_storage():
    res = po.optimize(CORRIDOR, qm=QM, space=FAST)
    lo, hi = _row(res, 3.0, 3, 0.0), _row(res, 3.0, 3, 0.5)
    assert hi["crops_per_pass"] <= lo["crops_per_pass"]
    assert hi["p_found"] <= lo["p_found"]


def test_recommendation_respects_budgets_and_is_near_best():
    b = po.Budgets(fps_max=3.0)
    res = po.optimize(CORRIDOR, qm=QM, budgets=b, space=FAST)
    rec = res["recommended"]
    assert rec["fps"] <= 3.0 and rec["within_budget"]
    assert rec["p_found"] >= FAST.near_best * res["best_p_found"]
    tight = po.optimize(CORRIDOR, qm=QM, budgets=po.Budgets(storage_bytes=0.12e9), space=FAST)
    assert tight["recommended"]["storage_bytes"] <= 0.12e9


def test_pareto_is_non_dominated():
    res = po.optimize(CORRIDOR, qm=QM, space=FAST)
    par = res["pareto"]
    for r in par:
        assert not any(o["fps"] <= r["fps"] and o["storage_bytes"] <= r["storage_bytes"]
                       and o["p_found"] > r["p_found"] for o in par if o is not r)


def test_quality_model_loads_calibration(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"source": "test", "base_samples": [0.9], "p_broken": 0.0,
                                "p_curve_q": [0, 1], "p_curve_p": [0.0, 1.0],
                                "unknown_key": 1}), encoding="utf-8")
    qm = po.QualityModel.load(str(path))
    assert qm.source == "test" and qm.p_match(0.5) == pytest.approx(0.5)
    assert po.QualityModel.load(str(tmp_path / "missing.json")).source == "default-assumption"


def test_unusable_camera_warns():
    res = po.optimize(sp.CameraGeometry(distance_m=10.0), qm=QM, space=FAST)
    assert any("không nhận diện được" in w for w in res["warnings"])
