"""Test thuật toán chọn fps (sampling_policy) và quên dần (forgetting) — không cần DB."""
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("."))

from backend.api import forgetting as fg  # noqa: E402
from backend.api import sampling_policy as sp  # noqa: E402
from backend.api.camera_tracks import bbox_iou  # noqa: E402

NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)


# ------------------------------------------------------------ sampling -------

def test_field_width_and_face_pixels():
    g = sp.CameraGeometry(hfov_deg=90, image_width_px=1920)
    assert sp.field_width_m(4.0, 90) == pytest.approx(8.0)
    assert sp.face_pixels(4.0, g) == pytest.approx(0.16 * 1920 / 8.0)
    d = sp.max_recognition_distance_m(g, sp.SamplingTargets())
    assert sp.face_pixels(d, g) == pytest.approx(48.0)


def test_hfov_for_distance_hits_min_face_px():
    g = sp.CameraGeometry(image_width_px=1920)
    t = sp.SamplingTargets()
    h = sp.hfov_for_distance(5.0, g, t)
    assert sp.face_pixels(5.0, sp.CameraGeometry(hfov_deg=h)) == pytest.approx(48.0)


def test_binomial_samples_needed_is_minimal():
    n = sp.samples_needed(3, 0.35, 0.95)
    assert sp.binomial_tail(n, 3, 0.35) >= 0.95 > sp.binomial_tail(n - 1, 3, 0.35)
    assert sp.samples_needed(1, 1.0, 0.99) == 1
    with pytest.raises(ValueError):
        sp.samples_needed(3, 0.0, 0.95)


def test_tracking_fps_keeps_iou_at_threshold():
    """Ở đúng fps_tracking, hai bbox mặt liên tiếp có IoU == ngưỡng tracker."""
    g = sp.CameraGeometry(walk_speed_mps=1.4)
    t = sp.SamplingTargets()
    fps = sp.tracking_fps(g, t)
    px_per_m = 100.0
    w = t.face_width_m * px_per_m
    dx = g.walk_speed_mps / fps * px_per_m
    assert bbox_iou([0, 0, w, w], [dx, 0, dx + w, w]) == pytest.approx(t.iou_threshold)
    # Không phụ thuộc khoảng cách camera.
    assert sp.tracking_fps(sp.CameraGeometry(distance_m=2), t) == pytest.approx(fps)


def test_tracking_fps_approaching_scale_iou():
    g = sp.CameraGeometry(motion="approaching", near_m=2.0, walk_speed_mps=1.0)
    t = sp.SamplingTargets()
    fps = sp.tracking_fps(g, t)
    s = (g.near_m + g.walk_speed_mps / fps) / g.near_m     # tỷ lệ phóng to bbox
    assert 1.0 / s ** 2 == pytest.approx(t.iou_threshold)


def test_plan_crossing_too_far_is_unusable_and_cheap():
    p = sp.plan_sampling(sp.CameraGeometry(distance_m=8.0))
    assert p["usable"] is False and p["binding"] == "unusable"
    assert p["fps"] == sp.SamplingTargets().fps_min
    assert any("ống kính" in w for w in p["warnings"])


def test_plan_crossing_close_is_tracking_bound_and_warns_fragmentation():
    p = sp.plan_sampling(sp.CameraGeometry(distance_m=3.0))
    assert p["usable"] is True
    assert p["fps_tracking"] > sp.SESSION_FPS_MAX
    assert p["fps"] == sp.SESSION_FPS_MAX and p["binding"] == "fps_max"
    assert p["tracklets_per_pass"] > 1
    assert any("tracklet riêng" in w for w in p["warnings"])


def test_plan_approaching_corridor_is_coverage_bound():
    g = sp.CameraGeometry(motion="approaching", image_width_px=3840, hfov_deg=60, far_m=15)
    p = sp.plan_sampling(g)
    assert p["binding"] == "coverage"
    assert p["fps"] == pytest.approx(p["fps_coverage"], abs=0.01)
    assert p["tracklets_per_pass"] == 1.0
    assert p["coverage_confidence"] >= 0.9


def test_plan_budget_caps_fps():
    g = sp.CameraGeometry(motion="approaching")
    p = sp.plan_sampling(g, sp.SamplingTargets(fps_budget=2.0))
    assert p["fps"] == 2.0 and p["binding"] == "budget"


def test_low_fps_extends_tracker_timeout():
    g = sp.CameraGeometry(motion="approaching", image_width_px=3840, hfov_deg=40, far_m=25,
                          walk_speed_mps=1.0)
    p = sp.plan_sampling(g, sp.SamplingTargets(fps_budget=0.6))
    assert p["tracker_timeout_sec"] == pytest.approx(3 / 0.6, abs=0.01)


def test_plan_from_dict_rejects_unknown_keys():
    assert sp.plan_from_dict({"distance_m": 3.0})["usable"] is True
    with pytest.raises(ValueError):
        sp.plan_from_dict({"foo": 1})


def test_people_per_hour_gives_daily_volume():
    p = sp.plan_sampling(sp.CameraGeometry(motion="approaching"),
                         sp.SamplingTargets(people_per_hour=100))
    assert p["tracklets_per_day"] == 2400
    assert p["bytes_per_day"] == round(2400 * p["bytes_per_pass"])


# ---------------------------------------------------------- forgetting -------

def test_params_validate():
    with pytest.raises(ValueError):
        fg.ForgettingParams(thin_threshold=0.1, forget_threshold=0.5).validate()
    with pytest.raises(ValueError):
        fg.ForgettingParams(floor_days=40, cap_days=30).validate()
    fg.ForgettingParams().validate()


def test_params_from_env(monkeypatch):
    monkeypatch.setenv("FORGET_CAP_DAYS", "14")
    monkeypatch.setenv("FORGET_FLOOR_DAYS", "3")
    p = fg.ForgettingParams.from_env()
    assert p.cap_days == 14 and p.floor_days == 3


def test_stability_grows_with_sightings_and_quality():
    p = fg.ForgettingParams()
    one = fg.stability_days(1, 0.5, None, p)
    assert one == pytest.approx(p.base_stability_days)
    assert fg.stability_days(5, 0.5, None, p) > one
    assert fg.stability_days(1, 0.9, None, p) > one > fg.stability_days(1, 0.1, None, p)


def test_pressure_shrinks_stability():
    p = fg.ForgettingParams()
    assert fg.pressure_factor(0.5, p) == 1.0
    assert fg.pressure_factor(p.stop_ratio, p) == pytest.approx(p.pressure_min_factor)
    mid = (p.warn_ratio + p.stop_ratio) / 2
    assert fg.pressure_factor(mid, p) == pytest.approx((1 + p.pressure_min_factor) / 2)


def test_schedule_respects_floor_and_cap():
    p = fg.ForgettingParams()
    short = fg.schedule(0.5, p)                      # độ bền rất thấp → chạm sàn
    assert short["forget_after_days"] == p.floor_days
    assert short["thin_after_days"] < p.floor_days
    long = fg.schedule(100.0, p)                     # độ bền rất cao → chạm trần
    assert long["forget_after_days"] == p.cap_days
    s = 5.0                                          # giữa sàn và trần → công thức thuần
    assert fg.schedule(s, p)["forget_after_days"] == pytest.approx(
        s * math.log(1 / p.forget_threshold), abs=0.01)


def test_schedule_cap_shrinks_under_pressure_but_not_below_floor():
    p = fg.ForgettingParams()
    assert fg.schedule(100.0, p, usage_ratio=1.0)["forget_after_days"] == pytest.approx(
        max(p.floor_days, p.cap_days * p.pressure_min_factor))


def test_decide_stages_over_time():
    p = fg.ForgettingParams()
    s = fg.stability_days(1, 0.5, None, p)
    sch = fg.schedule(s, p)
    def at(days):
        return fg.decide(fg.MemoryItem("t1", NOW - timedelta(days=days)), NOW, p)["stage"]
    assert at(0.5) == fg.KEEP
    assert at((sch["thin_after_days"] + sch["forget_after_days"]) / 2) == fg.THIN
    assert at(sch["forget_after_days"] + 0.1) == fg.FORGET


def test_decide_hold_overrides_forgetting_until_hold_expires():
    p = fg.ForgettingParams()
    old = NOW - timedelta(days=60)
    held = fg.MemoryItem("t1", old, last_match=NOW - timedelta(days=10))
    assert fg.decide(held, NOW, p)["stage"] == fg.HOLD
    stale = fg.MemoryItem("t1", old, last_match=NOW - timedelta(days=p.hold_days + 1))
    assert fg.decide(stale, NOW, p)["stage"] == fg.FORGET


def _conn_with(items_rows, crop_rows):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.side_effect = [items_rows, crop_rows]
    return conn, cur


def test_plan_thin_keeps_best_crop_and_forget_takes_all():
    p = fg.ForgettingParams()
    items = [
        ("thin-me", NOW - timedelta(days=4), 1, 0.5, None),     # 4 ngày → lược
        ("forget-me", NOW - timedelta(days=40), 1, 0.5, None),  # quá trần → quên
        ("keep-me", NOW - timedelta(hours=6), 1, 0.5, None),
        ("held", NOW - timedelta(days=40), 1, 0.5, NOW - timedelta(days=1)),
    ]
    crops = [  # SQL đã sắp theo chất lượng giảm dần trong từng tracklet
        ("c-best", "k/best.jpg", 100, "thin-me", 0.9),
        ("c-2", "k/2.jpg", 100, "thin-me", 0.5),
        ("c-3", "k/3.jpg", 100, "thin-me", 0.3),
        ("f-1", "k/f1.jpg", 50, "forget-me", 0.8),
        ("f-2", "k/f2.jpg", 50, "forget-me", 0.2),
    ]
    conn, _ = _conn_with(items, crops)
    plan = fg.plan_forgetting(conn, now=NOW, params=p)
    assert plan["counts"] == {"keep": 1, "thin": 1, "forget": 1, "hold": 1}
    assert [c["crop_id"] for c in plan["crops_thin"]] == ["c-2", "c-3"]
    assert [c["crop_id"] for c in plan["crops_forget"]] == ["f-1", "f-2"]
    assert plan["tracklets_forget"] == ["forget-me"]
    assert plan["bytes_to_free"] == 300


def test_plan_without_actions_skips_crop_query():
    conn, cur = _conn_with([("k", NOW, 1, 0.5, None)], [])
    plan = fg.plan_forgetting(conn, now=NOW, params=fg.ForgettingParams())
    assert plan["counts"]["keep"] == 1 and cur.execute.call_count == 1


def test_apply_dry_run_writes_nothing():
    plan = {"crops_thin": [{"crop_id": "a"}], "crops_forget": [], "tracklets_forget": ["t"],
            "bytes_to_free": 10}
    conn = MagicMock()
    out = fg.apply_forgetting(conn, plan, dry_run=True)
    assert out == {"dry_run": True, "thinned": 1, "forgotten": 0,
                   "tracklets_expired": 1, "bytes_to_free": 10}
    conn.cursor.assert_not_called()


def test_apply_real_requires_enable_flag(monkeypatch):
    plan = {"crops_thin": [], "crops_forget": [{"crop_id": "a"}], "tracklets_forget": ["t"],
            "bytes_to_free": 0}
    monkeypatch.delenv("FORGETTING_ENABLED", raising=False)
    with pytest.raises(PermissionError):
        fg.apply_forgetting(MagicMock(), plan, dry_run=False)
    monkeypatch.setenv("FORGETTING_ENABLED", "true")
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.rowcount = 1
    with patch("backend.api.retention.tombstone_crops", return_value=1) as tomb:
        out = fg.apply_forgetting(conn, plan, dry_run=False)
    tomb.assert_called_once_with(conn, plan["crops_forget"], "forget_full")
    assert out["forgotten"] == 1 and out["tracklets_expired"] == 1
