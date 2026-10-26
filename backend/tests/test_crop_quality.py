"""Test thuật toán chọn crop (crop_quality) và tích hợp vào CameraTracker."""
import math
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))

from backend.api import camera_tracks as ct  # noqa: E402
from backend.api import crop_quality as cq  # noqa: E402

# 5 điểm template ArcFace 112×112 (mắt T/P, mũi, miệng T/P) — mặt chính diện.
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]])


def _textured(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _face_at(x0, y0, size=112.0, kps=TEMPLATE):
    k = kps * (size / 112.0) + np.array([x0, y0])
    return [x0, y0, x0 + size, y0 + size], k


# ------------------------------------------------------------ thành phần -----

def test_blur_index_increases_with_blur():
    img = cv2.cvtColor(_textured(64, 64), cv2.COLOR_BGR2GRAY)
    vals = [cq.blur_crete(cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img)
            for k in (1, 5, 9, 15)]
    assert vals == sorted(vals) and vals[-1] - vals[0] > 0.2
    assert cq.blur_crete(np.full((64, 64), 128, np.uint8)) == 1.0


def test_pose_template_is_frontal():
    p = cq.pose_from_kps(TEMPLATE)
    assert p["plausible"] and abs(p["yaw_ratio"]) < 0.01
    assert p["pitch_ratio"] == pytest.approx(cq.PITCH_CENTER, abs=0.01)
    assert cq.pose_score(p) > 0.99


def test_pose_roll_does_not_look_like_yaw():
    c, s = math.cos(0.5), math.sin(0.5)
    rolled = (TEMPLATE - 56) @ np.array([[c, -s], [s, c]]).T + 56
    p = cq.pose_from_kps(rolled)
    assert abs(p["roll_deg"]) == pytest.approx(math.degrees(0.5), abs=1.0)
    assert abs(p["yaw_ratio"]) < 0.01
    assert cq.pose_score(p) > 0.85            # roll chỉ phạt nhẹ (căn chỉnh sửa được)


def test_turned_face_scores_lower_and_profile_is_gated():
    turned = TEMPLATE.copy()
    turned[2, 0] += 8                           # mũi lệch sang phải
    assert cq.pose_score(cq.pose_from_kps(turned)) < 0.5
    profile = TEMPLATE.copy()
    profile[2, 0] += 20
    frame = _textured()
    bbox, k = _face_at(100, 60, kps=profile)
    assert cq.assess_face(frame, bbox, kps=k).gate == "profile"


def test_implausible_landmarks_are_gated():
    bad = TEMPLATE.copy()
    bad[3:, 1] = 40                             # miệng nằm trên mắt
    frame = _textured()
    bbox, k = _face_at(100, 60, kps=bad)
    assert cq.assess_face(frame, bbox, kps=k).gate == "landmark_implausible"


def test_truncated_face_is_gated():
    frame = _textured()
    bbox, k = _face_at(260, 60)                 # 112px mặt, khung rộng 320 → bị cắt 46%
    r = cq.assess_face(frame, bbox, kps=k)
    assert r.gate == "truncated" and r.total == 0.0


def test_exposure_prefers_mid_tones():
    mid = cv2.cvtColor(_textured(64, 64), cv2.COLOR_BGR2GRAY) // 2 + 60
    dark = (mid * 0.15).astype(np.uint8)
    assert cq.exposure_score(mid) > 3 * cq.exposure_score(dark)


def test_combine_is_geometric_and_renormalizes():
    good = {"resolution": 1.0, "sharpness": 1.0, "pose": 1.0}
    assert cq.combine(good) == pytest.approx(1.0)
    one_bad = {**good, "sharpness": 0.0}
    assert cq.combine(one_bad) < 0.45           # cộng tuyến tính sẽ còn ~0.67
    assert cq.combine({"pose": 0.5}) == pytest.approx(0.5)


# -------------------------------------------------------------- assess -------

def test_sharper_and_bigger_faces_score_higher():
    frame = _textured()
    bbox, k = _face_at(100, 60)
    sharp = cq.assess_face(frame, bbox, kps=k, det_score=0.9, embedding_norm=22)
    blurred = cq.assess_face(cv2.GaussianBlur(frame, (11, 11), 0), bbox, kps=k,
                             det_score=0.9, embedding_norm=22)
    assert sharp.total > blurred.total
    small_bbox, small_k = _face_at(100, 60, size=40)
    small = cq.assess_face(frame, small_bbox, kps=small_k, det_score=0.9, embedding_norm=22)
    assert sharp.components["resolution"] > small.components["resolution"]


def test_low_embedding_norm_lowers_quality():
    frame = _textured()
    bbox, k = _face_at(100, 60)
    hi = cq.assess_face(frame, bbox, kps=k, embedding_norm=cq.NORM_GOOD)
    lo = cq.assess_face(frame, bbox, kps=k, embedding_norm=cq.NORM_LOW)
    assert hi.total > lo.total
    assert "recognizability" not in cq.assess_face(frame, bbox, kps=k).components


# -------------------------------------------------- nhất quán danh tính -----

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_centroid_flags_other_person():
    rng = np.random.default_rng(1)
    base = _unit(rng.normal(size=512))
    cen = cq.IdentityCentroid()
    assert cen.consistency(base) == 1.0         # chưa đủ quan sát → trung tính
    for _ in range(4):
        cen.add(_unit(base + 0.3 * rng.normal(size=512) / math.sqrt(512)), 0.8)
    assert cen.consistency(base) == 1.0
    assert cen.consistency(_unit(rng.normal(size=512))) == cq.CONSIST_FLOOR
    cen.add(_unit(rng.normal(size=512)), 0.1)   # quan sát chất lượng thấp không vào tâm
    assert cen.count == 4


def test_rank_faces_puts_best_first():
    frame = _textured()
    bbox, k = _face_at(100, 60)
    items = [
        {"frame_bgr": cv2.GaussianBlur(frame, (13, 13), 0), "bbox": bbox, "kps": k},
        {"frame_bgr": frame, "bbox": bbox, "kps": k},
        {"frame_bgr": frame, "bbox": _face_at(260, 60)[0], "kps": _face_at(260, 60)[1]},
    ]
    ranked = cq.rank_faces(items)
    assert ranked[0].index == 1 and ranked[-1].gate == "truncated"


# ----------------------------------------------------- tích hợp tracker -----

def _face_dict(bbox, kps, emb=None, det=0.9, norm=22.0):
    return {"bbox_orig": bbox, "det_score": det, "embedding": emb, "kps": kps,
            "embedding_norm": norm}


def test_tracker_prefers_sharp_frame_and_skips_profile():
    tracker = ct.CameraTracker()
    frame = _textured()
    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    bbox, k = _face_at(100, 60)
    prof = TEMPLATE.copy()
    prof[2, 0] += 20
    _, kp = _face_at(100, 60, kps=prof)
    tracker.update(blurred, [_face_dict(bbox, k)], now=0.0)
    tracker.update(frame, [_face_dict(bbox, k)], now=0.1)
    tracker.update(frame, [_face_dict(bbox, kp)], now=0.2)     # nghiêng hẳn → bỏ
    tr = tracker.open_tracks()[0]
    assert tr.rejected_gate == 1
    best, worst = tr.candidates[0], tr.candidates[-1]          # crop nét đứng đầu
    assert best.quality_detail["blur_index"] < worst.quality_detail["blur_index"]
    assert best.score > worst.score


def test_finalize_drops_crop_of_other_person():
    rng = np.random.default_rng(2)
    me = _unit(rng.normal(size=512))
    other = _unit(rng.normal(size=512))
    tr = ct.TrackState(local_id="t1", track_id=1, last_bbox=[0, 0, 1, 1],
                       last_seen_monotonic=0, started_monotonic=0)
    for _ in range(4):
        tr.centroid.add(me, 0.8)
    crop = np.zeros((4, 4, 3), np.uint8)
    mk = lambda q, e: ct.Candidate(crop_bgr=crop, quality=q, blur=0, exposure=0, frontal=0,
                                   face_area=0, det_score=0.9, bbox=[0, 0, 1, 1],
                                   embedding=e, score=q)
    tr.candidates = [mk(0.95, other), mk(0.7, me), mk(0.6, me)]
    ct.finalize_candidates(tr)
    assert [c.quality for c in tr.candidates] == [0.7, 0.6]
    assert tr.rejected_identity == 1


def test_detect_faces_in_frame_scales_kps_and_reports_norm(monkeypatch):
    from backend.api import ingest
    raw = np.full(512, 0.05)
    face = SimpleNamespace(det_score=0.95, bbox=np.array([10, 10, 60, 60], float),
                           normed_embedding=raw / np.linalg.norm(raw), embedding=raw,
                           kps=np.array([[20, 30]] * 5, float))
    emb = SimpleNamespace(detect_faces=lambda img: [face])
    monkeypatch.setattr(ingest, "_downscale_for_detect", lambda f: (f, 0.5))
    faces, _, _ = ingest.detect_faces_in_frame(emb, np.zeros((10, 10, 3), np.uint8),
                                               preprocess_on=False)
    assert faces[0]["bbox_orig"][0] == pytest.approx(20.0)
    assert faces[0]["kps"][0][0] == pytest.approx(40.0)        # cùng hệ tọa độ với bbox
    assert faces[0]["embedding_norm"] == pytest.approx(float(np.linalg.norm(raw)))


def test_blockiness_detects_compression_and_penalizes_sharpness():
    frame = cv2.cvtColor(_textured(), cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 5])
    blocky = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    assert cq.blockiness(frame) < 1.5 < cq.blockiness(blocky)
    # Lưới khối căn theo tọa độ frame: ROI lệch 3px vẫn đo đúng khi truyền offset.
    assert cq.blockiness(blocky[3:, 3:], 3, 3) == pytest.approx(cq.blockiness(blocky), rel=0.1)
    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    bgr_blocky = cv2.cvtColor(blocky, cv2.COLOR_GRAY2BGR)
    bbox, k = _face_at(100, 60)
    clean = cq.assess_face(bgr, bbox, kps=k)
    comp = cq.assess_face(bgr_blocky, bbox, kps=k)
    assert comp.blockiness > clean.blockiness
    assert comp.components["sharpness"] < clean.components["sharpness"]


def test_noise_is_not_mistaken_for_sharpness():
    rng = np.random.default_rng(3)
    smooth = cv2.GaussianBlur(_textured(), (7, 7), 0)
    noisy = np.clip(smooth.astype(np.float64) + rng.normal(0, 12, smooth.shape), 0, 255).astype(np.uint8)
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Nhiễu σ=12 độc lập từng kênh → ảnh xám (trung bình có trọng số) còn ~σ 8.
    assert cq.noise_sigma(g(smooth)) < 2.0 < 5.0 < cq.noise_sigma(g(noisy)) < 12.0
    bbox, k = _face_at(100, 60)
    a = cq.assess_face(smooth, bbox, kps=k)
    b = cq.assess_face(noisy, bbox, kps=k)
    assert b.noise_sigma > a.noise_sigma
    assert b.components["sharpness"] < a.components["sharpness"]
