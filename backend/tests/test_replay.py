"""Test 8 camera phát lại video: producer theo thời gian video + router /api/replay."""
import os
import queue
import sys
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))

from backend.api import cameras as cam  # noqa: E402
from backend.api.main import app  # noqa: E402
from backend.api.routers import replay  # noqa: E402


def _video(path, seconds=2, fps=20):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(seconds * fps):
        vw.write(np.full((120, 160, 3), i % 255, np.uint8))
    vw.release()
    return path


def _session(path, sample_fps=5.0, qsize=100, offset=10.0, base=1_700_000_000.0):
    return {"queue": queue.Queue(maxsize=qsize), "cancel": threading.Event(),
            "sample_fps": sample_fps, "max_frames": 10_000, "frame_count": 0,
            "queue_bytes": 0,
            "replay": {"path": str(path), "offset_sec": offset, "base_epoch": base,
                       "position_sec": 0.0, "finished": False}}


def test_replay_producer_samples_by_video_time(tmp_path):
    s = _session(_video(tmp_path / "v.mp4"))
    cam._producer(s)                                   # chuyển sang nhánh phát lại
    items = []
    while not s["queue"].empty():
        items.append(s["queue"].get())
    # 2s video × 5 fps lấy mẫu = 10 khung, cách nhau 0.2s thời gian video.
    assert len(items) == 10 and s["replay"]["finished"] and s["producer_done"]
    times = [it["video_time_sec"] for it in items]
    assert times[0] == 0.0 and np.allclose(np.diff(times), 0.2)
    first = datetime.fromisoformat(items[0]["captured_at"]).timestamp()
    assert first == pytest.approx(1_700_000_000.0 + 10.0)     # mốc chung + lệch camera
    assert s["replay"]["duration_sec"] == pytest.approx(2.0)


def test_replay_producer_blocks_instead_of_dropping(tmp_path):
    s = _session(_video(tmp_path / "v.mp4"), qsize=2)
    t = threading.Thread(target=cam._producer, args=(s,))
    t.start()
    got = []
    while t.is_alive() or not s["queue"].empty():
        try:
            got.append(s["queue"].get(timeout=0.5))
        except queue.Empty:
            pass
    t.join()
    assert len(got) == 10 and s.get("dropped", 0) == 0


def test_replay_producer_reports_missing_file(tmp_path):
    s = _session(tmp_path / "missing.mp4")
    cam._producer(s)
    assert "Không mở được" in s["error"] and s["producer_done"]


def test_replay_status_shape():
    s = {"replay": {"position_sec": 3.0, "duration_sec": 6.0, "finished": True,
                    "offset_sec": 5.0}, "consumer_done": False}
    st = cam._replay_status(s)
    assert st["progress"] == 0.5 and st["finished"] is False      # consumer chưa xong
    s["consumer_done"] = True
    assert cam._replay_status(s)["finished"] is True
    assert cam._replay_status({}) is None


# ---------------------------------------------------------------- router ----

def _post_start(client, ext=".mp4", data=b"video"):
    return client.post("/api/replay/start", data={"job_id": "job-1", "slot": "2"},
                       files={"file": (f"cam{ext}", data, "video/mp4")})


def test_start_requires_db():
    with patch.object(replay, "db_ping", return_value=False):
        assert _post_start(TestClient(app)).status_code == 503


def test_start_requires_generated_images():
    def no_job(job_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    with patch.object(replay, "db_ping", return_value=True), \
            patch.object(replay, "_require_generated", side_effect=no_job):
        assert _post_start(TestClient(app)).status_code == 404


def test_start_rejects_bad_extension():
    with patch.object(replay, "db_ping", return_value=True), \
            patch.object(replay, "_require_generated", return_value={25: "/x.png"}):
        assert _post_start(TestClient(app), ext=".txt").status_code == 400


def test_start_saves_file_and_starts_replay_session(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "REPLAY_DIR", str(tmp_path))
    with patch.object(replay, "db_ping", return_value=True), \
            patch.object(replay, "_require_generated", return_value={25: "/x.png"}), \
            patch.object(replay, "_camera_for_slot", return_value="cam-2"), \
            patch.object(replay.cam_service, "start_capture",
                         return_value={"source_id": "s2", "run_id": "r2"}) as start:
        r = _post_start(TestClient(app))
    assert r.status_code == 202, r.text
    kw = start.call_args.kwargs
    assert kw["camera_id"] == "cam-2" and kw["config_source"] == "replay"
    assert os.path.isfile(kw["replay_path"]) and open(kw["replay_path"], "rb").read() == b"video"
    assert r.json()["source_id"] == "s2"


def test_match_dedupes_timeline_per_appearance():
    rec = SimpleNamespace
    m = lambda url, score, t: SimpleNamespace(
        face_image_url=url, score=score, timestamp_sec=t,
        model_dump=lambda: {"face_image_url": url, "score": score, "timestamp_sec": t})
    matches = [m("a", 0.84, 1.6), m("b", 0.80, 1.6), m("c", 0.10, 4.0)]
    records = [{"url": u, "crop_id": f"crop-{u}"} for u in ("a", "b", "c")]
    with patch.object(replay, "db_ping", return_value=True), \
            patch.object(replay, "_require_generated", return_value={25: "/x.png"}), \
            patch("backend.api.dependencies.get_video_embedder", return_value=object()), \
            patch("backend.api.routers.video_verify._embed_references",
                  return_value=({25: np.zeros(512)}, {25: "/x.png"})), \
            patch("backend.api.routers.video_verify._load_source_records",
                  return_value=(records, [r["crop_id"] for r in records])), \
            patch("backend.api.routers.video_verify._match_records", return_value=matches), \
            patch("backend.api.routers.video_verify._save_search_run", return_value="run-1"), \
            patch.object(replay.cam_service, "capture_status", side_effect=LookupError):
        r = TestClient(app).post("/api/replay/match", json={
            "job_id": "job-1", "sources": [{"slot": 4, "source_id": "s4", "start_offset_sec": 30}]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["found_on_cameras"] == [4]
    assert len(j["timeline"]) == 1 and j["timeline"][0]["face_image_url"] == "a"
    assert j["timeline"][0]["camera_time_sec"] == pytest.approx(31.6)
    assert j["cameras"][0]["accepted"] == 2 and j["cameras"][0]["search_run_id"] == "run-1"
