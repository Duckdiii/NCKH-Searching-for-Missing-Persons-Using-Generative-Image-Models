"""Ghi quan sát theo lô: tải crop song song, 1 executemany/bảng, video crop-only.

Không cần DB/Supabase thật: conn và storage giả ghi lại lời gọi.
"""

import os
import sys
import threading
import time
from contextlib import contextmanager

import httpx
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))

from backend.api import ingest  # noqa: E402
from backend.api import repositories as repo  # noqa: E402
from backend.api.gallery import emit_outbox_many  # noqa: E402
from database.migrate import discover_migrations  # noqa: E402

UPLOAD_DELAY = 0.05


class FakeStorage:
    def __init__(self, fail_key_part=None):
        self.objects = {}
        self.threads = set()
        self.deleted = []
        self.signed_calls = 0
        self.fail_key_part = fail_key_part
        self._lock = threading.Lock()

    def put_bytes(self, data, key, *, mime_type):
        time.sleep(UPLOAD_DELAY)  # giả lập RTT mạng
        if self.fail_key_part and self.fail_key_part in key:
            raise IOError("upload hỏng")
        with self._lock:
            self.objects[key] = data
            self.threads.add(threading.current_thread().name)
        return key

    def delete_many_quiet(self, keys):
        with self._lock:
            self.deleted.extend(keys)
            for key in keys:
                self.objects.pop(key, None)

    def get_access_urls(self, keys):
        self.signed_calls += 1
        return {k: f"https://signed/{k}" for k in keys}


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.calls.append(("execute", sql, params))

    def executemany(self, sql, rows):
        rows = list(rows)
        if self.conn.fail_on and self.conn.fail_on in sql:
            raise RuntimeError("db hỏng")
        self.conn.calls.append(("executemany", sql, rows))


class FakeConn:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on
        self.savepoints = 0

    def cursor(self):
        return FakeCursor(self)

    @contextmanager
    def transaction(self):
        self.savepoints += 1
        yield

    def table_rows(self, table):
        return [rows for kind, sql, rows in self.calls
                if kind == "executemany" and f"face_media.{table}" in sql]


def _face(x, det=0.9):
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    return {"bbox_orig": [x, 10.0, x + 40.0, 60.0], "det_score": det,
            "kps": [[x + 10, 20], [x + 30, 20], [x + 20, 30], [x + 12, 45], [x + 28, 45]],
            "embedding": emb}


def _batch(storage, **kw):
    return ingest.ObservationBatch(
        storage, source_id="src", purpose="search", detector_name="insightface",
        detector_version="buffalo_l", run_id="run", **kw)


FRAME = np.full((120, 200, 3), 128, dtype=np.uint8)


def test_batch_one_executemany_per_table_and_parallel_uploads():
    storage, conn = FakeStorage(), FakeConn()
    batch = _batch(storage, workers=8)
    for i in range(4):
        batch.add_frame(FRAME, frame_index=i, offset_ms=i * 333, captured_at=None,
                        faces=[_face(10), _face(100)])
    start = time.perf_counter()
    counts = batch.flush(conn)
    elapsed = time.perf_counter() - start

    assert counts == {"assets": 12, "frames": 4, "detections": 8, "crops": 8, "embeddings": 8}
    assert not [c for c in conn.calls if c[0] == "execute"], "không còn INSERT từng hàng"
    for table in ("assets", "frames", "face_detections", "face_crops",
                  "face_embeddings", "outbox_events"):
        assert len(conn.table_rows(table)) == 1, table
    assert len(conn.table_rows("outbox_events")[0]) == 8
    assert conn.savepoints == 1  # outbox trong savepoint riêng
    # 12 upload × 50ms tuần tự = 0.6s; song song phải nhanh hơn hẳn.
    assert len(storage.threads) > 1
    assert elapsed < 12 * UPLOAD_DELAY / 2
    # Thứ tự FK: assets trước frames trước detections trước crops.
    order = [sql for kind, sql, _ in conn.calls if kind == "executemany"]
    tables = [t for t in ("assets", "frames", "face_detections", "face_crops")
              for i, sql in enumerate(order) if f"face_media.{t}" in sql]
    assert tables == ["assets", "frames", "face_detections", "face_crops"]


def test_crop_only_skips_frame_pixels_and_empty_frames():
    storage, conn = FakeStorage(), FakeConn()
    batch = _batch(storage, store_frames=False)
    empty = batch.add_frame(FRAME, frame_index=0, offset_ms=0, captured_at=None, faces=[])
    hit = batch.add_frame(FRAME, frame_index=1, offset_ms=333, captured_at=None,
                          faces=[_face(10), _face(100, det=0.1)])
    batch.flush(conn)

    assert empty == {"frame_id": None, "detections": []}
    assert len(hit["detections"]) == 1  # det 0.1 dưới ngưỡng
    assert all(k.startswith("search/crops/") for k in storage.objects)
    assert len(storage.objects) == 1
    (frames,) = conn.table_rows("frames")
    assert len(frames) == 1
    frame = frames[0]
    assert frame[3] is None and frame[7:9] == (200, 120)  # metadata-only + kích thước nguồn


def test_crop_only_rejected_for_reference():
    with pytest.raises(ValueError):
        ingest.ObservationBatch(FakeStorage(), source_id="s", purpose="reference",
                                detector_name="insightface", detector_version="v",
                                run_id="r", store_frames=False)


def test_upload_failure_cleans_files_and_writes_nothing():
    storage, conn = FakeStorage(fail_key_part="search/crops/"), FakeConn()
    batch = _batch(storage)
    batch.add_frame(FRAME, frame_index=0, offset_ms=0, captured_at=None,
                    faces=[_face(10), _face(100)])
    with pytest.raises(IOError):
        batch.flush(conn)
    assert conn.calls == []
    assert set(storage.deleted) == set(batch.keys)
    assert storage.objects == {}


def test_db_failure_cleans_uploaded_files():
    storage, conn = FakeStorage(), FakeConn(fail_on="face_media.face_crops")
    batch = _batch(storage)
    batch.add_frame(FRAME, frame_index=0, offset_ms=0, captured_at=None, faces=[_face(10)])
    with pytest.raises(RuntimeError):
        batch.flush(conn)
    assert storage.objects == {} and len(storage.deleted) == 2


def test_invalid_bbox_and_vector_skipped_not_fatal():
    storage, conn = FakeStorage(), FakeConn()
    batch = _batch(storage)
    bad_vec = _face(10)
    bad_vec["embedding"] = np.ones(512, dtype=np.float32)  # không chuẩn hóa
    off_frame = _face(10)
    off_frame["bbox_orig"] = [500.0, 500.0, 600.0, 600.0]  # bị kẹp về 0 px
    out = batch.add_frame(FRAME, frame_index=0, offset_ms=0, captured_at=None,
                          faces=[bad_vec, off_frame])
    counts = batch.flush(conn)
    assert len(out["detections"]) == 1
    assert counts["embeddings"] == 0 and counts["crops"] == 1


def test_persist_observation_frame_keeps_contract():
    storage, conn = FakeStorage(), FakeConn()
    out = ingest.persist_observation_frame(
        conn, storage, source_id="src", purpose="search", frame_bgr=FRAME,
        frame_index=0, offset_ms=0, captured_at=None, faces=[_face(10)],
        detector_name="insightface", detector_version="buffalo_l", run_id="run")
    assert out["frame_id"] and len(out["detections"]) == 1
    det = out["detections"][0]
    assert {"detection_id", "crop_id", "crop_key", "bbox", "det_score", "embedding"} <= set(det)
    assert any(k.startswith("search/frames/") for k in storage.objects)  # ảnh vẫn lưu frame


def test_ingest_sampled_frames_single_transaction_and_one_sign(monkeypatch):
    storage, conns = FakeStorage(), []

    class Pool:
        @contextmanager
        def connection(self):
            conn = FakeConn()
            conns.append(conn)
            yield conn

    monkeypatch.setattr("backend.api.database.get_pool", lambda: Pool())
    monkeypatch.setattr("backend.api.storage.get_storage", lambda: storage)
    faces_by_frame = [[_face(10)], [], [_face(10), _face(100)]]
    calls = iter(faces_by_frame)
    monkeypatch.setattr(ingest, "detect_faces_in_frame",
                        lambda emb, frame, preprocess_on: (next(calls), ["night"], frame))
    frames = [{"frame_bgr": FRAME, "frame_index": i, "timestamp_sec": i / 3}
              for i in range(3)]
    out = ingest.ingest_sampled_frames(source_id="src", frames=frames, embedder=object(),
                                       preprocess_on=True, run_id="run", store_frames=False)
    assert out["faces_found"] == 3 and out["conditions"] == {"night": 3}
    assert len(conns) == 1  # một transaction cho cả video
    assert storage.signed_calls == 1
    assert all(r["url"].startswith("https://signed/") for r in out["crops"])
    assert len(conns[0].table_rows("frames")[0]) == 2  # frame không mặt bị bỏ


def test_create_source_crop_only_video_rules():
    conn = FakeConn()
    repo.create_source(conn, purpose="search", kind="video", source_id="s1",
                       storage_policy="crop_only")
    assert conn.calls and conn.calls[0][2][3] is None
    with pytest.raises(ValueError):
        repo.create_source(FakeConn(), purpose="search", kind="video")
    with pytest.raises(ValueError):
        repo.create_source(FakeConn(), purpose="search", kind="image",
                           storage_policy="crop_only")


def test_outbox_failure_contained_in_savepoint():
    conn = FakeConn(fail_on="outbox_events")
    ok = emit_outbox_many(conn, [{"entity": "embedding", "entity_id": "e", "op": "upsert"}])
    assert ok is False and conn.savepoints == 1


def test_supabase_signs_batch_in_one_request(monkeypatch):
    from backend.api.supabase_storage import SupabaseMediaStorage

    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        import json
        paths = json.loads(request.content)["paths"]
        return httpx.Response(200, json=[
            {"path": p, "error": None, "signedURL": f"/object/sign/face-media/{p}?token=t"}
            for p in paths])

    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_test")
    store = SupabaseMediaStorage(client=httpx.Client(transport=httpx.MockTransport(handler)))
    keys = [f"search/crops/{i:032x}.jpg" for i in range(30)]
    urls = store.get_access_urls(keys)
    assert len(seen) == 1 and seen[0] == ("POST", "/storage/v1/object/sign/face-media")
    assert urls[keys[0]].startswith("https://x.supabase.co/storage/v1/object/sign/")


def test_migration_009_discovered():
    assert "009" in {version for version, _ in discover_migrations()}
