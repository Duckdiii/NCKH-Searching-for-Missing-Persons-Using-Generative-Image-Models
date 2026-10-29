"""Test POST /api/tools/rank-faces (embedder giả, không cần GPU)."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))

from backend.api.main import app  # noqa: E402

TEMPLATE = np.array([[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]])


def _png(seed, blur=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((240, 320, 3)) * 255).astype(np.uint8), (3, 3), 0)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return cv2.imencode(".png", img)[1].tobytes()


class FakeEmbedder:
    def detect_faces(self, path):
        img = cv2.imread(path)
        if img is None or float(img.std()) < 1:          # ảnh phẳng = không có mặt
            return []
        emb = np.ones(512) * 1.0
        return [SimpleNamespace(bbox=np.array([100, 60, 212, 172], float),
                                kps=TEMPLATE + np.array([100, 60]), det_score=0.9,
                                embedding=emb * 22 / np.linalg.norm(emb),
                                normed_embedding=emb / np.linalg.norm(emb))]


def test_rank_faces_orders_sharp_first_and_reports_no_face():
    flat = cv2.imencode(".png", np.full((120, 120, 3), 128, np.uint8))[1].tobytes()
    files = [("files", ("mo.png", _png(1, blur=31), "image/png")),
             ("files", ("net.png", _png(1), "image/png")),
             ("files", ("trong.png", flat, "image/png")),
             ("files", ("ghi-chu.txt", b"x", "text/plain"))]
    with patch("backend.api.dependencies.get_embedder", return_value=FakeEmbedder()):
        r = TestClient(app).post("/api/tools/rank-faces", files=files)
    assert r.status_code == 200, r.text
    j = r.json()
    assert [x["file"] for x in j["ranked"]] == ["net.png", "mo.png"]
    assert j["ranked"][0]["thumbnail"].startswith("data:image/jpeg;base64,")
    reasons = {n["file"]: n["reason"] for n in j["no_face"]}
    assert reasons == {"trong.png": "không thấy khuôn mặt", "ghi-chu.txt": "định dạng không hỗ trợ"}


def test_rank_faces_rejects_too_many_files():
    files = [("files", (f"{i}.png", b"x", "image/png")) for i in range(31)]
    r = TestClient(app).post("/api/tools/rank-faces", files=files)
    assert r.status_code == 400
