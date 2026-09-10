import io
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.session_store import sessions

client = TestClient(app)


def create_dummy_image_bytes(width=256, height=256, color=(200, 150, 100)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockFace:
    def __init__(self, bbox, det_score=0.95):
        self.bbox = bbox
        self.det_score = det_score
        # 5 keypoints: left eye, right eye, nose, left mouth, right mouth
        self.kps = np.array([
            [80.0, 90.0],
            [140.0, 90.0],
            [110.0, 120.0],
            [90.0, 150.0],
            [130.0, 150.0]
        ], dtype=np.float32)


@pytest.fixture(autouse=True)
def clean_sessions():
    sessions.clear()
    yield
    sessions.clear()


def test_upload_image_one_face():
    dummy_bytes = create_dummy_image_bytes()
    mock_faces = [MockFace([50.0, 50.0, 180.0, 180.0], det_score=0.98)]

    with patch("backend.api.routers.session.get_embedder") as mock_get_embedder:
        mock_embedder = MagicMock()
        mock_embedder.detect_faces.return_value = mock_faces
        mock_get_embedder.return_value = mock_embedder

        resp = client.post(
            "/api/sessions",
            files={"file": ("test.png", dummy_bytes, "image/png")}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert len(data["faces"]) == 1
    assert data["faces"][0]["index"] == 0
    assert data["faces"][0]["det_score"] == pytest.approx(0.98, 0.01)


def test_upload_image_multiple_faces():
    dummy_bytes = create_dummy_image_bytes()
    mock_faces = [
        MockFace([20.0, 20.0, 80.0, 80.0], det_score=0.91),
        MockFace([120.0, 120.0, 200.0, 200.0], det_score=0.96)
    ]

    with patch("backend.api.routers.session.get_embedder") as mock_get_embedder:
        mock_embedder = MagicMock()
        mock_embedder.detect_faces.return_value = mock_faces
        mock_get_embedder.return_value = mock_embedder

        resp = client.post(
            "/api/sessions",
            files={"file": ("multi.png", dummy_bytes, "image/png")}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["faces"]) == 2


def test_upload_image_no_face_fails():
    dummy_bytes = create_dummy_image_bytes()

    with patch("backend.api.routers.session.get_embedder") as mock_get_embedder:
        mock_embedder = MagicMock()
        mock_embedder.detect_faces.return_value = []
        mock_get_embedder.return_value = mock_embedder

        resp = client.post(
            "/api/sessions",
            files={"file": ("noface.png", dummy_bytes, "image/png")}
        )

    assert resp.status_code == 400
    assert "Không phát hiện khuôn mặt nào" in resp.json()["detail"]


def test_select_face_success_and_invalid_index():
    dummy_bytes = create_dummy_image_bytes()
    mock_faces = [MockFace([30.0, 30.0, 150.0, 150.0], det_score=0.92)]

    with patch("backend.api.routers.session.get_embedder") as mock_get_embedder:
        mock_embedder = MagicMock()
        mock_embedder.detect_faces.return_value = mock_faces
        mock_get_embedder.return_value = mock_embedder

        resp_upload = client.post(
            "/api/sessions",
            files={"file": ("test.png", dummy_bytes, "image/png")}
        )
    session_id = resp_upload.json()["session_id"]

    # Select invalid face index
    resp_invalid = client.post(
        f"/api/sessions/{session_id}/select-face",
        json={"selected_idx": 5}
    )
    assert resp_invalid.status_code == 400

    # Select valid face index
    resp_valid = client.post(
        f"/api/sessions/{session_id}/select-face",
        json={"selected_idx": 0}
    )
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert "cropped_preview_url" in data
    assert isinstance(data["warnings"], list)


def test_resolve_age_manual_and_validation():
    dummy_bytes = create_dummy_image_bytes()
    mock_faces = [MockFace([30.0, 30.0, 150.0, 150.0], det_score=0.92)]

    with patch("backend.api.routers.session.get_embedder") as mock_get_embedder:
        mock_embedder = MagicMock()
        mock_embedder.detect_faces.return_value = mock_faces
        mock_get_embedder.return_value = mock_embedder

        resp_upload = client.post(
            "/api/sessions",
            files={"file": ("test.png", dummy_bytes, "image/png")}
        )
    session_id = resp_upload.json()["session_id"]
    client.post(f"/api/sessions/{session_id}/select-face", json={"selected_idx": 0})

    # Mode manual missing manual_age
    resp_missing = client.post(
        f"/api/sessions/{session_id}/resolve-age",
        json={"mode": "manual", "manual_age": None, "gender_word": "man"}
    )
    assert resp_missing.status_code == 422

    # Mode manual valid
    resp_manual = client.post(
        f"/api/sessions/{session_id}/resolve-age",
        json={"mode": "manual", "manual_age": 15, "gender_word": "man"}
    )
    assert resp_manual.status_code == 200
    assert resp_manual.json()["initial_age"] == 15
