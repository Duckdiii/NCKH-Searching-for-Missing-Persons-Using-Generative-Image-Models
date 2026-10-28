"""CodeFormer phóng to ảnh → landmark phải nhân theo (nếu không, crop cắt nhầm góc ảnh)."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.utils import face_enhancement as fe  # noqa: E402
from src.utils.ffhq_align import align_to_ffhq  # noqa: E402


def test_kps_follow_codeformer_upscale():
    img = np.zeros((256, 256, 3), np.uint8)
    cv2.circle(img, (128, 140), 60, (200, 180, 160), -1)        # "khuôn mặt" ở giữa ảnh
    kps = np.array([[100, 120], [156, 120], [128, 150], [105, 180], [151, 180]], float)
    up4 = lambda rgb, **kw: cv2.resize(rgb, (rgb.shape[1] * 4, rgb.shape[0] * 4))
    with patch.object(fe, "run_codeformer", side_effect=up4):
        out, new_kps = fe.preprocess_face_image(img, kps=kps, bbox=(68, 80, 188, 200),
                                                padding_enabled=False, white_balance_enabled=False)
    assert out.shape[:2] == (1024, 1024)
    assert np.allclose(new_kps, kps * 4)
    # Crop sau align phải chứa "khuôn mặt" (vùng sáng) ở giữa, không phải góc tối.
    crop = align_to_ffhq(out, new_kps, output_size=512)
    assert crop[192:320, 192:320].mean() > 100


def test_kps_unchanged_when_codeformer_keeps_size():
    img = np.zeros((128, 128, 3), np.uint8)
    kps = np.array([[50, 60], [78, 60], [64, 75], [52, 90], [76, 90]], float)
    with patch.object(fe, "run_codeformer", side_effect=lambda rgb, **kw: rgb):
        _, new_kps = fe.preprocess_face_image(img, kps=kps, padding_enabled=False,
                                              white_balance_enabled=False)
    assert np.allclose(new_kps, kps)


def test_codeformer_runs_one_at_a_time_with_separate_temp_dirs(tmp_path, monkeypatch):
    import threading
    import time
    monkeypatch.chdir(tmp_path)
    os.makedirs("CodeFormer")
    open("CodeFormer/inference_codeformer.py", "w").close()
    state = {"running": 0, "max": 0, "inputs": []}
    lock = threading.Lock()

    def fake_run(cmd, **kw):
        with lock:
            state["running"] += 1
            state["max"] = max(state["max"], state["running"])
            state["inputs"].append(cmd[cmd.index("--input_path") + 1])
        time.sleep(0.05)
        with lock:
            state["running"] -= 1
        return SimpleNamespace(returncode=1)          # lỗi → trả ảnh gốc

    monkeypatch.setattr(fe.subprocess, "run", fake_run)
    img = np.zeros((32, 32, 3), np.uint8)
    threads = [threading.Thread(target=fe.run_codeformer, args=(img,)) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["max"] == 1                          # không chạy song song
    assert len(set(state["inputs"])) == 5             # mỗi lần một thư mục tạm riêng
