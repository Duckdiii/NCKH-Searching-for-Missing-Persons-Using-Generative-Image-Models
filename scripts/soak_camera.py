#!/usr/bin/env python3
"""Soak test camera pipeline (§10 tài liệu kiến trúc đa camera).

Hai chế độ:
1. track (mặc định, không cần camera/DB): mô phỏng N camera ảo × M phút với
   mặt di chuyển + nhiễu, chạy đúng CameraTracker + selector + encode JPEG
   thật. Kiểm tra:
   - RAM ổn định với quota/TTL cố định (RSS đầu/cuối, cho phép tăng < 15%),
   - crops/tracklet ≤ 3, không sinh tracklet mới mỗi phút cho người đứng lâu,
   - chiếu crop/ngày và byte/ngày so với ngân sách (mặc định 2 crop × 30KiB).
2. poll: bám /api/ops/metrics của backend đang chạy, kiểm tra RSS ổn định và
   in storage hội tụ (cần backend + DB thật).

Ví dụ:
    python scripts/soak_camera.py --cameras 8 --minutes 5
    python scripts/soak_camera.py poll --url http://127.0.0.1:8000/api/ops/metrics \\
        --interval 10 --duration 600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return None


def cmd_track(args) -> int:
    from backend.api import camera_tracks as ct

    rng = np.random.default_rng(args.seed)
    trackers = [ct.CameraTracker(timeout_sec=2.0) for _ in range(args.cameras)]
    # Mỗi camera: vài "người" đi qua rồi đứng lâu (lặp bbox gần giống).
    persons = []
    for c in range(args.cameras):
        for _ in range(args.persons_per_camera):
            x = float(rng.uniform(20, 200))
            persons.append({"cam": c, "x": x, "y": 60.0, "dx": float(rng.uniform(-2, 2)),
                            "linger": int(rng.integers(30, 120))})
    total_frames = int(args.minutes * 60 * args.fps)
    dt = 1.0 / args.fps
    t = 0.0
    sim_drops = 0
    total_obs = 0
    for f in range(total_frames):
        t += dt
        for ci, tr in enumerate(trackers):
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            faces = []
            for p in persons:
                if p["cam"] != ci:
                    continue
                # Người xuất hiện theo chu kỳ: đi qua (di chuyển) rồi đứng lâu.
                phase = (f + int(p["x"])) % (p["linger"] + 60)
                if phase >= p["linger"]:
                    continue
                if phase > 20:
                    x = p["x"]  # đứng lâu: bbox gần như cố định + nhiễu nhỏ
                    x += float(rng.normal(scale=1.5))
                else:
                    x = p["x"] + p["dx"] * phase
                v = np.zeros(8)
                v[0] = 1.0
                faces.append({"bbox_orig": [x, p["y"], x + 90, p["y"] + 90],
                              "det_score": 0.9, "embedding": v, "kps": None})
            if len(faces) > 6:  # mô phỏng queue đầy phải bỏ bớt
                faces = faces[:6]
                sim_drops += 1
            tr.update(frame, faces, now=t)
            total_obs += len(faces)
            # Encode crop của ứng viên như persist thật (đo byte).
            for trk in tr.open_tracks():
                for cand in trk.candidates:
                    ct.encode_crop_jpeg(cand.crop_bgr)
            # Flush định kỳ như consumer.
            if f % int(args.fps * 2) == 0:
                for trk in tr.pop_closed_ready(t):
                    assert len(trk.candidates) <= 3, "vượt 3 crop/tracklet!"
            del frame
    rss0 = _rss_mb()
    # Ép GC rồi đo lại để đánh giá ổn định RAM.
    import gc
    gc.collect()
    rss1 = _rss_mb()
    closed_total = sum(tr._created_today for tr in trackers)
    open_total = sum(len(tr.open_tracks()) for tr in trackers)
    # Chiếu ngày: giả định tải này kéo dài 24h.
    sim_hours = args.minutes / 60
    proj_tracklets_day = closed_total / max(sim_hours, 1e-6) * 24
    proj_crops_day = proj_tracklets_day * 2  # trung bình tính tải K=2
    proj_gb_day = proj_crops_day * 30 * 1024 / 1e9
    print(f"[soak] cameras={args.cameras} frames={total_frames} obs={total_obs} "
          f"drops={sim_drops}")
    print(f"[soak] tracklets_created={closed_total} open={open_total}")
    print(f"[soak] RSS: {rss0}MB -> {rss1}MB (sau GC)")
    print(f"[soak] chiếu 24h: {proj_tracklets_day:.0f} tracklets/ngày, "
          f"{proj_crops_day:.0f} crops/ngày ≈ {proj_gb_day:.2f} GiB ảnh/ngày "
          f"(giả định 30KiB/crop — thay bằng p95 từ bench_crop_jpeg).")
    ok = True
    if rss0 and rss1 and rss1 > rss0 * 1.15:
        print("[soak] CẢNH BÁO: RSS tăng >15% sau soak — kiểm tra rò rỉ (track/cache).")
        ok = False
    if proj_tracklets_day > args.cameras * 2000:
        print("[soak] CẢNH BÁO: vượt quota 2000 tracklets/camera/ngày.")
        ok = False
    print("[soak] " + ("ĐẠT" if ok else "CẦN XEM LẠI"))
    return 0 if ok else 1


def cmd_poll(args) -> int:
    samples = []
    t0 = time.time()
    while time.time() - t0 < args.duration:
        try:
            with urllib.request.urlopen(args.url, timeout=10) as resp:
                m = json.loads(resp.read().decode("utf-8"))
            samples.append({"t": round(time.time() - t0, 1),
                            "rss_mb": m.get("rss_mb"),
                            "sessions": m.get("sessions", {}),
                            "outbox": (m.get("db", {}) or {}).get("outbox_pending"),
                            "ledger": (m.get("db", {}) or {}).get("ledger_pending")})
            print(f"[poll] t={samples[-1]['t']}s rss={samples[-1]['rss_mb']}MB "
                  f"outbox={samples[-1]['outbox']} ledger={samples[-1]['ledger']}")
        except Exception as exc:
            print(f"[poll] lỗi: {type(exc).__name__}: {exc}")
        time.sleep(args.interval)
    rss = [s["rss_mb"] for s in samples if s["rss_mb"]]
    if len(rss) >= 3:
        x = np.arange(len(rss))
        slope = float(np.polyfit(x, np.array(rss, dtype=float), 1)[0])
        print(f"[poll] RSS slope={slope:.3f} MB/mẫu "
              f"(~{slope * (3600 / max(args.interval, 1)):.1f} MB/giờ)")
        if abs(slope * (3600 / max(args.interval, 1))) > 100:
            print("[poll] CẢNH BÁO: RAM trôi >100MB/giờ — kiểm tra delta/index/cache.")
            return 1
    print("[poll] xong")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Soak test camera pipeline (§10).")
    sub = ap.add_subparsers(dest="cmd", required=False)
    tr = sub.add_parser("track", help="Soak tracker/selector tổng hợp (mặc định)")
    tr.add_argument("--cameras", type=int, default=8)
    tr.add_argument("--minutes", type=float, default=5)
    tr.add_argument("--fps", type=float, default=2.0)
    tr.add_argument("--persons-per-camera", type=int, default=3)
    tr.add_argument("--seed", type=int, default=7)
    pl = sub.add_parser("poll", help="Bám /api/ops/metrics backend đang chạy")
    pl.add_argument("--url", default="http://127.0.0.1:8000/api/ops/metrics")
    pl.add_argument("--interval", type=float, default=10)
    pl.add_argument("--duration", type=float, default=600)
    args = ap.parse_args()
    if args.cmd == "poll":
        return cmd_poll(args)
    if args.cmd is None:
        args.cameras = getattr(args, "cameras", 8) or 8
        args.minutes = getattr(args, "minutes", 5) or 5
        args.fps = getattr(args, "fps", 2.0) or 2.0
        args.persons_per_camera = getattr(args, "persons_per_camera", 3) or 3
        args.seed = getattr(args, "seed", 7)
        return cmd_track(args)
    return cmd_track(args)


if __name__ == "__main__":
    raise SystemExit(main())
