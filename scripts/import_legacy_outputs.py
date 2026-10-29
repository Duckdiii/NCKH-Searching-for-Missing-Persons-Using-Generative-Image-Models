"""T14 — Import dữ liệu cũ từ outputs/ vào face_media.

Sự thật lineage: ảnh gốc upload cũ đã bị xóa (chỉ qua file tạm), crop từng bị
ghi đè theo {session_id}_crop.png — nên chuỗi cũ THIẾU lineage về ảnh gốc.
Script KHÔNG tạo bản ghi giả để vượt FK. Hai chế độ:

- dry-run (mặc định, không cần DB): quét outputs/, báo cáo mapping +
  checkpoint, chỉ ra chuỗi nào khôi phục được, chuỗi nào thiếu lineage.
- --apply (cần DATABASE_URL): import chuỗi khôi phục được. Vì ảnh gốc đã mất,
  file crop cũ chỉ import được ở chế độ --reroot: crop được re-root thành ảnh
  nguồn reference MỚI với provenance ghi rõ
  (parameters {"imported": true, "lineage": "re-rooted: ..."}), KHÔNG suy
  nguồn/danh tính khi thiếu bằng chứng. File video cũ chỉ báo cáo, không import
  (thiếu timestamp/ngữ cảnh ingest).

Idempotency: bỏ qua asset đã có cùng storage_key; checkpoint JSON ghi lại tiến
trình để chạy lại an toàn; import hai lần không nhân dữ liệu.

Cách chạy:
    python scripts/import_legacy_outputs.py --dry-run
    python scripts/import_legacy_outputs.py --apply --reroot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

APP_UPLOADS_CROP_RE = re.compile(r"^(.+)_crop\.png$")
APP_UPLOADS_PREVIEW_RE = re.compile(r"^(.+)_preview\.png$")
AGE_RE = re.compile(r"^age_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

PROVENANCE_REROOT = (
    "re-rooted: original upload missing (legacy temp file deleted); "
    "legacy crop file registered as new reference source image"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan(root: Path) -> dict:
    """Quét outputs/, trả report {sessions, jobs, videos, orphans}."""
    report: dict = {"sessions": {}, "jobs": {}, "videos": [], "orphans": []}
    app_uploads = root / "app_uploads"
    if app_uploads.is_dir():
        for path in sorted(app_uploads.iterdir()):
            if not path.is_file():
                continue
            crop_match = APP_UPLOADS_CROP_RE.match(path.name)
            preview_match = APP_UPLOADS_PREVIEW_RE.match(path.name)
            if crop_match:
                sid = crop_match.group(1)
                entry = report["sessions"].setdefault(
                    sid, {"crop": None, "preview": None})
                entry["crop"] = str(path)
            elif preview_match:
                sid = preview_match.group(1)
                entry = report["sessions"].setdefault(
                    sid, {"crop": None, "preview": None})
                entry["preview"] = str(path)
            elif path.name != ".gitkeep":
                report["orphans"].append(str(path))
    jobs_dir = root / "jobs"
    if jobs_dir.is_dir():
        for job_dir in sorted(jobs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            info: dict = {"input_crop": None, "ages": {}, "videos": [],
                          "video_faces": 0, "gallery_match": []}
            for path in sorted(job_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.name == "input_crop.png":
                    info["input_crop"] = str(path)
                elif (m := AGE_RE.match(path.name)):
                    info["ages"][int(m.group(1))] = str(path)
                elif path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                    info["videos"].append(str(path))
                    report["videos"].append(str(path))
                elif path.parent.name == "video_faces":
                    info["video_faces"] += 1
                elif path.name.startswith("gallery_match_"):
                    info["gallery_match"].append(str(path))
            report["jobs"][job_dir.name] = info
    return report


def plan(report: dict) -> dict:
    """Đánh giá chuỗi nào import được (khớp crop bytes job↔session vẫn thiếu gốc)."""
    crop_hash: dict[str, str] = {}
    for sid, sess in report["sessions"].items():
        if sess["crop"]:
            crop_hash[sid] = sha256_file(Path(sess["crop"]))
    decisions = {"importable_reference": [], "missing_lineage": [], "videos_only": []}
    for job_id, job in report["jobs"].items():
        if not job["input_crop"] or not job["ages"]:
            decisions["missing_lineage"].append(
                {"job_id": job_id, "reason": "thiếu input_crop hoặc age_*.png"})
            continue
        job_hash = sha256_file(Path(job["input_crop"]))
        matched = [sid for sid, digest in crop_hash.items() if digest == job_hash]
        decisions["importable_reference"].append(
            {"job_id": job_id, "input_crop": job["input_crop"],
             "ages": job["ages"], "matched_sessions": matched,
             "lineage": "re-root (original upload đã mất)"})
    for sid, sess in report["sessions"].items():
        if sess["crop"] and not any(
                sid in d.get("matched_sessions", [])
                for d in decisions["importable_reference"]):
            decisions["missing_lineage"].append(
                {"session_id": sid, "reason": "crop mồ côi, không job nào dùng"})
    for video in report["videos"]:
        decisions["videos_only"].append(
            {"path": video, "reason": "video cũ không import (thiếu ingest context)"})
    return decisions


def _load_checkpoint(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"imported_storage_keys": [], "imported_jobs": []}


def apply_import(
    decisions: dict, *, checkpoint_path: Path, dry_run: bool = True,
) -> dict:
    """Import chuỗi re-root (idempotent). dry_run chỉ mô phỏng."""
    from backend.api import repositories as repo
    from backend.api.database import get_pool
    from backend.api.storage import (
        PREFIX_REFERENCE_CROPS,
        PREFIX_REFERENCE_GENERATED,
        PREFIX_REFERENCE_ORIGINALS,
        build_key,
        get_storage,
        sha256_bytes,
        sniff_mime,
    )

    checkpoint = _load_checkpoint(checkpoint_path)
    done_keys = set(checkpoint["imported_storage_keys"])
    done_jobs = set(checkpoint["imported_jobs"])
    stats = {"assets": 0, "sources": 0, "jobs": 0, "generated": 0,
             "skipped": 0, "gaps": []}
    storage = get_storage()

    def asset_exists(conn, key: str) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM face_media.assets WHERE storage_key = %s", (key,))
            return cur.fetchone() is not None

    for item in decisions["importable_reference"]:
        job_id = item["job_id"]
        if job_id in done_jobs:
            stats["skipped"] += 1
            continue
        crop_path = Path(item["input_crop"])
        crop_bytes = crop_path.read_bytes()
        import cv2
        import numpy as np

        crop_bgr = cv2.imdecode(
            np.frombuffer(crop_bytes, np.uint8), cv2.IMREAD_COLOR)
        if crop_bgr is None:
            stats["gaps"].append({"job_id": job_id, "reason": "crop hỏng"})
            continue
        height, width = crop_bgr.shape[:2]
        if dry_run:
            stats["sources"] += 1
            stats["jobs"] += 1
            stats["generated"] += len(item["ages"])
            continue
        import uuid as _uuid

        source_id = str(_uuid.uuid4())
        asset_id = str(_uuid.uuid4())
        key = build_key(PREFIX_REFERENCE_ORIGINALS, asset_id, "image/png")
        if key in done_keys:
            stats["skipped"] += 1
            continue
        with get_pool().connection() as conn:
            if asset_exists(conn, key):
                stats["skipped"] += 1
                continue
            storage.put_bytes(crop_bytes, key, mime_type="image/png")
            try:
                repo.create_asset(
                    conn, storage_key=key, media_type="image",
                    mime_type="image/png", sha256=sha256_bytes(crop_bytes),
                    byte_size=len(crop_bytes), width=int(width),
                    height=int(height), role="original_image", asset_id=asset_id)
                repo.create_source(
                    conn, purpose="reference", kind="image",
                    original_asset_id=asset_id, source_id=source_id)
                frame_id = repo.create_frame(
                    conn, purpose="reference", source_id=source_id,
                    asset_id=asset_id, frame_index=0, offset_ms=0,
                    still_image=True)
                # Detection phủ toàn frame (crop re-root = toàn bộ "ảnh gốc").
                run_id = str(_uuid.uuid4())
                detection_id = repo.create_detection(
                    conn, purpose="reference", frame_id=frame_id,
                    detector_name="legacy-import", detector_version="reroot-v1",
                    run_id=run_id, face_index=0,
                    bbox=(0.0, 0.0, float(width), float(height)),
                    confidence=1.0, frame_width=int(width),
                    frame_height=int(height),
                    quality={"imported": True, "provenance": PROVENANCE_REROOT})
                crop_asset_id = str(_uuid.uuid4())
                crop_key = build_key(PREFIX_REFERENCE_CROPS, crop_asset_id,
                                     "image/png")
                storage.put_bytes(crop_bytes, crop_key, mime_type="image/png")
                repo.create_asset(
                    conn, storage_key=crop_key, media_type="image",
                    mime_type="image/png", sha256=sha256_bytes(crop_bytes),
                    byte_size=len(crop_bytes), width=int(width),
                    height=int(height), role="crop", asset_id=crop_asset_id)
                crop_id = repo.create_crop(
                    conn, purpose="reference", detection_id=detection_id,
                    asset_id=crop_asset_id, method="bbox",
                    preprocessing={"imported": True,
                                   "provenance": PROVENANCE_REROOT,
                                   "legacy_job": job_id})
                db_job_id = repo.create_generation_job(
                    conn, input_crop_id=crop_id,
                    model_name="stable-diffusion-v1-5",
                    model_version="fading-specialized-unet",
                    status="done",
                    parameters={"imported": True,
                                "provenance": PROVENANCE_REROOT,
                                "legacy_job": job_id})
                for age, age_path in item["ages"].items():
                    data = Path(age_path).read_bytes()
                    mime = sniff_mime(data, "image/png")
                    gen_asset = str(_uuid.uuid4())
                    gen_key = build_key(PREFIX_REFERENCE_GENERATED, gen_asset,
                                        mime)
                    storage.put_bytes(data, gen_key, mime_type=mime)
                    gen_bgr = cv2.imdecode(
                        np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    gh, gw = (gen_bgr.shape[:2] if gen_bgr is not None
                              else (512, 512))
                    repo.create_asset(
                        conn, storage_key=gen_key, media_type="image",
                        mime_type=mime, sha256=sha256_bytes(data),
                        byte_size=len(data), width=int(gw), height=int(gh),
                        role="generated", asset_id=gen_asset)
                    repo.create_generated_image(
                        conn, job_id=db_job_id, asset_id=gen_asset,
                        target_age=int(age), variant_index=0,
                        parameters={"imported": True, "legacy_job": job_id})
                    stats["generated"] += 1
                repo.set_job_status(conn, db_job_id, "done")
            except Exception:
                storage.delete_quiet(key)
                raise
            done_keys.add(key)
            done_jobs.add(job_id)
            stats["assets"] += 1
            stats["sources"] += 1
            stats["jobs"] += 1
    if not dry_run:
        checkpoint["imported_storage_keys"] = sorted(done_keys)
        checkpoint["imported_jobs"] = sorted(done_jobs)
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return stats


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Import outputs/ cũ vào face_media.")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reroot", action="store_true",
                        help="Cho phép re-root crop cũ (ghi rõ provenance).")
    parser.add_argument("--checkpoint", default="outputs/.import_checkpoint.json")
    args = parser.parse_args(argv)
    apply = args.apply
    if apply and not args.reroot:
        print("Từ chối --apply khi thiếu --reroot: chuỗi cũ thiếu lineage ảnh gốc, "
              "chỉ import khi chấp nhận re-root có provenance rõ ràng.")
        return 2
    root = Path(args.root)
    report = scan(root)
    decisions = plan(report)
    print(f"sessions: {len(report['sessions'])}, jobs: {len(report['jobs'])}, "
          f"videos: {len(report['videos'])}, orphans: {len(report['orphans'])}")
    print(f"importable (re-root): {len(decisions['importable_reference'])}, "
          f"thiếu lineage: {len(decisions['missing_lineage'])}, "
          f"video bỏ qua: {len(decisions['videos_only'])}")
    for gap in decisions["missing_lineage"][:20]:
        print(f"  GAP: {gap}")
    if apply:
        stats = apply_import(decisions, checkpoint_path=Path(args.checkpoint),
                             dry_run=False)
        print(f"APPLIED: {stats}")
    else:
        stats = apply_import(decisions, checkpoint_path=Path(args.checkpoint),
                             dry_run=True)
        print(f"DRY-RUN (không ghi/xóa): {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
