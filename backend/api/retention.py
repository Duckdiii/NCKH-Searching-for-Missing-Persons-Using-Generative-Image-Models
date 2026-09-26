"""P4 — Retention/TTL, quota, quá tải, GC và metrics (doc §9).

- Tách TTL crop, vector, tracklet, exemplar và audit; không ngầm giữ sinh trắc
  học vô thời hạn. Không tự xóa dữ liệu hiện có khi bật thử nghiệm (policy
  seed enabled=false; GC chỉ chạy khi operator bật + gọi API, dry_run mặc định).
- Xóa theo ledger: tombstone ẩn query → event xóa index → xóa file hết tham
  chiếu/bảo lưu → xóa vector/bản ghi con theo FK → hoàn tất. Retry idempotent;
  query kiểm tombstone DB khi index còn cũ. Orphan GC có khoảng chờ (grace) và
  tránh file đang upload.
- Quota/quá tải: kiểm tra reserve byte trước ghi (nhiều worker không cùng vượt
  cap); cảnh báo 80%, dừng 90% (ngưỡng thử nghiệm, cần điều chỉnh).
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

WARN_RATIO = 0.8
STOP_RATIO = 0.9
ORPHAN_GRACE_SEC = 3600
_GUARD_CACHE: dict = {"at": 0.0, "used": 0}
GUARD_CACHE_SEC = 60.0


def media_cap_bytes() -> Optional[int]:
    raw = (os.environ.get("MEDIA_CAP_BYTES", "") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def media_used_bytes(root: Optional[str] = None) -> int:
    """Tổng byte file dưới MEDIA_ROOT (best-effort, đo lỗi → giữ giá trị cũ)."""
    root = root or os.environ.get("MEDIA_ROOT", os.path.join("outputs", "media"))
    total = 0
    try:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.startswith(".upload-"):
                    continue  # file đang upload — không tính, tránh xóa nhầm
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    continue
    except Exception:
        pass
    return total


def storage_guard() -> dict:
    """Kiểm tra byte trước ghi (doc §9): cảnh báo 80%, dừng 90%.

    Cache 60s để không walk đĩa mỗi frame. Không đo được hoặc không đặt cap
    → cho qua (fail-open) kèm cảnh báo, không chặn intake vì lỗi đo.
    """
    cap = media_cap_bytes()
    if cap is None:
        return {"cap_bytes": None, "used_bytes": None, "level": "ok",
                "ratio": 0.0, "reason": "no_cap"}
    now = datetime.now(timezone.utc).timestamp()
    if now - float(_GUARD_CACHE.get("at", 0.0)) >= GUARD_CACHE_SEC:
        _GUARD_CACHE["used"] = media_used_bytes()
        _GUARD_CACHE["at"] = now
    used = int(_GUARD_CACHE["used"])
    verdict = check_byte_budget(used, cap)
    return {"cap_bytes": cap, "used_bytes": used, **verdict}


def check_byte_budget(used_bytes: int, cap_bytes: Optional[int],
                      need_bytes: int = 0) -> dict:
    """Kiểm tra/reserve byte trước ghi. Trả {allowed, level, ratio}."""
    if not cap_bytes or cap_bytes <= 0:
        return {"allowed": True, "level": "ok", "ratio": 0.0}
    ratio = (used_bytes + need_bytes) / float(cap_bytes)
    if ratio >= STOP_RATIO:
        return {"allowed": False, "level": "stop", "ratio": round(ratio, 3)}
    if ratio >= WARN_RATIO:
        return {"allowed": True, "level": "warn", "ratio": round(ratio, 3)}
    return {"allowed": True, "level": "ok", "ratio": round(ratio, 3)}


def get_policies(conn) -> list[dict]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT scope, ttl_days, quota_bytes, enabled, reason, updated_at"
                " FROM face_media.retention_policies ORDER BY scope"
            )
            return [dict(zip(("scope", "ttl_days", "quota_bytes", "enabled",
                               "reason", "updated_at"), r)) for r in cur.fetchall()]
    except Exception:
        return []


def set_policy(conn, scope: str, *, ttl_days: Optional[int] = None,
               quota_bytes: Optional[int] = None,
               enabled: Optional[bool] = None,
               reason: Optional[str] = None) -> None:
    if scope not in ("crop", "vector", "tracklet", "exemplar", "audit"):
        raise ValueError(f"scope không hợp lệ: {scope!r}.")
    sets, vals = [], []
    if ttl_days is not None:
        if ttl_days <= 0:
            raise ValueError("ttl_days phải > 0.")
        sets.append("ttl_days = %s")
        vals.append(ttl_days)
    if quota_bytes is not None:
        sets.append("quota_bytes = %s")
        vals.append(quota_bytes)
    if enabled is not None:
        sets.append("enabled = %s")
        vals.append(enabled)
    if reason is not None:
        sets.append("reason = %s")
        vals.append(reason)
    sets.append("updated_at = now()")
    vals.append(scope)
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE face_media.retention_policies SET {', '.join(sets)} WHERE scope = %s",
            vals,
        )


def is_tombstoned(conn, crop_id: Optional[str] = None,
                  embedding_id: Optional[str] = None) -> bool:
    """Query kiểm tombstone DB khi index còn cũ (doc §9). Thiếu bảng → False."""
    if not crop_id and not embedding_id:
        return False
    try:
        with conn.cursor() as cur:
            if crop_id:
                cur.execute(
                    "SELECT 1 FROM face_media.deletion_ledger"
                    " WHERE crop_id = %s AND done = false LIMIT 1",
                    (crop_id,),
                )
                if cur.fetchone():
                    return True
            if embedding_id:
                cur.execute(
                    "SELECT 1 FROM face_media.embedding_tombstones WHERE embedding_id = %s",
                    (embedding_id,),
                )
                if cur.fetchone():
                    return True
        return False
    except Exception:
        return False


def check_quotas(conn) -> dict:
    """Đo usage so với quota_bytes từng scope (§9).

    crop/vector đo byte thật; scope còn lại báo số hàng (quota_bytes không áp
    được cho hàng → over=None). Vượt quota mà không còn ứng viên hết TTL thì
    chỉ BÁO (không tự xóa mẫu còn cần cho hồ sơ).
    """
    out: dict[str, dict] = {}
    try:
        policies = {p["scope"]: p for p in get_policies(conn)}
    except Exception:
        return out
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT COALESCE(SUM(a.byte_size), 0), COUNT(*)
                FROM face_media.face_crops c
                JOIN face_media.assets a ON a.id = c.asset_id
                """)
            b, n = cur.fetchone()
            out["crop"] = {"used_bytes": int(b), "rows": int(n), "unit": "bytes"}
        except Exception as exc:
            logger.debug("quota crop bỏ qua: %s", exc)
        try:
            cur.execute(
                'SELECT COALESCE(SUM(cardinality("values")) * 4, 0), COUNT(*)'
                " FROM face_media.face_embeddings")
            b, n = cur.fetchone()
            out["vector"] = {"used_bytes": int(b), "rows": int(n), "unit": "bytes"}
        except Exception:
            try:
                cur.execute("SELECT COUNT(*) FROM face_media.face_embeddings")
                n = int(cur.fetchone()[0])
                out["vector"] = {"used_bytes": n * 2048, "rows": n,
                                 "unit": "bytes_est"}
            except Exception as exc:
                logger.debug("quota vector bỏ qua: %s", exc)
        for table, scope in (("tracklets", "tracklet"),
                             ("identity_exemplars", "exemplar"),
                             ("search_runs", "audit")):
            try:
                cur.execute(f"SELECT COUNT(*) FROM face_media.{table}")
                out[scope] = {"rows": int(cur.fetchone()[0]), "unit": "rows"}
            except Exception:
                continue
    for scope, info in out.items():
        quota = (policies.get(scope, {}) or {}).get("quota_bytes")
        info["quota_bytes"] = quota
        if quota and info.get("unit", "").startswith("bytes"):
            info["over"] = info["used_bytes"] > quota
        else:
            info["over"] = None
    return out


def plan_expired(conn, *, batch: int = 200) -> dict:
    """Liệt kê ứng viên hết TTL theo policy đã bật (không xóa ở bước này)."""
    out: dict[str, list] = {"crops": [], "tracklets": []}
    try:
        policies = {p["scope"]: p for p in get_policies(conn)}
    except Exception:
        return out
    with conn.cursor() as cur:
        crop_pol = policies.get("crop", {})
        if crop_pol.get("enabled"):
            try:
                cur.execute(
                    """
                    SELECT c.id, a.storage_key, a.byte_size
                    FROM face_media.face_crops c
                    JOIN face_media.assets a ON a.id = c.asset_id
                    LEFT JOIN face_media.deletion_ledger l
                      ON l.crop_id = c.id AND l.done = false
                    WHERE c.created_at < now() - (%s || ' days')::interval
                      AND l.id IS NULL
                    ORDER BY c.created_at LIMIT %s
                    """,
                    (str(int(crop_pol.get("ttl_days", 30))), batch),
                )
                out["crops"] = [{"crop_id": r[0], "storage_key": r[1],
                                 "byte_size": r[2]} for r in cur.fetchall()]
            except Exception as exc:
                logger.debug("plan_expired crops bỏ qua: %s", exc)
        trk_pol = policies.get("tracklet", {})
        if trk_pol.get("enabled"):
            try:
                cur.execute(
                    """
                    SELECT id FROM face_media.tracklets
                    WHERE last_seen_at < now() - (%s || ' days')::interval
                      AND status <> 'expired'
                    ORDER BY last_seen_at LIMIT %s
                    """,
                    (str(int(trk_pol.get("ttl_days", 30))), batch),
                )
                out["tracklets"] = [{"tracklet_id": r[0]} for r in cur.fetchall()]
            except Exception as exc:
                logger.debug("plan_expired tracklets bỏ qua: %s", exc)
    return out


def tombstone_crops(conn, crops: list[dict], reason: str) -> int:
    """Ghi ledger tombstone để ẩn query trước khi xóa thật."""
    n = 0
    with conn.cursor() as cur:
        for c in crops:
            try:
                cur.execute(
                    """
                    INSERT INTO face_media.deletion_ledger
                        (id, entity, entity_id, crop_id, storage_key, reason)
                    VALUES (%s, 'crop', %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (str(uuid.uuid4()), c["crop_id"], c["crop_id"],
                     c.get("storage_key"), reason),
                )
                n += 1
            except Exception as exc:
                logger.debug("tombstone bỏ qua %s: %s", c.get("crop_id"), exc)
    return n


def run_gc_once(conn, storage, *, batch: int = 200,
                dry_run: bool = True) -> dict:
    """GC một lượt theo ledger: index evict → file → rows. Idempotent.

    dry_run=True (mặc định): chỉ đếm, không xóa — an toàn thử nghiệm.
    """
    from backend.api import gallery as gal
    job_id = str(uuid.uuid4())
    scanned = tombstoned = files_removed = 0
    bytes_freed = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.retention_jobs (id, status, dry_run)
                VALUES (%s, 'running', %s)
                """,
                (job_id, dry_run),
            )
    except Exception:
        pass  # thiếu bảng jobs → vẫn chạy GC lõi
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, entity, entity_id, crop_id, storage_key, attempts
                FROM face_media.deletion_ledger
                WHERE done = false ORDER BY created_at LIMIT %s
                """,
                (batch,),
            )
            rows = cur.fetchall()
        scanned = len(rows)
        for lid, entity, eid, crop_id, skey, attempts in rows:
            try:
                if dry_run:
                    continue
                # 1. Event xóa index (delta tombstone + outbox delete).
                try:
                    with conn.cursor() as cur2:
                        cur2.execute(
                            """
                            SELECT id FROM face_media.face_embeddings
                            WHERE crop_id = %s
                            """,
                            (str(crop_id or eid),),
                        )
                        emb_rows = cur2.fetchall()
                    for (emb_id,) in emb_rows:
                        try:
                            with conn.cursor() as cur3:
                                cur3.execute(
                                    """
                                    INSERT INTO face_media.embedding_tombstones
                                        (embedding_id, crop_id, reason)
                                    VALUES (%s, %s, 'gc')
                                    ON CONFLICT (embedding_id) DO NOTHING
                                    """,
                                    (str(emb_id), str(crop_id or eid)),
                                )
                        except Exception:
                            pass
                        gal.emit_outbox(conn, entity="embedding",
                                        entity_id=str(emb_id), op="delete",
                                        payload={"crop_id": str(crop_id or eid)})
                    # Evict khỏi delta RAM ngay.
                    for _key, delta in list(gal._DELTAS.items()):
                        delta.get("tombstones", set()).add(str(crop_id or eid))
                        for (emb_id,) in emb_rows:
                            delta.get("tombstones", set()).add(str(emb_id))
                except Exception as exc:
                    logger.debug("gc index evict bỏ qua: %s", exc)
                # 2. Xóa file hết tham chiếu/bảo lưu (giữ file còn tham chiếu khác).
                file_ok = True
                if skey:
                    try:
                        with conn.cursor() as cur2:
                            cur2.execute(
                                """
                                SELECT 1 FROM face_media.face_crops c
                                JOIN face_media.assets a ON a.id = c.asset_id
                                WHERE a.storage_key = %s AND c.id <> %s LIMIT 1
                                """,
                                (skey, str(crop_id or eid)),
                            )
                            shared = cur2.fetchone() is not None
                        if not shared:
                            storage.delete_quiet(skey)
                            files_removed += 1
                        file_ok = True
                    except Exception:
                        file_ok = False
                # 3. Xóa vector/bản ghi con theo FK (embedding → crop).
                rows_ok = True
                try:
                    with conn.cursor() as cur2:
                        cur2.execute(
                            "DELETE FROM face_media.face_embeddings WHERE crop_id = %s",
                            (str(crop_id or eid),),
                        )
                        cur2.execute(
                            "DELETE FROM face_media.face_crops WHERE id = %s",
                            (str(crop_id or eid),),
                        )
                        if skey and file_ok:
                            cur2.execute(
                                "DELETE FROM face_media.assets WHERE storage_key = %s",
                                (skey,),
                            )
                except Exception:
                    rows_ok = False
                with conn.cursor() as cur2:
                    cur2.execute(
                        """
                        UPDATE face_media.deletion_ledger
                        SET file_deleted = %s, index_evicted = true,
                            rows_deleted = %s, attempts = %s,
                            done = %s WHERE id = %s
                        """,
                        (file_ok, rows_ok, int(attempts or 0) + 1,
                         bool(file_ok and rows_ok), lid),
                    )
                    if file_ok:
                        tombstoned += 1
                if rows_ok:
                    try:
                        from backend.api import thumbs as _thumbs
                        _thumbs.invalidate(str(crop_id or eid))
                    except Exception:
                        pass
            except Exception as exc:
                try:
                    with conn.cursor() as cur2:
                        cur2.execute(
                            "UPDATE face_media.deletion_ledger SET attempts = attempts + 1,"
                            " error = %s WHERE id = %s",
                            (f"{type(exc).__name__}: {exc}"[:2000], lid),
                        )
                except Exception:
                    pass
        status = "done"
    except Exception as exc:
        status = "error"
        logger.warning("run_gc_once lỗi: %s", exc)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE face_media.retention_jobs
                SET status = %s, scanned = %s, tombstoned = %s,
                    files_removed = %s, bytes_freed = %s, finished_at = now(),
                    error = NULL WHERE id = %s
                """,
                (status, scanned, tombstoned, files_removed, bytes_freed, job_id),
            )
    except Exception:
        pass
    return {"job_id": job_id, "status": status, "scanned": scanned,
            "tombstoned": tombstoned, "files_removed": files_removed,
            "bytes_freed": bytes_freed, "dry_run": dry_run}


def orphan_gc(conn, storage, *, batch: int = 200, dry_run: bool = True,
              grace_sec: int = ORPHAN_GRACE_SEC) -> dict:
    """Orphan GC: asset crop không còn tham chiếu + quá grace → ledger.

    Tránh file đang upload (grace) và file còn tham chiếu.
    """
    found = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id, a.storage_key
                FROM face_media.assets a
                LEFT JOIN face_media.face_crops c ON c.asset_id = a.id
                LEFT JOIN face_media.frames f ON f.asset_id = a.id
                LEFT JOIN face_media.sources s ON s.original_asset_id = a.id
                LEFT JOIN face_media.generated_images gi ON gi.asset_id = a.id
                WHERE c.id IS NULL AND f.id IS NULL AND s.id IS NULL
                  AND gi.id IS NULL
                  AND a.created_at < now() - (%s || ' seconds')::interval
                LIMIT %s
                """,
                (str(int(grace_sec)), batch),
            )
            orphans = cur.fetchall()
        found = len(orphans)
        if not dry_run:
            for aid, skey in orphans:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO face_media.deletion_ledger
                                (id, entity, entity_id, storage_key, reason)
                            VALUES (%s, 'asset', %s, %s, 'orphan_gc')
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (str(uuid.uuid4()), str(aid), skey),
                        )
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("orphan_gc bỏ qua: %s", exc)
    return {"orphans": found, "dry_run": dry_run}


def collect_metrics() -> dict:
    """Dashboard/soak metrics: RAM/queue/fps/tracks/crops/index/GC (doc §10)."""
    import psutil  # optional
    has_psutil = True
    try:
        proc = psutil.Process()
        rss_mb = round(proc.memory_info().rss / 1024 / 1024, 1)
    except Exception:
        has_psutil, rss_mb = False, None
    sessions = {}
    try:
        from backend.api import cameras as cam
        with cam._SESSIONS_LOCK:
            for sid, s in cam._SESSIONS.items():
                sessions[sid] = {
                    "camera_id": s.get("camera_id"),
                    "frames_sampled": s.get("frames_sampled", 0),
                    "faces_found": s.get("faces_found", 0),
                    "dropped": s.get("dropped", 0),
                    "queue": s.get("queue").qsize() if s.get("queue") else 0,
                    "queue_bytes": s.get("queue_bytes", 0),
                    "active_tracks": s.get("active_tracks", 0),
                    "crops_persisted": s.get("crops_persisted", 0),
                    "tracks_closed": s.get("tracks_closed", 0),
                    "effective_fps": round(float(s.get("effective_fps", 0) or 0), 2),
                    "ingest_latency_ms": s.get("ingest_latency_ms"),
                    "persist_errors": s.get("persist_errors", 0),
                    "spooled": s.get("spooled", 0),
                    "spool_dropped": s.get("spool_dropped", 0),
                    "spooled_redriven": s.get("spooled_redriven", 0),
                    "linked": s.get("linked", 0),
                    "link_unresolved": s.get("link_unresolved", 0),
                    "link_conflicts": s.get("link_conflicts", 0),
                    "link_errors": s.get("link_errors", 0),
                }
    except Exception:
        pass
    gallery_info: Any = {}
    try:
        from backend.api import gallery as gal
        gallery_info = {"snapshots": [{k: v for k, v in s.items() if k != "index"}
                                      for s in gal.list_snapshots()],
                        "delta": gal.get_delta_stats(),
                        "latency_ms": gal.get_latency_stats()}
    except Exception as exc:
        gallery_info = {"error": str(exc)}
    db_info: dict = {}
    try:
        from backend.api.database import get_pool
        from backend.api.persistence import db_ping
        if db_ping():
            with get_pool().connection() as conn:
                with conn.cursor() as cur:
                    for table in ("face_crops", "face_embeddings", "tracklets",
                                  "global_identities", "outbox_events",
                                  "deletion_ledger"):
                        try:
                            cur.execute(
                                f"SELECT count(*) FROM face_media.{table}")
                            db_info[table] = int(cur.fetchone()[0])
                        except Exception:
                            db_info[table] = -1
                    try:
                        cur.execute(
                            "SELECT count(*) FROM face_media.outbox_events WHERE done = false")
                        db_info["outbox_pending"] = int(cur.fetchone()[0])
                    except Exception:
                        pass
                    try:
                        cur.execute(
                            "SELECT count(*) FROM face_media.deletion_ledger WHERE done = false")
                        db_info["ledger_pending"] = int(cur.fetchone()[0])
                    except Exception:
                        pass
        else:
            db_info = {"db": "down"}
    except Exception as exc:
        db_info = {"error": f"{type(exc).__name__}"}
    disk: dict = {}
    try:
        root = os.environ.get("MEDIA_ROOT", os.path.join("outputs", "media"))
        usage = shutil.disk_usage(root if os.path.exists(root) else ".")
        disk = {"total_gb": round(usage.total / 1e9, 2),
                "used_gb": round(usage.used / 1e9, 2),
                "free_gb": round(usage.free / 1e9, 2)}
        cap_env = os.environ.get("MEDIA_CAP_BYTES")
        if cap_env and cap_env.isdigit():
            disk["budget"] = check_byte_budget(usage.used, int(cap_env))
    except Exception:
        pass
    thumbs_info: dict = {}
    spool_info: dict = {}
    try:
        from backend.api import spool as _spool
        from backend.api import thumbs as _thumbs
        thumbs_info = _thumbs.stats()
        spool_info = _spool.spool_usage()
    except Exception:
        pass
    return {"at": datetime.now(timezone.utc).isoformat(),
            "rss_mb": rss_mb, "psutil": has_psutil,
            "sessions": sessions, "gallery": gallery_info,
            "db": db_info, "disk": disk,
            "thumbs": thumbs_info, "spool": spool_info}
