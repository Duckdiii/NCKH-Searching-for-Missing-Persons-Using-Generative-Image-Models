"""T10 + P3 — Embedding và gallery FAISS dựng từ DB.

- Vector search-crop được giữ lúc ingest hoặc bù qua ensure_crop_embedding();
  retry cùng (entity, model, version, preprocessing) không tạo trùng.
- Gallery FAISS CHỈ chứa crop purpose=search; reference crop và generated là query.
- Snapshot version hóa theo bộ ba (model, version, preprocessing): rebuild tạo
  snapshot mới rồi công bố atomically (swap dict dưới lock), không trộn mapping
  của hai snapshot. Mapping: FAISS position → embedding UUID (+ crop_id).
- Không so vector khác bộ ba; vector NaN/Infinity/zero bị loại kèm ngữ cảnh.
- P3 (doc §6): rebuild theo batch (mặc định 2000 vector) với watermark keyset
  (created_at, id) — không lấy phần đầu cũ bằng ORDER BY ... LIMIT; công bố
  watermark/phạm vi index rõ ràng. Query dùng base snapshot bất biến + delta
  snapshot version hóa (outbox worker, một writer/shard, dedupe event_id).
  Update/delete có tombstone; query lọc tombstone khi index còn cũ.
  Flat là baseline retrieval; ANN/PQ (nếu có) tạo ứng viên rồi re-rank bằng
  float32 và phải đo recall so với Flat (measure_recall).
"""

from __future__ import annotations

import functools
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_GALLERIES: dict[str, dict] = {}
_GALLERY_LOCK = threading.Lock()
# P3 delta: key -> {mapping, vectors(list float32 1-D), tombstones(set),
#                   version, updated_at}. Cap RAM bằng MAX_DELTA.
_DELTAS: dict[str, dict] = {}
# Latency query (§10 P3 acceptance): key -> deque ms (giữ 128 mẫu gần nhất).
_LATENCY: dict[str, Any] = {}
LATENCY_KEEP = 128
# TTL RAM của delta (§6: TTL RAM khác TTL lưu bền). Delta quá tuổi hoặc quá
# số lượng → compact (rebuild base) thay vì tăng RAM vô hạn.
DELTA_TTL_SEC = float(os.environ.get("GALLERY_DELTA_TTL_SEC", "1800"))
REBUILD_BATCH = int(os.environ.get("GALLERY_REBUILD_BATCH", "2000"))
MAX_DELTA = int(os.environ.get("GALLERY_MAX_DELTA", "5000"))


def current_embed_triple() -> tuple[str, str, str]:
    """Bộ ba (model_name, model_version, preprocessing_version) hiện hành.

    model từ config embedding.model_name (vd buffalo_l); preprocessing mã hóa
    det_size cấu hình để không trộn không gian vector khác nhau.
    """
    try:
        from backend.api.dependencies import get_config

        cfg = get_config()
        emb = cfg.get("embedding", {})
        model_version = str(emb.get("model_name", "buffalo_l"))
        det = emb.get("det_size", [256, 256])
        preproc = f"det{det[0]}x{det[1]}_crop112_v1"
    except Exception:
        model_version, preproc = "buffalo_l", "det256x256_crop112_v1"
    return ("insightface", model_version, preproc)


def snapshot_key(model_name: str, model_version: str, preprocessing_version: str) -> str:
    return f"{model_name}/{model_version}/{preprocessing_version}"


def _weights_fingerprint_cached(name: str) -> str:
    import hashlib as _hl
    try:
        root = os.path.join(os.path.expanduser("~"), ".insightface", "models", name)
        if os.path.isdir(root):
            parts = []
            for dirpath, _, files in os.walk(root):
                for f in sorted(files):
                    if f.lower().endswith(".onnx"):
                        p = os.path.join(dirpath, f)
                        try:
                            st = os.stat(p)
                            parts.append(f"{f}:{st.st_size}:{int(st.st_mtime)}")
                        except OSError:
                            continue
            if parts:
                return _hl.sha256("|".join(parts).encode()).hexdigest()[:12]
    except Exception:
        pass
    try:
        import insightface as _is
        ver = getattr(_is, "__version__", "") or "pkg"
        return _hl.sha256(f"insightface-{ver}".encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


_weights_fingerprint_cached = functools.lru_cache(maxsize=4)(
    _weights_fingerprint_cached)


def weights_fingerprint(model_name: str = "buffalo_l") -> str:
    """Fingerprint weights model (doc §6: tách không gian theo weights).

    Hash (tên, size, mtime) các file .onnx của pack InsightFace — rẻ, không đọc
    toàn bộ weights. Thiếu model dir → version package → "unknown" (fail-open:
    caller dùng space legacy, không vỡ triển khai cũ).
    """
    try:
        return _weights_fingerprint_cached(str(model_name))
    except Exception:
        return "unknown"


def ensure_crop_embedding(crop_id: str) -> Optional[str]:
    """Tính + lưu embedding cho crop tham chiếu hoặc crop quan sát.

    Trả embedding id, hoặc None khi DB/storage/model không sẵn sàng.
    """
    try:
        from backend.api import repositories as repo
        from backend.api.database import get_pool
        from backend.api.dependencies import get_embedder
        from backend.api.persistence import db_ping
        from backend.api.storage import get_storage

        if not db_ping():
            return None
        model_name, model_version, preproc = current_embed_triple()
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.storage_key, c.purpose FROM face_media.face_crops c
                    JOIN face_media.assets a ON a.id = c.asset_id
                    WHERE c.id = %s
                    """,
                    (crop_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                storage_key, purpose = row
                if purpose not in ("reference", "search"):
                    logger.warning("từ chối embed crop có purpose không hợp lệ: %s", crop_id)
                    return None
                cur.execute(
                    """
                    SELECT id FROM face_media.face_embeddings
                    WHERE crop_id = %s AND model_name = %s
                      AND model_version = %s AND preprocessing_version = %s
                    """,
                    (crop_id, model_name, model_version, preproc),
                )
                existing = cur.fetchone()
                if existing is not None:
                    return existing[0]
        with get_storage().open(storage_key) as handle:
            data = handle.read()
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            vector = np.asarray(get_embedder().embed(tmp), dtype=np.float64)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Embedding crop {crop_id} chứa NaN/Infinity.")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError(f"Embedding crop {crop_id} là vector zero.")
        values = (vector / norm).tolist()
        with get_pool().connection() as conn:
            embedding_id = repo.create_embedding(
                conn, model_name=model_name, model_version=model_version,
                preprocessing_version=preproc, dimensions=len(values),
                values=[float(v) for v in values], normalized=True,
                crop_id=crop_id,
            )
            emit_outbox(conn, entity="embedding", entity_id=embedding_id,
                        op="upsert", triple=(model_name, model_version, preproc),
                        payload={"crop_id": crop_id})
        return embedding_id
    except Exception as exc:
        logger.warning("ensure_crop_embedding bỏ qua (%s): %s", crop_id, exc)
        return None


def _next_version(key: str) -> int:
    with _GALLERY_LOCK:
        snap = _GALLERIES.get(key)
        cur = int(snap.get("version", 0)) if snap else 0
        delta = _DELTAS.get(key, {})
        dver = int(delta.get("version", 0))
        return max(cur, dver) + 1


def _delta_for(key: str) -> dict:
    with _GALLERY_LOCK:
        delta = _DELTAS.get(key)
        if delta is None:
            delta = {"mapping": [], "vectors": [], "added_at": [],
                     "tombstones": set(), "version": 0,
                     "updated_at": datetime.now(timezone.utc).isoformat()}
            _DELTAS[key] = delta
        if "added_at" not in delta:
            # Tương thích delta tạo bởi bản cũ (không có timestamp).
            delta["added_at"] = [0.0] * len(delta.get("mapping", []))
        return delta


def delta_age_sec(key: str) -> Optional[float]:
    """Tuổi của entry delta cũ nhất (None khi delta rỗng)."""
    import time as _time
    with _GALLERY_LOCK:
        delta = _DELTAS.get(key)
        if not delta or not delta.get("added_at"):
            return None
        oldest = min(delta["added_at"]) if delta["added_at"] else None
        if not oldest:
            return None
        return max(0.0, _time.time() - oldest)


def delta_needs_compact(key: str) -> dict:
    """Delta có cần compact (rebuild base) không: quá tuổi RAM hoặc quá số lượng."""
    with _GALLERY_LOCK:
        delta = _DELTAS.get(key, {})
        pending = len(delta.get("mapping", []))
    age = delta_age_sec(key)
    by_size = pending >= MAX_DELTA
    by_age = age is not None and age >= DELTA_TTL_SEC
    return {"key": key, "pending": pending, "age_sec": age,
            "compact_needed": bool(by_size or by_age),
            "reason": ("size" if by_size else ("age" if by_age else "ok"))}


def compact_index(key: Optional[str] = None) -> dict:
    """Compact delta → base bằng rebuild (giữ nguyên triple của snapshot)."""
    results = []
    keys = [key] if key else list(_DELTAS.keys())
    if not keys:
        snap = get_snapshot()
        keys = [snap["key"]] if snap else []
    for k in keys:
        need = delta_needs_compact(k)
        snap = get_snapshot(k)
        if snap is None:
            # Chưa có base: dựng từ triple của key (model/version/preproc).
            parts = k.split("/")
            try:
                info = rebuild_gallery(model_name=parts[0], model_version=parts[1],
                                       preprocessing_version=parts[2])
            except Exception as exc:
                results.append({"key": k, "compacted": False,
                                "reason": f"{type(exc).__name__}: {exc}"})
                continue
        else:
            try:
                info = rebuild_gallery(model_name=snap["model_name"],
                                       model_version=snap["model_version"],
                                       preprocessing_version=snap["preprocessing_version"])
            except Exception as exc:
                results.append({"key": k, "compacted": False,
                                "reason": f"{type(exc).__name__}: {exc}"})
                continue
        results.append({"key": k, "compacted": True, "size": info.get("size"),
                        "was": need})
    return {"results": results}


def rebuild_gallery(
    *, model_name: Optional[str] = None, model_version: Optional[str] = None,
    preprocessing_version: Optional[str] = None, limit: int = 100000,
    batch_size: int = REBUILD_BATCH,
) -> dict:
    """Dựng lại snapshot gallery từ DB theo batch (P3).

    - Đọc keyset (created_at, id) từng batch để peak RAM bounded; add增量 vào
      cùng index thay vì np.stack toàn bộ + FAISS copy.
    - Lọc tombstone DB (embedding_tombstones) ngay lúc đọc.
    - Công bố atomically kèm version + watermark_to + scope (phạm vi đã lập
      chỉ mục); snapshot cũ giữ nguyên cho query đang chạy.
    - Giữ chữ ký cũ (limit) để tương thích; thêm batch_size tùy chọn.
    """
    from backend.api.database import get_pool
    from backend.api.persistence import db_ping
    from src.search.faiss_index import build_index

    if not db_ping():
        raise ConnectionError("Database face_media không sẵn sàng.")
    triple = (
        model_name or current_embed_triple()[0],
        model_version or current_embed_triple()[1],
        preprocessing_version or current_embed_triple()[2],
    )
    key = snapshot_key(*triple)
    index = None
    dim = None
    mapping: list[dict] = []
    skipped = 0
    scanned = 0
    batches = 0
    last_ts = None
    last_id = None
    first_ts = None
    base_table_has_tomb = True
    truncated = False
    with get_pool().connection() as conn:
        while len(mapping) < limit:
            with conn.cursor() as cur:
                page = min(batch_size, limit - len(mapping))
                try:
                    if last_ts is None:
                        cur.execute(
                            """
                            SELECT e.id, e.crop_id, e.dimensions, e."values",
                                   e.normalized, e.created_at, e.id
                            FROM face_media.face_embeddings e
                            JOIN face_media.face_crops c ON c.id = e.crop_id
                            LEFT JOIN face_media.embedding_tombstones t
                              ON t.embedding_id = e.id
                            WHERE e.crop_id IS NOT NULL
                              AND c.purpose = 'search'
                              AND e.model_name = %s AND e.model_version = %s
                              AND e.preprocessing_version = %s
                              AND t.embedding_id IS NULL
                            ORDER BY e.created_at, e.id LIMIT %s
                            """,
                            (*triple, page),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT e.id, e.crop_id, e.dimensions, e."values",
                                   e.normalized, e.created_at, e.id
                            FROM face_media.face_embeddings e
                            JOIN face_media.face_crops c ON c.id = e.crop_id
                            LEFT JOIN face_media.embedding_tombstones t
                              ON t.embedding_id = e.id
                            WHERE e.crop_id IS NOT NULL
                              AND c.purpose = 'search'
                              AND e.model_name = %s AND e.model_version = %s
                              AND e.preprocessing_version = %s
                              AND t.embedding_id IS NULL
                              AND (e.created_at, e.id) > (%s::timestamptz, %s::uuid)
                            ORDER BY e.created_at, e.id LIMIT %s
                            """,
                            (*triple, last_ts, last_id, page),
                        )
                except Exception as exc:
                    if "embedding_tombstones" in str(exc) and base_table_has_tomb:
                        base_table_has_tomb = False
                        continue  # thử lại batch này không lọc tombstone
                    raise
                rows = cur.fetchall()
            # Tôn trọng limit ngay cả khi driver/mock trả thừa (phòng thủ).
            rows = list(rows[:page])
            if not rows:
                break
            batches += 1
            batch_vecs: list[np.ndarray] = []
            batch_map: list[dict] = []
            for emb_id, crop_id, dims, values, normalized, cts, _eid in rows:
                scanned += 1
                last_ts, last_id = str(cts), emb_id
                if first_ts is None:
                    first_ts = str(cts)
                vec = np.asarray(values, dtype=np.float64)
                if vec.shape != (dims,) or not np.all(np.isfinite(vec)):
                    skipped += 1
                    continue
                if normalized:
                    norm = float(np.linalg.norm(vec))
                    if norm == 0.0 or abs(norm - 1.0) > 1e-3:
                        skipped += 1
                        continue
                    vec = vec / norm
                else:
                    skipped += 1  # gallery cosine yêu cầu vector chuẩn hóa
                    continue
                if dim is None:
                    dim = int(vec.shape[0])
                elif int(vec.shape[0]) != dim:
                    skipped += 1  # không trộn chiều khác nhau
                    continue
                batch_vecs.append(vec.astype(np.float32))
                batch_map.append({"embedding_id": emb_id, "crop_id": crop_id})
            if batch_vecs:
                import numpy as _np
                mat = _np.stack(batch_vecs)  # chỉ 1 batch trong RAM
                if index is None:
                    index = build_index([row for row in mat])
                else:
                    index.add(mat)
                mapping.extend(batch_map)
                del mat, batch_vecs
            if len(rows) < page:
                break
        if len(mapping) >= limit:
            truncated = True
    if index is None or not mapping:
        raise ValueError(
            f"Không có embedding search hợp lệ cho bộ ba {triple} "
            f"(bỏ {skipped} vector lỗi)."
        )
    version = _next_version(key)
    watermark_to = last_ts
    snapshot = {
        "key": key,
        "version": version,
        "model_name": triple[0], "model_version": triple[1],
        "preprocessing_version": triple[2],
        "index": index, "mapping": mapping,
        "size": len(mapping), "skipped": skipped,
        "scanned": scanned, "batches": batches,
        "watermark_to": watermark_to, "watermark_id": last_id,
        "scope": {"from": first_ts, "to": watermark_to, "scanned": scanned,
                  "limit": limit, "truncated": truncated,
                  "tombstone_filtered": base_table_has_tomb},
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with _GALLERY_LOCK:
        _GALLERIES[key] = snapshot
        # Replay phần mới hơn: giữ delta chưa có trong base (dedupe).
        delta = _DELTAS.get(key)
        if delta:
            base_ids = {m["embedding_id"] for m in mapping}
            kept_map, kept_vec, kept_at = [], [], []
            times = delta.get("added_at", [])
            for i, (m, v) in enumerate(zip(delta.get("mapping", []),
                                           delta.get("vectors", []))):
                if m.get("embedding_id") not in base_ids and \
                        m.get("embedding_id") not in delta.get("tombstones", set()) and \
                        m.get("crop_id") not in delta.get("tombstones", set()):
                    kept_map.append(m)
                    kept_vec.append(v)
                    kept_at.append(times[i] if i < len(times) else 0.0)
            delta["mapping"], delta["vectors"] = kept_map, kept_vec
            delta["added_at"] = kept_at
            delta["version"] = version
    _record_snapshot_row(snapshot)
    return {k: v for k, v in snapshot.items() if k != "index"}


def _record_snapshot_row(snapshot: dict) -> None:
    """Best-effort: ghi index_snapshots để công bố watermark (thiếu bảng → bỏ)."""
    try:
        from backend.api.database import get_pool
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO face_media.index_snapshots
                        (id, triple_key, version, model_name, model_version,
                         preprocessing_version, watermark_to, watermark_id,
                         scope, size, skipped, active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::timestamptz, %s::uuid,
                            %s::jsonb, %s, %s, true)
                    ON CONFLICT (triple_key, version) DO NOTHING
                    """,
                    (str(uuid.uuid4()), snapshot["key"], snapshot["version"],
                     snapshot["model_name"], snapshot["model_version"],
                     snapshot["preprocessing_version"], snapshot.get("watermark_to"),
                     snapshot.get("watermark_id"),
                     __import__("json").dumps(snapshot.get("scope", {})),
                     snapshot["size"], snapshot["skipped"]),
                )
    except Exception as exc:
        logger.debug("bỏ qua ghi index_snapshots: %s", exc)


def get_snapshot(key: Optional[str] = None) -> Optional[dict]:
    with _GALLERY_LOCK:
        if key is not None:
            return _GALLERIES.get(key)
        if not _GALLERIES:
            return None
        return max(_GALLERIES.values(), key=lambda s: s["built_at"])


def list_snapshots() -> list[dict]:
    with _GALLERY_LOCK:
        return [
            {k: v for k, v in snap.items() if k != "index"}
            for snap in sorted(_GALLERIES.values(), key=lambda s: s["built_at"])
        ]


def get_delta_stats(key: Optional[str] = None) -> dict:
    with _GALLERY_LOCK:
        if key is not None:
            d = _DELTAS.get(key, {})
            single = {"key": key, "pending": len(d.get("mapping", [])),
                      "tombstones": len(d.get("tombstones", set())),
                      "version": d.get("version", 0),
                      "updated_at": d.get("updated_at")}
        else:
            single = None
    # Tính age/compact ngoài lock (delta_age_sec tự lock).
    if single is not None:
        need = delta_needs_compact(key)
        single.update({"age_sec": need["age_sec"],
                       "compact_needed": need["compact_needed"],
                       "compact_reason": need["reason"]})
        return single
    with _GALLERY_LOCK:
        out = {k: {"pending": len(v.get("mapping", [])),
                   "tombstones": len(v.get("tombstones", set())),
                   "version": v.get("version", 0)}
               for k, v in _DELTAS.items()}
    for k in out:
        need = delta_needs_compact(k)
        out[k].update({"age_sec": need["age_sec"],
                       "compact_needed": need["compact_needed"],
                       "compact_reason": need["reason"]})
    return out


def get_latency_stats(key: Optional[str] = None) -> dict:
    """Latency query gallery (§10 P3): count/p50/p95/last_ms theo snapshot key."""
    def _summ(samples: list[float]) -> dict:
        if not samples:
            return {"count": 0, "p50_ms": None, "p95_ms": None, "last_ms": None}
        s = sorted(samples)
        pick = lambda p: s[min(len(s) - 1, int(p / 100 * len(s)))]
        return {"count": len(s), "p50_ms": pick(50), "p95_ms": pick(95),
                "last_ms": samples[-1]}

    with _GALLERY_LOCK:
        if key is not None:
            return {"key": key, **_summ(list(_LATENCY.get(key, [])))}
        return {k: _summ(list(v)) for k, v in _LATENCY.items()}


def _delta_search(delta: dict, query: np.ndarray, k: int) -> list[dict]:
    vecs = delta.get("vectors", [])
    mapping = delta.get("mapping", [])
    tombs = delta.get("tombstones", set())
    if not vecs or not mapping:
        return []
    import numpy as _np
    mat = _np.stack([_np.asarray(v, dtype=_np.float32).reshape(-1) for v in vecs])
    q = _np.asarray(query, dtype=_np.float32).reshape(-1)
    q = q / (float(_np.linalg.norm(q)) + 1e-9)
    norms = _np.linalg.norm(mat, axis=1) + 1e-9
    scores = (mat / norms[:, None]) @ q
    order = _np.argsort(-scores)[:k]
    out = []
    for i in (order.tolist() if hasattr(order, "tolist") else list(order)):
        m = mapping[int(i)]
        if m.get("embedding_id") in tombs or m.get("crop_id") in tombs:
            continue
        out.append({"embedding_id": m["embedding_id"], "crop_id": m["crop_id"],
                    "score": float(scores[int(i)]), "from": "delta"})
    return out


def search_gallery(
    query_vector: np.ndarray, *, k: int = 5, snapshot_key_: Optional[str] = None,
    include_delta: bool = True,
) -> list[dict]:
    """Tìm top-k trong snapshot + delta (re-rank gộp, lọc tombstone)."""
    import time as _time
    from collections import deque as _deque
    t0 = _time.perf_counter()
    snap = get_snapshot(snapshot_key_)
    if snap is None:
        raise ValueError("Chưa có gallery snapshot — gọi rebuild trước.")
    query = np.asarray(query_vector, dtype=np.float64)
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ValueError("Query vector phải 1-D và hữu hạn.")
    norm = float(np.linalg.norm(query))
    if norm == 0.0:
        raise ValueError("Query vector zero.")
    query = (query / norm).astype(np.float32).reshape(1, -1)
    with _GALLERY_LOCK:
        delta = _DELTAS.get(snap["key"])
        tombs = set(delta.get("tombstones", set())) if delta else set()
    scores, indices = snap["index"].search(query, min(k, snap["index"].ntotal))
    out = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if int(idx) == -1:
            continue
        entry = snap["mapping"][int(idx)]
        if entry.get("embedding_id") in tombs or entry.get("crop_id") in tombs:
            continue  # query kiểm tombstone DB khi index còn cũ (doc §9)
        out.append(
            {
                "embedding_id": entry["embedding_id"],
                "crop_id": entry["crop_id"],
                "score": float(score),
            }
        )
    if include_delta and delta:
        out.extend(_delta_search(delta, query.reshape(-1), k))
        out.sort(key=lambda x: x["score"], reverse=True)
        # Dedupe theo crop, giữ score cao nhất.
        seen: dict[str, dict] = {}
        for h in out:
            old = seen.get(h["crop_id"])
            if old is None or h["score"] > old["score"]:
                seen[h["crop_id"]] = h
        out = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    out = out[:k]
    # Ghi latency (§10 P3: theo dõi p95 query).
    try:
        ms = (_time.perf_counter() - t0) * 1000.0
        with _GALLERY_LOCK:
            dq = _LATENCY.get(snap["key"])
            if dq is None:
                dq = _deque(maxlen=LATENCY_KEEP)
                _LATENCY[snap["key"]] = dq
            dq.append(round(ms, 3))
    except Exception:
        pass
    return out


def measure_recall(flat_hits: list[dict], ann_hits: list[dict], k: int = 5) -> float:
    """Recall@k của ANN so với Flat: |ANN ∩ Flat| / |Flat| (doc §6).

    Re-rank không cứu ứng viên bị ANN bỏ sót — đo để quyết định có dùng ANN.
    """
    flat_ids = [h["crop_id"] for h in flat_hits[:k]]
    if not flat_ids:
        return 1.0
    ann_ids = {h["crop_id"] for h in ann_hits[:k]}
    return sum(1 for i in flat_ids if i in ann_ids) / len(flat_ids)


# ------------------------- outbox (P3) -------------------------

def emit_outbox(conn, *, entity: str, entity_id: str, op: str,
                triple: tuple[str, str, str] = ("", "", ""),
                payload: Optional[dict] = None) -> bool:
    """Ghi outbox cùng transaction vector/metadata (best-effort, thiếu bảng → False)."""
    try:
        import json as _json
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO face_media.outbox_events
                    (event_id, entity, entity_id, op, model_name, model_version,
                     preprocessing_version, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (str(uuid.uuid4()), entity, entity_id, op, triple[0],
                 triple[1], triple[2], _json.dumps(payload or {})),
            )
        return True
    except Exception as exc:
        logger.debug("bỏ qua emit_outbox (%s): %s", entity_id, exc)
        return False


def claim_outbox(conn, limit: int = 200) -> list[dict]:
    """Lấy event pending (một writer/shard gọi; best-effort)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, entity, entity_id, op, model_name,
                       model_version, preprocessing_version, payload, attempts
                FROM face_media.outbox_events
                WHERE done = false ORDER BY created_at LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            cur.execute(
                """
                UPDATE face_media.outbox_events SET claimed_at = now(),
                    attempts = attempts + 1
                WHERE event_id = ANY(%s)
                """,
                ([r[0] for r in rows],),
            )
        keys = ("event_id", "entity", "entity_id", "op", "model_name",
                "model_version", "preprocessing_version", "payload", "attempts")
        return [dict(zip(keys, r)) for r in rows]
    except Exception as exc:
        logger.debug("claim_outbox bỏ qua: %s", exc)
        return []


def mark_outbox_done(conn, event_id: str, error: Optional[str] = None) -> None:
    try:
        with conn.cursor() as cur:
            if error:
                cur.execute(
                    "UPDATE face_media.outbox_events SET error = %s WHERE event_id = %s",
                    (error[:2000], event_id),
                )
            else:
                cur.execute(
                    "UPDATE face_media.outbox_events SET done = true WHERE event_id = %s",
                    (event_id,),
                )
    except Exception as exc:
        logger.debug("mark_outbox_done bỏ qua: %s", exc)


def outbox_lag(conn) -> int:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM face_media.outbox_events WHERE done = false")
            return int(cur.fetchone()[0])
    except Exception:
        return -1


def drain_outbox_once(limit: int = 200) -> dict:
    """Worker: áp dụng event vào delta (ít nhất một lần, dedupe event_id)."""
    from backend.api.database import get_pool
    from backend.api.persistence import db_ping
    if not db_ping():
        return {"applied": 0, "reason": "db_down"}
    applied, deleted, errors = 0, 0, 0
    with get_pool().connection() as conn:
        events = claim_outbox(conn, limit)
        for ev in events:
            try:
                key = snapshot_key(ev["model_name"] or current_embed_triple()[0],
                                   ev["model_version"] or current_embed_triple()[1],
                                   ev["preprocessing_version"] or current_embed_triple()[2])
                delta = _delta_for(key)
                if ev["op"] == "delete":
                    delta["tombstones"].add(str(ev["entity_id"]))
                    payload = ev.get("payload") or {}
                    if isinstance(payload, dict) and payload.get("crop_id"):
                        delta["tombstones"].add(str(payload["crop_id"]))
                    # Evict khỏi delta mapping (base sẽ sạch sau rebuild).
                    keep_m, keep_v, keep_t = [], [], []
                    times = delta.get("added_at", [])
                    for i, (m, v) in enumerate(zip(delta["mapping"],
                                                   delta["vectors"])):
                        if m.get("embedding_id") == str(ev["entity_id"]):
                            continue
                        keep_m.append(m)
                        keep_v.append(v)
                        keep_t.append(times[i] if i < len(times) else 0.0)
                    delta["mapping"], delta["vectors"] = keep_m, keep_v
                    delta["added_at"] = keep_t
                    deleted += 1
                else:
                    # Upsert embedding: tải vector gốc rồi add vào delta.
                    with conn.cursor() as cur:
                        cur.execute(
                            'SELECT "values", crop_id FROM face_media.face_embeddings WHERE id = %s',
                            (str(ev["entity_id"]),),
                        )
                        row = cur.fetchone()
                    if row is None:
                        mark_outbox_done(conn, str(ev["event_id"]))
                        continue
                    values, crop_id = row
                    vec = np.asarray(values, dtype=np.float64)
                    n = float(np.linalg.norm(vec))
                    if n == 0 or not np.all(np.isfinite(vec)):
                        mark_outbox_done(conn, str(ev["event_id"]),
                                         error="invalid_vector")
                        continue
                    vec = (vec / n).astype(np.float32)
                    with _GALLERY_LOCK:
                        import time as _time2
                        ids = {m.get("embedding_id") for m in delta["mapping"]}
                        if str(ev["entity_id"]) not in ids:
                            if len(delta["mapping"]) >= MAX_DELTA:
                                # Cap delta: compact bằng rebuild thay vì tăng RAM.
                                mark_outbox_done(
                                    conn, str(ev["event_id"]),
                                    error="delta_full_needs_compact")
                                errors += 1
                                continue
                            delta["mapping"].append(
                                {"embedding_id": str(ev["entity_id"]),
                                 "crop_id": str(crop_id)})
                            delta["vectors"].append(vec)
                            delta.setdefault("added_at", []).append(_time2.time())
                    applied += 1
                delta["updated_at"] = datetime.now(timezone.utc).isoformat()
                mark_outbox_done(conn, str(ev["event_id"]))
            except Exception as exc:
                errors += 1
                try:
                    mark_outbox_done(conn, str(ev["event_id"]),
                                     error=f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass
    return {"applied": applied, "deleted": deleted, "errors": errors}
