"""T10 — Embedding và gallery FAISS dựng từ DB.

- Vector search-crop được giữ miễn phí lúc ingest (ingest.persist_observation_frame)
  hoặc bù qua ensure_crop_embedding(); retry cùng (entity, model, version,
  preprocessing) không tạo trùng (ON CONFLICT DO NOTHING).
- Gallery FAISS CHỈ chứa crop purpose=search; reference crop và generated là query.
- Snapshot version hóa theo bộ ba (model, version, preprocessing): rebuild tạo
  snapshot mới rồi công bố atomically (swap dict dưới lock), không trộn mapping
  của hai snapshot. Mapping: FAISS position → embedding UUID (+ crop_id).
- Không so vector khác bộ ba; vector NaN/Infinity/zero bị loại kèm ngữ cảnh.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_GALLERIES: dict[str, dict] = {}
_GALLERY_LOCK = threading.Lock()


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


def ensure_crop_embedding(crop_id: str) -> Optional[str]:
    """Tính + lưu embedding cho 1 search crop (bù cho crop nạp trước T10).

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
                if purpose != "search":
                    logger.warning("từ chối embed crop không phải search: %s", crop_id)
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
        return embedding_id
    except Exception as exc:
        logger.warning("ensure_crop_embedding bỏ qua (%s): %s", crop_id, exc)
        return None


def rebuild_gallery(
    *, model_name: Optional[str] = None, model_version: Optional[str] = None,
    preprocessing_version: Optional[str] = None, limit: int = 100000,
) -> dict:
    """Dựng lại snapshot gallery từ DB (chỉ crop purpose=search, đúng bộ ba).

    Công bố atomically; snapshot cũ giữ nguyên cho query đang chạy.
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
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.id, e.crop_id, e.dimensions, e."values", e.normalized
                FROM face_media.face_embeddings e
                JOIN face_media.face_crops c ON c.id = e.crop_id
                WHERE e.crop_id IS NOT NULL
                  AND c.purpose = 'search'
                  AND e.model_name = %s AND e.model_version = %s
                  AND e.preprocessing_version = %s
                ORDER BY e.created_at LIMIT %s
                """,
                (*triple, limit),
            )
            rows = cur.fetchall()
    vectors: list[np.ndarray] = []
    mapping: list[dict] = []
    skipped = 0
    for emb_id, crop_id, dims, values, normalized in rows:
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
        vectors.append(vec.astype(np.float32))
        mapping.append({"embedding_id": emb_id, "crop_id": crop_id})
    if not vectors:
        raise ValueError(
            f"Không có embedding search hợp lệ cho bộ ba {triple} "
            f"(bỏ {skipped} vector lỗi)."
        )
    index = build_index(vectors)
    snapshot = {
        "key": snapshot_key(*triple),
        "model_name": triple[0], "model_version": triple[1],
        "preprocessing_version": triple[2],
        "index": index, "mapping": mapping,
        "size": len(mapping), "skipped": skipped,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with _GALLERY_LOCK:
        _GALLERIES[snapshot["key"]] = snapshot
    return {k: v for k, v in snapshot.items() if k != "index"}


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


def search_gallery(
    query_vector: np.ndarray, *, k: int = 5, snapshot_key_: Optional[str] = None,
) -> list[dict]:
    """Tìm top-k trong snapshot (query phải cùng không gian: caller đảm bảo
    vector đã normalize và cùng model/preprocessing với snapshot)."""
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
    scores, indices = snap["index"].search(query, min(k, snap["index"].ntotal))
    out = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if int(idx) == -1:
            continue
        entry = snap["mapping"][int(idx)]
        out.append(
            {
                "embedding_id": entry["embedding_id"],
                "crop_id": entry["crop_id"],
                "score": float(score),
            }
        )
    return out
