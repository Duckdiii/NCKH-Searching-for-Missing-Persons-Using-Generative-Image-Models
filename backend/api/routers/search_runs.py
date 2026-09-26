"""T10/T11 — Gallery FAISS (rebuild/snapshot/query) + lượt search đã lưu.

- Rebuild snapshot từ DB (chỉ crop search, đúng bộ ba model/version/preproc).
- Query gallery bằng crop quan sát hoặc ảnh tạo sinh; kết quả lưu thành
  search_runs/search_results (accepted theo ngưỡng TÁCH BIỆT human_confirmed).
- Xem lại run sau restart; xác nhận con người qua /confirm.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.api import gallery as gallery_service
from backend.api import repositories as repo
from backend.api.database import get_pool
from backend.api.gallery import current_embed_triple
from backend.api.persistence import db_ping
from backend.api.schemas import (
    ConfirmRequest,
    GalleryQueryRequest,
    GallerySnapshotInfo,
    SearchResultItem,
    SearchRunDetail,
)
from backend.api.storage import get_storage

router = APIRouter(prefix="/api", tags=["search"])


def _require_db() -> None:
    if not db_ping():
        raise HTTPException(
            status_code=503, detail="Database face_media không sẵn sàng.")


@router.post("/gallery/rebuild", response_model=GallerySnapshotInfo)
def rebuild_gallery_view():
    """T10: dựng lại snapshot gallery từ DB (atomic swap)."""
    _require_db()
    try:
        return GallerySnapshotInfo(**gallery_service.rebuild_gallery())
    except (ConnectionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/gallery/snapshots")
def list_snapshots_view():
    _require_db()
    return {"snapshots": gallery_service.list_snapshots(),
            "current_triple": list(current_embed_triple())}


def _query_vector_from_crop(crop_id: str) -> np.ndarray:
    model_name, model_version, preproc = current_embed_triple()
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e."values", e.dimensions FROM face_media.face_embeddings e
                JOIN face_media.face_crops c ON c.id = e.crop_id
                WHERE e.crop_id = %s AND e.model_name = %s
                  AND e.model_version = %s AND e.preprocessing_version = %s
                  AND c.purpose IN ('reference', 'search')
                """,
                (crop_id, model_name, model_version, preproc))
            row = cur.fetchone()
    if row is None:
        embedding_id = gallery_service.ensure_crop_embedding(crop_id)
        if embedding_id is None:
            raise HTTPException(
                status_code=400,
                detail="Không có embedding cho crop (crop không tồn tại "
                "hoặc model/storage không sẵn sàng).")
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT "values" FROM face_media.face_embeddings WHERE id = %s',
                    (embedding_id,))
                row = cur.fetchone()
    return np.asarray(row[0], dtype=np.float64)


def _query_vector_from_generated(generated_image_id: str) -> tuple[np.ndarray, dict]:
    from backend.api.dependencies import get_embedder

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.storage_key, gi.job_id, gi.target_age
                FROM face_media.generated_images gi
                JOIN face_media.assets a ON a.id = gi.asset_id
                WHERE gi.id = %s
                """,
                (generated_image_id,))
            row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Ảnh tạo sinh không tồn tại.")
    storage_key, job_id, target_age = row
    model, version, preproc = current_embed_triple()
    with get_pool().connection() as conn:
        cached = conn.execute(
            'SELECT "values" FROM face_media.face_embeddings WHERE generated_image_id=%s '
            'AND model_name=%s AND model_version=%s AND preprocessing_version=%s',
            (generated_image_id, model, version, preproc)).fetchone()
    if cached is not None:
        return np.asarray(cached[0], dtype=np.float64), {"job_id": job_id, "target_age": target_age}
    try:
        with get_storage().open(storage_key) as handle:
            data = handle.read()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="File ảnh tạo sinh đã mất.")
    fd, tmp = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        vector = np.asarray(get_embedder().embed(tmp), dtype=np.float64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Không embed được ảnh query: {exc}")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return vector, {"job_id": job_id, "target_age": target_age}


@router.post("/gallery/query", response_model=SearchRunDetail)
def query_gallery_view(body: GalleryQueryRequest):
    """Đối chiếu 1 query (crop hoặc ảnh tạo sinh) với gallery search."""
    _require_db()
    if (body.crop_id is None) == (body.generated_image_id is None):
        raise HTTPException(
            status_code=400,
            detail="Chỉ định đúng một trong crop_id / generated_image_id.")
    snapshot = gallery_service.get_snapshot()
    if snapshot is None:
        raise HTTPException(
            status_code=400,
            detail="Chưa có gallery snapshot — gọi POST /api/gallery/rebuild trước.")
    if body.crop_id is not None:
        query_vector = _query_vector_from_crop(body.crop_id)
        query_kind, query_crop_id = "crop", body.crop_id
        generation_job_id, extra_params = None, {}
    else:
        assert body.generated_image_id is not None
        query_vector, gen_info = _query_vector_from_generated(body.generated_image_id)
        query_kind, query_crop_id = "generated_set", None
        generation_job_id = gen_info["job_id"]
        extra_params = {"generated_image_id": body.generated_image_id,
                        "target_age": gen_info["target_age"]}
    try:
        hits = gallery_service.search_gallery(query_vector, k=max(body.top_k * 3, 10))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    model_name, model_version, preproc = current_embed_triple()
    with get_pool().connection() as conn:
        if body.scope_source_id is not None:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, f.source_id FROM face_media.face_crops c
                    JOIN face_media.face_detections d ON d.id = c.detection_id
                    JOIN face_media.frames f ON f.id = d.frame_id
                    WHERE c.id = ANY(%s)
                    """,
                    ([h["crop_id"] for h in hits],))
                source_of = {r[0]: r[1] for r in cur.fetchall()}
            hits = [h for h in hits
                    if source_of.get(h["crop_id"]) == body.scope_source_id]
        hits = hits[:body.top_k]
        run_id = repo.create_search_run(
            conn, query_kind=query_kind, query_crop_id=query_crop_id,
            scope_source_id=body.scope_source_id,
            generation_job_id=generation_job_id,
            model_name=model_name, model_version=model_version,
            preprocessing_version=preproc,
            index_version=snapshot["key"], threshold=body.threshold,
            parameters={"top_k": body.top_k, **extra_params})
        for rank, hit in enumerate(hits, start=1):
            repo.add_search_result(
                conn, run_id=run_id, candidate_crop_id=hit["crop_id"],
                best_generated_image_id=body.generated_image_id,
                score=hit["score"], rank=rank,
                accepted_by_threshold=bool(hit["score"] >= body.threshold))
        detail = repo.get_search_run_detail(conn, run_id)
    return _detail_to_schema(detail)


def _detail_to_schema(detail: dict) -> SearchRunDetail:
    storage = get_storage()
    results = []
    for item in detail["results"]:
        results.append(SearchResultItem(
            result_id=item["result_id"],
            candidate_crop_id=item["candidate_crop_id"],
            crop_key=item["crop_key"],
            crop_url=storage.get_access_url(item["crop_key"]),
            best_generated_image_id=item["best_generated_image_id"],
            score=item["score"], rank=item["rank"],
            accepted_by_threshold=item["accepted_by_threshold"],
            human_confirmed=item["human_confirmed"],
            frame_index=item["frame_index"], offset_ms=item["offset_ms"],
            captured_at=item["captured_at"], source_id=item["source_id"],
            source_kind=item["source_kind"], camera_id=item["camera_id"]))
    return SearchRunDetail(
        run_id=detail["run_id"], query_kind=detail["query_kind"],
        query_crop_id=detail["query_crop_id"],
        scope_source_id=detail["scope_source_id"],
        generation_job_id=detail["generation_job_id"],
        model_name=detail["model_name"], model_version=detail["model_version"],
        preprocessing_version=detail["preprocessing_version"],
        index_version=detail["index_version"], threshold=detail["threshold"],
        parameters=detail["parameters"], created_at=detail["created_at"],
        results=results)

@router.get("/search-runs")
def list_search_runs_view(
    scope_source_id: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """T12: lịch sử lượt đối chiếu (phân trang, lọc theo nguồn)."""
    _require_db()
    with get_pool().connection() as conn:
        items, total = repo.list_search_runs(
            conn, scope_source_id=scope_source_id,
            limit=max(1, min(limit, 200)), offset=max(0, offset))
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/search-runs/{run_id}", response_model=SearchRunDetail)
def get_search_run_view(run_id: str):
    """T11: mở kết quả cũ sau restart (truy ngược nguồn quan sát + query)."""
    _require_db()
    try:
        with get_pool().connection() as conn:
            return _detail_to_schema(repo.get_search_run_detail(conn, run_id))
    except repo.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/search-runs/confirm")
def confirm_result_view(body: ConfirmRequest):
    """T11: xác nhận con người (không đồng nghĩa similarity cao = đúng người)."""
    _require_db()
    try:
        with get_pool().connection() as conn:
            return repo.confirm_search_result(
                conn, result_id=body.result_id, confirmed=body.confirmed)
    except repo.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
