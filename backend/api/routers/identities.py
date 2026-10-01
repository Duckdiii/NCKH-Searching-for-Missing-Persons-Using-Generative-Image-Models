"""P2 — API duy trì ID xuyên camera (giả thuyết, có revision).

- POST /api/tracklets/{id}/link: liên kết 1 tracklet → global ID.
- GET /api/identities + /{id}/timeline: xem ID và timeline có khoảng trống.
- POST /api/identities/merge + /split: sửa gán nhầm (tách/gộp có revision).
- GET/POST /api/topology: gợi ý mềm không-thời gian (version hóa).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.api import identity_link as link
from backend.api import repositories as repo
from backend.api.database import get_pool
from backend.api.persistence import db_ping
from backend.api.schemas import (
    IdentityInfo,
    IdentityListResponse,
    LinkResult,
    LinkTrackletRequest,
    MergeRequest,
    SplitRequest,
    TopologyItem,
)

router = APIRouter(prefix="/api", tags=["identities"])


def _require_db() -> None:
    if not db_ping():
        raise HTTPException(
            status_code=503, detail="Database face_media không sẵn sàng.")


@router.post("/tracklets/{tracklet_id}/link", response_model=LinkResult)
def link_tracklet_view(tracklet_id: str, body: LinkTrackletRequest):
    """Liên kết 1 tracklet đã persist (lấy members từ crop/embedding của nó)."""
    _require_db()
    cfg = link.LinkConfig(
        accept_threshold=(body.accept_threshold
                          if body.accept_threshold is not None
                          else link.UNCALIBRATED_ACCEPT),
        margin=(body.margin if body.margin is not None else link.UNCALIBRATED_MARGIN),
        calibrated=bool(body.calibrated),
    )
    try:
        with get_pool().connection() as conn:
            # Members: embedding + quality của từng crop thuộc tracklet.
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e."values", e.id, c.id,
                           COALESCE((d.quality->>'det_score')::float, 0.5)
                    FROM face_media.face_detections d
                    JOIN face_media.face_crops c ON c.detection_id = d.id
                    JOIN face_media.face_embeddings e ON e.crop_id = c.id
                    WHERE (d.tracklet_id = %s OR d.track_id = %s)
                      AND e.model_name = %s AND e.model_version = %s
                      AND e.preprocessing_version = %s
                    LIMIT 10
                    """,
                    (tracklet_id, tracklet_id, cfg.model_name,
                     cfg.model_version, cfg.preprocessing_version),
                )
                rows = cur.fetchall()
            # Fallback khi preprocessing_version trống (triple hiện hành khác).
            if not rows and not cfg.preprocessing_version:
                from backend.api.gallery import current_embed_triple
                m, mv, pv = current_embed_triple()
                cfg.model_name, cfg.model_version, cfg.preprocessing_version = m, mv, pv
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT e."values", e.id, c.id,
                               COALESCE((d.quality->>'det_score')::float, 0.5)
                        FROM face_media.face_detections d
                        JOIN face_media.face_crops c ON c.detection_id = d.id
                        JOIN face_media.face_embeddings e ON e.crop_id = c.id
                        WHERE (d.tracklet_id = %s OR d.track_id = %s)
                          AND e.model_name = %s AND e.model_version = %s
                          AND e.preprocessing_version = %s
                        LIMIT 10
                        """,
                        (tracklet_id, tracklet_id, m, mv, pv),
                    )
                    rows = cur.fetchall()
            if not rows:
                raise HTTPException(
                    status_code=400,
                    detail="Tracklet không có embedding để tạo descriptor.")
            members = [{"embedding": np.asarray(r[0], dtype=np.float64),
                        "embedding_id": r[1], "crop_id": r[2],
                        "quality": float(r[3] or 0.5)} for r in rows]
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT camera_id FROM face_media.tracklets WHERE id = %s",
                        (tracklet_id,))
                    trow = cur.fetchone()
                camera_id = trow[0] if trow else None
            except Exception:
                camera_id = None
            descriptor = link.tracklet_descriptor(members)
            result = link.link_tracklet(
                conn, tracklet_id=tracklet_id, descriptor=descriptor,
                camera_id=camera_id,
                at_time=datetime.now(timezone.utc),
                config=cfg, site=body.site)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return LinkResult(action=result.get("action", "unresolved"),
                      global_id=result.get("global_id"),
                      assignment_id=result.get("assignment_id"),
                      reason=str(result.get("reason", ""))[:500],
                      margin=result.get("margin"))


@router.get("/identities", response_model=IdentityListResponse)
def list_identities_view(status: str = None, limit: int = 50, offset: int = 0):
    _require_db()
    try:
        with get_pool().connection() as conn:
            items, total = repo.list_identities(
                conn, status=status, limit=max(1, min(limit, 200)),
                offset=max(0, offset))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return IdentityListResponse(
        items=[IdentityInfo(**i) for i in items], total=total)


@router.get("/identities/{identity_id}/timeline")
def identity_timeline_view(identity_id: str, gap_sec: float = 600.0):
    _require_db()
    try:
        from backend.api.storage import get_storage
        storage = get_storage()
        with get_pool().connection() as conn:
            repo.get_identity(conn, identity_id)
            timeline = repo.get_timeline(conn, identity_id, gap_sec=gap_sec)
            for seg in timeline.get("segments", []):
                key = seg.get("evidence_crop_key")
                seg["evidence_crop_url"] = (
                    storage.get_access_url(key) if key else None)
            return timeline
    except repo.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/identities/merge")
def merge_identities_view(body: MergeRequest):
    _require_db()
    try:
        with get_pool().connection() as conn:
            return link.merge_identities(
                conn, winner_id=body.winner_id, loser_id=body.loser_id,
                reason=body.reason)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/identities/split")
def split_identity_view(body: SplitRequest):
    _require_db()
    try:
        with get_pool().connection() as conn:
            return link.split_identity(
                conn, assignment_id=body.assignment_id, reason=body.reason)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/topology")
def list_topology_view(site: str = "default"):
    _require_db()
    with get_pool().connection() as conn:
        return {"site": site, "links": repo.get_topology(conn, site=site)}


@router.post("/topology")
def add_topology_view(body: TopologyItem):
    _require_db()
    try:
        with get_pool().connection() as conn:
            topo_id = repo.upsert_topology(
                conn, site=body.site, camera_a=body.camera_a,
                camera_b=body.camera_b, travel_sec_min=body.travel_sec_min,
                travel_sec_typical=body.travel_sec_typical,
                travel_sec_max=body.travel_sec_max,
                overlapping=body.overlapping, version=body.version,
                note=body.note)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"topology_id": topo_id}
