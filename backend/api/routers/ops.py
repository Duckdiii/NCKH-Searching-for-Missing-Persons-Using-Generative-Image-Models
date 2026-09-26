"""P4 — API vận hành: metrics, retention policies, GC (dry_run mặc định).

- GET /api/ops/metrics: RAM, queue/fps/tracks/crops, index watermark/lag, GC.
- GET/PUT /api/ops/retention/policies: xem/bật TTL theo scope (mặc định tắt).
- POST /api/ops/retention/plan: liệt kê hết TTL (không xóa).
- POST /api/ops/retention/gc: chạy GC ledger một lượt (dry_run mặc định true).
- POST /api/ops/retention/orphans: orphan GC (grace, dry_run mặc định true).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api import retention as ret
from backend.api.database import get_pool
from backend.api.persistence import db_ping
from backend.api.schemas import GCRunRequest, RetentionPolicyUpdate
from backend.api.storage import get_storage

router = APIRouter(prefix="/api/ops", tags=["ops"])


def _require_db() -> None:
    if not db_ping():
        raise HTTPException(
            status_code=503, detail="Database face_media không sẵn sàng.")


@router.get("/metrics")
def metrics_view():
    """Dashboard/soak metrics (phần không cần DB vẫn trả khi DB down)."""
    return ret.collect_metrics()


@router.get("/gpu")
def gpu_status_view():
    """Admission GPU chung: diffusion bận, tải camera, VRAM, hàng chờ."""
    from backend.api import gpu_admission
    return gpu_admission.status()


@router.get("/retention/policies")
def list_policies_view():
    _require_db()
    with get_pool().connection() as conn:
        policies = ret.get_policies(conn)
        quotas = ret.check_quotas(conn)
        for p in policies:
            q = quotas.get(p["scope"], {})
            p["usage"] = q
        return {"policies": policies}


@router.put("/retention/policies/{scope}")
def update_policy_view(scope: str, body: RetentionPolicyUpdate):
    _require_db()
    try:
        with get_pool().connection() as conn:
            ret.set_policy(conn, scope, ttl_days=body.ttl_days,
                           quota_bytes=body.quota_bytes,
                           enabled=body.enabled, reason=body.reason)
            return {"policies": ret.get_policies(conn)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/retention/plan")
def retention_plan_view(batch: int = 200):
    _require_db()
    with get_pool().connection() as conn:
        return ret.plan_expired(conn, batch=max(1, min(batch, 2000)))


@router.post("/retention/tombstone")
def tombstone_view(reason: str = "retention_ttl", batch: int = 200):
    """Ghi tombstone cho crop hết TTL (ẩn query trước khi GC xóa thật)."""
    _require_db()
    with get_pool().connection() as conn:
        plan = ret.plan_expired(conn, batch=max(1, min(batch, 2000)))
        n = 0 if not plan["crops"] else ret.tombstone_crops(
            conn, plan["crops"], reason)
        return {"candidates": len(plan["crops"]), "tombstoned": n}


@router.post("/retention/gc")
def gc_once_view(body: GCRunRequest):
    _require_db()
    with get_pool().connection() as conn:
        return ret.run_gc_once(conn, get_storage(), batch=body.batch,
                               dry_run=body.dry_run)


@router.post("/retention/orphans")
def orphans_view(body: GCRunRequest):
    _require_db()
    with get_pool().connection() as conn:
        return ret.orphan_gc(conn, get_storage(), batch=body.batch,
                             dry_run=body.dry_run)
