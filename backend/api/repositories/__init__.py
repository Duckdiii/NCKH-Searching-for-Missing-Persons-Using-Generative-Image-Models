"""T02 — Repository face_media (Postgres/Supabase).

Dùng query tham số hóa + transaction theo nghiệp vụ. Purpose luôn truyền rõ
theo endpoint; service không tin giá trị frontend để đưa dữ liệu search vào
reference (DB giữ FK ghép chặn cuối).
"""

from . import generation, history, media, search, validation
from .errors import ConflictError, NotFoundError, PurposeError
from .generation import (
    claim_idempotency_key,
    create_embedding,
    create_generated_image,
    create_generation_job,
    save_idempotency_result,
    set_job_status,
)
from .history import (
    get_session_record,
    list_generation_jobs,
    list_search_runs,
    reconcile_interrupted,
    upsert_session,
)
from .media import (
    create_asset,
    create_crop,
    create_detection,
    create_frame,
    create_source,
    get_crop_purpose,
    new_id,
)
from .search import (
    add_search_result,
    confirm_search_result,
    create_camera,
    create_ingestion_run,
    create_search_run,
    get_camera,
    get_ingestion_run,
    get_search_run_detail,
    list_cameras,
    list_source_crops,
    list_sources,
    update_ingestion_run,
)

__all__ = [
    "validation",
    "media",
    "generation",
    "search",
    "ConflictError",
    "NotFoundError",
    "PurposeError",
    "add_search_result",
    "claim_idempotency_key",
    "confirm_search_result",
    "create_asset",
    "create_camera",
    "create_crop",
    "create_detection",
    "create_embedding",
    "create_frame",
    "create_generated_image",
    "create_generation_job",
    "create_ingestion_run",
    "create_search_run",
    "create_source",
    "get_camera",
    "get_crop_purpose",
    "get_ingestion_run",
    "get_search_run_detail",
    "get_session_record",
    "list_cameras",
    "list_generation_jobs",
    "list_search_runs",
    "list_source_crops",
    "list_sources",
    "new_id",
    "reconcile_interrupted",
    "save_idempotency_result",
    "set_job_status",
    "update_ingestion_run",
    "upsert_session",
]
