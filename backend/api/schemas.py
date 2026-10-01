from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FaceBox(BaseModel):
    index: int
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")
    det_score: float


class UploadResponse(BaseModel):
    session_id: str
    faces: List[FaceBox]
    # T04: lineage bền vững (None khi DB không sẵn sàng — client không được
    # coi thiếu IDs là lỗi, luồng legacy vẫn dùng session_id).
    source_id: Optional[str] = None
    detection_ids: List[str] = Field(default_factory=list)


class SelectFaceRequest(BaseModel):
    selected_idx: int


class SelectFaceResponse(BaseModel):
    warnings: List[str]
    cropped_preview_url: str
    # T05: revision crop đã áp dụng (None khi DB không sẵn sàng).
    crop_id: Optional[str] = None


class RestorePreviewRequest(BaseModel):
    mode: Literal["auto", "manual"] = "auto"
    padding_enabled: bool = True
    white_balance_enabled: bool = True
    click_x: Optional[int] = None
    click_y: Optional[int] = None
    fidelity_weight: float = 0.7


class RestorePreviewResponse(BaseModel):
    preview_url: str
    wb_info: Optional[Dict[str, Any]] = None


class ApplyRestoreRequest(BaseModel):
    mode: Literal["auto", "manual"] = "auto"
    use_restored: bool = True
    padding_enabled: bool = True
    white_balance_enabled: bool = True
    click_x: Optional[int] = None
    click_y: Optional[int] = None
    fidelity_weight: float = 0.7
    cropped_preview_url: Optional[str] = None


class ResolveAgeRequest(BaseModel):
    mode: Literal["manual", "mivolo"]
    manual_age: Optional[int] = Field(None, ge=0, le=120)
    gender_word: Literal["man", "woman"] = "man"
    photo_year: Optional[int] = Field(None, ge=1900)


class ResolveAgeResponse(BaseModel):
    initial_age: int
    gender_word: str
    warning_text: Optional[str] = None


class RunPipelineRequest(BaseModel):
    gallery_dir: Optional[str] = None
    photo_year: Optional[int] = Field(None, ge=1900)


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "done", "error"]
    stage: str
    error_message: Optional[str] = None


class GeneratedVariant(BaseModel):
    """T06: một biến thể tạo sinh có ID bền vững (nhiều variant cùng tuổi)."""
    id: str
    target_age: int
    variant_index: int = 0
    seed: Optional[int] = None
    image_url: str


class JobResult(BaseModel):
    job_id: str
    status: Literal["done", "error"]
    edited_images: Dict[int, str] = Field(default_factory=dict, description="Age -> image URL")
    age_scores: Dict[int, float] = Field(default_factory=dict, description="Age -> ID score for top identity")
    final_scores: Dict[str, float] = Field(default_factory=dict)
    accepted: bool = False
    top_identity: str = ""
    top_score: float = 0.0
    best_age: Optional[int] = None
    matched_gallery_image: Optional[str] = None
    pipeline_params: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    # T06: danh sách biến thể có ID (adapter giữ edited_images cho UI cũ).
    variants: List[GeneratedVariant] = Field(default_factory=list)


class GenerationJobDetail(BaseModel):
    """T06: xem lại job sau restart (đọc từ DB, không phụ thuộc RAM)."""
    job_id: str
    input_crop_id: str
    status: str
    model_name: str
    model_version: str
    initial_age: Optional[int] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    variants: List[GeneratedVariant] = Field(default_factory=list)


class VideoFaceMatch(BaseModel):
    face_image_url: str
    frame_index: int
    timestamp_sec: float
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")
    det_score: float
    best_age: int
    best_age_image_url: str
    score: float


class VideoVerifyResponse(BaseModel):
    job_id: str
    frames_sampled: int
    faces_found: int
    best_match: Optional[VideoFaceMatch] = None
    matches: List[VideoFaceMatch] = Field(default_factory=list)
    conditions: Dict[str, int] = Field(
        default_factory=dict,
        description="Thong ke dieu kien frame (normal/night/rain/glare/fog/backlight/blur)",
    )
    # T08/T11: nguồn search độc lập + lượt search đã lưu (None ở luồng legacy).
    source_id: Optional[str] = None
    search_run_id: Optional[str] = None
    processing: Dict[str, Any] = Field(
        default_factory=dict,
        description="{frames_total, duration_sec, truncated} — phạm vi đã xử lý",
    )


# ---------------- T07/T08: nạp gallery search ----------------

class SearchSourceItem(BaseModel):
    source_id: str
    purpose: str
    kind: str
    original_asset_id: Optional[str] = None
    camera_id: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    created_at: str
    # P0: chính sách lưu ('full' | 'crop_only'); camera mới luôn crop_only.
    storage_policy: str = "full"


class SourceListResponse(BaseModel):
    items: List[SearchSourceItem] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class SourceCropItem(BaseModel):
    crop_id: str
    crop_key: str
    crop_url: Optional[str] = None
    method: str
    bbox: List[float]
    det_score: float
    frame_index: int
    offset_ms: int
    captured_at: Optional[str] = None
    frame_id: str
    # P0 viewer: luồng camera crop-only không có file toàn khung.
    frame_available: bool = True
    frame_url: Optional[str] = None


class CropListResponse(BaseModel):
    items: List[SourceCropItem] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class IngestedCrop(BaseModel):
    crop_id: str
    crop_key: str
    url: str
    bbox: List[float]
    det_score: float


class ImageIngestItem(BaseModel):
    filename: str
    status: Literal["done", "error", "no_face"]
    source_id: Optional[str] = None
    faces_found: int = 0
    crops: List[IngestedCrop] = Field(default_factory=list)
    conditions: Dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None


class ImageIngestResponse(BaseModel):
    items: List[ImageIngestItem] = Field(default_factory=list)


class VideoIngestResponse(BaseModel):
    source_id: str
    run_id: str
    status: str


class IngestionRunStatus(BaseModel):
    run_id: str
    source_id: str
    status: str
    fps_target: float
    max_frames: int
    frames_sampled: int
    faces_found: int
    frames_total: Optional[int] = None
    duration_sec: Optional[float] = None
    truncated: bool = False
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None


# ---------------- T09: camera ----------------

class CameraCreate(BaseModel):
    name: str
    location: Optional[str] = None
    # Tham chiếu secret (tên biến môi trường giữ URL), KHÔNG gửi URL/mật khẩu.
    connection_secret_ref: Optional[str] = None


class CameraInfo(BaseModel):
    camera_id: str
    name: str
    location: Optional[str] = None
    has_connection_ref: bool = False
    created_at: str


class CameraListResponse(BaseModel):
    items: List[CameraInfo] = Field(default_factory=list)
    total: int = 0


class CaptureStartRequest(BaseModel):
    sample_fps: float = Field(1.0, gt=0, le=10)
    max_frames: int = Field(900, gt=0, le=100000)
    capture_seconds: Optional[float] = Field(None, gt=0)


class CaptureSessionStatus(BaseModel):
    source_id: str
    camera_id: str
    run_id: str
    status: str
    frames_sampled: int = 0
    faces_found: int = 0
    dropped_frames: int = 0
    reconnects: int = 0
    error_message: Optional[str] = None


# ---------------- P2: identities ----------------

class LinkTrackletRequest(BaseModel):
    accept_threshold: Optional[float] = Field(None, ge=-1, le=1)
    margin: Optional[float] = Field(None, ge=0, le=2)
    site: str = "default"
    calibrated: bool = False


class LinkResult(BaseModel):
    action: str
    global_id: Optional[str] = None
    assignment_id: Optional[str] = None
    reason: Optional[str] = None
    margin: Optional[float] = None


class IdentityInfo(BaseModel):
    identity_id: str
    status: str
    revision: int
    last_seen_at: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str


class IdentityListResponse(BaseModel):
    items: List[IdentityInfo] = Field(default_factory=list)
    total: int = 0


class MergeRequest(BaseModel):
    winner_id: str
    loser_id: str
    reason: str = "manual_merge"


class SplitRequest(BaseModel):
    assignment_id: str
    reason: str = "manual_split"


class TopologyItem(BaseModel):
    site: str = "default"
    camera_a: Optional[str] = None
    camera_b: Optional[str] = None
    travel_sec_min: Optional[float] = None
    travel_sec_typical: Optional[float] = None
    travel_sec_max: Optional[float] = None
    overlapping: bool = False
    version: int = 1
    note: Optional[str] = None


# ---------------- P4: retention/ops ----------------

class RetentionPolicyItem(BaseModel):
    scope: str
    ttl_days: int
    quota_bytes: Optional[int] = None
    enabled: bool = False
    reason: Optional[str] = None


class RetentionPolicyUpdate(BaseModel):
    ttl_days: Optional[int] = Field(None, gt=0)
    quota_bytes: Optional[int] = Field(None, gt=0)
    enabled: Optional[bool] = None
    reason: Optional[str] = None


class GCRunRequest(BaseModel):
    batch: int = Field(200, ge=1, le=2000)
    dry_run: bool = True


# ---------------- T10/T11: gallery + search runs ----------------

class GallerySnapshotInfo(BaseModel):
    key: str
    model_name: str
    model_version: str
    preprocessing_version: str
    size: int
    skipped: int = 0
    built_at: str
    # P3: version + watermark/phạm vi index (công bố rõ ràng).
    version: int = 1
    watermark_to: Optional[str] = None
    scope: Dict[str, Any] = Field(default_factory=dict)


class GalleryQueryRequest(BaseModel):
    crop_id: Optional[str] = None
    generated_image_id: Optional[str] = None
    scope_source_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=50)
    threshold: float = Field(0.6, ge=0, le=1)


class SearchResultItem(BaseModel):
    result_id: str
    candidate_crop_id: str
    crop_key: str
    crop_url: Optional[str] = None
    best_generated_image_id: Optional[str] = None
    score: float
    rank: int
    accepted_by_threshold: bool = False
    human_confirmed: bool = False
    frame_index: int
    offset_ms: int
    captured_at: Optional[str] = None
    source_id: str
    source_kind: str
    camera_id: Optional[str] = None


class SearchRunDetail(BaseModel):
    run_id: str
    query_kind: str
    query_crop_id: Optional[str] = None
    scope_source_id: Optional[str] = None
    generation_job_id: Optional[str] = None
    model_name: str
    model_version: str
    preprocessing_version: str
    index_version: str
    threshold: float
    parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    results: List[SearchResultItem] = Field(default_factory=list)


class ConfirmRequest(BaseModel):
    result_id: str
    confirmed: bool
