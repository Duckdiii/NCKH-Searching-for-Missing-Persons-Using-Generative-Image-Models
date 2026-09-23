from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FaceBox(BaseModel):
    index: int
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")
    det_score: float


class UploadResponse(BaseModel):
    session_id: str
    faces: List[FaceBox]


class SelectFaceRequest(BaseModel):
    selected_idx: int


class SelectFaceResponse(BaseModel):
    warnings: List[str]
    cropped_preview_url: str


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
