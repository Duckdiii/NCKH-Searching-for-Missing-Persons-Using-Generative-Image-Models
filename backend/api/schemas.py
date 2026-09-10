from typing import Literal, Optional, List, Dict
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


class ResolveAgeRequest(BaseModel):
    mode: Literal["manual", "mivolo"]
    manual_age: Optional[int] = Field(None, ge=0, le=120)
    gender_word: Literal["man", "woman"] = "man"


class ResolveAgeResponse(BaseModel):
    initial_age: int
    gender_word: str
    warning_text: Optional[str] = None


class RunPipelineRequest(BaseModel):
    gallery_dir: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["running", "done", "error"]
    stage: str
    error_message: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    status: Literal["done", "error"]
    edited_images: Dict[int, str] = Field(default_factory=dict, description="Age -> image URL")
    final_scores: Dict[str, float] = Field(default_factory=dict)
    accepted: bool = False
    top_identity: str = ""
    top_score: float = 0.0
    error_message: Optional[str] = None
