import dataclasses
from typing import Any, Dict, List, Optional
import numpy as np


@dataclasses.dataclass
class SessionState:
    session_id: str
    image_bgr: np.ndarray
    image_rgb: np.ndarray
    faces: List[Any]
    chosen_face: Optional[Any] = None
    cropped_path: Optional[str] = None
    gender_word: str = "man"
    initial_age: Optional[int] = None
    file_name: str = ""


@dataclasses.dataclass
class JobState:
    job_id: str
    session_id: str
    status: str = "running"  # running, done, error
    stage: str = "pending"   # specialization, inversion, editing, search
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# In-memory stores indexed by UUID string
sessions: Dict[str, SessionState] = {}
jobs: Dict[str, JobState] = {}


def get_session(session_id: str) -> Optional[SessionState]:
    return sessions.get(session_id)


def save_session(session: SessionState) -> None:
    sessions[session.session_id] = session


def delete_session(session_id: str) -> None:
    sessions.pop(session_id, None)


def get_job(job_id: str) -> Optional[JobState]:
    return jobs.get(job_id)


def save_job(job: JobState) -> None:
    jobs[job.job_id] = job
