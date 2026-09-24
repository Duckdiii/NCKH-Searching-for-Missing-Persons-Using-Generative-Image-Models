import os
import secrets
import socket
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from contextlib import asynccontextmanager

from backend.api.database import close_pool, init_pool
from backend.api.routers import cameras, health, jobs, search_runs, search_sources, session, video_verify


@asynccontextmanager
async def lifespan(app: FastAPI):
    # T01: mở pool lúc startup (thiếu DATABASE_URL vẫn cho app chạy để
    # endpoint không-DB hoạt động; lỗi cấu hình chỉ nêu tên biến, không in DSN).
    try:
        init_pool()
    except Exception as exc:
        print(f"[database] pool chưa khởi tạo: {exc}")
    # T12: job/run đang dở khi process chết → error 'interrupted'
    # (không tự chạy lại diffusion khi chưa có chiến lược resume).
    try:
        from backend.api import repositories as repo
        from backend.api.database import get_pool

        with get_pool().connection() as conn:
            counts = repo.reconcile_interrupted(conn)
        if counts["generation_jobs"] or counts["ingestion_runs"]:
            print(f"[database] reconcile interrupted: {counts}")
    except Exception as exc:
        print(f"[database] reconcile bỏ qua: {exc}")
    yield
    # Đóng pool lúc shutdown để không rò connection.
    try:
        close_pool()
    except Exception:
        pass


app = FastAPI(
    title="Missing Person Search API",
    description="FastAPI backend bọc quanh pipeline FADING cho desktop application",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Tauri webview and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# T12: mô hình truy cập. Mặc định local một người (không guard).
# Đặt FACE_MEDIA_API_KEY để yêu cầu header X-API-Key trên mọi /api/*
# (trừ health). So sánh constant-time; key sai/thiếu → 401, không lộ gì thêm.
@app.middleware("http")
async def _api_key_guard(request: Request, call_next):
    required = os.environ.get("FACE_MEDIA_API_KEY", "")
    path = request.url.path
    if required and path.startswith("/api") and not path.startswith("/api/health"):
        provided = request.headers.get("x-api-key", "")
        if not provided or not secrets.compare_digest(provided, required):
            return JSONResponse(
                status_code=401,
                content={"detail": "Thiếu hoặc sai API key."})
    return await call_next(request)


# Mount outputs and data static directories
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

# Include API routers
app.include_router(health.router)
app.include_router(session.router)
app.include_router(jobs.router)
app.include_router(video_verify.router)
app.include_router(search_sources.router)
app.include_router(cameras.router)
app.include_router(search_runs.router)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_server():
    port_env = os.environ.get("BACKEND_PORT")
    if port_env and port_env.isdigit():
        port = int(port_env)
    else:
        port = 8000

    # Dòng đầu tiên in ra stdout để Tauri Rust runner đọc được port
    print(f"PORT:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    run_server()
