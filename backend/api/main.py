import os
import socket
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from backend.api.routers import health, jobs, session

app = FastAPI(
    title="Missing Person Search API",
    description="FastAPI backend bọc quanh pipeline FADING cho desktop application",
    version="1.0.0"
)

# CORS middleware for Tauri webview and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs and data static directories
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

# Include API routers
app.include_router(health.router)
app.include_router(session.router)
app.include_router(jobs.router)


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
