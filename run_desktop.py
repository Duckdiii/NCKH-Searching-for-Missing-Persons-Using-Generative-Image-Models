"""
Script khởi động trọn gói ứng dụng Desktop từ CLI của VS Code:
1. Khởi động Backend FastAPI (Port 8000).
2. Đợi Backend sẵn sàng qua health check (timeout 60s, hiển thị tiến trình).
3. Khởi động Frontend và tự động mở Cửa sổ Ứng dụng độc lập (App Window - không URL bar, không tab web).
4. Tự động dọn dẹp và tắt sạch tiến trình con khi nhấn Ctrl+C, bảo vệ VRAM GPU.
"""

import os
import sys
import time
import subprocess
import signal
import webbrowser
import urllib.request
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DESKTOP_DIR = os.path.join(REPO_DIR, "desktop")
PYTHON_EXE = sys.executable

BACKEND_PORT = 8000
FRONTEND_PORT = 1420
APP_URL = f"http://localhost:{FRONTEND_PORT}"
LOG_DIR = os.path.join(REPO_DIR, "outputs")
os.makedirs(LOG_DIR, exist_ok=True)
BACKEND_LOG_PATH = os.path.join(LOG_DIR, "backend_startup.log")


def is_backend_alive() -> bool:
    health_url = f"http://127.0.0.1:{BACKEND_PORT}/api/health"
    try:
        with urllib.request.urlopen(health_url, timeout=1) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    return False


def wait_for_backend(proc, timeout_sec=60) -> bool:
    start_t = time.time()
    while time.time() - start_t < timeout_sec:
        # Kiểm tra xem tiến trình backend có bị crash sớm không
        if proc and proc.poll() is not None:
            return False

        if is_backend_alive():
            elapsed = time.time() - start_t
            print(f"\r  ✅ Backend API server đã sẵn sàng! ({elapsed:.1f}s)                ", flush=True)
            return True

        elapsed = int(time.time() - start_t)
        print(f"\r  • Đang nạp mô hình & khởi động server AI... ({elapsed}s/{timeout_sec}s)", end="", flush=True)
        time.sleep(1)

    return False


def open_app_window(url: str):
    """Mở giao diện dưới dạng Cửa sổ Ứng dụng Desktop độc lập (App Mode - không URL bar, không tabs)."""
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]

    for p in edge_paths + chrome_paths:
        if os.path.exists(p):
            print(f"[Launcher] Mở Cửa sổ Ứng dụng Desktop riêng biệt ({os.path.basename(p)} --app)...")
            subprocess.Popen([p, f"--app={url}", "--window-size=1280,860"])
            return

    # Fallback nếu không tìm thấy browser hỗ trợ app mode
    print(f"[Launcher] Mở giao diện tại: {url}")
    webbrowser.open(url)


def main():
    print("=" * 70)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG MISSING PERSON SEARCH (FADING DESKTOP)")
    print("=" * 70)

    backend_proc = None
    frontend_proc = None

    # Kiểm tra xem backend đã chạy sẵn từ trước chưa
    if is_backend_alive():
        print(f"ℹ️ Backend API server đã đang chạy sẵn trên cổng {BACKEND_PORT}.")
    else:
        print(f"[1/3] Đang khởi động Backend API server trên cổng {BACKEND_PORT}...")
        env = os.environ.copy()
        env["BACKEND_PORT"] = str(BACKEND_PORT)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        log_file = open(BACKEND_LOG_PATH, "w", encoding="utf-8")
        backend_proc = subprocess.Popen(
            [PYTHON_EXE, "-m", "backend.api.main"],
            cwd=REPO_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

        ready = wait_for_backend(backend_proc, timeout_sec=60)
        if not ready:
            print("\n❌ Lỗi: Backend không khởi động được!")
            if backend_proc and backend_proc.poll() is not None:
                print(f"  • Tiến trình Python thoát với mã lỗi: {backend_proc.returncode}")
            print(f"  • Chi tiết log xem tại: {BACKEND_LOG_PATH}")
            if os.path.exists(BACKEND_LOG_PATH):
                with open(BACKEND_LOG_PATH, "r", encoding="utf-8") as f:
                    print("--- [Nội dung log backend] ---")
                    print(f.read().strip())
                    print("------------------------------")
            if backend_proc:
                backend_proc.kill()
            sys.exit(1)

    try:
        # 2. Khởi động Frontend Vite
        print(f"[2/3] Đang khởi động giao diện Desktop Frontend...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        frontend_proc = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=DESKTOP_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        time.sleep(2)

        # 3. Mở Cửa sổ Ứng dụng
        print(f"[3/3] Đang kích hoạt Cửa sổ Ứng dụng Desktop...")
        open_app_window(APP_URL)

        print("\n" + "=" * 70)
        print("✨ HỆ THỐNG ĐÃ SẴN SÀNG HOẠT ĐỘNG!")
        print(f"  • Cửa sổ ứng dụng đã được mở trên màn hình.")
        print(f"  • Nhấn Ctrl + C trên terminal này để dừng toàn bộ hệ thống.")
        print("=" * 70 + "\n")

        # Chờ người dùng nhấn Ctrl+C
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Launcher] Nhận tín hiệu dừng (Ctrl+C). Đang tắt các tiến trình...")
    finally:
        if frontend_proc:
            try:
                frontend_proc.terminate()
                frontend_proc.kill()
            except Exception:
                pass
        if backend_proc:
            try:
                backend_proc.terminate()
                backend_proc.kill()
            except Exception:
                pass
        print("🛑 Đã tắt sạch Backend và Frontend. GPU VRAM đã được giải phóng.")


if __name__ == "__main__":
    main()
