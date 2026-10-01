#!/usr/bin/env python3
"""Phục vụ frontend đã build (desktop/dist) trên cổng 1420 — thay `vite dev` khi demo trên vast.

- File tĩnh lấy từ dist/; route không tồn tại trả index.html (SPA).
- /outputs/* và /data/* chuyển tiếp sang backend 127.0.0.1:8000 (giống proxy trong vite.config.ts).
- Chỉ nghe 127.0.0.1: truy cập qua SSH tunnel, không mở ra internet.
"""
import http.server
import os
import sys
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "desktop/dist")
PORT = int(os.environ.get("FRONTEND_PORT", "1420"))
BACKEND = f"http://127.0.0.1:{os.environ.get('BACKEND_PORT', '8000')}"
PROXY_PREFIXES = ("/outputs/", "/data/")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_GET(self):
        if self.path.startswith(PROXY_PREFIXES):
            return self._proxy()
        path = self.translate_path(self.path.split("?", 1)[0])
        if not os.path.exists(path):
            self.path = "/index.html"
        return super().do_GET()

    def _proxy(self):
        try:
            with urllib.request.urlopen(BACKEND + self.path, timeout=30) as r:
                body = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/octet-stream"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            self.send_error(e.code)
        except Exception:
            self.send_error(502, "Backend không phản hồi")


if __name__ == "__main__":
    print(f"[frontend] phục vụ {ROOT} tại http://127.0.0.1:{PORT} (proxy /outputs,/data -> {BACKEND})")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
