#!/bin/bash
# Bật backend (cổng 8000) + frontend (cổng 1420) trong tmux. Dừng: bash deploy/vast/stop.sh
cd "$(dirname "$0")/../.."
APP=$(pwd)
mkdir -p outputs

tmux has-session -t backend 2>/dev/null || tmux new -d -s backend \
  "source /venv/main/bin/activate && cd '$APP' && BACKEND_PORT=8000 PYTHONUNBUFFERED=1 python -m backend.api.main 2>&1 | tee outputs/backend.log"
tmux has-session -t frontend 2>/dev/null || tmux new -d -s frontend \
  "cd '$APP' && python3 deploy/vast/serve_frontend.py desktop/dist 2>&1 | tee outputs/frontend.log"

echo "Đang chờ backend nạp mô hình (lần đầu 1-3 phút)..."
for i in $(seq 1 180); do
  if curl -fs http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "Backend sẵn sàng sau ${i}s."
    break
  fi
  if ! tmux has-session -t backend 2>/dev/null; then
    echo "Backend đã thoát — xem outputs/backend.log"; tail -30 outputs/backend.log; exit 1
  fi
  sleep 1
done
curl -fs http://127.0.0.1:8000/api/health >/dev/null || { echo "Backend chưa phản hồi — xem: tail -f outputs/backend.log"; exit 1; }

cat <<EOF

Demo đang chạy trên máy vast.
Trên máy tính của bạn, mở tunnel (thay PORT/IP bằng lệnh SSH của instance):
  ssh -p PORT root@IP -L 1420:localhost:1420 -L 8000:localhost:8000
rồi mở trình duyệt:  http://localhost:1420

Log:  tail -f outputs/backend.log      Dừng:  bash deploy/vast/stop.sh
EOF
