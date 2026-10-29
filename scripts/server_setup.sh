#!/bin/bash
# server_setup.sh - Chay TREN GPU SERVER sau khi da rsync xong
# Cach dung: bash server_setup.sh [--backend-only] [--stop]
#
# Script nay duoc chay tu xa qua SSH hoac copy len server roi chay truc tiep.

set -e

REMOTE_DIR="${REMOTE_DIR:-/workspace/NCKH}"
BACKEND_PORT="${BACKEND_PORT:-8080}"
LOG_DIR="$REMOTE_DIR/logs"

# ============================================================
# Parse args
# ============================================================
BACKEND_ONLY=false
STOP=false
for arg in "$@"; do
    case $arg in
        --backend-only) BACKEND_ONLY=true ;;
        --stop)         STOP=true ;;
    esac
done

cd "$REMOTE_DIR"
mkdir -p "$LOG_DIR" outputs/media outputs/jobs outputs/generation_only

# ============================================================
# STOP
# ============================================================
if [ "$STOP" = "true" ]; then
    echo "==> Dung tat ca server..."
    screen -S nckh_backend -X quit 2>/dev/null || true
    screen -S nckh_frontend -X quit 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    pkill -f "python -m backend" 2>/dev/null || true
    pkill -f "python -m http.server" 2>/dev/null || true
    echo "Da dung."
    exit 0
fi

# ============================================================
# Kiem tra GPU
# ============================================================
echo "==> Kiem tra GPU..."
source venv/bin/activate || { python3 -m venv venv && source venv/bin/activate; }
python -c "
import torch
cuda = torch.cuda.is_available()
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {cuda}')
if cuda:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# ============================================================
# Tao .env neu chua co
# ============================================================
if [ ! -f .env ]; then
    echo "==> Tao file .env (local storage mode, khong can DB)..."
    cp .env.example .env
    sed -i 's|^DATABASE_URL=.*|DATABASE_URL=|' .env
    sed -i 's|^#.*STORAGE_BACKEND=.*|STORAGE_BACKEND=local|' .env
    # Them neu chua co
    grep -q "^STORAGE_BACKEND=" .env || echo "STORAGE_BACKEND=local" >> .env
    grep -q "^MEDIA_ROOT=" .env || echo "MEDIA_ROOT=outputs/media" >> .env
fi

# ============================================================
# Dung server cu neu dang chay
# ============================================================
echo "==> Dung server cu (neu co)..."
screen -S nckh_backend -X quit 2>/dev/null || true
screen -S nckh_frontend -X quit 2>/dev/null || true
sleep 1

# ============================================================
# Khoi dong FastAPI Backend
# ============================================================
echo "==> Khoi dong FastAPI backend (port $BACKEND_PORT)..."
screen -dmS nckh_backend bash -c "
    cd $REMOTE_DIR
    source venv/bin/activate
    export BACKEND_PORT=$BACKEND_PORT
    # HF offline - dung checkpoint local, khong download
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo '[backend] Khoi dong...'
    python -m backend 2>&1 | tee $LOG_DIR/backend.log
"

sleep 3

# Kiem tra backend da chay chua
if curl -s "http://localhost:${BACKEND_PORT}/api/health" > /dev/null 2>&1; then
    echo "=== Backend ONLINE tai http://localhost:${BACKEND_PORT} ==="
elif screen -list | grep -q "nckh_backend"; then
    echo "=== Backend dang khoi dong (screen session ton tai)... ==="
    echo "    Xem log: screen -r nckh_backend"
else
    echo "!!! Backend khoi dong that bai! Xem log:"
    tail -30 "$LOG_DIR/backend.log" 2>/dev/null || echo "Khong co log."
    exit 1
fi

# ============================================================
# Serve frontend (desktop/dist) neu co
# ============================================================
if [ "$BACKEND_ONLY" = "false" ] && [ -d "desktop/dist" ]; then
    FRONTEND_PORT=3000
    echo "==> Serve React frontend tai port $FRONTEND_PORT..."
    screen -dmS nckh_frontend bash -c "
        cd $REMOTE_DIR/desktop/dist
        python3 -m http.server $FRONTEND_PORT --bind 127.0.0.1 2>&1 | tee $LOG_DIR/frontend.log
    "
    sleep 1
    echo "=== Frontend tai http://localhost:${FRONTEND_PORT} ==="
    echo "    (Can SSH tunnel: -L ${FRONTEND_PORT}:localhost:${FRONTEND_PORT})"
fi

# ============================================================
# Tong ket
# ============================================================
echo ""
echo "============================================================"
echo "  NCKH GPU Demo - Dang chay"
echo "============================================================"
echo "  Backend API:  http://localhost:${BACKEND_PORT}"
echo "  Swagger UI:   http://localhost:${BACKEND_PORT}/docs"
echo "  Health:       http://localhost:${BACKEND_PORT}/api/health"
if [ -d "desktop/dist" ]; then
echo "  Frontend:     http://localhost:3000"
fi
echo ""
echo "  Xem log backend:  screen -r nckh_backend"
echo "  Xem log frontend: screen -r nckh_frontend"
echo "  Dung tat ca:      bash server_setup.sh --stop"
echo "============================================================"
