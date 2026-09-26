# deploy_gpu_server.ps1
# Script deploy project len GPU server vast.ai va chay backend FastAPI + frontend
#
# SSH: ssh -p 20615 root@ssh1.vast.ai -L 8080:localhost:8080
#      (redirect port 8080 tu server ve localhost:8080 tren may local)
#
# Cach dung:
#   .\scripts\deploy_gpu_server.ps1          # rsync + install + start server
#   .\scripts\deploy_gpu_server.ps1 -SkipSync  # chi restart server (da sync roi)

param(
    [switch]$SkipSync,
    [switch]$StopServer
)

# ============================================================
# CAU HINH - chinh sua cho phu hop voi vast.ai instance cua ban
# ============================================================
$SSH_HOST     = "ssh1.vast.ai"
$SSH_PORT     = "20615"
$SSH_USER     = "root"
$REMOTE_DIR   = "/workspace/NCKH"          # thu muc tren server
$BACKEND_PORT = "8080"                      # port FastAPI (phai trung voi -L 8080:localhost:8080)
$LOCAL_ROOT   = $PSScriptRoot | Split-Path -Parent  # thu muc goc project

# ============================================================
# HELPER
# ============================================================
function Invoke-SSH {
    param([string]$Command)
    ssh -p $SSH_PORT "${SSH_USER}@${SSH_HOST}" $Command
}

function Write-Step {
    param([string]$Msg)
    Write-Host "`n==> $Msg" -ForegroundColor Cyan
}

# ============================================================
# STOP SERVER (neu co flag)
# ============================================================
if ($StopServer) {
    Write-Step "Dung server tren GPU..."
    Invoke-SSH "pkill -f 'uvicorn' || true; pkill -f 'python -m backend' || true"
    Write-Host "Server da dung." -ForegroundColor Green
    exit 0
}

# ============================================================
# STEP 1: SYNC FILES
# ============================================================
if (-not $SkipSync) {
    Write-Step "Dong bo code len server (rsync)..."

    # Kiem tra rsync co san hay khong
    $rsync = Get-Command rsync -ErrorAction SilentlyContinue
    if (-not $rsync) {
        Write-Host "CAUTION: rsync khong tim thay. Dung scp thay the..." -ForegroundColor Yellow
        # Tao thu muc tren server truoc
        Invoke-SSH "mkdir -p $REMOTE_DIR"

        # Cac thu muc can thiet (khong bao gom node_modules, .venv, __pycache__)
        $dirs = @("backend", "src", "configs", "desktop/dist", "outputs", "data")
        foreach ($d in $dirs) {
            $local = Join-Path $LOCAL_ROOT $d
            if (Test-Path $local) {
                Write-Host "  Copying $d ..." -ForegroundColor Gray
                scp -P $SSH_PORT -r "$local" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"
            }
        }

        # Copy cac file don le
        $files = @("app.py", "generation_only_app.py", "main.py", "requirements.txt", "requirements-generation-gpu.txt", ".env.example", "yolov8n.pt")
        foreach ($f in $files) {
            $local = Join-Path $LOCAL_ROOT $f
            if (Test-Path $local) {
                Write-Host "  Copying $f ..." -ForegroundColor Gray
                scp -P $SSH_PORT "$local" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"
            }
        }
    } else {
        # rsync: nhanh hon, chi copy file da thay doi, bo qua thu muc nang
        $rsync_excludes = @(
            "--exclude=.venv",
            "--exclude=.git",
            "--exclude=node_modules",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.pytest_cache",
            "--exclude=outputs/jobs",       # jobs output lon
            "--exclude=desktop/src-tauri",  # khong can Tauri binary
            "--exclude=desktop/node_modules"
        )

        rsync -avz --progress `
            -e "ssh -p $SSH_PORT" `
            @rsync_excludes `
            "${LOCAL_ROOT}/" `
            "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"
    }

    Write-Step "Dong bo checkpoints (co the mat nhieu thoi gian)..."
    $ckptLocal = Join-Path $LOCAL_ROOT "checkpoints"
    if (Test-Path $ckptLocal) {
        if ($rsync) {
            rsync -avz --progress `
                -e "ssh -p $SSH_PORT" `
                "--exclude=*.zip" `
                "${ckptLocal}/" `
                "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/checkpoints/"
        } else {
            scp -P $SSH_PORT -r "$ckptLocal" "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"
        }
    } else {
        Write-Host "  [SKIP] Khong tim thay checkpoints/ - can copy thu cong!" -ForegroundColor Yellow
    }
}

# ============================================================
# STEP 2: CAI DAT MOI TRUONG TREN SERVER
# ============================================================
Write-Step "Cai dat Python dependencies tren server GPU..."

$setup_cmd = @"
set -e
cd $REMOTE_DIR

# Tao venv neu chua co
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Nang cap pip
pip install --upgrade pip -q

# Cai PyTorch voi CUDA (cu124 phu hop T4/A10/RTX30xx tro len)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 -q

# Cai cac package con lai
pip install -r requirements.txt -q

# Cai MiVOLO (khong co tren PyPI)
pip install --no-deps "git+https://github.com/WildChlamydia/MiVOLO.git" -q 2>/dev/null || true

echo "=== Cai dat hoan tat ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
"@

Invoke-SSH $setup_cmd

# ============================================================
# STEP 3: CHAY FASTAPI BACKEND
# ============================================================
Write-Step "Khoi dong FastAPI backend tren port $BACKEND_PORT..."

$start_server_cmd = @"
set -e
cd $REMOTE_DIR
source venv/bin/activate

# Tao file .env neu chua co (dung local storage, khong can DB)
if [ ! -f .env ]; then
    cp .env.example .env
    # Dung local storage - khong can Supabase
    sed -i 's/^STORAGE_BACKEND=.*/STORAGE_BACKEND=local/' .env
    sed -i 's/^# MEDIA_ROOT=.*/MEDIA_ROOT=outputs\/media/' .env
    echo "DATABASE_URL=" >> .env
fi

mkdir -p outputs/media outputs/jobs logs

# Dung screen de chay ngam (screen -dmS ten_session lenh)
screen -dmS nckh_backend bash -c "
    source venv/bin/activate
    cd $REMOTE_DIR
    BACKEND_PORT=$BACKEND_PORT python -m backend 2>&1 | tee logs/backend.log
"

sleep 2

# Kiem tra server co chay khong
if screen -list | grep -q 'nckh_backend'; then
    echo '=== Backend dang chay trong screen session [nckh_backend] ==='
    echo '=== API: http://localhost:$BACKEND_PORT ==='
    echo '=== Log: screen -r nckh_backend ==='
else
    echo 'CAUTION: screen session khong tim thay! Kiem tra log:'
    cat logs/backend.log 2>/dev/null || true
fi
"@

Invoke-SSH $start_server_cmd

# ============================================================
# STEP 4: HUONG DAN TRUY CAP
# ============================================================
Write-Step "HOAN TAT! Huong dan truy cap:"
Write-Host @"

+------------------------------------------------------------------+
|  CACH TRUY CAP DEMO                                             |
+------------------------------------------------------------------+
|                                                                  |
|  1. Giu SSH tunnel dang chay:                                   |
|     ssh -p $SSH_PORT ${SSH_USER}@${SSH_HOST} -L ${BACKEND_PORT}:localhost:${BACKEND_PORT} -N  |
|                                                                  |
|  2. Mo trinh duyet va vao:                                      |
|     http://localhost:$BACKEND_PORT/docs    <- Swagger API UI   |
|     http://localhost:$BACKEND_PORT/api/health  <- Health check  |
|                                                                  |
|  3. Frontend (neu dung desktop/dist):                           |
|     Mo file: desktop/dist/index.html bang trinh duyet           |
|     (static HTML - se goi ve localhost:$BACKEND_PORT)          |
|                                                                  |
|  4. Xem log server:                                             |
|     ssh -p $SSH_PORT ${SSH_USER}@${SSH_HOST} "screen -r nckh_backend"  |
|                                                                  |
|  5. Dung server:                                                |
|     .\scripts\deploy_gpu_server.ps1 -StopServer                 |
+------------------------------------------------------------------+
"@ -ForegroundColor Green
