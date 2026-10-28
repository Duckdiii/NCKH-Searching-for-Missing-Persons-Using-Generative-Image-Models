#!/bin/bash
# Cài đặt demo trên vast.ai (image PyTorch, RTX 3090). Chạy 1 lần sau khi giải nén:
#   cd /workspace/nckh_demo && bash deploy/vast/setup.sh
set -e
cd "$(dirname "$0")/../.."
APP=$(pwd)
source /venv/main/bin/activate

echo "== [1/5] Thư viện Python =="
if [ "${SKIP_PIP:-0}" = "1" ]; then
  echo "  (SKIP_PIP=1: dùng venv có sẵn, bỏ qua cài đặt)"
else
uv pip install -q setuptools wheel
# Giữ torch có sẵn của image. onnxruntime-gpu phải khớp CUDA của torch:
#   torch CUDA 12.x → onnxruntime-gpu 1.22 (bản CUDA 12); CUDA 13 → bản mới nhất (CUDA 13).
# Lệch bản → InsightFace âm thầm rơi về CPU (rất chậm trên vast: hàng trăm luồng ORT).
CUDA_MAJOR=$(python -c "import torch; print((torch.version.cuda or '12').split('.')[0])")
if [ "$CUDA_MAJOR" -ge 13 ]; then ORT_SPEC="onnxruntime-gpu"; else ORT_SPEC="onnxruntime-gpu==1.22.0"; fi
echo "  torch CUDA $CUDA_MAJOR → $ORT_SPEC"
grep -vE '^\s*(torch|torchvision|onnxruntime-gpu)\b' requirements.txt > /tmp/req_vast.txt
# Không dùng -q: lần trước uv treo im lặng; in log để biết kẹt ở gói nào.
uv pip install -r /tmp/req_vast.txt "$ORT_SPEC"
# setup.py của MiVOLO import pkg_resources — setuptools >= 80 đã bỏ module này.
uv pip install "setuptools<80"
uv pip install --no-deps --no-build-isolation "git+https://github.com/WildChlamydia/MiVOLO.git"
fi
python - <<'EOF'
# Tạo phiên thật trên GPU: thiếu thư viện CUDA thì ORT vẫn liệt kê CUDAExecutionProvider
# nhưng tự rơi về CPU khi tạo phiên.
import onnxruntime as o
try:
    o.preload_dlls()          # nạp cuBLAS/cuDNN từ gói nvidia-* đi kèm torch (ORT >= 1.21)
except Exception:
    pass
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
app.prepare(ctx_id=0, det_size=(256, 256))
prov = app.models["detection"].session.get_providers()
assert prov[0] == "CUDAExecutionProvider", f"InsightFace đang chạy trên {prov}: kiểm tra bản onnxruntime-gpu"
print("onnxruntime", o.__version__, "GPU OK:", prov)
EOF

echo "== [2/5] Cấu hình cho máy GPU 24GB =="
# runwayml/stable-diffusion-v1-5 đã bị gỡ khỏi HuggingFace -> dùng bản mirror chính thức.
sed -i 's#"runwayml/stable-diffusion-v1-5"#"stable-diffusion-v1-5/stable-diffusion-v1-5"#' configs/config.yaml
# Embedding trên GPU: host vast nhiều lõi nhưng container ít lõi -> ORT CPU tạo hàng trăm luồng, rất chậm.
sed -i 's|^  ctx_id: -1 .*|  ctx_id: 0  # [vast] GPU|' configs/config.yaml
grep -nE "pretrained_model_name_or_path|^  ctx_id" configs/config.yaml

echo "== [3/5] Tải trước Stable Diffusion 1.5 (VAE + text encoder, ~0.9GB) =="
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
    allow_patterns=["*.json", "*.txt", "tokenizer/*", "text_encoder/*.safetensors",
                    "vae/*.safetensors", "scheduler/*"],  # UNet gốc không cần: dùng checkpoints/specialized_unet
    ignore_patterns=["*fp16*", "*.ckpt", "*ema*"])
print("SD1.5 OK")
EOF
# InsightFace buffalo_l (~280MB, tải từ GitHub khá chậm): tải sẵn để request đầu tiên không phải chờ.
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l'); print('InsightFace buffalo_l OK')" 2>&1 | grep -v "KB/s\]"

echo "== [4/5] CodeFormer (làm nét ảnh — tùy chọn, thiếu thì backend tự bỏ qua) =="
if [ ! -d CodeFormer ]; then
  git clone -q https://github.com/sczhou/CodeFormer.git CodeFormer
  grep -rl "torchvision.transforms.functional_tensor" CodeFormer/basicsr | xargs -r sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/g'
  [ -f CodeFormer/basicsr/version.py ] || printf '__version__ = "1.4.2"\n__gitsha__ = "unknown"\n' > CodeFormer/basicsr/version.py
fi
# basicsr đi kèm CodeFormer import lpips (không có trong requirements chính).
python -c "import lpips" 2>/dev/null || uv pip install lpips
if [ ! -f CodeFormer/weights/CodeFormer/codeformer.pth ]; then
  # scripts/ không nằm cùng thư mục basicsr đi kèm -> cần PYTHONPATH trỏ về gốc CodeFormer
  (cd CodeFormer && PYTHONPATH=. python scripts/download_pretrained_models.py facelib      && PYTHONPATH=. python scripts/download_pretrained_models.py CodeFormer)      || echo "  ! tải weights CodeFormer lỗi — demo vẫn chạy, chỉ bỏ bước làm nét"
fi

echo "== [5/5] Kiểm tra =="
mkdir -p outputs
python - <<'EOF'
import os, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ")
for p in ["checkpoints/specialized_unet/diffusion_pytorch_model.safetensors",
          "checkpoints/mivolo/yolov8x_person_face.pt", "checkpoints/mivolo/mivolo_imdb.pth.tar",
          "data/test_gallery", "desktop/dist/index.html"]:
    print(("OK   " if os.path.exists(p) else "THIẾU ") + p)
EOF
echo "Xong. Chạy demo: bash deploy/vast/start.sh"
