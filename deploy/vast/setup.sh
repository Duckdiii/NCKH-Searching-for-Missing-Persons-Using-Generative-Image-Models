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
# Giữ torch có sẵn của image (CUDA 12.x). onnxruntime-gpu 1.22 = bản build cho CUDA 12
# (bản mới nhất cần CUDA 13 -> InsightFace rơi về CPU, rất chậm trên vast).
grep -vE '^\s*(torch|torchvision|onnxruntime-gpu)\b' requirements.txt > /tmp/req_vast.txt
uv pip install -q -r /tmp/req_vast.txt "onnxruntime-gpu==1.22.0"
uv pip install -q --no-deps --no-build-isolation "git+https://github.com/WildChlamydia/MiVOLO.git"
fi
python - <<'EOF'
import onnxruntime as o
assert "CUDAExecutionProvider" in o.get_available_providers(), o.get_available_providers()
print("onnxruntime GPU OK")
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
