# Demo trên vast.ai (RTX 3090)

Gói `nckh_vast_demo.tar` chứa: backend (FastAPI), frontend đã build (`desktop/dist`),
checkpoint UNet chuyên biệt + MiVOLO, gallery mẫu, và các script trong thư mục này.
Stable Diffusion 1.5 và InsightFace tự tải trên máy vast (nhanh hơn upload).

## 1. Thuê máy
- Template **PyTorch** (có `/venv/main`, CUDA 12.x), GPU **RTX 3090 24GB**, đĩa ≥ 40GB.
- Launch mode: **SSH** (hoặc Jupyter + SSH). Thêm SSH public key của bạn trước khi tạo instance.

## 2. Tải gói lên (từ máy Windows, PowerShell)
```powershell
scp -P <PORT> "outputs\nckh_vast_demo.tar" root@<IP>:/workspace/
```
~3.7GB; tốc độ upload phụ thuộc mạng nhà bạn.

## 3. Cài đặt (trên máy vast, một lần)
```bash
cd /workspace && tar -xf nckh_vast_demo.tar && cd nckh_demo
bash deploy/vast/setup.sh        # ~10 phút: thư viện, SD1.5, CodeFormer
```

## 4. Chạy
```bash
bash deploy/vast/start.sh        # backend :8000 + frontend :1420 trong tmux
```
Trên máy bạn, mở tunnel rồi vào trình duyệt:
```powershell
ssh -p <PORT> root@<IP> -L 1420:localhost:1420 -L 8000:localhost:8000
```
→ http://localhost:1420

Dừng: `bash deploy/vast/stop.sh` · Log: `tail -f outputs/backend.log`

## Ghi chú
- **Không có DB:** gói không chứa `.env` (mật khẩu Supabase). Backend vẫn chạy tìm kiếm bình
  thường, chỉ tắt phần lưu lịch sử. Muốn bật: `scp -P <PORT> src\.env root@<IP>:/workspace/nckh_demo/src/.env`
  rồi `bash deploy/vast/stop.sh && bash deploy/vast/start.sh`.
- **Gallery:** mặc định `data/test_gallery`. Muốn đổi, nhập đường dẫn *trên máy vast* vào ô
  "Thư mục Gallery" (nút "Chọn thư mục" chỉ hoạt động trong app desktop).
- **Camera:** webcam của máy bạn không truyền được tới máy vast qua tunnel; demo phần camera
  bằng file video hoặc luồng RTSP mà máy vast truy cập được.
- **Tắt instance** khi demo xong — vast tính tiền theo giờ kể cả lúc rảnh.
