# Nhật ký triển khai demo trên vast.ai

Ghi lại trạng thái và các lỗi đã gặp khi chạy demo FE + BE trên vast.ai (RTX 3090),
để lần kết nối sau làm lại nhanh. Hướng dẫn cài đặt đầy đủ: [README.md](README.md).

## Trạng thái lần gần nhất (2026-10-01)

| Hạng mục | Giá trị |
|---|---|
| Instance | RTX 3090 24GB, template PyTorch, container `C.53654007` |
| SSH trực tiếp | `ssh -p 50403 root@154.64.230.50` — **bị ngắt sau ~1 phút**, chỉ dùng cho lệnh ngắn |
| SSH qua proxy | `ssh -p 28655 root@ssh5.vast.ai` — **ổn định, dùng cái này** |
| Thư mục app | `/workspace/nckh_demo` |
| Dịch vụ | tmux `backend` (cổng 8000) + `frontend` (cổng 1420) |
| Venv | `/venv/main` của image đã có đủ thư viện (kể cả onnxruntime-gpu 1.22, MiVOLO) |
| Đã tải sẵn | SD 1.5 (VAE + text encoder), InsightFace `buffalo_l` |
| Chưa chạy | CodeFormer (lỗi import `basicsr` đi kèm) — backend tự bỏ bước làm nét |
| DB | Không có `.env` → tắt lưu lịch sử (log "face_media DB không sẵn sàng" là bình thường) |

Port và IP **đổi mỗi lần tạo/mở lại instance** — lấy lệnh mới ở nút **Connect** trên vast.
Nếu instance đã bị destroy, `/workspace` mất hết (không phải volume) → làm lại từ bước 1 bên dưới.

## Lần sau: mở lại demo (instance vẫn còn)

1. **Tắt bản local** trên máy Windows (app desktop / `run_desktop.py`) — chúng chiếm cổng 1420/8000
   và làm trình duyệt chạy GPU máy mình thay vì vast. Kiểm tra:
   ```powershell
   Get-NetTCPConnection -LocalPort 1420,8000 -State Listen | Select LocalPort,OwningProcess
   ```
2. **Bật dịch vụ trên vast** (nếu đã tắt):
   ```powershell
   ssh -p <PORT_PROXY> root@ssh5.vast.ai "cd /workspace/nckh_demo && bash deploy/vast/start.sh"
   ```
3. **Mở tunnel tự nối lại** (PowerShell, giữ cửa sổ mở):
   ```powershell
   while ($true) { ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -N -p <PORT_PROXY> root@ssh5.vast.ai -L 1420:localhost:1420 -L 8000:localhost:8000; Start-Sleep 2 }
   ```
4. Trình duyệt: **http://localhost:1420** (Ctrl+F5 nếu trước đó từng mở bản local).

## Lần sau: instance mới hoàn toàn

1. Thêm SSH public key của máy vào vast (Account → SSH Keys) **trước** khi tạo instance.
2. Tạo gói ở máy Windows (nếu chưa có `outputs/nckh_vast_demo.tar`) — xem README.
3. **Upload có thể nối tiếp** (scp bị ngắt giữa chừng sẽ phải gửi lại từ đầu):
   ```bash
   # Git Bash trên Windows; thay PORT/HOST
   F=outputs/nckh_vast_demo.tar; T=$(stat -c %s $F); R="ssh -o ServerAliveInterval=15 -p PORT root@HOST"
   while :; do D=$($R 'stat -c %s /workspace/nckh_vast_demo.tar 2>/dev/null || echo 0' | tail -1)
     [ "$D" = "$T" ] && break; tail -c +$((D+1)) $F | $R 'cat >> /workspace/nckh_vast_demo.tar'; sleep 5; done
   ```
   Sau đó so `sha256sum` hai đầu.
4. Trên vast:
   ```bash
   cd /workspace && tar -xf nckh_vast_demo.tar && cd nckh_demo
   SKIP_PIP=1 bash deploy/vast/setup.sh   # bỏ SKIP_PIP nếu venv chưa có thư viện
   bash deploy/vast/start.sh
   ```

## Lỗi đã gặp và cách xử lý

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Upload scp đứt sau ~1 phút | Đường SSH trực tiếp bị vast ngắt định kỳ | Upload nối tiếp (bước 3 ở trên); dùng proxy `ssh*.vast.ai` |
| `uv pip install` treo, 0% CPU, không kết nối mạng | Không rõ; venv của image đã đủ thư viện | `SKIP_PIP=1` |
| Tải SD 1.5 đứng ở ~500MB | HuggingFace không xác thực chậm/đứt; tải cả UNet gốc 3.4GB không cần | Chỉ tải VAE + text encoder (UNet dùng `checkpoints/specialized_unet`) |
| `sed: unknown option to s` | Dấu phân cách `#` trùng ký tự trong chuỗi thay thế | Dùng `|` làm dấu phân cách |
| GPU vast 0%, trình duyệt dùng GPU máy mình | Bản local chiếm cổng 1420 (`::1`) và 8000 (`127.0.0.1`), tunnel chỉ giành được nửa còn lại | Tắt bản local, nối lại tunnel |
| GPU vast 0% vài phút sau khi gửi ảnh | Request đầu tiên tải InsightFace `buffalo_l` từ GitHub (~600KB/s) | Đã thêm bước tải sẵn vào `setup.sh` |
| `POST /api/sessions/<id>/run` → 404 "Phiên làm việc không tồn tại" | Giao diện giữ ID phiên tạo ở backend local trước đó | Ctrl+F5, tải ảnh và chạy lại từ đầu |
| CodeFormer: `No module named basicsr` / lỗi import trong `basicsr/losses` | `basicsr` đi kèm CodeFormer không hợp torch/torchvision mới | Chưa sửa; demo chạy được không cần CodeFormer |
| Lệnh SSH tự chết khi dùng `pkill -f "<mẫu>"` | Mẫu khớp luôn chính dòng lệnh SSH | Dùng `tmux kill-session` hoặc `pgrep` + `kill PID` |
| Gõ `tail`, `cd /workspace` báo lỗi | Gõ vào PowerShell Windows thay vì phiên SSH | Dấu nhắc `root@...#` mới là máy vast |

## Việc còn dở

- Sửa CodeFormer trên vast (cần cài `basicsr` tương thích hoặc vá `basicsr/losses`).
- Chưa thử camera (backend mở camera phía vast → chỉ RTSP hoặc file video).
- Gói hiện tại trên vast dùng `setup.sh` trước khi thêm bước tải InsightFace; instance mới sẽ có sẵn.
- **Destroy instance** khi demo xong — vast tính tiền theo giờ.
