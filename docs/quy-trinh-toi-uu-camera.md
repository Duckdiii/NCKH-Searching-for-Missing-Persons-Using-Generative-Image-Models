# Quy trình tối ưu camera: lấy mẫu → chọn crop → số crop giữ → quên

Ghép bốn thuật toán thành một quy trình. Mỗi bước có mã, API và số đo riêng.

| Bước | Câu hỏi | Mã | Tài liệu |
|---|---|---|---|
| 1. Lấy mẫu | Mỗi giây phân tích bao nhiêu khung? | `sampling_policy.py` | [thuat-toan-quen-va-lay-mau.md](thuat-toan-quen-va-lay-mau.md) §1 |
| 2. Chọn crop | Khung nào là ảnh mặt tốt nhất? | `crop_quality.py` (+ `camera_tracks.py`) | mục 2 bên dưới |
| 3. Tối ưu | Bao nhiêu khung × bao nhiêu crop là đủ? | `pipeline_optimizer.py` | mục 3 bên dưới |
| 4. Quên | Giữ crop bao lâu? | `forgetting.py` | [thuat-toan-quen-va-lay-mau.md](thuat-toan-quen-va-lay-mau.md) §2 |

```
camera ──fps (bước 1/3)──▶ detector ──▶ chấm q từng khung (bước 2) ──▶ tracker giữ K crop tốt nhất
                                                                       (K, q_min: bước 3)
           ──▶ lưu crop (rank 1 = tốt nhất) ──▶ lược sau t_lược, quên sau t_quên (bước 4)
```

---

## 2. Chọn crop tốt nhất (đánh giá chất lượng ảnh mặt — FIQA)

### Bảy thành phần, mỗi thành phần ∈ [0, 1]

| Thành phần | Đo bằng | Trọng số |
|---|---|---|
| Khả năng nhận dạng | Norm embedding ArcFace **chưa chuẩn hóa**; khoảng 18 → 25 | 0.30 |
| Độ nét | 1 − chỉ số mờ Crete (2007) trên vùng mặt chuẩn hóa 64×64, **nhân hệ số phạt vỡ khối nén và nhiễu** | 0.20 |
| Độ phân giải | Khoảng cách hai mắt (px), 10 → 36 | 0.15 |
| Phơi sáng | Độ sáng, độ tương phản, tỷ lệ pixel cháy/tối trên vùng mặt | 0.10 |
| Đầy đủ | Phần bbox nằm ngoài khung hình | 0.10 |
| Góc mặt | yaw / pitch / roll từ 5 điểm mốc | 0.10 |
| Phát hiện | det_score | 0.05 |

**Điểm tổng** `Q = Π sᵢ^wᵢ` (trung bình nhân có trọng số): chỉ một thành phần rất tệ là ảnh
không thể đứng đầu. Thành phần không đo được thì bỏ và chuẩn hóa lại trọng số.

**Loại cứng** (Q = 0): mặt nghiêng hẳn (|yaw| > 0.45), mặt bị cắt > 15%, điểm mốc vô lý.

**Hai hệ số phạt độ nét**, thêm vì đo mờ hiểu nhầm cả hai thứ này là "chi tiết nét":
- **Vỡ khối nén** (JPEG/H.264): tỷ lệ chênh lệch tại biên lưới 8px của frame so với chỗ khác.
  Ảnh thường ~1.5–2, nén Q15 ~4, Q5 ~8–13. Phạt từ 2, độ nét về 0 ở 6.
- **Nhiễu cảm biến**: σ ước lượng theo Immerkær (1996). Ảnh rõ ~0.5, camera thiếu sáng 5–20.
  Phạt từ 3, độ nét về 0 ở 15.

**Nhất quán danh tính**: điểm chọn = Q × c, với c là cosine (đã giãn thang) giữa embedding crop và
tâm embedding của tracklet (có trọng số Q). Khi đóng tracklet, crop có c ở mức sàn (gần như chắc
chắn là người khác, ví dụ tracker nhảy sang người đi cạnh) bị loại nếu còn crop khác.

### Kiểm chứng (`scripts/eval_crop_quality.py`)

**FG-NET ảnh sạch, 1002 ảnh: không dùng để kết luận.** Lỗi so khớp ở đây chủ yếu do chênh tuổi
(FNMR 2.3% khi chênh ≤ 3 tuổi, 38% khi chênh > 20 tuổi), không phải do chất lượng ảnh. Chỉ có 29
cặp cùng tuổi bị lỗi, quá ít để hiệu chuẩn. Dữ liệu này chỉ dùng để chọn khoảng norm embedding
(p5 = 20.3, p50 = 22.7, p95 = 25.6).

**Suy giảm kiểu camera** (50 ảnh × nhòe chuyển động / nhiễu / nén / giảm phân giải × 4 mức = 850 mẫu;
"khớp" = khớp với **ảnh khác** của cùng người trong ±3 tuổi ở FMR 1%). Bỏ 2 ảnh vốn đã khớp sai
do tuổi, còn 21 lỗi do suy giảm trên 816 mẫu:

| Loại bỏ x% crop có điểm thấp nhất | 0% | 5% | 10% | 20% | AUC ERC |
|---|---|---|---|---|---|
| **Q (crop_quality)** | 2.6% | 0.4% | **0.0%** | 0.0% | **0.0059** |
| Norm embedding đơn lẻ | 2.6% | 0.6% | 0.1% | 0.0% | 0.0067 |
| Phơi sáng đơn lẻ | 2.6% | 0.6% | 0.4% | 0.2% | 0.0079 |
| Ngẫu nhiên (50 lần) | | | | | 0.0258 ± 0.0016 |

Q dự báo crop xấu tốt hơn ngẫu nhiên khoảng 4.4 lần, và tốt hơn bất kỳ thành phần đơn lẻ nào.

**Phép thử chọn crop** (mỗi người coi 17 phiên bản như các khung của một lượt đi qua):

| Cách chọn | Tỷ lệ khớp ảnh khác cùng người |
|---|---|
| Crop Q cao nhất | **98.5%** |
| Biết trước, chọn ảnh gốc | 98.5% |
| Ngẫu nhiên (kỳ vọng) | 95.7% |
| Crop Q thấp nhất | 65.1% |

Quá trình hiệu chuẩn đã phát hiện và sửa hai lỗi thật: trước khi có hệ số phạt, bộ chọn ưu tiên bản
**nén JPEG** (42/48 người), sau đó ưu tiên bản **thêm nhiễu** (37/48 người). Cả hai đều do đo mờ coi
nhiễu và cạnh khối là chi tiết.

**Giới hạn:** suy giảm là nhân tạo trên ảnh chân dung; chưa có crop camera thật có nhãn. Trọng số
góc mặt và độ phân giải chưa kiểm chứng được, vì FG-NET đều là ảnh to, chụp thẳng.

---

## 3. Tối ưu số khung × số crop (`pipeline_optimizer.py`)

### Mô hình một lượt đi qua (Monte Carlo, cùng số ngẫu nhiên cho mọi cấu hình)

```
n     = fps · T                                        T: thời gian trong vùng (bước 1)
q_j   = clip(b + ε_j),  b ~ phân bố Q thật,  ε_j ~ N(0, 0.12)
        khung hỏng (quay lưng, che, nghiêng) với xác suất 0.4 → q_j = 0
giữ   = K khung q cao nhất có q ≥ q_min      (fps < fps_bám → mỗi khung một tracklet)
P     = 1 − (1 − p(q₁)) · Π_{k≥2} (1 − (1 − ρ)·p(q_k)),   ρ = 0.6
lưu   = (crop/lượt × t_lược + tracklet/lượt × (t_quên − t_lược)) × lượt/ngày × byte/crop
báo động giả/truy vấn = số crop người lạ trong index × FMR (1e-4)
```

- `p(q)` = xác suất một crop chất lượng q khớp đúng, hiệu chuẩn bằng hồi quy isotonic trên 850 mẫu:
  p(0) = 0, p(0.25) = 0.85, p(0.5) = 0.93, p(0.75) = 0.97, p(1) = 1
  (`outputs/eval_crop/calibration.json`, tự nạp nếu có).
- Lưới tìm kiếm: fps ∈ {0.5 … 10} × K ∈ {1 … 5} × q_min ∈ {0, 0.3, 0.5}.
- Khuyến nghị: trong các cấu hình đạt ≥ 99% P tốt nhất và nằm trong ngân sách, chọn cấu hình
  **ít crop trong index nhất** (ổ đĩa + báo động giả), rồi ít fps nhất.

### Kết quả (120 người/giờ × 12 giờ/ngày, người lạ thấy 1 lần: lược 2 ngày, quên 7 ngày)

| Kịch bản | Khuyến nghị | P(tìm thấy) | Lưu/camera | Báo động giả/truy vấn |
|---|---|---|---|---|
| Hành lang 4K, 60°, đi tới | 5 fps, K = 2, q_min = 0.3 | 98.9% | 0.27GB | 1.5 |
| Như trên, GPU 2 fps | 1.5 fps, K = 3 | 98.9% | 0.33GB | 1.9 |
| Cửa 1080p, 90°, đi tới | 8 fps, K = 3 | 98.8% | 0.33GB | 1.9 |
| Đi ngang cách 3m, 1080p | 3 fps (tách 6 tracklet/lượt) | 99.5% | 1.25GB | 7.0 |

Lợi ích theo K ở hành lang 4K, 5 fps:

| K | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| P(tìm thấy) | 98.3% | 98.9% | 99.3% | 99.6% |
| Lưu | 0.21GB | 0.27GB | 0.33GB | 0.46GB |

**Kết luận:**
1. **K = 2–3 là điểm tối ưu.** Lợi ích giảm dần: từ K = 3 lên K = 5 chỉ thêm 0.3 điểm nhưng tốn thêm
   40% ổ đĩa và báo động giả.
2. **fps và K bù trừ cho nhau:** GPU yếu thì lấy mẫu thưa hơn và giữ nhiều crop hơn, P không đổi.
3. **Hướng đặt camera quan trọng hơn cả fps lẫn K:** đi ngang bị tách tracklet, tốn gấp 4 lần ổ đĩa
   và tăng gấp 4 lần báo động giả.

### Giả định cần đo trên camera thật

| Tham số | Giá trị hiện tại | Đo bằng |
|---|---|---|
| Tỷ lệ khung hỏng | 0.4 (giả định) | Đếm khung mất mặt hoặc bị gate trong một tracklet thật |
| Dao động q giữa các khung (σ) | 0.12 (giả định) | Độ lệch chuẩn q trong cùng tracklet |
| Tương quan lỗi giữa các crop (ρ) | 0.6 (giả định) | Khớp nhiều crop cùng lượt với ảnh tham chiếu |
| Phân bố q, p(q) | FG-NET + suy giảm nhân tạo | `eval_crop_quality.py` trên crop camera có nhãn |

---

## 4. Quy trình triển khai một camera

1. **Đặt camera**: gọi `POST /api/cameras/sampling-plan`. Nếu `usable = false` thì sửa vị trí hoặc
   ống kính trước. Ưu tiên hướng người đi tới camera.
2. **Tối ưu**: gọi `POST /api/ops/pipeline/optimize` với ngân sách GPU (`fps_budget`), ổ đĩa
   (`storage_bytes`) và lưu lượng (`people_per_hour`). Lấy `recommended.fps`, `k`, `q_min`.
3. **Bật camera** với `geometry` và `"auto_optimize": true`: backend tự chạy bộ tối ưu rồi áp dụng
   `fps`, `K` (`max_candidates`) và `q_min` (`min_quality`) được khuyến nghị. Tracker cũng tự
   nới timeout theo fps. Tham số nào truyền tay (`sample_fps`, `max_candidates`, `min_quality`)
   thì luôn được ưu tiên. Nếu khuyến nghị bị tách tracklet (`k = null`), dùng K = 3.
   ```bash
   curl -X POST localhost:8000/api/cameras/<id>/start -H 'content-type: application/json'      -d '{"auto_optimize": true, "geometry": {"motion": "approaching", "image_width_px": 3840,
          "hfov_deg": 60, "far_m": 15, "people_per_hour": 120, "fps_budget": 4}}'
   ```
   Phản hồi và `GET /api/capture-sessions/{source_id}` báo cấu hình đang chạy (`sample_fps`,
   `max_candidates`, `min_quality`, `config_source` = manual | sampling_plan | optimizer[+manual]),
   cùng số crop bị loại (`rejected_gate`, `rejected_low_quality`) để chỉnh lại q_min nếu loại quá nhiều.
4. **Quên**: kiểm tra `GET /api/ops/forgetting/schedule`, chạy `POST /api/ops/forgetting/run`
   (dry run), rồi bật `FORGETTING_ENABLED=true` khi đã đồng ý chính sách.
5. **Hiệu chuẩn lại** khi có crop camera thật: chạy `eval_crop_quality.py --calibrate`. Sau mỗi
   lần đổi công thức Q, chỉ cần `--calibrate --rescore` (dùng cache detector, vài giây), rồi tối
   ưu lại.

### Lệnh

```bash
# Chấm, hiệu chuẩn, phân tích (FG-NET)
python scripts/eval_crop_quality.py --images data/FGNET/FGNET/images        # per_image.csv
python scripts/eval_crop_quality.py --calibrate --degrade-n 50              # ~40 phút CPU
python scripts/eval_crop_quality.py --calibrate --rescore                   # vài giây
python scripts/eval_crop_quality.py --analyze

# Chọn ảnh tốt nhất trong một thư mục (ảnh người nhà gửi)
python scripts/rank_crops.py --dir <thư_mục> --copy-best outputs/best.jpg
```
