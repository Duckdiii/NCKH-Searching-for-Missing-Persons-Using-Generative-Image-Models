# Thuật toán chọn tốc độ lấy mẫu camera và quên dần dữ liệu

Hai thuật toán bổ sung cho luồng camera crop-only
([kien-truc-da-camera-tiet-kiem-bo-nho.md](kien-truc-da-camera-tiet-kiem-bo-nho.md)):

| Thuật toán | Trả lời câu hỏi | Mã | API |
|---|---|---|---|
| Lấy mẫu | Mỗi giây nên phân tích bao nhiêu khung? | `backend/api/sampling_policy.py` | `POST /api/cameras/sampling-plan`; `POST /api/cameras/{id}/start` với `geometry` |
| Quên dần | Sau bao lâu thì máy quên một người? | `backend/api/forgetting.py` | `GET /api/ops/forgetting/schedule`, `POST /api/ops/forgetting/plan`, `POST /api/ops/forgetting/run` |

Mọi hằng số mặc định là **giả định khởi điểm**, cần hiệu chuẩn bằng dữ liệu camera thật
(đặc biệt `usable_ratio` và các tham số quên).

---

## 1. Chọn tốc độ lấy mẫu (khung/giây)

### Đầu vào

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `hfov_deg` | Góc nhìn ngang của camera | 90° (ống 2.8mm) |
| `image_width_px` | Độ rộng khung hình gốc | 1920 |
| `motion` | `crossing` (đi ngang) hoặc `approaching` (đi về phía camera) | crossing |
| `distance_m` | Crossing: khoảng cách vuông góc từ camera tới lối đi | 4m |
| `near_m`, `far_m` | Approaching: đoạn đường người đi trong khung | 1.5–8m |
| `walk_speed_mps` | Tốc độ thiết kế (nên lấy người đi nhanh) | 1.4 m/s |
| `camera_fps`, `fps_budget` | Fps gốc camera, fps GPU dành cho camera này | 25, không giới hạn |
| `usable_ratio` | Tỷ lệ khung lấy mẫu dùng được (mặt thẳng, nét) | 0.35 |
| `confidence` | Xác suất mong muốn có đủ 3 ảnh tốt mỗi lượt | 0.95 |

Hằng số: bề rộng mặt `w = 0.16m`, mặt tối thiểu `48px`, ngưỡng IoU tracker `t = 0.3`,
fps tối đa phiên camera `10`.

### Bước 1 — Vùng nhận diện được

```
Bề rộng khung ở khoảng cách d:   W(d) = 2·d·tan(hfov/2)
Số pixel của mặt:                face_px(d) = w · image_width_px / W(d)
Xa nhất còn nhận diện:           d_max = w · image_width_px / (48 · 2·tan(hfov/2))
```

- Crossing: dùng được khi `d ≤ d_max`; quãng đường `L = W(d)·(1 − 2·lề) − w` (bỏ 10% mỗi mép).
- Approaching: quãng đường `L = min(far, d_max) − near`.
- Thời gian ở trong vùng: `T = L / v`.

Nếu `d > d_max`, thuật toán đề xuất góc nhìn tối đa `hfov ≤ 2·atan(w·image_width_px / (2·48·d))`.

### Bước 2 — Ràng buộc phủ (đủ ảnh tốt)

Mỗi khung lấy mẫu dùng được với xác suất `p = usable_ratio`. Chọn `n` nhỏ nhất sao cho

```
P(X ≥ 3) ≥ confidence,  X ~ Binomial(n, p)
fps_phủ = n / T
```

Với `p = 0.35`, `confidence = 0.95` ⇒ `n = 16` khung/lượt.

### Bước 3 — Ràng buộc bám vết (không tách một người thành nhiều tracklet)

Tracker ghép khung bằng IoU ≥ t (`camera_tracks.py`).

- **Đi ngang:** giữa hai mẫu, mặt dịch `Δ = v / fps`. Hai bbox cùng cỡ lệch ngang Δ có
  `IoU = (w − Δ)/(w + Δ)`. Giữ IoU ≥ t cần `Δ/w ≤ (1 − t)/(1 + t)`:
  ```
  fps_bám = v·(1 + t) / (w·(1 − t))
  ```
  Kết quả **không phụ thuộc khoảng cách camera**, vì số pixel trên mỗi mét triệt tiêu.
  Với v = 1.4 m/s, t = 0.3 ⇒ **16.2 fps**.
- **Đi tới camera:** bbox phóng to `s = (d + Δ)/d`, `IoU = 1/s²` ⇒ cần `s ≤ 1/√t`:
  ```
  fps_bám = v / (near · (1/√t − 1))
  ```
  Với near = 1.5m ⇒ **1.1 fps**.

### Bước 4 — Chọn fps

```
fps_cần = max(fps_phủ, fps_bám)
giới hạn = min(10, camera_fps, fps_budget)
fps     = max(fps_cần, fps_min)    nếu fps_cần ≤ giới hạn
        = giới hạn                 nếu không (kèm cảnh báo)
        = fps_min                  nếu không có vùng nhận diện được (lấy mẫu dày chỉ phí GPU)
```

Kết quả trả về cả `binding`, tức ràng buộc quyết định (`coverage`, `tracking`, `fps_max`,
`budget`, `camera_fps`, `fps_min`, `unusable`), cùng các ước lượng phụ:
- `tracker_timeout_sec = max(3, (2 + 1)/fps)`: fps thấp thì tracker chờ lâu hơn trước khi đóng track;
- số tracklet/lượt (bằng số khung khi fps < fps_bám), byte/lượt, byte/ngày (nếu có `people_per_hour`);
- độ nhòe chuyển động `v·shutter / w`, cảnh báo khi > 10% bề rộng mặt.

### Ví dụ

| Kịch bản | Mặt | Ở trong vùng | fps_phủ | fps_bám | **fps** | Ràng buộc |
|---|---|---|---|---|---|---|
| 1080p, 90°, đi ngang cách 4m | 38px | — | — | 16.2 | **0.5** | không nhận diện được (cần ≤ 3.2m hoặc góc ≤ 77°) |
| 1080p, 90°, đi ngang cách 3m | 51px | 3.3s | 4.8 | 16.2 | **10** | trần fps → ~33 tracklet/lượt |
| 1080p, 90°, đi tới 1.5–8m | 102px ở 1.5m | 1.2s (1.5–3.2m) | 13.2 | 1.1 | **10** | trần fps → xác suất đủ ảnh ~85% |
| 4K, 60°, hành lang đi tới 1.5–15m | 355px ở 1.5m | 6.9s | 2.3 | 1.1 | **2.3** | phủ |

**Kết luận thiết kế:**
1. Camera nhìn người **đi tới** (cửa ra vào, hành lang) rẻ hơn hẳn camera nhìn người đi ngang,
   vì yêu cầu bám vết giảm từ ~16 fps xuống ~1 fps.
2. Với tracker hiện tại, người **đi ngang** luôn bị tách thành nhiều tracklet dưới 16 fps.
   Hướng khắc phục là tracker dự đoán chuyển động (vận tốc không đổi), hiện chưa làm.
3. Mặc định cũ `sample_fps = 1` quá thấp cho đi ngang và vừa đủ cho đi tới ở near ≥ 1.5m.

### Dùng API

```bash
# Chỉ xem kế hoạch
curl -X POST localhost:8000/api/cameras/sampling-plan -H 'content-type: application/json' \
  -d '{"motion":"approaching","image_width_px":3840,"hfov_deg":60,"far_m":15,"people_per_hour":120}'

# Bật camera, tự chọn fps + timeout tracker (bỏ sample_fps)
curl -X POST localhost:8000/api/cameras/<id>/start -H 'content-type: application/json' \
  -d '{"geometry":{"motion":"approaching","far_m":10}}'
```

Truyền `sample_fps` thì giá trị đó được dùng; kế hoạch chỉ nới timeout tracker.

---

## 2. Quên dần dữ liệu

Bổ sung cho TTL cố định của `retention.py`. Đơn vị nhớ là **một tracklet**, tức một lượt
một người đi qua một camera.

### Mô hình (đường cong quên Ebbinghaus)

```
Độ nhớ:   R(Δt) = exp(−Δt / S)
Độ bền:   S = S0 · (1 + α·ln(số ngày xuất hiện)) · (0.5 + q) · g_áp_lực
```

| Thành phần | Ý nghĩa | Mặc định |
|---|---|---|
| Δt | Số ngày từ lần **cuối** người đó được thấy (theo danh tính toàn cục nếu đã liên kết) | — |
| S0 | Độ bền cơ sở: thấy 1 lần, ảnh chất lượng trung bình | 3 ngày |
| α · ln(số ngày) | **Củng cố**: xuất hiện lại vào nhiều ngày khác nhau thì nhớ lâu hơn (tăng chậm dần) | α = 0.7 |
| 0.5 + q | **Chất lượng**: q ∈ [0, 1] của ảnh tốt nhất; ảnh tồi bị quên sớm | — |
| g_áp_lực | **Áp lực ổ đĩa**: 1 dưới 80% dung lượng, giảm tuyến tính còn 0.4 ở 90% | 0.4 |

### Hai bậc quên

```
Lược (thin):   t_lược = S·ln(1/0.5)                          → chỉ giữ crop tốt nhất
Quên (forget): t_quên = clamp(S·ln(1/0.1), sàn, trần)        → xóa mọi crop, tracklet = expired
```

- **Sàn 7 ngày**: không quên hẳn trước 7 ngày, vì gia đình thường báo mất tích sau vài ngày.
  Bậc lược không bị chặn bởi sàn vì vẫn còn giữ ảnh tốt nhất.
- **Trần 30 ngày** cho người lạ chưa từng khớp truy vấn. Khi ổ đĩa chạm ngưỡng dừng, trần
  co lại theo cùng hệ số áp lực (không dưới sàn).

**Không bao giờ quên tự động:**
- crop từng **khớp truy vấn tìm người** (`search_results.accepted_by_threshold`) trong
  180 ngày gần nhất, vì đó là bằng chứng của hồ sơ;
- crop đang là **mẫu định danh** (`identity_exemplars`), do policy exemplar riêng quản lý.

### Lịch quên mặc định (ổ dưới 80%)

| Người | Độ bền S | Lược sau | Quên sau |
|---|---|---|---|
| Thấy 1 ngày, ảnh trung bình (q = 0.5) | 3.0 ngày | 2.1 ngày | 7 ngày (sàn) |
| Thấy 1 ngày, ảnh tồi (q = 0.1) | 1.8 ngày | 1.2 ngày | 7 ngày (sàn) |
| Thấy 5 ngày, ảnh tốt (q = 0.9) | 8.9 ngày | 6.2 ngày | 20.6 ngày |
| Thấy 20 ngày, ảnh rất tốt (q = 1.0) | 13.9 ngày | 9.7 ngày | 30 ngày (trần) |
| Bất kỳ, ổ đầy 90% | × 0.4 | sớm hơn | trần còn 12 ngày |

### Dung lượng ổn định

Kết hợp hai thuật toán: dung lượng lưu trữ ở trạng thái ổn định ≈
`byte/ngày × thời gian giữ trung bình`, trong đó `byte/ngày` lấy từ kế hoạch lấy mẫu
(`bytes_per_day`). Ví dụ hành lang 4K ở trên với 120 người/giờ: ~103MB/ngày; nếu phần lớn
là người lạ chỉ thấy 1 lần (lược sau 2 ngày, quên sau 7 ngày) thì cần khoảng 0.3–0.7GB.

### Vận hành (an toàn mặc định)

```bash
GET  /api/ops/forgetting/schedule?sighting_days=5&quality=0.9   # xem lịch quên
POST /api/ops/forgetting/plan?include_decisions=true             # kế hoạch, không xóa
POST /api/ops/forgetting/run   {"dry_run": true}                 # đếm, không xóa
POST /api/ops/forgetting/run   {"dry_run": false}                # cần FORGETTING_ENABLED=true
POST /api/ops/retention/gc     {"dry_run": false}                # GC xóa file/vector theo ledger
```

Quên thật chỉ ghi **tombstone** vào `deletion_ledger` (truy vấn ẩn crop ngay), sau đó
`retention/gc` mới xóa file và vector. Có thể thử lại an toàn, và chạy lại không xóa trùng.

Tham số đổi qua biến môi trường `FORGET_*` (xem `.env.example`).

### Giới hạn

- Các tham số quên là lựa chọn chính sách, không phải số đo; nên chốt cùng yêu cầu pháp lý
  về lưu dữ liệu sinh trắc học.
- "Củng cố" dựa vào liên kết danh tính xuyên camera. Liên kết sai (gộp nhầm hai người) sẽ
  làm người lạ được nhớ lâu hơn.
- Chưa có bậc "chỉ giữ vector, bỏ ảnh": GC hiện xóa crop và vector cùng lúc.
