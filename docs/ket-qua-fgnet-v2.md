# Kết quả đánh giá FG-NET v2 — ảnh tạo sinh (FADING) so với ảnh gốc

Chạy ngày 2026-09-26 trên vast.ai (RTX 3090), notebook `notebooks/FADING_eval_fgnet.ipynb`
(giao thức v2) chuyển bằng `scripts/notebook_to_vast.py`. Dữ liệu thô: `outputs/fgnet_v2/`
(không commit — CSV, embedding `.npy`, ảnh sinh theo cặp).

## Thiết lập

- **156 cặp** nguồn→đích cùng người, giống hệt lần chạy Kaggle trước (`fgnet_eval_results.csv`);
  đầu vào diffusion là ảnh nguồn đã tiền xử lý + căn chỉnh của lần chạy đó (tải từ Drive).
- UNet: `checkpoints/specialized_unet` (= `specialized_unet_v3_v2_600samples/final`).
- Embedder: InsightFace buffalo_l, det 256 — **cùng một embedder** cho ảnh sinh, ảnh nguồn, ảnh đích.
- Gallery Rank-k: 1002 ảnh / 82 người FG-NET, **đã loại chính ảnh nguồn**.
- 154/156 cặp thành công; 2 cặp lỗi vì ảnh sinh không còn khuôn mặt nhận diện được
  (033A02→033A25, 042A01→042A24 — đều từ ảnh 1–2 tuổi sinh lên ~25 tuổi).

## Kết quả

| Nhóm | n | cos gốc→đích | cos sinh→đích | Sinh thắng | R1 gốc | R1 sinh | R1 kết hợp (mean) | R5 gốc | R5 sinh |
|---|---|---|---|---|---|---|---|---|---|
| Tất cả | 154 | **0.475** | 0.361 | 2.6% | **93.5%** | 89.6% | 94.2% | 96.8% | 94.2% |
| Nguồn < 15 tuổi | 102 | 0.413 | 0.315 | 3.9% | 90.2% | 84.3% | 91.2% | 95.1% | 91.2% |
| Nguồn ≥ 15 tuổi | 52 | 0.598 | 0.451 | 0.0% | 100% | 100% | 100% | 100% | 100% |
| Cách tuổi ≤ 10 | 96 | 0.556 | 0.430 | 1.0% | 97.9% | 95.8% | 97.9% | 99.0% | 97.9% |
| Cách tuổi > 10 | 58 | 0.342 | 0.247 | 5.2% | 86.2% | 79.3% | 87.9% | 93.1% | 87.9% |

Kiểm định:
- Cosine ảnh sinh vs ảnh gốc (Wilcoxon ghép cặp): p ≈ 1e-26 — ảnh sinh **kém hơn có ý nghĩa**.
- Rank-1 ảnh sinh vs ảnh gốc (McNemar): sinh đúng/gốc sai = 1, ngược lại = 7, p = 0.07.
- Rank-1 kết hợp (mean) vs ảnh gốc (McNemar): 2 vs 1, p = 1.0 — **không có cải thiện**.
- Rank-1 theo cách tính cũ (gallery chứa ảnh nguồn): 100% — xác nhận rò rỉ; không dùng con số này.

Độ tái lập so với lần chạy Kaggle: tương quan id_score giữa hai lần r = 0.82; trung bình
0.361 (v2) vs 0.325 (cũ). Kết luận không đổi giữa hai lần chạy.

## Kết luận

1. Trên FG-NET, truy vấn bằng ảnh tạo sinh **không tốt hơn** truy vấn thẳng bằng ảnh gốc — kể cả
   ở nhóm được kỳ vọng có lợi nhất (trẻ em, cách tuổi > 10).
2. Kết hợp ảnh sinh với ảnh gốc (trung bình vector hoặc max điểm) không cải thiện Rank-1 có ý nghĩa.
3. Con số Rank-1 94.2% trong báo cáo trước bị thổi phồng do gallery chứa ảnh nguồn; giá trị
   công bằng của ảnh sinh là 89.6%, của ảnh gốc là 93.5%.
4. Nguyên tắc hiện có của hệ thống — ảnh tạo sinh **không** được dùng làm mẫu định danh — được
   số liệu này ủng hộ.

## Giới hạn

- FG-NET nhỏ (82 người, gallery 1002 ảnh); ở nhóm nguồn ≥ 15 tuổi mọi phương pháp đạt trần 100%.
- Chỉ một cấu hình FADING (checkpoint, guidance 4.0, 50 bước). Chưa thử sinh nhiều mốc tuổi rồi
  gộp, hay tinh chỉnh để giữ danh tính.
- Embedder ArcFace (buffalo_l) vốn đã khá bền với thay đổi tuổi, nên dư địa cải thiện nhỏ.
