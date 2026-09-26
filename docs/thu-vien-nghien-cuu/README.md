# Thư viện nghiên cứu: Duy trì danh tính xuyên camera & Phương thức lưu trữ

Tổng hợp các bài báo / nghiên cứu khoa học (survey, phương pháp, hệ thống, benchmark) phục vụ đề tài
**"Tìm kiếm người mất tích bằng mô hình sinh ảnh"** — cụm chủ đề:

1. **Duy trì danh tính xuyên camera** (person re-identification, cross-camera tracking, face recognition giữa nhiều camera).
2. **Các phương thức lưu trữ hoạt động như thế nào** (lưu embedding/vector, chỉ mục ANN, lưu template sinh trắc học, lưu trữ video giám sát).

> **Bản thiết kế áp dụng cho dự án:** [Kiến trúc đa camera tiết kiệm bộ nhớ](../kien-truc-da-camera-tiet-kiem-bo-nho.md).
> Theo yêu cầu hiện tại, luồng camera chỉ lưu crop khuôn mặt, embedding và metadata;
> không lưu video/ảnh toàn khung. Bản thiết kế có chính sách chọn mẫu, giới hạn RAM,
> ước lượng dung lượng và lộ trình thay đổi mã nguồn; các con số là giả định cần đo thực tế.

---

## Cấu trúc thư mục

| File | Chủ đề |
|------|--------|
| [`01-duy-tri-danh-tinh-xuyen-camera.md`](01-duy-tri-danh-tinh-xuyen-camera.md) | Survey & phương pháp Person Re-ID (trích xuất đặc trưng, metric learning, re-ranking) |
| [`02-tracking-xuyen-camera-mtmc.md`](02-tracking-xuyen-camera-mtmc.md) | Multi-Camera Multi-Object Tracking, liên kết tracklet giữa các camera |
| [`03-nhan-dien-khuon-mat-xuyen-camera.md`](03-nhan-dien-khuon-mat-xuyen-camera.md) | Nhận diện khuôn mặt xuyên camera / hệ thống ở quy mô lớn |
| [`04-luu-tru-vector-embedding.md`](04-luu-tru-vector-embedding.md) | **Cách hoạt động của lưu trữ vector / embedding**: FAISS, PQ, IVF, HNSW, vector DB |
| [`05-luu-tru-bien-the-sinh-trac-hoc.md`](05-luu-tru-bien-the-sinh-trac-hoc.md) | Lưu trữ & bảo vệ template sinh trắc học (cancelable biometrics, template protection) |
| [`06-kien-truc-luu-tru-video.md`](06-kien-truc-luu-tru-video.md) | Kiến trúc lưu trữ video giám sát: edge/cloud, chọn lọc khung hình, nén |
| [`07-person-search-va-he-thong.md`](07-person-search-va-he-thong.md) | Person search (tìm người trong ảnh nguyên khung) & hệ thống triển khai thực tế |
| [`08-datasets-benchmark.md`](08-datasets-benchmark.md) | Dataset / benchmark cho Re-ID, tracking, tìm kiếm người |

---

## Tóm tắt: các phương thức lưu trữ hoạt động như thế nào

Một hệ thống duy trì danh tính xuyên camera gồm **3 tầng lưu trữ** hoạt động nối tiếp:

```
Camera → Detect/Track → Trích xuất embedding → [Tầng 1] Kho vector
                                                   + [Tầng 2] DB metadata
                                                   + [Tầng 3] Kho video thô/đã nén
                              Query (ảnh người mất tích) → tìm ANN → ngưỡng → trả về ID / cảnh báo
```

### Tầng 1 — Kho vector (embedding index): "bộ nhớ danh tính"

- **Đầu vào**: mỗi người được detect → model (FaceNet/ArcFace/OSNet/ResNet-ReID) sinh ra vector
  128–2048 chiều, L2-normalized. *FaceNet (CVPR 2015)* chứng minh chỉ **128 byte/người**
  là đủ để nhận diện, nhờ đó một triệu người chỉ tốn ~128 MB.
- **Lưu trữ**: đưa vector vào chỉ mục ANN (approximate nearest neighbor):
  - `IndexFlat` (brute-force): chính xác 100%, O(N) mỗi truy vấn → chỉ hợp < 100k vector.
  - **IVF** (inverted file): K-means chia không gian thành K ô, mỗi truy vấn chỉ quét nprobe ô gần nhất → giảm số phép tính.
  - **PQ / OPQ** (product quantization): nén vector 1024D → 8–64 byte, tra soát bằng bảng tra (lookup table); *Jégou et al. TPAMI 2011*, *Johnson et al. VLDB 2018 (FAISS)*.
  - **HNSW**: đồ thị nhiều tầng "navigable small world" → truy vấn O(log N), độ trễ sub-ms.
  - Cấu trúc lai IVF+PQ, graph+quantization (L&C), DiskANN cho dữ liệu tỷ.
- **Ghi/đọc**: mọi frame mới → embedding → `add()` vào index; truy vấn → `search(k)` → so
  cosine/L2 với ngưỡng (vd 0.7–0.9). Vượt ngưỡng → **gán lại ID cũ**, không thì **tạo ID mới**
  (open-set). Đổi ID tích lũy bằng **centroid/EMA** để giảm nhiễu (RAMOT, 2026).
- **Metadata song song**: ID, camera_id, timestamp, bbox, track_id, path ảnh lưu ở DB quan hệ
  (SQLite/Postgres) hoặc object storage; chỉ lưu embedding trong index.

### Tầng 2 — Bảo vệ template (khi dữ liệu sinh trắc học phải lưu lâu dài)

- Template gốc **không thể thu hồi/đổi được** nếu lộ → cần *template protection*:
  **Cancelable Biometrics** (biến đổi một chiều: salting, non-invertible transform),
  **Biometric Cryptosystem** (trộn khóa vào template: fuzzy vault/commitment, helper data),
  **Homomorphic encryption** (so khớp trong miền mã hóa).
- Tiêu chí ISO/IEC 24745: *irreversibility, unlinkability, renewability/cancelability*.

### Tầng 3 — Lưu trữ video & ảnh gốc

- **Không lưu tất cả**: chọn lọc theo "information amount", downsampling (temporal/spatial/fidelity),
  ROI/tile streaming, tách nhịp edge-cloud để giảm băng thông.
- Kiến trúc 3 lớp: **camera (edge) → edge server → cloud**: xử lý trước ở biên, chỉ đẩy frame
  "đáng chú ý" lên cloud; lưu trữ hiệu năng (compression → encryption → redundancy/RAID).
- Riêng với mục tiêu ảnh sinh ra từ model (Gen-AI): lưu ảnh gốc + embedding + prompt/provenance
  để truy vết; đối chiếu ảnh query với kho embedding.

---

## Đọc nhanh theo mục đích

| Mục đích | Đọc trước |
|---|---|
| Hiểu tổng quan bài toán Re-ID | `01` (Ye et al. TPAMI 2022) |
| Chọn model trích xuất đặc trưng | `01`, `03` (OSNet, TransReID, ArcFace) |
| Thiết kế kho vector/index | `04` (FAISS, HNSW, PQ) + `07` (RAMOT) |
| Lưu ảnh/video lâu dài, ít tốn dung lượng | `06` |
| Bảo mật dữ liệu nhận diện | `05` |
| Đánh giá/training | `08` |

> **Ghi chú**: các link là arXiv / open-access. Bài trên IEEE/ACM/Elsevier có thể cần truy cập qua
> thư viện trường; link arXiv tương ứng (nếu có) đã được ưu tiên.
> 4 link bị nhà xuất bản chặn robot (MDPI, ScienceDirect, ACM DL, ResearchGate) — mở bằng
> trình duyệt web bình thường là được, nội dung link là thật và đúng bài.
>
> Tổng số link đã kiểm tra: 76 (72 truy cập được bằng request tự động, 4 link nêu trên).
