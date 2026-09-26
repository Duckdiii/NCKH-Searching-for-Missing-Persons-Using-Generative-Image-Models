# 04 — Các phương thức lưu trữ hoạt động như thế nào (Kho vector / Embedding / ANN index)

Đây là tầng **lưu trữ danh tính**: biến mỗi người thành một vector và tổ chức để truy vấn
"người này từng xuất hiện ở đâu?" trong vài ms.

## 0. Tại sao lưu vector chứ không chỉ lưu ảnh?

| | Lưu ảnh gốc | Lưu embedding |
|---|---|---|
| Dung lượng | MB/frame, 60–400 GB/ngày/camera | **128 B – 2 KB/người** |
| So khớp | Chậm (phải chạy lại model) | So sánh số học trực tiếp |
| Quyền riêng tư | Rủi ro cao | Rủi ro thấp hơn |
| Mất gì | — | Khó phân tích lại từ vector (giữ thêm crop nhỏ nếu cần) |

Thực tế hệ thống lưu **cả hai**: crop ảnh đại diện + vector + metadata trong DB quan hệ;
video thô chỉ giữ cửa sổ trượt (rolling window) hoặc nén/downsample (xem file `06`).

## 1. Chỉ mục tìm láng giềng gần đúng (ANN) — các bài báo nền tảng

| # | Bài báo | Ý nghĩa | Link |
|---|---------|---------|------|
| 1 | **Product Quantization for Nearest Neighbor Search** — M. Jégou, M. Douze, C. Schmid, P. Pérez (TPAMI 2011) | **PQ**: chia vector thành m phụ chiều, mỗi phụ chiều có codebook 256 centroid → vector 1024D còn **8 byte**; so sánh bằng bảng tra (ADC) | https://doi.org/10.1109/TPAMI.2010.57 |
| 2 | **Billion-scale similarity search with GPUs** — J. Johnson, M. Douze, H. Jégou (VLDB 2018), thương mại hóa trong **FAISS** | IVF + PQ + k-selection trên GPU; chuẩn đánh giá ở quy mô tỷ vector | https://arxiv.org/abs/1702.08734 |
| 3 | **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (HNSW)** — Y. Malkov, D. Yashunin (TPAMI 2018) | **HNSW**: đồ thị nhiều tầng, truy vấn ~O(log N), độ trễ sub-ms, độ chính xác cao | https://arxiv.org/abs/1603.09320 |
| 4 | **Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors** (IVF + HNSW) | Codebook hàng triệu ô vẫn hiệu quả nhờ tìm centroid bằng HNSW | https://arxiv.org/abs/1802.02422 |
| 5 | **Link and Code: Fast indexing with graphs and compact regression codes** — M. Douze et al. (CVPR 2018) | Gộp đồ thị + mã hóa nén (graph + OPQ) cho tỷ vector trên một máy chủ | https://arxiv.org/abs/1804.09996 |
| 6 | **Similarity search in the blink of an eye with compressed indices (LVQ)** — C. Aguerrebere et al. (VLDB 2023) | Nén kiểu "locally adaptive vector quantization": giảm bộ nhớ mà không mất accuracy | https://www.vldb.org/pvldb/vol16/p3433-aguerrebere.pdf |
| 7 | **HM-ANN: Efficient Billion-Point Nearest Neighbor Search on Heterogeneous Memory** (NeurIPS 2020) | Truy vấn tỷ điểm trên bộ nhớ lai (DRAM + PMem) không cần nén | https://proceedings.neurips.cc/paper/2020/file/788d986905533aba051261497ecffcbb-Paper.pdf |
| 8 | **Improving Bilayer Product Quantization for Billion-Scale ANN (FBPQ / HBPQ)** (arXiv:1404.1831) | Tối ưu 2 lớp PQ: nhanh hơn tới 15×, recall cao hơn 10–17% | https://ar5iv.labs.arxiv.org/html/1404.1831 |

## 2. Các chỉ mục hoạt động như thế nào (mô tả trực quan)

**a) Exhaustive / Flat (`IndexFlatL2`, `IndexFlatIP`)**
- Lưu vector thô, mỗi truy vấn tính khoảng cách với **mọi** vector rồi sắp xếp.
- Chính xác 100%, đơn giản, nhưng O(N) mỗi truy vấn → chỉ hợp N < 100.000.

**b) IVF (inverted file — file đảo ngược)**
- K-means chia không gian thành K "ô" (centroid); mỗi vector thuộc 1 ô.
- Truy vấn: tìm `nprobe` ô gần nhất rồi chỉ quét trong đó, bỏ qua ô xa.
- Tham số `nprobe` đổi giữa tốc độ và độ chính xác. Cấu trúc kết hợp phổ biến: **IVF-PQ**.

**c) PQ / OPQ (nén dữ liệu)**
- Chia vector thành m đoạn, mỗi đoạn thay bằng chỉ số trong codebook (1 byte/đoạn).
- Nhờ bảng tra (lookup table) ước lượng khoảng cách mà **không cần giải mã** vector → rất nhanh.
- OPQ: thêm phép xoay/tuyến tính trước khi nén để giảm lỗi nén.

**d) HNSW (đồ thị láng giềng)**
- Nhiều lớp đồ thị "navigable small world"; truy vấn đi từ lớp trên xuống dưới bằng tìm tham lam (greedy).
- Tham số: `M` (số cạnh), `efConstruction`, `efSearch` → đánh đổi chính xác/thời gian.
- Ưu điểm: giữ chất lượng tốt ngay cả khi dữ liệu nhỏ/nhẹ (khác PQ vốn có lỗi nén).

**e) Cấu trúc lai đồ thị + nén** (Link&Code, *DiskANN/Vamana*…)
- Dùng đồ thị để tạo ứng viên, dùng mã nén để ước lượng khoảng cách, sau đó **re-rank** bằng vector gốc.

**Chọn chỉ mục:** N nhỏ → Flat; cần nhanh + tiết kiệm RAM → IVF+PQ; cần chính xác/độ trễ thấp → HNSW.

## 3. Hệ thống quản lý vector (vector database)

| Công cụ | Ghi chú | Link |
|---|---|---|
| **FAISS** (Meta) | Thư viện C++/Python, chuẩn mực trong nghiên cứu & sản phẩm | https://github.com/facebookresearch/faiss |
| **Milvus** | Hệ quản lý dữ liệu vector riêng (SIGMOD 2021) | https://milvus.io/ |
| **pgvector** | Vector search ngay trong PostgreSQL (kết hợp metadata + vector) | https://github.com/pgvector/pgvector |
| **ann-benchmarks** | Benchmark so sánh các chỉ mục ANN | https://github.com/erikbern/ann-benchmarks |

> Lưu ý vận hành: chỉ mục ANN thường giữ **trong RAM**; metadata (ID, camera_id, timestamp,
> bbox, đường dẫn ảnh) giữ ở SQLite/Postgres; ảnh crop giữ ở ổ cứng/object storage.
> FAISS không update từng mục dễ dàng → hay dùng `IndexIDMap`/phân mảnh, hoặc rebuild định kỳ.

## 4. Vòng đời dữ liệu trong kho vector (open-set)

```
frame → detect/track → crop → encoder → vector (L2-normalized)
      → search(top-k) trên index
           ├─ cosine ≥ ngưỡng (0.7–0.9) → gán lại ID cũ, cập nhật centroid (EMA)
           └─ cosine < ngưỡng           → tạo ID mới, insert vào index
      → ghi metadata (id, camera_id, ts, bbox, ảnh) vào DB
```

- **Centroid/EMA** cho mỗi ID: làm trung bình cộng dồn mọi embedding đã thấy → giảm nhiễu do
  che khuất/đổi góc/đổi sáng, đồng thời giảm bộ nhớ (chỉ lưu 1 vector/ID).
- Tham số ngưỡng quyết định đánh đổi false match ↔ ID mới giả (xem RAMOT, file `02`).
- **Re-ranking** (k-reciprocal, CA-Jaccard — file `01`) nâng chất lượng top-k sau truy vấn.
