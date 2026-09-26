# 02 — Tracking & liên kết danh tính xuyên camera (MTMC)

Bài toán: theo dõi nhiều người qua **nhiều camera** (multi-target multi-camera tracking — MTMC),
bao gồm 2 giai đoạn: (1) tracking trong từng camera (intra-camera), (2) **gắn tracklet của các
camera lại với nhau** (inter-camera / handover) — bước "giữ danh tính" khi người rời camera này
và xuất hiện ở camera khác.

## A. Survey

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 1 | **Multi-camera multi-object tracking: A review of current trends and future advances** — Neurocomputing (đánh giá 30 thuật toán MCT) | 2023 | https://www.sciencedirect.com/science/article/pii/S0925231223006811 |
| 2 | **Technique and Challenge for Multi-Camera Tracking** — (arXiv:1702.01507) | 2017 | https://arxiv.org/pdf/1702.01507 |
| 3 | **In Pursuit of Many: A Review of Modern Multiple Object Tracking Systems** — (arXiv:2209.04796) | 2022 | https://arxiv.org/abs/2209.04796 |
| 4 | **Connected Vision Systems: survey on multi-view multi-camera tracking + Re-ID + action understanding** — (arXiv:2510.09731) | 2025 | https://arxiv.org/pdf/2510.09731 |
| 5 | **Learning deep features for online person tracking using non-overlapping cameras: A survey** | — | (tra cứu qua Google Scholar / thư viện trường) |

**Điểm chính của survey (1):**
- MTMC = *tracking-by-detection* (dominant) + bước **handover** bằng Re-ID.
- Liên kết liên camera dựa trên: **gợi ý ngoại hình (appearance)** + **ràng buộc không-thời gian**
  (spatio-temporal constraints) + **topology camera** (camera nào có thể đi tới camera nào,
  thời gian chuyển tối thiểu) → cắt bớt ứng viên không hợp lệ (pruning).
- Phân loại theo 6 tiêu chí: công thức bài toán, cách giải, nhu cầu liên kết dữ liệu,
  ràng buộc loại trừ lẫn nhau, benchmark, chỉ số (IDF1, HOTA, MOTA, ID switches).

## B. Phương pháp liên kết liên camera tiêu biểu

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 6 | **Performance Measures and a Data Set for Multi-target, Multi-camera Tracking** (DukeMTMC) — E. Ristani et al. ECCV Workshops | 2016 | https://arxiv.org/abs/1609.01775 |
| 7 | **Towards Effective Multi-Moving-Camera Tracking** (MMCT dataset, Linker + color transfer) — (arXiv:2312.11035) | 2023 | https://arxiv.org/html/2312.11035v3 |
| 8 | **GMT: Effective Global Framework for Multi-Camera Multi-Target Tracking** — CVPR | 2026 | https://openaccess.thecvf.com/content/CVPR2026/html/Zhen_GMT_Effective_Global_Framework_for_Multi-Camera_Multi-Target_Tracking_CVPR_2026_paper.html |
| 9 | **MTMMC: A Large-Scale Real-World Multi-Modal Camera Tracking Benchmark** (RGB + thermal, 16 camera) — (arXiv:2403.20225) | 2024 | https://arxiv.org/html/2403.20225v1 |
| 10 | **DyGLIP: Dynamic Graph Model with Link Prediction for Accurate Multi-Camera Multiple Object Tracking** | 2021 | (xem danh mục [paper-MTMC](https://github.com/SherryJYC/paper-MTMC)) |

**Cơ chế chung của bước handover:**
1. Mỗi tracklet trong camera → vector đặc trưng ngoại hình (Re-ID embedding) + đặc trưng chuyển động.
2. Với cặp tracklet liên camera, tính điểm tương đồng (cosine / KISSME / XQDA) kết hợp
   **khoảng thời gian + khoảng cách địa lý hợp lý** (nếu có topology).
3. Giải bài toán gán (Hungarian / clustering / graph link-prediction) với ràng buộc
   *một người chỉ có một tracklet tại một thời điểm*, *không được gán 2 người cùng lúc*.
4. Bộ lọc (streaming / dominant set clustering) cập nhật online khi có camera mới.

**GMT (CVPR 2026)** đi ngược paradigm 2 giai đoạn: ghép **global trajectory** ngay từ đầu với
module Cross-View Feature Consistency Enhancement → cải thiện tới 13.1% CVMA, 19.2% CVIDF1.

**Hệ thống thu nhỏ tài nguyên: ReXCam** (mục C) dùng thống kê không-thời-gian giữa các camera để
**cắt bớt camera/khung hình cần xử lý** → giảm mạnh chi phí tính toán, có cơ chế "replay search"
tìm lại phần đã bỏ qua khi có sai sót.

## C. Hệ thống cross-camera ở quy mô thực tế

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 11 | **ReXCam: Resource-Efficient Cross-Camera Video Analytics at Scale** | 2018/2021 | https://arxiv.org/pdf/1811.01268v4.pdf |
| 12 | **RAMOT: Retrieval Augmented Continuous Person Tracking and Re-Identification** (ACM MM/ICMR) | 2026 | https://dl.acm.org/doi/10.1145/3774521.3774616 |

- **ReXCam**: mô hình "spatio-temporal profile" học (learn) từ dữ liệu lịch sử — hỏi *camera C2 có
  liên hệ mạnh với C1 trong khoảng thời gian T không?* → chỉ chạy detect/Re-ID ở camera liên quan.
- **RAMOT**: YOLOv8 (detect) + ByteTrack (track) + CNN encoder (embedding) + **FAISS vector DB**;
  duy trì **centroid embedding** theo EMA cho mỗi ID; ngưỡng cosine ~0.90 → khớp lại ID cũ,
  ngược lại tạo ID mới (open-set) → đây là mẫu hệ thống lưu trữ/truy vấn danh tính điển hình.

## D. Chỉ số đánh giá xuyên camera

- **IDF1 / IDP / IDR**: độ bền danh tính (quan trọng nhất với cross-camera).
- **ID switches (IDSW)**: số lần đổi nhầm ID.
- **HOTA, MOTA**: tổng thể detection + association.
- **mAP / Rank-k** cho bước Re-ID handover riêng lẻ.
