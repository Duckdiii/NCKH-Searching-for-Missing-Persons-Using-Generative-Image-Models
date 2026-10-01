# 01 — Duy trì danh tính xuyên camera: Person Re-Identification

Bài toán: cho một ảnh "probe" của người quan tâm, tìm lại người đó trong các khung hình do
**camera khác** (thường không chồng lấp trường nhìn) đã ghi lại. Đây là cốt lõi của việc
"duy trì danh tính xuyên camera".

## A. Survey / tổng quan

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 1 | **Deep Learning for Person Re-identification: A Survey and Outlook** — M. Ye, J. Shen, L. Shao, B. Lin, T. Xiang, S.C.H. Hoi (TPAMI 2022) | 2020/2022 | https://arxiv.org/abs/2001.04193 |
| 2 | **Survey on Reliable Deep Learning-Based Person Re-Identification Models: Are We There Yet?** — N. B. Bakalarczyk, H. Boujnah, M. Megherbi (arXiv:2005.00355) | 2020 | https://arxiv.org/html/2005.00355 |
| 3 | **Deep learning-based person re-identification methods: A survey and outlook of recent works** — Z. Ming, M. Zhu, et al. (Image and Vision Computing) | 2021 | http://arxiv.org/pdf/2110.04764v2 |
| 4 | **Advancements and Challenges in Deep Learning-Based Person Re-Identification: A Review** — (Electronics, MDPI, 14(22):4398) | 2025 | https://www.mdpi.com/2079-9292/14/22/4398 |
| 5 | **Deep video-based person re-identification (Deep Vid-ReID): comprehensive survey** — (EURASIP J. Advances in Signal Processing) | 2024 | https://link.springer.com/article/10.1186/s13634-024-01139-x |
| 6 | **Person Re-Identification: Past, Present and Future** — L. Zheng, Y. Yang, A.G. Hauptmann | 2016 | https://arxiv.org/abs/1610.02565 |

**Ghi chú nội dung:**
- Ye et al. (TPAMI 2022) là survey được trích nhiều nhất: phân tách Re-ID thành **closed-world**
  (đóng: tập gallery có sẵn, query chắc chắn nằm trong gallery) và **open-world**
  (mở: domain shift, gallery lớn, mẫu chưa từng thấy — sát với thực tế giám sát).
  Gồm 3 thành phần: *feature representation learning → deep metric learning → ranking optimization*.
  Cũng đề xuất baseline **AGW** và chỉ số **mINP** (chi phí tìm hết các match đúng).
- Survey của Electronics 2025 tổng kết tới 08/2024, gồm Transformer/LLM-based Re-ID, đa mô-đun
  (RGB–infrared, text–image), và các vấn đề đạo đức/quyền riêng tư.

## B. Phương pháp cốt lõi (đặc trưng + khoảng cách)

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 7 | **Scalable Person Re-identification: A Benchmark** (Market-1501) — L. Zheng et al. ICCV | 2015 | https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf |
| 8 | **DeepReID: Deep Filter Pairing Neural Network for Person Re-identification** (CUHK03) — L. Li et al. CVPR | 2014 | https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf |
| 9 | **Omni-Scale Feature Learning for Person Re-Identification (OSNet)** — K. Zhou et al. ICCV | 2019 | https://arxiv.org/abs/1905.03220 |
| 10 | **Bag of Tricks and a Strong Baseline for Deep Person Re-Identification (BoT)** — H. Luo et al. | 2019 | https://arxiv.org/abs/1903.07028 |
| 11 | **TransReID: Deep Transformer Branching and Aggregating for Person Re-Identification** — S. He et al. ICCV | 2021 | https://arxiv.org/abs/2105.06722 |
| 12 | **A Pose-Sensitive Embedding for Person Re-Identification Full Body Re-ID** — M.S. Sarfraz et al. CVPR | 2018 | https://openaccess.thecvf.com/content_cvpr_2018/papers/Sarfraz_A_Pose-Sensitive_Embedding_CVPR_2018_paper.pdf |
| 13 | **Unsupervised Cross-camera Person Re-identification (ACAN)** — (Adversarial Camera Alignment Network) | 2019 | https://arxiv.org/pdf/1908.00862v2.pdf |

**Cơ chế hoạt động (chuẩn chung):**
1. Detect/track → crop người → CNN/ViT → vector đặc trưng (đại diện ngoại hình).
2. **Metric learning**: triplet loss, contrastive, classification loss để cùng ID xích lại,
   khác ID tách ra (quan trọng nhất là *cross-camera positive pair*).
3. **Re-ranking** sau khi truy vấn: k-reciprocal neighbors, Jaccard distance để nâng top-k.
4. **Đối chiếu cross-camera**: huấn luyện đối kháng theo camera ID (ACAN) để "gộp" phân bố
   các camera về không gian chung, giảm chênh lệch ánh sáng/góc nhìn.

## C. Re-ranking & xử lý chênh lệch giữa camera

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 14 | **CA-Jaccard: Camera-aware Jaccard Distance for Person Re-identification** (CVPR 2024) | 2024 | https://arxiv.org/abs/2311.10605 |
| 15 | **Person Re-Identification Under Non-Overlapping Cameras Based on Advanced Contextual Embeddings** (TransReID + CA-Jaccard) | 2025 | https://exa.ai/library/publication/l7nv29534st |

- **CA-Jaccard**: dùng thông tin `camera_id` làm ràng buộc khi so sánh danh sách hàng xóm —
  hàng xóm *liên camera* được tin cậy/ưu tiên hơn hàng xóm cùng camera → cải thiện mAP đáng kể
  (Market-1501: mAP 88.2% → 93.58%). Ý tưởng "so khớp khác camera được ưu tiên" rất hợp
  bài toán tìm người xuyên giám sát.

## D. Đo lường

- **CMC / Rank-k**: xác suất xuất hiện match đúng trong top-k.
- **mAP**: trung bình độ chính xác theo từng query.
- **mINP** (Ye et al.): đo chi phí tìm được *tất cả* match đúng.
- Phải luôn báo cáo kèm số lượng distractor (người lạ) — ảnh hưởng rất lớn (xem MegaFace, file `03`).

> Liên quan trực tiếp tới đề tài: ảnh query từ ảnh sinh (Gen-AI) được coi là "probe",
> kho gallery = embedding của các frame camera giám sát → toàn bộ pipeline trên áp dụng trực tiếp.
