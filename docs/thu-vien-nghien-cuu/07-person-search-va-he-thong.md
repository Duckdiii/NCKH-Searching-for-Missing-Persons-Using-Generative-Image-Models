# 07 — Person Search & hệ thống triển khai thực tế

**Person search** = tìm một người cụ thể trong **ảnh nguyên khung** (không có bbox sẵn):
phải vừa *detect* vừa *nhận diện*. Gần với thực tế giám sát nhất, vì camera không tự khoanh vùng
người mất tích.

## A. Person search (end-to-end)

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 1 | **End-to-End Deep Learning for Person Search** (CUHK-SYSU dataset; random sampling softmax) — X. Xiao et al. | 2017 | http://www.ee.cuhk.edu.hk/~xgwang/PS/paper.pdf |
| 2 | **Person Re-Identification in the Wild** (PRW dataset; phát hiện ảnh hưởng của detector tới Re-ID) — L. Zheng et al. CVPR | 2017 | https://openaccess.thecvf.com/content_cvpr_2017/papers/Zheng_Person_Re-Identification_in_CVPR_2017_paper.pdf |
| 3 | **Sequential End-to-end Network for Efficient Person Search (SeqNet)** | 2021 | https://arxiv.org/abs/2103.10148 |
| 4 | **PSTR: End-to-End One-Step Person Search With Transformers** — J. Cao et al. CVPR | 2022 | https://openaccess.thecvf.com/content/CVPR2022/html/Cao_PSTR_End-to-End_One-Step_Person_Search_With_Transformers_CVPR_2022_paper.html |
| 5 | **Sequential Transformer for End-to-End Person Search (SeqTR)** | 2022 | https://arxiv.org/pdf/2211.04323 |
| 6 | **Query-Guided End-To-End Person Search (QEEPS)** — B. Munjal et al. CVPR | 2019 | https://openaccess.thecvf.com/content_CVPR_2019/papers/Munjal_Query-Guided_End-To-End_Person_Search_CVPR_2019_paper.pdf |
| 7 | **Diverse Knowledge Distillation for End-to-End Person Search** | 2020 | https://arxiv.org/pdf/2012.11187v1.pdf |

**Điểm chung & bài học:**
- Hai hướng: **2 bước** (detector riêng + Re-ID riêng → chính xác hơn) và **1 bước** (một mạng
  chung → nhanh hơn, đang bắt kịp về accuracy).
- **Chất lượng detector quyết định Re-ID**: PRW cho thấy ngưỡng IoU > 0.7 phản ánh ảnh hưởng
  tới Re-ID tốt hơn IoU > 0.5 → phải tối ưu localization, không chỉ embedding.
- Gallery lớn (nhiều người lạ) → mAP giảm; cần **re-ranking** và chỉ số đủ độ phân biệt.

## B. Hệ thống triển khai có thật (dùng FAISS/ vector store)

| # | Hệ thống / bài báo | Nội dung | Link |
|---|---|---|---|
| 8 | **RAMOT: Retrieval Augmented Multi-Object Tracking** | YOLOv8 + ByteTrack + ResNet ReID + **FAISS**; centroid EMA; open-set; sub-ms latency hàng nghìn ID | https://dl.acm.org/doi/10.1145/3774521.3774616 |
| 9 | **Real-Time Person Re-Identification with OSNet and FAISS** (repo) | RetinaNet + OSNet (512-D) + `IndexIDMap(IndexFlatIP)`; ngưỡng cosine 0.7 → ID mới/nhận lại; pipeline minh họa tốt cho đề tài | https://github.com/ortatepeyusuf/Real-Time-Person-ReID |
| 10 | **Human Congestion Monitoring System Using YOLOv8, DeepSORT, ResNet-50 ReID, FAISS** | Kho `IndexFlatL2` dim=2048 kèm metadata (camera, timestamp, bbox); 2 chế độ: match realtime người mất tích + truy vấn 1-shot trên lịch sử; cảnh báo qua WebSocket + SQLite | https://ijirt.org/publishedpaper/IJIRT205228_PAPER.pdf |
| 11 | **Video Surveillance face clustering (MSDB + ODIVC, VISAPP 2021)** | Đăng ký ID mới + cập nhật online, 40 camera/GPU, **không lưu thông tin sinh trắc học thô** | https://www.scitepress.org/PublicationsDetail.aspx?ID=Glp7lm89wDg%3D&t=1 |

**Bài học thiết kế từ các hệ thống trên:**
1. Chuỗi chuẩn: `detect → track → crop → embedding → index → ngưỡng → ID/ cảnh báo`.
2. Lưu **centroid** thay vì mọi frame embedding → tiết kiệm bộ nhớ, tăng độ ổn định.
3. Metadata luôn lưu cạnh index (camera_id, timestamp, bbox, path ảnh) để **truy vết** kết quả.
4. Ngưỡng ~0.7 (cosine) cho Re-ID toàn thân, ~0.9 cho hệ thống chặt hơn → phải tinh chỉnh
   trên dữ liệu thật (FAR/FRR).
5. Cảnh báo realtime nên đẩy qua kênh sự kiện (WebSocket/MQTT), không polling DB.

## C. Ứng dụng trực tiếp cho "tìm người mất tích"

- **Ảnh query** = ảnh thật của người hoặc **ảnh sinh từ model** (đề tài) → trích cùng embedding.
- Xếp hạng top-k trên toàn kho → hiển thị nghi phạm kèm điểm tương đồng + camera/thời điểm;
  re-ranking (CA-Jaccard, file `01`) để nâng chất lượng danh sách.
- Nếu **không khớp** (open-set): cảnh báo "chưa từng xuất hiện" → có thể mở rộng tìm theo
  thuộc tính mềm (giới tính, màu áo — attribute-based retrieval, xem dataset RAP file `08`).
- Hạn chế cần nêu trong báo cáo: thay đổi quần áo theo thời gian, chất lượng camera thấp,
  che khuất → cần kết hợp nhiều nguồn (mặt + toàn thân + thuộc tính + lịch sử xuất hiện).
