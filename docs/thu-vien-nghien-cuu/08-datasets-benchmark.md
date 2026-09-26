# 08 — Dataset & benchmark (đánh giá duy trì danh tính xuyên camera)

| Dataset | Quy mô / đặc điểm | Liên kết |
|---|---|---|
| **Market-1501** (Zheng et al. ICCV 2015) | 1.501 ID, 32.668 ảnh, 6 camera, có distractor | https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf |
| **CUHK03** (Li et al. CVPR 2014) | 1.360 người, 13.164 ảnh, 10 camera (5 cặp), có bản detect tự động | https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf |
| **DukeMTMC / DukeMTMC-reID** (Ristani et al.) | 8 camera 1080p60, 2.834 ID gốc; bộ re-ID: 1.404 ID, 16.5k train / 2.2k query / 17.6k gallery. *Lưu ý: bộ dữ liệu đã bị rút vì vấn đề đồng thuận/quyền riêng tư* | https://arxiv.org/abs/1609.01775 |
| **PRW** (Zheng et al. CVPR 2017) | 6 camera, 932 ID, 11.816 frame, 43.110 bbox — cho phép đánh giá **end-to-end** (detect + re-ID) | https://openaccess.thecvf.com/content_cvpr_2017/papers/Zheng_Person_Re-Identification_in_CVPR_2017_paper.pdf |
| **CUHK-SYSU** (Xiao et al. 2017) | 18.184 ảnh street/movie, 8.432 ID — benchmark person search | http://www.ee.cuhk.edu.hk/~xgwang/PS/paper.pdf |
| **MegaFace** (Kemelmacher-Shlizerman et al. CVPR 2016) | 1 triệu ảnh / 690K người — đo độ sụt giảm khi gallery lớn (10 → 10^6 distractor) | https://arxiv.org/pdf/1512.00596 |
| **RAP** (Richly Annotated Pedestrian) | 84.928 ảnh, 72 thuộc tính, 2.589 ID từ 25 camera — hỗ trợ **attribute-based search** | https://zhangzhang80.github.io/RAP.pdf |
| **MTMMC** (2024) | 16 camera RGB+nhiệt, >3 triệu frame, 3.669 ID — MTMC tracking thực tế | https://arxiv.org/html/2403.20225v1 |
| **CHIRLA** (2025) | 7 camera / 7 tháng, thay quần áo, ~1M bbox — re-ID **dài hạn**, indoor | https://arxiv.org/html/2502.06681 |
| **Multi-Scene Cross-Camera Facial Dataset** | 4.200 clip đã gắn ID toàn cục, 4 camera (thang máy, escalator, hành lang) | https://www.surfing.ai/free-samples/multi-scene-cross-camera-facial-dataset/ |
| **FaceSurv** (2019) | 252 người / 460 video, RGB + NIR, khoảng cách 36ft — nhận diện mặt trong giám sát | https://iab-rubric.org/images/pdf/papers/2019_FG_faceSurv.pdf |

**Dataset phổ biến khác** (chi tiết trong survey file `01`): MARS, DukeMTMC-VideoReID, iLIDS-VID,
PRID2011 (video Re-ID); MSMT17, VIPeR, CUHK01/02 (ảnh Re-ID); SYSU-MM01, LLCM (RGB–infrared);
IJB-C (video/ảnh mặt mở rộng); ChokePoint (mặt đa camera).

**Chỉ số đánh giá chuẩn**
- Re-ID: **mAP**, **Rank-1/5/10**, **mINP**.
- Tracking liên camera: **IDF1**, **HOTA**, **MOTA**, **ID switches**.
- Mở-set: **DIR@FAR** (Detection & Identification Rate), TAR@FAR, EER.
- Luôn báo cáo kèm **số distractor** — với 1 triệu distractor, Rank-1 giảm mạnh (MegaFace).

## Bộ dữ liệu cho bài toán "người mất tích" cụ thể
- Không có benchmark chuẩn quốc tế cho *missing person search*; cách tiếp cận thực tế:
  1) dùng Person Re-ID benchmark (Market-1501/PRW) làm proxy,
  2) dựng bộ riêng từ camera thật + ảnh query sinh (Gen-AI) — nên ghi rõ provenance ảnh,
  3) đánh giá cả phương án **ảnh sinh → ảnh thật** (domain gap giữa ảnh model sinh và ảnh camera).
- Dự án tham chiếu: FG-NET (lão hóa khuôn mặt, đã có `fgnet_eval_results.csv` trong repo)
  → phục vụ đánh giá tính bền vững của danh tính theo tuổi tác.
