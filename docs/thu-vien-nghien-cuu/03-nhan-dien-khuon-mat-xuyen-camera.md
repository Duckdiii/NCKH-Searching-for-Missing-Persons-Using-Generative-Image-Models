# 03 — Nhận diện khuôn mặt xuyên camera & hệ thống quy mô lớn

Phương án "nhận diện bằng mặt" thay vì ngoại hình toàn thân: trực tiếp, chính xác cao khi camera
đủ nét, nhưng chịu nhiều hạn chế về góc nhìn/khoảng cách trong giám sát.

## A. Nhận diện khuôn mặt trong video giám sát đa camera

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 1 | **Face Recognition in Multi-Camera Surveillance** (Dynamic Bayesian Network kết hợp nhiều camera + tín hiệu thời gian) — M. Kafai, B. Bhanu et al. (ICDSC) | — | http://alumni.cs.ucr.edu/~mkafai/papers/Paper_icdsc.pdf |
| 2 | **Real-Time Face Recognition from Surveillance Video** (đặc trưng shape-based, kiến trúc node biên → back-end) | — | https://pureadmin.qub.ac.uk/ws/files/6903843/FaceRecognition_colour.pdf |
| 3 | **Multi-Stage Dynamic Batching and On-Demand I-Vector Clustering for Cost-effective Video Surveillance** (VISAPP 2021: nhận diện + clustering online, 40 camera/GPU, **không lưu dữ liệu sinh trắc học thô**) | 2021 | https://www.scitepress.org/PublicationsDetail.aspx?ID=Glp7lm89wDg%3D&t=1 |

**Cơ chế của (1):** mỗi camera sinh vector mặt → DBN (mạng Bayes động) ghép thông tin liên camera
và liên frame → nhận diện theo đa số/thời gian thay vì 1 frame đơn lẻ → tăng độ chính xác.

**Cơ chế của (3):** trích **i-vector** theo lô (dynamic batching) rồi **clustering online** để
tự động đăng ký ID mới và cập nhật khi người đó xuất hiện lại → mô hình lưu trữ "không giữ
thông tin sinh trắc học", chỉ giữ vector đặc trưng đã gán.

## B. Trích xuất embedding mặt (nền tảng lưu trữ)

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 4 | **DeepFace: Closing the Gap to Human-Level Performance in Face Verification** — Y. Taigman et al. CVPR | 2014 | https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf |
| 5 | **FaceNet: A Unified Embedding for Face Recognition and Clustering** — F. Schroff et al. CVPR | 2015 | https://openaccess.thecvf.com/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf |
| 6 | **ArcFace: Additive Angular Margin Loss for Deep Face Recognition** — J. Deng et al. CVPR | 2019 | https://arxiv.org/pdf/1801.07698.pdf |
| 7 | **Web-Scale Training for Face Identification** (học trên 10 triệu ID, semantic bootstrapping) — N. Taigman et al. CVPR | 2014 | https://arxiv.org/pdf/1406.5266 |

**Liên quan trực tiếp tới việc lưu trữ:**
- **FaceNet**: mỗi khuôn mặt chỉ chiếm **128 byte** (128-D embedding), khoảng cách L2 = mức độ
  tương đồng; nhận diện = *k-NN / clustering* trên vector đã lưu. → 1 triệu người ≈ 128 MB
  (chưa tính overhead index).
- **ArcFace**: loss góc phụt (angular margin) → embedding phân biệt hơn, cho phép 1:N với
  hàng triệu gallery; sub-center ArcFace tự làm sạch nhãn nhiễu — quan trọng khi dữ liệu
  ảnh web/thu từ camera có nhiễu.
- Cả FaceNet/ArcFace đều là **closed-set→open-set**: lưu embedding đại diện, so bằng
  cosine + ngưỡng (không dùng softmax số lớp cố định).

## C. Bài toán quy mô lớn (gallery triệu-đến-tỷ người)

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 8 | **The MegaFace Benchmark: 1 Million Faces for Recognition at Scale** — I. Kemelmacher-Shlizerman et al. CVPR | 2016 | https://arxiv.org/pdf/1512.00596 |
| 9 | **WebFace260M: A Benchmark for Million-Scale Deep Face Recognition** — D. Wang et al. TPAMI | 2021/2023 | https://www.computer.org/csdl/journal/tp/2023/02/09763004/1CT4Sf3ptmM |

**Kết luận then chốt từ MegaFace** (rất quan trọng cho thiết kế kho lưu trữ):
- Với 10 distractor: >95% độ chính xác; với **1 triệu distractor: chỉ còn 35–75%**.
- → Kích thước gallery làm **sụt giảm mạnh** độ chính xác ⇒ phải có **chỉ mục ANN +
  re-ranking + ngưỡng open-set**, không thể so sánh tuần tự thô.
- Dữ liệu huấn luyện càng lớn càng tốt khi ở quy mô lớn (FaceNet huấn luyện >500 triệu ảnh).

## D. Face template — lưu trữ & bảo mật

Xem file [`05-luu-tru-bien-the-sinh-trac-hoc.md`](05-luu-tru-bien-the-sinh-trac-hoc.md):
bảo vệ template mặt (cancelable biometrics, homomorphic encryption) khi phải lưu lâu dài.
