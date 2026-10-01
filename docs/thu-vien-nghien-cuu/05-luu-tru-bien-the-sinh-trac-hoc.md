# 05 — Lưu trữ & bảo vệ template sinh trắc học (Biometric Template Protection)

Khi phải **lưu danh tính lâu dài** (người mất tích có thể tái xuất hiện nhiều tháng sau),
biến thể sinh trắc học (khuôn mặt, dấu vân tay…) là dữ liệu **không thể thu hồi** nếu lộ:
dấu vân tay/mặt sống với người đó suốt đời. Do đó kho lưu trữ phải đáp ứng tiêu chí
**ISO/IEC 24745**: `irreversibility` (không đảo được), `unlinkability` (không nối được các bản sao),
`renewability/cancelability` (đổi/hủy được khi bị lộ).

## A. Survey tổng quan

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 1 | **Cancelable Biometrics: a comprehensive survey** — Artificial Intelligence Review (>120 kỹ thuật, taxonomy mới) | 2019 | https://doi.org/10.1007/s10462-019-09767-8 |
| 2 | **A survey on biometric cryptosystems and cancelable biometrics** — C. Rathgeb, A. Uhl (EURASIP J. Information Security) | 2011 | https://wavelab.at/papers/Rathgeb11e.pdf |
| 3 | **A survey on face template protection methods** (CECIIS) — phân loại: Key-binding / Key-generating / Cancelable / Homomorphic Encryption | 2022 | https://archive.ceciis.foi.hr/app/public/conferences/2022/Proceedings/IS/IS3.pdf |
| 4 | **A Survey on Biometrics and Cancelable Biometrics Systems** — Choudhury et al. | — | https://research.tees.ac.uk/ws/files/4230794/621491.pdf |
| 5 | **A Survey on Biometric Template Security Schemes and Protection In Cloud Environment** — IJAER | — | https://www.ripublication.com/ijaer10/ijaerv10n9_105.pdf |
| 6 | **A Review on Protection and Cancelable Techniques in Biometric Systems** — Bernal-Romero et al. (5 nhóm: cryptosystem, cancelable, ML/DL-based, hybrid, multibiometric) | 2023 | https://exa.ai/library/publication/zl6858mpk4v |
| 7 | **Feature extraction and learning approaches for cancellable biometrics: A survey** — Y. Yang, S. Wang, J. Hu, X. Tao, Y. Li | 2024 | https://exa.ai/library/publication/ng5dkl798vg |
| 8 | **Biometric Template Protection for Neural-Network-based Face Recognition Systems: A Survey of Methods and Evaluation Techniques** | 2021 | https://www.researchgate.net/publication/355221642_Biometric_Template_Protection_for_Neural-Network-based_Face_Recognition_Systems_A_Survey_of_Methods_and_Evaluation_Techniques |

## B. Các phương thức lưu trữ template **hoạt động như thế nào**

### 1. Cancelable Biometrics (biến đổi một chiều) — *so khớp trong miền đã biến đổi*
- **Nguyên tắc**: áp phép biến đổi `T_k` (tham số là khóa/PIN ngẫu nhiên) lên template rồi **mới lưu**.
  Khi so khớp: biến đổi query bằng đúng `T_k` rồi so trong miền biến đổi — **không bao giờ** lưu/decrypt bản gốc.
- **Hai nhánh con**:
  - *Salting* (xát muối): biến đổi có thể đảo được nhưng phụ thuộc khóa bí mật → an toàn nếu khóa giữ kín.
  - *Non-invertible transform*: hàm một chiều (sinh ngẫu nhiên có kiểm soát, DCT + Huffman, random permutation…)
    → đảo lại rất khó; đổi khóa → sinh template mới (cancel/revoke).
- **Ưu/nhược**: chất lượng nhận diện thường giảm nhẹ; bài toán *căn chỉnh* (alignment) template sau biến đổi rất khó.

### 2. Biometric Cryptosystem (trộn khóa vào template)
- **Key binding**: trộn khóa bí mật vào template bằng *fuzzy commitment* / *fuzzy vault*
  (kèm helper data để sửa lỗi) → truy vấn đúng sinh trắc học mới giải được khóa.
- **Key generation**: sinh khóa trực tiếp từ template (helper data chứa thông tin sửa lỗi).
- Đặc tính: match phải **chính xác 100%** (không phải ngưỡng như so khớp thường) → dùng ECC sửa lỗi
  do biến thiên nội lớp (nội trạng thái khuôn mặt khác nhau giữa các lần chụp).

### 3. Homomorphic Encryption (so khớp trong miền mã hóa)
- Mã hóa vector đặc trưng, server **tính khoảng cách/cosine trên dữ liệu đã mã hóa**, trả về kết quả
  mà không từng nhìn thấy template → phù hợp lưu trữ đám mây (không tin server).
- Nhược: chậm hơn nhiều so với so khớp plaintext.

### 4. Deep-learning template protection (mới nhất)
- Dùng CNN học hàm biến đổi: sinh template bảo mật nhưng vẫn giữ độ phân biệt cao;
  phải kèm cơ chế "sinh lại template mới" nếu bị lộ (xem mục 8 ở trên).

### 5. Các kiểu tấn công kho template cần phòng ngừa
1. Tấn công giao diện người dùng (sensor giả).
2. Tấn công giữa 2 module (chặn/điều khiển truyền template).
3. Tấn công phần mềm (sửa module trả kết quả giả).
4. **Tấn công kho template** (trộm/sửa/spoofing) — nhóm này là lý do tồn tại của file này.
   - *Record multiplicity*: ghép nhiều bản sao template đã biến đổi để tái dựng bản gốc.

## C. Gợi ý cho hệ thống của đề tài

- Ảnh **crop mặt đã đăng ký**: có thể chỉ lưu **embedding đã băm/biến đổi** + ngưỡng;
  ảnh hiển thị nên ở dạng đã xóa nhận dạng vùng mặt nhạy cảm hoặc lưu ở nơi truy cập có kiểm soát.
- Mẫu đối chiếu (ảnh người mất tích từ model sinh): cần **lưu provenance** (prompt, seed, model)
  để phân biệt ảnh sinh với ảnh thật — tránh "tuồn" ảnh giả vào kho nhận diện.
- Nếu triển khai cloud: ưu tiên pgvector + kiểm soát truy cập theo hàng (row-level),
  template mã hóa trước khi rời biên (edge).
