# Missing Person Search via FADING

Đề tài NCKH: tìm kiếm người mất tích bằng cách kết hợp **FADING** (Face Aging via
Diffusion-based Editing, BMVC 2023) — mô phỏng khuôn mặt già đi qua nhiều mốc tuổi từ 1 ảnh
duy nhất — với **InsightFace** (trích xuất embedding khuôn mặt) và **FAISS** (tìm kiếm tương
đồng), nhằm đối chiếu ảnh đã "làm già" với cơ sở dữ liệu ảnh nghi vấn.

**Ý tưởng**: 1 người mất tích khi còn nhỏ/trẻ có thể đã thay đổi ngoại hình rất nhiều sau
nhiều năm — hệ thống sinh ra nhiều phiên bản khuôn mặt ở các mốc tuổi khác nhau từ 1 ảnh cũ,
rồi tìm kiếm từng phiên bản đó trong cơ sở dữ liệu ảnh để tăng khả năng nhận diện đúng người,
thay vì chỉ so khớp trực tiếp ảnh cũ với ảnh hiện tại (chênh lệch tuổi tác quá lớn khiến
face recognition thông thường thất bại).

## Pipeline

Ảnh input → 5 module chính (+ 1 module tiền xử lý bắt buộc):

| Module | Chức năng | File |
|---|---|---|
| 1.5 — Align | Căn chỉnh khuôn mặt về đúng bố cục FFHQ (công thức gốc NVIDIA) | `src/utils/ffhq_align.py` |
| 1 — Specialization | Fine-tune UNet Stable Diffusion v1.5 trên 1 người cụ thể (double-prompt) | `src/fading/specialization.py` |
| 2 — Null-text Inversion | Đảo ngược ảnh input về latent + null-text embeddings (Mokady et al., CVPR 2023) | `src/fading/inversion.py` |
| 3 — Editing | Sinh ảnh ở các mốc tuổi mục tiêu qua cross-attention injection | `src/fading/editing.py` |
| 4 — Embedding | Trích xuất vector embedding khuôn mặt (buffalo_l) | `src/search/embedding.py` |
| 5 — FAISS Search | Tìm identity gần nhất trong gallery theo cosine similarity | `src/search/faiss_index.py` |

**Bắt buộc phải align (Module 1.5) trước khi vào Module 2** — UNet ở Module 1 chỉ học trên bố
cục ảnh đã align kiểu FFHQ; ảnh "trong tự nhiên" (chưa align, như FG-NET) nếu đưa thẳng vào sẽ
khiến Module 2/3 sinh sai hoàn toàn cấu trúc khuôn mặt (đã kiểm chứng thực nghiệm, xem phần
*Hiện trạng / giới hạn đã biết* bên dưới).

### Các lớp add-on (bọc quanh 5 module trên, không đổi logic core)

- **Ensemble** (`src/search/ensemble.py`) — gộp kết quả search qua nhiều mốc tuổi.
- **Rejection threshold** (`src/search/rejection.py`) — từ chối kết quả nếu độ tin cậy thấp.
- **Age Estimator / MiVOLO** (`src/utils/age_estimator.py`) — ước tính tuổi khi không có tuổi
  thật (ảnh người mất tích không có nhãn); `resolve_initial_age()` tách 3 nguồn tuổi ban đầu
  (nhập tay / tra CSV / MiVOLO) không trộn lẫn.
- **Cảnh báo chất lượng ảnh** (`src/utils/head_pose.py`) — cảnh báo yaw/roll/độ tin cậy detect
  quá lớn (pitch đã bỏ do sai số đo thực tế quá cao, MAE ~14.6°).
- **De-duplicate** (`src/search/deduplicate.py`) — gộp các lần phát hiện liên tiếp cùng 1
  identity (chuẩn bị cho nhánh video, chưa nối vào pipeline chính).

## Cấu trúc project

```
main.py                 # Pipeline chính (CLI) - nguồn sự thật duy nhất của luồng xử lý
app.py                  # Giao diện Streamlit - tái dùng trực tiếp các hàm trong main.py
configs/config.yaml     # Toàn bộ tham số (đường dẫn dữ liệu/checkpoint, hyperparameter...)
src/
  fading/               # Module 1-3 (Specialization, Inversion, Editing)
  search/                # Module 4-5 + ensemble/rejection/deduplicate
  utils/                 # align, age estimator, head pose, prompt helper
tests/                  # pytest cho từng module
scripts/                # Script chẩn đoán/đo đạc rời (không phải pipeline chính thức)
notebooks/              # Bản tự chứa để chạy trên Google Colab (GPU miễn phí)
docs/                   # Ghi chú giải thích kỹ thuật chi tiết
```

## Cài đặt

```bash
pip install -r requirements.txt

# MiVOLO không có trên PyPI, cài riêng (--no-deps để không hạ cấp ultralytics/timm):
pip install --no-deps "git+https://github.com/WildChlamydia/MiVOLO.git"
```

Yêu cầu GPU CUDA (đã verify chạy được trên RTX 3050 6GB VRAM nhờ fp16 + 8-bit Adam
`bitsandbytes`); có thể chạy trên CPU nhưng rất chậm.

### Dữ liệu & checkpoint (KHÔNG kèm trong repo — tự chuẩn bị)

Sửa lại đường dẫn trong `configs/config.yaml` cho đúng máy của bạn:

- **Base model**: `runwayml/stable-diffusion-v1-5` (tự tải qua HuggingFace Hub khi chạy lần đầu).
- **FFHQ-Aging**: tập ảnh mẫu có nhãn tuổi/giới tính (`sampled_labels.csv`) dùng để train
  Module 1 và làm gallery test.
- **FG-NET**: dataset ảnh thật (chưa align) dùng đánh giá độ chính xác trên "ảnh trong tự
  nhiên" — tên file dạng `<person_id 3 số>A<tuổi 2 số>.JPG` (vd `001A05.JPG`).
- **MiVOLO checkpoints**: `yolov8x_person_face.pt` + `mivolo_imdb.pth.tar` (tải từ release
  chính thức của MiVOLO, đường dẫn Google Drive ghi chú trong `config.yaml`).

## Chạy

```bash
python main.py              # Pipeline CLI - dùng 1 ảnh test cấu hình sẵn trong main.py
streamlit run app.py        # Giao diện web - upload ảnh, chọn mặt, xem kết quả trực quan
pytest tests/ -v             # Chạy toàn bộ test
```

Notebook `notebooks/FADING_pipeline_colab_1.ipynb` tự chứa toàn bộ code (không phụ thuộc
`src/`), dữ liệu lấy từ Google Drive — dùng để chạy trên Colab khi cần GPU mạnh hơn máy cá
nhân, gồm cả vòng lặp đánh giá định lượng trên toàn bộ FG-NET (ID Score + Age MAE).

## Hiện trạng / giới hạn đã biết

- **Bắt buộc phải align ảnh input** (xem Module 1.5) — thiếu bước này, ảnh sinh ra sai lệch
  hoàn toàn về cấu trúc khuôn mặt so với người gốc.
- **Cặp ảnh "trẻ nhỏ → người lớn" (chênh tuổi vượt xa giai đoạn phát triển khuôn mặt)** vẫn cho
  kết quả nhận dạng kém dù đã align đúng — giới hạn cấu trúc đã biết trước của FADING, không
  phải lỗi của bước align.
- Ngưỡng chấp nhận (`rejection_threshold` trong `config.yaml`) và các hằng số cảnh báo chất
  lượng ảnh đã được hiệu chỉnh dựa trên đo đạc thực nghiệm trên tập FFHQ mẫu, chưa phải con số
  tối ưu tuyệt đối — có thể cần tinh chỉnh thêm khi mở rộng dữ liệu đánh giá.
- Các file trong `scripts/` (tiền tố `_diag_*`) là script chẩn đoán tạm thời dùng trong quá
  trình phát triển, không phải một phần của pipeline chính thức.
