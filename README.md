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

> Repo này dùng nội bộ trong nhóm — khi tạo trên GitHub hãy để **Private** và thêm thành viên
> nhóm làm collaborator, thay vì để Public (repo chưa gắn giấy phép mã nguồn mở).

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

main.py                 # Pipeline chính (CLI) - nguồn sự thật duy nhất của luồng xử lý
app.py                  # Giao diện Streamlit (Legacy entrypoint)
backend/                # FastAPI backend bọc pipeline (Session store, GPU Mutex, WebSocket)
desktop/                # Ứng dụng Desktop (React + Vite + TypeScript & Tauri v2 shell)
configs/config.yaml     # Toàn bộ tham số (đường dẫn dữ liệu/checkpoint, hyperparameter...)
src/
  fading/               # Module 1-3 (Specialization, Inversion, Editing)
  search/               # Module 4-5 + ensemble/rejection/deduplicate
  utils/                # align, age estimator, head pose, prompt helper
tests/                  # pytest cho từng module core src/
scripts/                # Script đo đạc rời & chuẩn bị dữ liệu (validate_head_pose, prepare_test_gallery...)
notebooks/              # Notebooks nghiên cứu (Kaggle/Colab)
docs/                   # Ghi chú giải thích kỹ thuật chi tiết
data/                   # Dữ liệu ảnh và nhãn CSV (sạch sẽ, không chứa file mã nguồn .py)
checkpoints/            # Checkpoint model weights (specialized_unet, mivolo)
outputs/                # Thư mục sinh ra khi chạy pipeline (jobs, app_uploads)
```

## Bắt đầu (dành cho thành viên nhóm)

### 1. Clone và cài môi trường

```bash
git clone <URL repo>
cd MissingPersonSearch_AI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt

# MiVOLO không có trên PyPI, cài riêng (--no-deps để không hạ cấp ultralytics/timm):
pip install --no-deps "git+https://github.com/WildChlamydia/MiVOLO.git"
```

Yêu cầu **Python 3.10+**, **GPU CUDA** (đã verify chạy được trên RTX 3050 6GB VRAM nhờ fp16 +
8-bit Adam `bitsandbytes`); có thể chạy trên CPU nhưng rất chậm, không khuyến khích cho Module
1-3 (diffusion).

### 2. Chuẩn bị dữ liệu & checkpoint (KHÔNG kèm trong repo — dung lượng lớn, tự tải)

`configs/config.yaml` mặc định trỏ vào các thư mục tương đối bên dưới — đặt đúng dữ liệu vào
đây thì chạy được ngay, không cần sửa file config:

| Đường dẫn mặc định | Nội dung cần có | Nguồn |
|---|---|---|
| `data/ffhq_aging_150_samples/` | Ảnh mẫu FFHQ đã align sẵn + `sampled_labels.csv` (nhãn tuổi/giới tính) | Tập dữ liệu nội bộ nhóm — xin từ thành viên đã có |
| `data/test_gallery/` | ~26 ảnh (gồm 1 ảnh test + ảnh nhiễu) để sanity-check Top-1 search | Tự tạo bằng `data/check_data.py` (cần sửa lại đường dẫn nguồn trong script) hoặc xin từ nhóm |
| `checkpoints/specialized_unet/` | Checkpoint UNet đã fine-tune (Module 1) | Tự train qua `main.py` (150 step, mất vài phút trên GPU) — thư mục sẽ tự sinh nếu để trống |
| `checkpoints/mivolo/yolov8x_person_face.pt` | Detector người+mặt của MiVOLO | [Release chính thức MiVOLO](https://github.com/WildChlamydia/MiVOLO) — link Google Drive ghi trong `config.yaml` |
| `checkpoints/mivolo/mivolo_imdb.pth.tar` | Model age/gender của MiVOLO | Như trên |
| — | Base model `runwayml/stable-diffusion-v1-5` | Tự tải qua HuggingFace Hub khi chạy lần đầu (cần đăng nhập `huggingface-cli login` nếu bị chặn) |
| `data/FGNET.../` (tuỳ chọn) | Dataset FG-NET (ảnh thật, chưa align) — dùng đánh giá định lượng, không bắt buộc để chạy demo | [FG-NET](http://yanweifu.github.io/FG_NET_data/) — tên file dạng `<person_id 3 số>A<tuổi 2 số>.JPG`, vd `001A05.JPG` |

Nếu dữ liệu của bạn nằm ở chỗ khác, sửa trực tiếp các đường dẫn trong `configs/config.yaml`
(mục `paths` và `age_estimator`).

### 3. Chạy thử

```bash
python main.py              # Pipeline CLI - dùng 1 ảnh test cấu hình sẵn (TEST_IMAGE_NAME trong main.py)
streamlit run app.py        # Giao diện web - upload ảnh, chọn mặt, xem kết quả trực quan
pytest tests/ -v             # Chạy toàn bộ test (một số test tự skip nếu thiếu checkpoint MiVOLO)
```

Lần chạy đầu `python main.py` sẽ tự train Module 1 (Specialization, ~150 step) nếu
`checkpoints/specialized_unet/` chưa có gì — các lần sau tự động dùng lại checkpoint đã có
(xoá thư mục nếu muốn train lại).

Notebook `notebooks/FADING_pipeline_colab_1.ipynb` tự chứa toàn bộ code (không phụ thuộc
`src/`), dữ liệu lấy từ Google Drive — dùng để chạy trên Colab khi cần GPU mạnh hơn máy cá
nhân, gồm cả vòng lặp đánh giá định lượng trên toàn bộ FG-NET (ID Score + Age MAE).

> **Lưu ý đồng bộ**: notebook Colab và code trong `src/` là 2 bản sao độc lập (notebook tự
> chứa để không phụ thuộc Drive/GitHub lúc chạy) — sửa code ở 1 bên KHÔNG tự động áp dụng cho
> bên kia. Nếu sửa logic pipeline trong `src/`/`main.py`, nhớ đối chiếu lại notebook nếu cần
> dùng trên Colab.

## Hiện trạng / giới hạn đã biết

- **Bắt buộc phải align ảnh input** (xem Module 1.5) — thiếu bước này, ảnh sinh ra sai lệch
  hoàn toàn về cấu trúc khuôn mặt so với người gốc (đã đo thực nghiệm: ID Score gần 0 hoặc âm
  khi bỏ qua align trên ảnh chưa align sẵn như FG-NET).
- **Cặp ảnh "trẻ nhỏ → người lớn" (chênh tuổi vượt xa giai đoạn phát triển khuôn mặt)** vẫn cho
  kết quả nhận dạng kém dù đã align đúng — giới hạn cấu trúc đã biết trước của FADING, không
  phải lỗi của bước align.
- Ngưỡng chấp nhận (`rejection_threshold` trong `config.yaml`) và các hằng số cảnh báo chất
  lượng ảnh đã được hiệu chỉnh dựa trên đo đạc thực nghiệm trên tập FFHQ mẫu, chưa phải con số
  tối ưu tuyệt đối — có thể cần tinh chỉnh thêm khi mở rộng dữ liệu đánh giá.
- Các file trong `scripts/` là các script công cụ hỗ trợ chuẩn bị dữ liệu gallery và kiểm thử góc quay khuôn mặt (`prepare_test_gallery.py`, `check_fgnet_data.py`, `validate_head_pose.py`).

## Sự cố thường gặp

- **`UnicodeEncodeError` khi redirect output ra file trên Windows** — đã fix bằng
  `sys.stdout.reconfigure(encoding="utf-8")` ở đầu `main.py`; nếu vẫn gặp ở script khác, thêm
  2 dòng tương tự vào đầu script đó.
- **`torch.load` lỗi `UnpicklingError` khi load checkpoint MiVOLO** — checkpoint gốc dùng định
  dạng cũ, không tương thích mặc định `weights_only=True` của torch ≥ 2.6; đã xử lý sẵn trong
  `src/utils/age_estimator.py` (chỉ tin checkpoint từ release chính thức MiVOLO).
  - Cài chậm/lỗi ONNX Runtime GPU → kiểm tra bản CUDA cài đặt khớp với `onnxruntime-gpu`
  yêu cầu, hoặc tạm dùng `ctx_id: -1` (CPU) trong `config.yaml` cho `embedding`/insightface.
- **`ValueError` từ `Editor.edit()` báo lệch `num_inference_steps`** — Module 2 và Module 3
  bắt buộc dùng chung `inversion.num_inference_steps`, không được cấu hình riêng cho Module 3
  (xem comment trong `config.yaml`).

## Lưu trữ face_media (Supabase) — vận hành

Hai luồng: REFERENCE (ảnh gốc → crop → ảnh tạo sinh) và SEARCH (ảnh/video/camera →
crop quan sát). Chi tiết schema: `docs/face-media-database.md`; backlog: 
`docs/face-media-implementation-tasks.md`.

### 1. Cấu hình

```bash
copy .env.example .env   # rồi điền giá trị, KHÔNG commit .env
```

| Biến | Mô tả |
|---|---|
| `DATABASE_URL` | Postgres pooler Supabase (`postgresql://...`, bắt buộc SSL; runner tự ép `sslmode=require`, `connect_timeout=10`, bỏ cờ `pgbouncer`) |
| `DATABASE_URL_TEST` | Database test RIÊNG (runner/test từ chối khi trùng production) |
| `DATABASE_POOL_SIZE` | Pool FastAPI (mặc định 10) |
| `STORAGE_BACKEND` | `local` (thư mục `MEDIA_ROOT`, mặc định `outputs/media`) hoặc `supabase` (bucket private tạo riêng qua infra) |
| `CAMERA_<TEN>_URL` | URL/secret từng camera (DB chỉ giữ tên biến, không lưu URL) |
| `CAMERA_SAVE_SESSION_VIDEO` | `false` mặc định (không lưu video toàn phiên) |
| `FACE_MEDIA_API_KEY` | Để trống = local một người; đặt giá trị để yêu cầu header `X-API-Key` trên mọi `/api/*` (trừ `/api/health`) |

### 2. Migration (không chạy lại 001 trên DB hiện tại)

```bash
python database/migrate.py status     # xem trạng thái + phát hiện drift checksum
python database/migrate.py baseline   # 1 lần duy nhất: xác minh 9 bảng/2 view/12 FK rồi ghi nhận 001
python database/migrate.py migrate    # áp dụng 002, 003, 004, ... (mỗi migration 1 transaction, chạy 1 lần)
```

### 3. Chạy backend và worker

```bash
python -m backend.api.main            # FastAPI (lifespan mở/đóng pool)
```

- Nạp video/camera chạy ở thread nền + hàng `face_media.ingestion_runs` (tiến độ, hủy, retry):
  `POST /api/search-sources/videos` → `GET /api/ingestion-runs/{id}` →
  `POST /api/ingestion-runs/{id}/cancel|retry`.
- GPU mutex: một pipeline diffusion tại một thời điểm (409 khi bận).
- Job đang `running` khi process chết được đánh `error/interrupted` lúc startup —
  không tự chạy lại (xem T12).

### 4. Gallery FAISS và đối chiếu

```bash
# Sau khi nạp crop search: bù embedding thiếu rồi rebuild snapshot (chỉ crop search)
POST /api/gallery/rebuild
POST /api/gallery/query   # {crop_id | generated_image_id, scope_source_id?, top_k, threshold}
```

- Snapshot version theo bộ ba (model, version, preprocessing), công bố atomically.
- Mọi lượt verify video (`POST /api/jobs/{id}/video-verify` với `file` hoặc
  `source_id` đã nạp) đều lưu `search_runs`/`search_results`; `accepted` theo
  ngưỡng tách biệt xác nhận con người (`POST /api/search-runs/confirm` —
  similarity cao không phải danh tính đã xác nhận).
- Video dài hơn `max_frames` (mặc định 90 frame @3fps) bị cắt phạm vi: xem cờ
  `truncated` trong response, không coi là đã xử lý toàn bộ.

### 5. Lịch sử, backup/restore, rollback

- Lịch sử phân trang: `GET /api/search-sources`, `/api/search-sources/{id}/crops`,
  `/api/generation-jobs`, `/api/search-runs`, `GET /api/sessions/{id}` (dựng lại
  sau restart), `GET /api/generation-jobs/{id}`.
- Backup gồm cả metadata Postgres (`pg_dump`) và thư mục `MEDIA_ROOT`; kiểm thử
  restore trên môi trường test rồi mới chạy production.
- Rollback ứng dụng: checkout tag/commit cũ + `migrate` chỉ tiến tới (không có
  down-migration; schema mới tương thích đọc cũ). Không sửa file migration đã apply.

### 6. Chuyển dữ liệu cũ (T14)

```bash
python scripts/import_legacy_outputs.py --dry-run            # báo cáo mapping + gap, không ghi/xóa
python scripts/import_legacy_outputs.py --apply --reroot     # import chuỗi khôi phục được
```

Chuỗi cũ thiếu ảnh gốc (file tạm đã xóa) nên chỉ import ở chế độ `--reroot`
(crop cũ thành nguồn reference mới, provenance ghi rõ trong parameters);
`--apply` thiếu `--reroot` bị từ chối. Idempotent qua `storage_key` +
checkpoint `outputs/.import_checkpoint.json`; video cũ chỉ báo cáo, không import.
