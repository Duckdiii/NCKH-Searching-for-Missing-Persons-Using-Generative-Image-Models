# Báo cáo kiểm thử lưu trữ với pipeline thực tế — 26/09/2026

## Kết luận

Đã chạy model thật, API FastAPI thật và PostgreSQL/Supabase thật bằng cấu hình `src/.env`. Nhánh ảnh tham chiếu, crop, ảnh quan sát, video, tạo sinh và truy vấn gallery đều có dữ liệu được commit. Kết quả cuối: **đạt các kiểm tra dữ liệu, có một lần truy vấn ảnh sinh phải chạy lại bằng kết nối mới**. Chưa xác định nguyên nhân lần chờ kết nối đó; không coi đây là kiểm chứng độ ổn định liên tục.

Test gọi ứng dụng qua FastAPI TestClient, không qua giao diện desktop. Không kích hoạt lifespan để tránh tác động các job khác thông qua reconcile_interrupted. Không mock model hoặc DB trong bài smoke test. Bộ test hồi quy offline được cô lập khỏi DB thật.

## Các lỗi đã sửa và cấu hình đã bổ sung

1. Backend chỉ tìm `.env` ở vị trí mặc định, không đọc `src/.env`, khiến persistence rơi về RAM/local. Đã nạp đường dẫn theo gốc repository, ưu tiên biến môi trường tiến trình → `.env` gốc → `src/.env`.
2. DB chỉ có 9 bảng nền tảng. Đã baseline schema 001 hiện hữu, áp dụng migration 002–004; hiện có 14 bảng nghiệp vụ và `schema_migrations` trong schema **face_media**. Không tạo lại các bảng nền tảng.
3. API nhập ảnh dùng schema `conditions: Dict[str, int]`, nhưng engine trả `list[str]`. Đã chuyển thành bảng đếm cho cả ảnh có mặt và không có mặt.
4. psycopg trả UUID dạng đối tượng, gây lỗi validation khi API đọc trạng thái video/lịch sử. Đã chuẩn hóa UUID nhận từ truy vấn text thành chuỗi tại connection adapter.
5. API truy vấn crop và helper tạo embedding chỉ chấp nhận `purpose=search`, từ chối ảnh tham chiếu. Đã cho phép embedding/query crop reference; bộ lọc gallery vẫn chỉ lấy crop search.

## Dữ liệu và kết quả

- Mẫu: `outputs/aligned_input.png`. Tuổi đầu vào nhập thủ công 30, năm ảnh 2020, không dùng ước lượng tuổi trong lần thử này.
- Video: tạo 8 frame từ ảnh mẫu, 4 FPS; đây là clip thử nghiệm tĩnh, không phải camera trực tiếp.
- Reference: lưu ảnh gốc, phát hiện mặt, lưu crop; xóa cache phiên trong RAM rồi đọc lại từ DB thành công.
- Search image: lưu nguồn ảnh, frame, detection, crop và embedding thành công.
- Video: nhập nguồn video, lấy 2 frame ở 1 FPS, phát hiện/lưu 2 khuôn mặt; run kết thúc `done`.
- Gallery: dựng từ 5 embedding quan sát, gồm dữ liệu của những lần thử trước. Không có vector bị loại.
- Truy vấn crop tham chiếu: lưu một search_run và 5 search_results. Điểm đầu bảng khoảng 0,9314; đây không phải đánh giá độ chính xác nhận dạng vì dùng cùng ảnh làm fixture.
- Tạo sinh: sử dụng lại `checkpoints/specialized_unet`; không huấn luyện lại. Inversion 50 bước, Editing CFG 4,0; sinh một ảnh tuổi 36, job trong DB `done`.
- Truy vấn ảnh tạo sinh: lưu thêm một search_run và 5 search_results, sau khi chạy lại riêng bằng kết nối mới.
- 10 file ảnh/video của lần smoke chính: SHA-256 và byte_size khớp metadata.
- 1 ảnh tạo sinh: SHA-256, byte_size và kích thước ảnh khớp metadata; lineage trỏ đúng crop đầu vào.
- Embedding crop tham chiếu: 512 chiều, chuẩn hóa L2 đạt kiểm tra sai số.
- Toàn bộ candidate của hai lượt tìm kiếm đều là crop `purpose=search`.
- **69 test backend đạt**, 2 cảnh báo deprecation từ thư viện TestClient. `git diff --check` trên các file đã sửa đạt.

## Mã để đối chiếu trên Supabase

Chọn schema **face_media**, không phải `public`.

| Đối tượng | ID |
| --- | --- |
| Lần smoke chính | af7ff385-a7ff-4eb8-9d32-b720b8611fcb |
| Session | 94a134e5-0bcc-4134-8509-99dbafb0d3d7 |
| Reference crop | 12aba08e-bd2e-4ea3-9c4e-43592e6914ae |
| Video ingestion run | ed913b95-10e8-4454-852b-19cd55061c92 |
| Generation job | a14aa5a4-9e0e-41a9-bd92-97410df5160d |
| Generated image | 3b89905d-58b5-4cc3-bc6f-7a13abe1d3b9 |
| Search run — crop | d7e88a56-d6f0-48fc-bce7-16ec2882102d |
| Search run — generated | c05634c4-4f1a-420d-875a-8b505e493646 |

Bằng chứng JSON: `outputs/live_storage_tests/af7ff385-a7ff-4eb8-9d32-b720b8611fcb/report.json`.
Log test offline: `outputs/storage_regression.log`.
Log pipeline: `outputs/live_storage_smoke.log`.
Log truy vấn lại: `outputs/generated_query_retry.log`.
Dữ liệu thử được giữ lại để kiểm tra, tên ảnh đăng ký bắt đầu bằng `live-smoke-`. Run thử bị ngắt do lỗi UUID đã được đánh dấu error, không để trạng thái running giả.

## Giới hạn và điểm còn cần xử lý

- File ảnh/video hiện lưu tại `outputs/media` trên máy, metadata/embedding lưu trên Supabase PostgreSQL. `SupabaseMediaStorage` chưa triển khai; đây không phải kiểm thử Supabase bucket.
- Một request sau khi tạo sinh chờ bất thường: PostgreSQL quan sát thấy kết nối idle in transaction sau `BEGIN`. Chỉ dừng tiến trình smoke của lần thử, không dừng backend khác. Request chạy lại với process mới thành công. Cần kiểm tra thêm vòng đời connection/pooler và timeout; nguyên nhân gốc chưa xác lập.
- Bước tìm kiếm tự động trong `job_runner` vẫn dùng `gallery_dir` và `main.run_embedding_and_search`. Hai bản ghi search_run trong bài thử được tạo qua `/api/gallery/query` riêng; chưa chứng minh job tự ghi lịch sử FAISS vào DB.
- `duration_sec` của clip 2 giây được ghi là 1 giây, vì implementation lấy timestamp frame được lấy mẫu cuối. Cần tách thời lượng nguồn khỏi thời điểm lấy mẫu cuối.
- Chưa kiểm thử camera trực tiếp, mất mạng giữa transaction, rollback nhiều ảnh tạo sinh, tải đồng thời hoặc xóa dữ liệu thử. Lần thử chỉ sinh một mốc tuổi.

## Chạy lại

Dùng Python của `.venv`; Python hệ thống kiểm tra ban đầu thiếu psycopg. Các dependency FastAPI cần thiết đã được cài vào `.venv` theo requirements của dự án.

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\python.exe -u scripts/smoke_face_media_live.py --generation
```

Lệnh smoke **ghi thêm dữ liệu thật** vào DB cấu hình trong `src/.env` và lưu file local. Mỗi lần chạy có UUID riêng. Sau khi thành công:

```powershell
.\.venv\Scripts\python.exe scripts/verify_face_media_smoke.py outputs/live_storage_tests/<run-id>/report.json
```
