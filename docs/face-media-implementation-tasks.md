# Backlog bàn giao: tích hợp lưu trữ khuôn mặt với Supabase

## Mục tiêu và hiện trạng

Hai luồng bắt buộc:

- REFERENCE: ảnh gốc người cần tìm → ảnh crop → ảnh tạo sinh.
- SEARCH: video camera / video tải lên / ảnh → ảnh crop quan sát để tìm kiếm.
- Crop reference và ảnh tạo sinh được đối chiếu với crop search. Không tạo sinh
  từ crop search; không đưa ảnh tạo sinh vào gallery quan sát.

Đã tạo và xác minh trên Supabase: schema face_media có 9 bảng, 2 view, 12 FK.
DDL: database/001_face_media.sql. Thiết kế: docs/face-media-database.md.
Chưa tích hợp API ghi/đọc DB; chưa tạo bucket Supabase Storage trong công việc này.
Session/job hiện ở RAM, ảnh nằm trong outputs. Driver psycopg và python-dotenv
đã cài ở .venv nhưng chưa khai báo trong requirements.txt.

Tài liệu là danh sách việc cần làm, không phải báo cáo các tính năng đã triển khai.
Không chạy lại 001 trên database hiện tại. Mọi thay đổi schema dùng migration mới.
Không gửi .env cho người khác hoặc đưa secret vào frontend, log, commit.

## Quy ước bàn giao

P0 = nền tảng bắt buộc; P1 = hoàn thiện nghiệp vụ; P2 = chuyển dữ liệu/vận hành.
Mỗi task bàn giao bằng PR gồm code/migration, cách chạy, bằng chứng nghiệm thu,
biến cấu hình cần thiết chỉ ghi tên và mô tả, cùng giới hạn còn lại.
Các đường dẫn mới dưới đây là đề xuất. Kiểm tra thay đổi đang có trước khi sửa;
đặc biệt video_verify.py, schemas.py, preprocessing và VideoVerify.tsx.

## T01 — Kết nối DB và quản lý migration [P0]

**Phụ trách:** backend/database. **Phụ thuộc:** không.

- Tạo module backend/api/database.py, đọc DATABASE_URL phía backend; kiểm tra
  cấu hình thiếu bằng lỗi rõ ràng, không in DSN/mật khẩu.
- Chuẩn hóa URL: bỏ tham số pgbouncer không thuộc libpq; giữ các tham số hợp lệ;
  cấu hình SSL bắt buộc và prepare_threshold=None cho kết nối pooler hiện có.
- Chọn pool kết nối phù hợp worker FastAPI; đóng pool khi ứng dụng dừng; không
  chia sẻ một connection giữa request và thread job_runner.
- Khai báo dependency có phiên bản tương thích trong requirements; tạo config
  mẫu không chứa secret, bảo đảm .gitignore cho phép theo dõi đúng file mẫu.
- Lập migration runner và bảng lịch sử/checksum. Với môi trường Supabase đã có,
  kiểm tra cấu trúc rồi baseline 001, không thực thi lại CREATE SCHEMA.
- Tách database test khỏi database chứa dữ liệu thật. Không cho fixture dọn DB
  chạy trên DATABASE_URL production.

**File:** requirements.txt, backend/api/main.py, database/, module mới.
**Nghiệm thu:** máy mới cài dependency chạy được; lỗi kết nối không lộ secret;
restart không rò connection; migration mới chạy một lần; phát hiện drift baseline.

## T02 — Hoàn thiện hợp đồng dữ liệu và ràng buộc [P0]

**Phụ trách:** database/backend. **Phụ thuộc:** T01.

- Viết repository cho assets, sources, frames, detections, crops, generation_jobs,
  generated_images và embeddings; dùng query tham số hóa và transaction theo nghiệp vụ.
- Purpose phải truyền rõ và cố định theo endpoint; không tin giá trị frontend
  để đưa dữ liệu search vào reference. Giữ FK ghép hiện có.
- Kiểm tra loại asset, kích thước/bbox, frame tĩnh index=0, vector hữu hạn và
  số chiều/chuẩn hóa. Xác định điều kiện nào đưa vào DB, điều kiện nào ở service.
- Thiết kế idempotency bền vững: request key, run ID, variant ID, quy tắc conflict;
  retry cùng thao tác không tạo lại source/job/crop. Mỗi lần chỉnh crop thực sự
  phải tạo revision mới. Thêm migration nếu cần bảng operation riêng.
- Bổ sung metadata model/checkpoint và preprocessing đủ để tái hiện xử lý.

**File:** backend/api/repositories/ (mới), database/002_*.sql (đề xuất).
**Nghiệm thu:** DB chặn job dùng crop search; service chặn sai media/bbox/vector;
retry và hai request đồng thời không tạo trùng; rollback không để chuỗi FK dở dang.

## T03 — Lớp lưu file thống nhất [P0]

**Phụ trách:** backend/storage. **Phụ thuộc:** T01; thống nhất contract với T02.

- Tạo MediaStorage với put/open/get_access_url/delete; storage_key tương đối,
  không phụ thuộc đường dẫn máy. Chốt local disk hay private Supabase Storage
  cho môi trường chạy; Supabase DB không đồng nghĩa file đã nằm trên Storage.
- Nếu dùng Supabase Storage: tạo bucket private bằng cấu hình triển khai riêng;
  key dịch vụ chỉ ở backend, frontend nhận URL có hạn hoặc qua endpoint có quyền.
- Tổ chức prefix reference/originals, reference/crops, reference/generated,
  search/originals, search/frames, search/crops.
- Lưu MIME thực tế, SHA-256, byte_size, kích thước; kiểm tra decode; stream upload
  có giới hạn dung lượng, không đọc toàn bộ video lớn rồi mới kiểm tra.
- File được ghi thành công trước metadata; ghi nhận/dọn file mồ côi sau lỗi;
  đọc lại file từ storage cho pipeline hiện cần filesystem bằng cache tạm có cleanup.
- Ngăn path traversal và tên file do người dùng điều khiển; kiểm tra cv2.imwrite
  trả thành công trước tạo asset.

**Nghiệm thu:** upload/download đúng bytes/hash; hết hạn URL xử lý được; storage
lỗi không sinh metadata hợp lệ giả; DB lỗi có cơ chế dọn file; cùng key không bị ghi đè.

## T04 — Lưu ảnh gốc luồng reference [P0]

**Phụ trách:** backend. **Phụ thuộc:** T02, T03.

- Sửa upload_image: giữ file ảnh gốc bền vững, tạo asset → source reference/image
  → frame tĩnh → detections. Hiện ảnh chỉ qua file tạm rồi bị xóa.
- Lưu tất cả khuôn mặt phát hiện trong ảnh với detector/version và tọa độ ảnh gốc.
- Trả source_id/detection_id ổn định; giữ tương thích session_id trong thời gian chuyển đổi.
- Quy định ảnh không có mặt: báo lỗi rõ và cleanup hoặc lưu nguồn với trạng thái
  xử lý thất bại đã định nghĩa; không để orphan ngoài ý muốn.

**File:** backend/api/routers/session.py, schemas.py, session_store.py.
**Nghiệm thu:** ảnh nhiều mặt có đủ detections; ảnh gốc tải lại giống input;
khởi động lại backend vẫn truy được nguồn; purpose luôn reference.

## T05 — Lưu crop, align và restore theo revision [P0]

**Phụ trách:** backend/CV. **Phụ thuộc:** T04.

- Sửa select-face, apply-restore: lưu crop asset mới và face_crops thay vì ghi
  đè {session_id}_crop.png. Preview tạm tách khỏi crop đã áp dụng.
- Ghi method, padding, white balance, restoration model/version và transform
  để truy tọa độ về frame gốc; không thay đổi bbox nguồn theo tọa độ sau padding.
- Lưu lựa chọn crop hiện hành bền vững; mỗi job chốt input_crop_id tại lúc chạy.
- Khi đổi crop, job cũ tiếp tục tham chiếu đúng crop cũ.

**File:** session.py, session_store.py, src/utils/face_enhancement.py,
src/utils/ffhq_align.py; chỉ sửa module CV nếu cần xuất metadata.
**Nghiệm thu:** chọn 2 mặt hoặc restore nhiều lần không mất bản trước; preview
không tự thay input; lineage job giữ nguyên sau khi người dùng chọn crop khác.

## T06 — Lưu job và ảnh tạo sinh [P0]

**Phụ trách:** backend/ML. **Phụ thuộc:** T05.

- Tạo generation_jobs trước chạy pipeline; ghi pending/running/done/error,
  model/version/checkpoint, initial_age, cấu hình thực tế, thời gian và lỗi đã lọc secret.
- Ghi từng ảnh vào assets + generated_images với target_age, seed, variant_index.
- Chỉ done khi đủ output yêu cầu đã lưu; output đã thành công được giữ nếu job lỗi.
- Hỗ trợ nhiều biến thể cùng tuổi: API mới trả danh sách có ID thay vì chỉ dict
  age→URL; giữ adapter cho UI cũ khi chưa chuyển xong.
- Giữ GPU mutex; cập nhật WebSocket sau khi transaction tương ứng đã thành công.

**File:** job_runner.py, routers/jobs.py, schemas.py.
**Nghiệm thu:** xem lại job sau restart; hai biến thể cùng tuổi không ghi đè;
job lỗi giữa chừng có trạng thái đúng; view lineage truy về đúng ảnh gốc reference.

## T07 — Nạp ảnh vào gallery search [P1]

**Phụ trách:** backend/CV. **Phụ thuộc:** T02, T03.

- Tạo endpoint nhận ảnh search độc lập upload reference; lưu source search/image,
  frame tĩnh, detections, crops. Hỗ trợ kết quả xử lý từng ảnh nếu upload batch.
- Ảnh nhiều mặt tạo nhiều crop; ảnh không có mặt trả trạng thái cụ thể.
- Cho phép xem nguồn và crop đã nạp bằng API phân trang.
- Không yêu cầu tồn tại job tạo sinh để thêm ảnh vào gallery.

**File:** router search_sources.py mới, schemas.py.
**Nghiệm thu:** ảnh search xuất hiện trong search_face_gallery, không xuất hiện
trong nguồn reference và không thể dùng làm input generation_jobs.

## T08 — Nạp video và lưu crop quan sát [P1]

**Phụ trách:** backend/CV. **Phụ thuộc:** T02, T03; dùng contract API của T07.

- Tách ingest video khỏi verify job: video tạo nguồn search độc lập; verify nhận
  source_id để đối chiếu mà không xử lý lại hoặc lưu trùng video mỗi lần tìm.
- Lưu video gốc, frame gốc trước preprocessing, detections, crop cùng cấu hình xử lý.
- Lưu timestamp thực từ decoder; bbox quy về frame gốc; giữ thông tin quality.
- Lưu mọi crop đạt điều kiện thu nhận đã công bố trước khi lọc top matches.
- Công bố fps sampling, max frames, đoạn thời gian đã xử lý. Hiện giới hạn 90 frame
  có thể chỉ xử lý một phần video; không báo hoàn tất toàn bộ nếu còn đoạn chưa xử lý.
- Đưa xử lý dài vào worker/job có tiến độ, hủy, retry, trạng thái bền vững; cần
  migration cho ingestion_runs nếu dùng. generation_jobs chỉ dành cho tạo sinh.
- Crop key dùng UUID/run ID; tránh trùng frame_... khi nhiều video cùng job.

**File:** routers/video_verify.py, src/preprocessing/, service ingest mới.
**Nghiệm thu:** nạp hai video không ghi đè crop; đối chiếu lại không nhân dữ liệu;
video FPS biến đổi có timestamp đúng; file lỗi/hủy không báo done giả.

## T09 — Thu nhận camera [P1]

**Phụ trách:** backend/CV. **Phụ thuộc:** T08.

- Tạo quản lý cameras và start/stop capture; mỗi phiên tạo source search/camera.
- Kết nối camera theo loại triển khai đã chốt (RTSP hoặc webcam); credentials
  ở backend/secret store, bảng chỉ giữ connection_secret_ref.
- Lưu started_at, ended_at, captured_at UTC, frame index tăng trong phiên.
- Tái sử dụng ingest frame của T08; bounded queue/backpressure để tránh tăng RAM
  vô hạn; giới hạn lấy mẫu và chính sách lưu video toàn phiên là cấu hình rõ ràng.
- Xử lý mất kết nối/reconnect, hủy, đóng thiết bị; track ID chỉ dùng trong phiên/run.

**Nghiệm thu:** stop đóng capture; mất mạng được ghi trạng thái; reconnect không
đụng key/frame; crop truy được camera và thời điểm; URL camera không lộ mật khẩu.

## T10 — Embedding và gallery FAISS từ DB [P1]

**Phụ trách:** ML/backend. **Phụ thuộc:** T06, T07; T08/T09 bổ sung nguồn sau.

- Ghi face_embeddings cho crop/ảnh tạo sinh với model/version/preprocessing,
  số chiều và trạng thái L2 normalization; chống retry tạo trùng vector.
- Gallery FAISS chỉ lấy crop purpose=search; reference crop và generated là query.
- Tạo mapping FAISS ID → embedding UUID; version hóa snapshot và rebuild index
  từ DB. Công bố index mới atomically, không trộn mapping của hai snapshot.
- Cập nhật khi thêm/xóa crop; không so vector khác model/version/preprocessing.
- Loại vector NaN/Infinity, vector zero khi yêu cầu chuẩn hóa; báo lỗi có ngữ cảnh.

**File:** src/search/embedding.py, faiss_index.py, service index mới.
**Nghiệm thu:** rebuild có mapping chính xác; generated không lọt gallery;
thêm/xóa dữ liệu phản ánh vào search; không ghép vector từ không gian khác nhau.

## T11 — Lưu lượt tìm kiếm và kết quả đối soát [P1]

**Phụ trách:** backend/database. **Phụ thuộc:** T10.

- Thiết kế migration search_runs, search_results: 9 bảng hiện tại chưa lưu các
  lượt search/match. Giữ query embedding/crop/generated ID, candidate crop ID,
  model/index version, threshold, score, rank, thời điểm và tham số ensemble.
- Lưu best_age bằng liên kết ảnh tạo sinh cụ thể; không chỉ lưu URL hoặc tuổi.
- Phân biệt accepted theo ngưỡng và xác nhận của con người; không gán danh tính
  đã xác nhận chỉ vì similarity cao.
- Trả kết quả có ảnh nguồn/crop, camera, frame/timestamp để kiểm chứng.

**File:** routers/video_verify.py, routers/jobs.py, schemas.py, migration mới.
**Nghiệm thu:** mở kết quả cũ sau restart; truy ngược đúng nguồn quan sát và query;
chạy threshold khác không ghi đè lịch sử trước; kết quả rỗng được biểu diễn rõ.

## T12 — Khôi phục trạng thái, API lịch sử và quyền truy cập [P0/P1]

**Phụ trách:** backend. **Phụ thuộc:** T04–T06; mở rộng cho T08–T11.

- DB là nguồn trạng thái bền vững; RAM chỉ cache. Thiết kế lưu session metadata/
  selected_crop_id nếu cần tiếp tục session sau restart; không serialize numpy/Face
  object vào DB. Tái dựng runtime từ asset và metadata.
- Job đang running khi process chết chuyển trạng thái gián đoạn/lỗi phù hợp;
  không tự chạy lại diffusion gây trùng output khi chưa có chiến lược resume.
- API lịch sử/phân trang/filter purpose, source, camera, ngày; dùng IDs và URL
  do storage layer sinh, không quét glob outputs để làm nguồn sự thật.
- Chốt mô hình truy cập local một người hay nhiều tài khoản. Backend kiểm tra
  quyền trước đọc/sửa/xóa file; nếu dùng Supabase Auth/Data API, thêm ownership,
  RLS/policy bằng migration và test. Không mở toàn bộ schema cho anon để chạy thử.
- Rà soát static mount /outputs để tránh bỏ qua kiểm tra quyền nếu triển khai nhiều người.

**File:** session_store.py, main.py, routers/jobs.py, routers/session.py.
**Nghiệm thu:** restart vẫn mở lịch sử và crop đã chọn; request không có quyền
không đọc được file qua URL trực tiếp; phân trang ổn định, không trả credential.

## T13 — Giao diện tách rõ hai luồng [P1]

**Phụ trách:** frontend. **Phụ thuộc:** contract T04–T12; có thể làm mock trước.

- Khu vực người cần tìm: ảnh gốc, chọn mặt/crop revision, các ảnh tạo sinh theo tuổi.
- Khu vực nguồn tìm kiếm: ảnh/video/camera, trạng thái ingest, số frame/crop,
  danh sách crop và lọc nguồn/thời gian.
- Kết quả hiển thị query cạnh crop quan sát, nguồn, timestamp, score; nhãn ảnh
  tạo sinh rõ ràng. Có trạng thái empty/loading/error/canceled và retry phù hợp.
- Chuyển API types sang IDs bền vững; hỗ trợ nhiều biến thể cùng tuổi và URL hết hạn.
- Khi mở app lại tải lịch sử từ API, không phụ thuộc session RAM cũ.

**File:** desktop/src/types/api.ts, components/FaceSelector.tsx,
PhotoRestoration.tsx, ResultsGallery.tsx, GalleryPicker.tsx, VideoVerify.tsx.
**Nghiệm thu:** không nhầm ảnh search với reference; không mất lịch sử khi restart;
video nhiều lần không đổi thumbnail cũ; tuổi giống nhau vẫn thấy các variant.

## T14 — Chuyển dữ liệu cũ, xóa và dọn dẹp [P2]

**Phụ trách:** backend/ops. **Phụ thuộc:** T03, T06, T08, T12.

- Viết import outputs/app_uploads và outputs/jobs với dry-run, báo cáo mapping,
  checkpoint và idempotency; không suy nguồn/danh tính khi thiếu bằng chứng.
- File crop từng bị ghi đè hoặc thiếu ảnh gốc phải báo thiếu lineage, không tạo
  bản ghi giả để vượt FK. Chỉ import các chuỗi khôi phục được; phần thiếu có báo cáo.
- Thiết kế xóa theo phụ thuộc, tham chiếu dùng chung, cập nhật FAISS và storage;
  retry an toàn, khoảng chờ cho file mồ côi, chính sách retention cấu hình được.
- Backup gồm cả metadata DB và file ảnh/video; kiểm thử restore trên môi trường test.

**Nghiệm thu:** dry-run không ghi/xóa; import hai lần không nhân dữ liệu; không xóa
asset còn tham chiếu; restore xem lại được ảnh và lineage; có báo cáo file thiếu.

## T15 — Kiểm thử tích hợp và bàn giao vận hành [P0/P1]

**Phụ trách:** QA + backend + frontend. **Phụ thuộc:** kiểm thử tăng dần theo từng task;
nghiệm thu đầy đủ sau T01–T13, bổ sung T14 nếu chuyển dữ liệu cũ.

- Fixture PostgreSQL/storage test riêng, media tổng hợp hoặc dữ liệu được phép dùng.
- Test end-to-end: ảnh reference nhiều mặt → chọn crop → sinh nhiều tuổi/variant →
  nạp ảnh/video search → embedding → đối chiếu → restart → xem lại nguồn và kết quả.
- Test ràng buộc purpose, retry đồng thời, DB/storage mất kết nối, pipeline chết,
  URL hết hạn, camera reconnect, video bị cắt phạm vi xử lý, file bị thiếu.
- Mock model nặng cho CI; một smoke test thực trên máy có model/GPU, ghi rõ cấu hình.
- Đo thời gian ghi theo batch, RAM ingest, tốc độ/frame, số connection và dung lượng;
  báo số đo kèm môi trường thay vì cam kết chỉ tiêu chưa đo.
- Cập nhật README: cấu hình, migrate baseline/new, chạy worker, chọn storage,
  khôi phục index, xử lý job gián đoạn, backup/restore và rollback ứng dụng.

**Nghiệm thu:** checklist có bằng chứng; không chứa secret; CI pass; smoke test
hai luồng pass; người nhận mới có thể dựng môi trường theo tài liệu.

## Thứ tự thực hiện và chia việc

1. Nền tảng: T01 → T02 và T03 → thống nhất ID/API/storage contract.
2. Nhánh reference: T04 → T05 → T06; làm T12 phần session/job cùng giai đoạn.
3. Nhánh search: T07 và T08 → T09. Nhánh này có thể do người khác làm sau nền tảng.
4. T10 → T11; T12 hoàn thiện quyền/API lịch sử; T13 tích hợp UI.
5. T14 nếu chuyển dữ liệu cũ; T15 kiểm thử xuyên suốt và nghiệm thu cuối.

Gợi ý giao người: A phụ trách T01/T02/T12; B T03/T04/T05/T06;
C T07/T08/T09/T10/T11; D T13; QA phối hợp T15, ops/backend làm T14.
Đây là gợi ý phân công cho người nhận, không có agent nào được chạy tự động.

## Definition of Done toàn bộ

- [ ] Cả hai luồng lưu và đọc lại đầy đủ sau khi backend restart.
- [ ] Crop search không thể trở thành input tạo sinh; generated không nằm trong gallery.
- [ ] Mỗi crop truy về ảnh/frame gốc; mỗi ảnh tạo sinh truy về đúng crop reference.
- [ ] Video/camera có thông tin thời điểm, phạm vi xử lý, không ghi đè giữa các nguồn.
- [ ] File và metadata nhất quán khi lỗi/retry; không lộ secret hay file ngoài quyền.
- [ ] Migration, kiểm thử, cấu hình mẫu và hướng dẫn vận hành đã bàn giao.
