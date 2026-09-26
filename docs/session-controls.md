# Dừng phiên, xóa phiên và mở lại tác vụ

- Dừng phiên gửi POST /api/sessions/{id}/stop, đặt cờ hủy cho tác vụ thuộc phiên.
  Worker kiểm tra tại các vòng lặp huấn luyện, nghịch đảo, tạo sinh, embedding và
  đối soát video. Một phép tính GPU hoặc lệnh gọi model đang thực thi phải trả về
  trước khi worker kiểm tra cờ tiếp theo; không cưỡng bức kết thúc thread.
- DELETE /api/sessions/{id} dừng tác vụ trước. Khi còn worker, trả 202 và tiếp tục
  xóa trong nền sau khi các worker kết thúc. Giao diện theo dõi qua GET
  /api/sessions/{id}/lifecycle; không reset giao diện trước khi máy chủ xác nhận.
- Xóa phiên gỡ bản ghi sessions, RAM và các tệp crop/preview legacy của phiên;
  nguồn, crop, job và ảnh tạo sinh đã lưu trong face_media được giữ để truy nguồn.
- Chọn lịch sử mở cả job running/error chưa có result. Job running mở màn tiến độ;
  job done mở kết quả. WebSocket và HTTP polling được quản lý theo job_id,
  nên đổi màn hình không dừng job và không ghi tiến độ vào phiên khác.
- Registry hủy đang ở bộ nhớ của một tiến trình backend, phù hợp desktop hiện tại.
  Nếu triển khai nhiều worker/process, phải thay bằng điều phối dùng chung.

Kiểm thử: backend/tests/test_session_lifecycle.py và
 desktop/src/__tests__/SessionControls.test.tsx; model nặng được mock trong test.
