# 06 — Kiến trúc lưu trữ video giám sát hoạt động như thế nào

Tầng **lưu trữ thô**: video/ảnh từ camera. Khác với kho vector (file `04`), tầng này lo
*dung lượng, băng thông, vòng đời dữ liệu (giữ/bỏ/nén)*.

## A. Bài toán cơ bản

- 1 camera 1080p ~ 1–2 Mbps → **1 camera ≈ 10–20 GB/ngày**; 10 camera 1 tuần ≈ 1.4 TB.
- Dữ liệu thô 1 ngày 60fps 1080p ≈ **117 TiB (raw)** → phải nén (H.264/H.265/neural codec).
- Do đó hệ thống thực tế luôn chọn: **giữ cái gì, nén mức nào, đẩy lên đâu**.

## B. Bài báo theo chủ đề

### B1. Chọn lọc & tối ưu lưu theo "mức thông tin" (analytics-aware storage)

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 1 | **Analytics-Aware Storage of Surveillance Videos** (SmartComp 2020, testbed ĐH NTHU) | 2020 | https://ics.uci.edu/~dsm/publications/papers/2020/analysis_aware_storage_SmartComp.pdf |
| 2 | **Salient Store: Enabling Smart Storage for Continuous Learning Edge Servers** | 2024 | https://arxiv.org/html/2410.05435 |
| 3 | **TileClipper: Lightweight Selection of Regions of Interest from Videos for Traffic Surveillance** (USENIX ATC 2024) | 2024 | https://www.usenix.org/system/files/atc24-chaudhary.pdf |

**Cơ chế (1):**
- Ước lượng **"information amount"** của từng clip bằng cách **lấy mẫu khung hình** (không chạy
  hết analytics → không nghẽn máy chủ).
- Khi hết chỗ: clip có thông tin thấp nhất bị **downsample** thay vì xóa — 3 kiểu:
  *temporal* (bỏ bớt khung), *spatial* (giảm分辨率), *fidelity* (giảm bitrate/chất lượng).
- Kết quả: số clip lưu được tăng ~35%, thông tin mỗi truy vấn tăng ~4×.

**Cơ chế (3) TileClipper:** chia frame thành các ô (tile) theo chuẩn HEVC, chỉ gửi lên cloud
các ô chứa đối tượng quan tâm → giảm ~22% dữ liệu gửi lên mà vẫn giữ detection ~92%.

### B2. Kiến trúc biên (edge) — cloud

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 4 | **Video Surveillance Architecture from the Cloud to the Edge** (ĐH Limerick; CI/CD, microservice, container tại biên) | — | https://pure.ul.ie/en/publications/video-surveillance-architecture-from-the-cloud-to-the-edge/ |
| 5 | **Elastic Urban Video Surveillance System Using Edge Computing** (kiến trúc 3 tầng: application / edge computing / data; SDN+NFn) | 2017 | https://mason.gmu.edu/~jpan22/papers/smartiot2017.pdf |
| 6 | **Croesus: Multi-stage edge-cloud video processing** (tính ở biên bằng model nhỏ → cloud sửa lỗi model lớn) | 2022 | http://arxiv.org/pdf/2201.00063 |
| 7 | **Clownfish: Edge and Cloud Symbiosis for Video Stream Analytics** (gửi lên cloud **nhịp sự** theo ngữ cảnh, hợp nhất kết quả 2 bên) | 2020 | https://vnigade.github.io/assets/pdf/clownfish_sec20.pdf |

**Đường đi dữ liệu chuẩn:**
```
Camera ──(encode H.264/H.265)──► Edge node: detect + trích embedding + lọc frame
        │                                   │
        │                                   ├─► lưu local: video nén + index (rolling window)
        │                                   └─► đẩy frame "đáng chú ý" ──► Cloud (lưu lâu dài / tìm kiếm lớn)
        └─ băng thông WAN giảm vì chỉ gửi phần quan trọng
```

### B3. Bảo mật & toàn vẹn khi lưu/transfer

| # | Bài báo | Năm | Link |
|---|---------|-----|------|
| 8 | **Optimizing network performance and security in video surveillance data storage: a solution with blockchain and mobile edge computing** (IPFS + smart contract + SegWit) — Computing (Springer) | 2025 | https://link.springer.com/article/10.1007/s00607-025-01487-y |

## C. Quy trình lưu trữ 3 pha (áp dụng cho hệ thống của đề tài)

1. **Compression** — mã hóa H.264/H.265 (hoặc neural codec như Salient Store, tận dụng motion vector).
2. **Encryption** — mã hóa lưu trữ (AES) / hậu lượng tử (lattice) ở biên.
3. **Redundancy** — RAID / đa bản sao để chống hỏng ổ.

**Chính sách lưu đề xuất cho đề tài tìm người mất tích:**
- **Giữ vĩnh viễn**: crop ảnh người + embedding + metadata (nhẹ, ~KB/người) → kho vector (`04`).
- **Giữ cửa sổ trượt N ngày**: video gốc đã nén (đủ để trích lại bằng chứng).
- **Lưu "sự kiện"** (khi khớp query): snapshot + đoạn ±N giây + log nguồn camera/thời gian (chain of custody).
- **Không lưu**: khung hình nền lặp lại → downsampling theo thông tin (`B1`).

> Kết nối với hệ thống hiện có trong repo: xem `docs/face-media-database.md`,
> `docs/EUREKA_3.8_Ha-tang-luu-tru-face-media*.docx` (thiết kế hạ tầng lưu face-media).
