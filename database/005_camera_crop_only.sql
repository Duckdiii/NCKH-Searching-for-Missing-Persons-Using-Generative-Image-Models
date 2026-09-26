-- 005 — Camera crop-only + tracklet (P0/P1 kiến trúc đa camera tiết kiệm bộ nhớ).
-- Không sửa 001–004, không xóa dữ liệu cũ. Chạy qua migration runner
-- (python database/migrate.py migrate). Idempotent với IF NOT EXISTS /
-- kiểm tra constraint trước khi thêm.
--
-- Nội dung:
-- 1. sources.storage_policy: 'full' (mặc định, giữ hành vi image/video cũ)
--    hoặc 'crop_only' (luồng camera mới: chỉ crop JPEG + vector + metadata).
-- 2. frames.asset_id cho phép NULL (metadata-only cho camera crop-only);
--    thêm kích thước nguồn + timestamp provenance (received_at, source_timestamp,
--    timestamp_uncertainty_ms) để đo độ trễ và thể hiện khoảng trống timeline.
-- 3. face_detections.tracklet_id FK nullable về tracklets mới (giữ tương thích
--    hàng cũ chưa có tracklet).
-- 4. tracklets: đoạn theo dõi liên tục trong một camera/source (P1).
BEGIN;

-- 1. Chính sách lưu phiên camera.
ALTER TABLE face_media.sources
    ADD COLUMN IF NOT EXISTS storage_policy text NOT NULL DEFAULT 'full'
        CHECK (storage_policy IN ('full', 'crop_only'));

-- 2. Frame metadata-only: cho phép asset_id NULL.
-- 001 tạo asset_id NOT NULL; DROP NOT NULL an toàn (giữ hàng cũ nguyên).
ALTER TABLE face_media.frames
    ALTER COLUMN asset_id DROP NOT NULL;

ALTER TABLE face_media.frames
    ADD COLUMN IF NOT EXISTS source_width integer CHECK (source_width IS NULL OR source_width > 0),
    ADD COLUMN IF NOT EXISTS source_height integer CHECK (source_height IS NULL OR source_height > 0),
    ADD COLUMN IF NOT EXISTS received_at timestamptz,
    ADD COLUMN IF NOT EXISTS source_timestamp timestamptz,
    ADD COLUMN IF NOT EXISTS timestamp_uncertainty_ms integer
        CHECK (timestamp_uncertainty_ms IS NULL OR timestamp_uncertainty_ms >= 0);

-- 3. Tracklet trong một camera/source.
CREATE TABLE IF NOT EXISTS face_media.tracklets (
    id uuid PRIMARY KEY,
    source_id uuid NOT NULL REFERENCES face_media.sources(id),
    camera_id uuid REFERENCES face_media.cameras(id),
    local_track_id text NOT NULL,
    status text NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'closed', 'expired')),
    started_at timestamptz NOT NULL DEFAULT now(),
    ended_at timestamptz,
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    observation_count integer NOT NULL DEFAULT 0 CHECK (observation_count >= 0),
    quality_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_id, local_track_id),
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);
CREATE INDEX IF NOT EXISTS tracklets_source_idx
    ON face_media.tracklets (source_id, last_seen_at);
CREATE INDEX IF NOT EXISTS tracklets_camera_idx
    ON face_media.tracklets (camera_id, last_seen_at);
CREATE INDEX IF NOT EXISTS tracklets_status_idx
    ON face_media.tracklets (status, expires_at);

-- 4. Gắn detection đã persist vào tracklet (nullable để tương thích hàng cũ).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'face_media' AND table_name = 'face_detections'
          AND column_name = 'tracklet_id'
    ) THEN
        ALTER TABLE face_media.face_detections
            ADD COLUMN tracklet_id uuid REFERENCES face_media.tracklets(id);
    END IF;
END
$$;
CREATE INDEX IF NOT EXISTS detections_tracklet_idx
    ON face_media.face_detections (tracklet_id);

-- 5. Ràng buộc crop-only ở tầng ứng dụng (repository validation), không dùng
-- CHECK tham chiếu bảng khác. Ghi chú vận hành:
-- - Hàng frames của source storage_policy='crop_only' phải có asset_id IS NULL
--   (metadata-only) và source_width/source_height NOT NULL để truy vết.
-- - Không ghi file vào prefix search/frames cho luồng này; chỉ search/crops.

COMMIT;
