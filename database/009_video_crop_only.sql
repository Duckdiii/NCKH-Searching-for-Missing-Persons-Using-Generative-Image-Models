-- 009 — Đối soát video chỉ lưu crop mặt (không lưu video gốc / frame gốc).
-- Không sửa 001–008, không xóa dữ liệu. Idempotent.
--
-- 001 bắt buộc nguồn image/video có original_asset_id. Luồng đối soát video
-- mới tạo source search/video với storage_policy='crop_only' và
-- original_asset_id NULL; frame là hàng metadata-only (asset_id NULL,
-- source_width/source_height) như camera crop-only của 005.
BEGIN;

DO $$
DECLARE
    c record;
BEGIN
    -- CHECK hình dạng nguồn của 001 không đặt tên → tìm theo định nghĩa.
    FOR c IN
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'face_media.sources'::regclass AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%original_asset_id IS NOT NULL%'
    LOOP
        EXECUTE format('ALTER TABLE face_media.sources DROP CONSTRAINT %I', c.conname);
    END LOOP;
END
$$;

ALTER TABLE face_media.sources
    ADD CONSTRAINT sources_kind_shape_check CHECK (
        (kind = 'image' AND original_asset_id IS NOT NULL AND camera_id IS NULL)
        OR (kind = 'video' AND camera_id IS NULL
            AND (original_asset_id IS NOT NULL OR storage_policy = 'crop_only'))
        OR (kind = 'camera' AND camera_id IS NOT NULL AND started_at IS NOT NULL)
    );

COMMIT;
