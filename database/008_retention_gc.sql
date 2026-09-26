-- 008 — P4: retention/TTL, deletion ledger, retention jobs (doc §9).
-- Không sửa 001–007, không tự xóa dữ liệu hiện có khi bật thử nghiệm:
-- seed policy với enabled=false; GC chỉ chạy khi operator bật + gọi API.
-- Tách TTL crop, vector, tracklet, exemplar và audit.
BEGIN;

CREATE TABLE IF NOT EXISTS face_media.retention_policies (
    scope text PRIMARY KEY CHECK (scope IN (
        'crop', 'vector', 'tracklet', 'exemplar', 'audit')),
    ttl_days integer NOT NULL CHECK (ttl_days > 0),
    quota_bytes bigint CHECK (quota_bytes IS NULL OR quota_bytes > 0),
    enabled boolean NOT NULL DEFAULT false,
    reason text,
    updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO face_media.retention_policies (scope, ttl_days, enabled, reason)
VALUES
    ('crop', 30, false, 'P4 seed: crop theo lượt xuất hiện trong cửa sổ retention'),
    ('vector', 30, false, 'P4 seed: vector đi cùng crop'),
    ('tracklet', 30, false, 'P4 seed: tracklet metadata'),
    ('exemplar', 90, false, 'P4 seed: mẫu ID giữ lâu hơn, quota riêng cho hồ sơ'),
    ('audit', 365, false, 'P4 seed: audit/search_runs giữ phục vụ truy vết')
ON CONFLICT (scope) DO NOTHING;

-- Xóa theo ledger: tombstone ẩn query → event xóa index → xóa file hết tham
-- chiếu/bảo lưu → xóa vector/bản ghi con theo FK → hoàn tất. Retry idempotent.
CREATE TABLE IF NOT EXISTS face_media.deletion_ledger (
    id uuid PRIMARY KEY,
    entity text NOT NULL CHECK (entity IN ('crop', 'embedding', 'tracklet', 'asset')),
    entity_id uuid NOT NULL,
    crop_id uuid,
    storage_key text,
    reason text,
    file_deleted boolean NOT NULL DEFAULT false,
    index_evicted boolean NOT NULL DEFAULT false,
    rows_deleted boolean NOT NULL DEFAULT false,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    done boolean NOT NULL DEFAULT false,
    error text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ledger_pending_idx
    ON face_media.deletion_ledger (done, created_at) WHERE done = false;
CREATE INDEX IF NOT EXISTS ledger_crop_idx
    ON face_media.deletion_ledger (crop_id) WHERE crop_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS face_media.retention_jobs (
    id uuid PRIMARY KEY,
    status text NOT NULL DEFAULT 'done'
        CHECK (status IN ('running', 'done', 'error')),
    scanned integer NOT NULL DEFAULT 0 CHECK (scanned >= 0),
    tombstoned integer NOT NULL DEFAULT 0 CHECK (tombstoned >= 0),
    files_removed integer NOT NULL DEFAULT 0 CHECK (files_removed >= 0),
    bytes_freed bigint NOT NULL DEFAULT 0 CHECK (bytes_freed >= 0),
    dry_run boolean NOT NULL DEFAULT true,
    error text,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

COMMIT;
