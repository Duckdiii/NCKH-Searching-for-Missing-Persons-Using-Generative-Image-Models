-- T02 — Idempotency bền vững cho retry (không tạo trùng source/job/crop).
-- Chạy qua migration runner (python database/migrate.py migrate), không chạy tay từng phần.
BEGIN;

CREATE TABLE IF NOT EXISTS face_media.idempotency_keys (
    key text PRIMARY KEY,
    scope text NOT NULL CHECK (scope IN (
        'reference_upload', 'select_crop', 'generation_run',
        'search_upload', 'video_ingest', 'camera_session'
    )),
    result jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idempotency_scope_idx
    ON face_media.idempotency_keys (scope, created_at);

COMMIT;
