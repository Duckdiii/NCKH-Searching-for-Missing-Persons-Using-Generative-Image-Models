-- T08/T09/T11 — ingestion_runs (tiến độ/hủy/retry nạp video-camera),
-- search_runs + search_results (lưu lượt tìm kiếm và đối soát).
-- Chạy qua migration runner (python database/migrate.py migrate).
BEGIN;

-- Một run cho mỗi lần nạp video/camera session: trạng thái bền vững,
-- phạm vi đã xử lý (không báo done giả khi còn đoạn chưa xử lý).
CREATE TABLE IF NOT EXISTS face_media.ingestion_runs (
    id uuid PRIMARY KEY,
    source_id uuid NOT NULL REFERENCES face_media.sources(id),
    status text NOT NULL CHECK (status IN ('pending', 'running', 'done', 'error', 'canceled')),
    fps_target real NOT NULL DEFAULT 3 CHECK (fps_target > 0),
    max_frames integer NOT NULL DEFAULT 90 CHECK (max_frames > 0),
    frames_sampled integer NOT NULL DEFAULT 0 CHECK (frames_sampled >= 0),
    faces_found integer NOT NULL DEFAULT 0 CHECK (faces_found >= 0),
    frames_total integer CHECK (frames_total IS NULL OR frames_total >= 0),
    duration_sec real CHECK (duration_sec IS NULL OR duration_sec >= 0),
    truncated boolean NOT NULL DEFAULT false,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    CHECK (finished_at IS NULL OR finished_at >= created_at)
);
CREATE INDEX IF NOT EXISTS ingestion_source_idx
    ON face_media.ingestion_runs (source_id, created_at);

-- Một lượt đối chiếu: query là tập ảnh tạo sinh (verify sau FADING) hoặc
-- một crop; candidate luôn là crop quan sát (purpose=search).
CREATE TABLE IF NOT EXISTS face_media.search_runs (
    id uuid PRIMARY KEY,
    query_kind text NOT NULL CHECK (query_kind IN ('generated_set', 'crop')),
    query_crop_id uuid REFERENCES face_media.face_crops(id),
    scope_source_id uuid REFERENCES face_media.sources(id),
    generation_job_id uuid REFERENCES face_media.generation_jobs(id),
    model_name text NOT NULL,
    model_version text NOT NULL,
    preprocessing_version text NOT NULL DEFAULT 'video_preprocess_v1',
    index_version text NOT NULL DEFAULT 'adhoc',
    threshold real NOT NULL CHECK (threshold BETWEEN 0 AND 1),
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((query_kind = 'crop' AND query_crop_id IS NOT NULL)
        OR (query_kind = 'generated_set' AND query_crop_id IS NULL))
);
CREATE INDEX IF NOT EXISTS search_runs_scope_idx
    ON face_media.search_runs (scope_source_id, created_at);

-- Kết quả từng candidate: liên kết ảnh tạo sinh cụ thể (best match theo tuổi),
-- accepted theo ngưỡng TÁCH BIỆT xác nhận của con người (mặc định false:
-- similarity cao không phải danh tính đã xác nhận).
CREATE TABLE IF NOT EXISTS face_media.search_results (
    id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES face_media.search_runs(id) ON DELETE CASCADE,
    candidate_crop_id uuid NOT NULL REFERENCES face_media.face_crops(id),
    best_generated_image_id uuid REFERENCES face_media.generated_images(id),
    score real NOT NULL CHECK (score BETWEEN -1 AND 1),
    rank integer NOT NULL CHECK (rank >= 1),
    accepted_by_threshold boolean NOT NULL DEFAULT false,
    human_confirmed boolean NOT NULL DEFAULT false,
    confirmed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, candidate_crop_id),
    CHECK (confirmed_at IS NULL OR human_confirmed)
);
CREATE INDEX IF NOT EXISTS search_results_run_rank_idx
    ON face_media.search_results (run_id, rank);

COMMIT;
