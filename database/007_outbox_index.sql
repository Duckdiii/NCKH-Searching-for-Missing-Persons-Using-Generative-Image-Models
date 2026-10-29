-- 007 — P3: outbox đồng bộ index + snapshot version + tombstone.
-- Không sửa 001–006. Chạy qua migration runner.
--
-- - outbox_events: commit vector + metadata + outbox cùng transaction (doc §6).
--   Index worker nhận ít nhất một lần, dedupe event_id; một writer/shard.
-- - index_snapshots: công bố base snapshot bất biến + watermark nhất quán +
--   phạm vi index rõ ràng (tránh rebuild ORDER BY ... LIMIT lấy phần đầu cũ).
-- - embedding_tombstones: update/delete có revision và tombstone; query kiểm
--   tombstone DB khi index còn cũ (doc §9).
BEGIN;

CREATE TABLE IF NOT EXISTS face_media.outbox_events (
    event_id uuid PRIMARY KEY,
    entity text NOT NULL CHECK (entity IN ('embedding', 'crop')),
    entity_id uuid NOT NULL,
    op text NOT NULL CHECK (op IN ('upsert', 'delete')),
    model_name text NOT NULL DEFAULT '',
    model_version text NOT NULL DEFAULT '',
    preprocessing_version text NOT NULL DEFAULT '',
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    claimed_at timestamptz,
    done boolean NOT NULL DEFAULT false,
    error text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS outbox_pending_idx
    ON face_media.outbox_events (done, created_at) WHERE done = false;
CREATE INDEX IF NOT EXISTS outbox_entity_idx
    ON face_media.outbox_events (entity, entity_id);

CREATE TABLE IF NOT EXISTS face_media.index_snapshots (
    id uuid PRIMARY KEY,
    triple_key text NOT NULL,
    version integer NOT NULL CHECK (version > 0),
    model_name text NOT NULL,
    model_version text NOT NULL,
    preprocessing_version text NOT NULL,
    watermark_to timestamptz,
    watermark_id uuid,
    scope jsonb NOT NULL DEFAULT '{}'::jsonb,
    size integer NOT NULL DEFAULT 0 CHECK (size >= 0),
    skipped integer NOT NULL DEFAULT 0 CHECK (skipped >= 0),
    recall_vs_flat real CHECK (recall_vs_flat IS NULL OR recall_vs_flat BETWEEN 0 AND 1),
    active boolean NOT NULL DEFAULT true,
    built_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (triple_key, version)
);
CREATE INDEX IF NOT EXISTS snapshots_triple_idx
    ON face_media.index_snapshots (triple_key, version DESC);

CREATE TABLE IF NOT EXISTS face_media.embedding_tombstones (
    embedding_id uuid PRIMARY KEY,
    crop_id uuid,
    model_name text NOT NULL DEFAULT '',
    model_version text NOT NULL DEFAULT '',
    preprocessing_version text NOT NULL DEFAULT '',
    reason text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tombstones_crop_idx
    ON face_media.embedding_tombstones (crop_id);

COMMIT;
