-- 006 — P2: duy trì ID xuyên camera (giả thuyết, có revision).
-- Không sửa 001–005, không xóa dữ liệu cũ. Chạy qua migration runner.
--
-- - camera_topology: gợi ý mềm vùng tìm (site, cặp camera, thời gian di chuyển
--   kỳ vọng, chồng lấn). Version hóa để assignment ghi lại topology_version.
-- - global_identities: ID giả thuyết (KHÔNG phải tên người, không đồng nghĩa
--   human_confirmed). Có revision để tách/gộp lại.
-- - identity_assignments: 1 assignment hiện hành/tracklet (UNIQUE tracklet_id
--   cho hàng còn hiệu lực qua partial index); lịch sử sửa bằng valid_to +
--   superseded_by. Ghi score, margin, evidence, model, topology version.
-- - identity_exemplars: tối đa 3 slot/ID/không gian vector, tham chiếu
--   crop/embedding sẵn có (không copy ảnh/vector cho mỗi liên kết).
BEGIN;

CREATE TABLE IF NOT EXISTS face_media.camera_topology (
    id uuid PRIMARY KEY,
    site text NOT NULL DEFAULT 'default',
    camera_a uuid REFERENCES face_media.cameras(id),
    camera_b uuid REFERENCES face_media.cameras(id),
    travel_sec_min real CHECK (travel_sec_min IS NULL OR travel_sec_min >= 0),
    travel_sec_typical real CHECK (travel_sec_typical IS NULL OR travel_sec_typical >= 0),
    travel_sec_max real CHECK (travel_sec_max IS NULL OR travel_sec_max >= 0),
    overlapping boolean NOT NULL DEFAULT false,
    version integer NOT NULL DEFAULT 1 CHECK (version > 0),
    note text,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (camera_a IS DISTINCT FROM camera_b)
);
CREATE INDEX IF NOT EXISTS topology_site_idx
    ON face_media.camera_topology (site, camera_a, camera_b);

CREATE TABLE IF NOT EXISTS face_media.global_identities (
    id uuid PRIMARY KEY,
    status text NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'merged', 'split', 'archived', 'unresolved')),
    revision integer NOT NULL DEFAULT 1 CHECK (revision > 0),
    last_seen_at timestamptz,
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (expires_at IS NULL OR last_seen_at IS NULL OR expires_at >= last_seen_at)
);
CREATE INDEX IF NOT EXISTS identities_status_idx
    ON face_media.global_identities (status, last_seen_at);

CREATE TABLE IF NOT EXISTS face_media.identity_assignments (
    id uuid PRIMARY KEY,
    tracklet_id uuid NOT NULL REFERENCES face_media.tracklets(id) ON DELETE CASCADE,
    global_id uuid NOT NULL REFERENCES face_media.global_identities(id) ON DELETE CASCADE,
    score real NOT NULL CHECK (score BETWEEN -1 AND 1),
    margin real CHECK (margin IS NULL OR margin BETWEEN -2 AND 2),
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    reason text,
    model_name text NOT NULL,
    model_version text NOT NULL,
    preprocessing_version text NOT NULL,
    topology_version integer,
    revision integer NOT NULL DEFAULT 1 CHECK (revision > 0),
    valid_from timestamptz NOT NULL DEFAULT now(),
    valid_to timestamptz,
    superseded_by uuid REFERENCES face_media.identity_assignments(id),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (valid_to IS NULL OR valid_to >= valid_from)
);
-- Một assignment hiện hành/tracklet; nhiều tracklet có thể cùng global ID.
CREATE UNIQUE INDEX IF NOT EXISTS assignments_tracklet_active_uidx
    ON face_media.identity_assignments (tracklet_id) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS assignments_global_idx
    ON face_media.identity_assignments (global_id, valid_from);

CREATE TABLE IF NOT EXISTS face_media.identity_exemplars (
    global_id uuid NOT NULL REFERENCES face_media.global_identities(id) ON DELETE CASCADE,
    embedding_space text NOT NULL,
    slot integer NOT NULL CHECK (slot BETWEEN 0 AND 2),
    crop_id uuid REFERENCES face_media.face_crops(id) ON DELETE SET NULL,
    embedding_id uuid REFERENCES face_media.face_embeddings(id) ON DELETE SET NULL,
    weight real NOT NULL DEFAULT 1.0 CHECK (weight >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (global_id, embedding_space, slot)
);
CREATE INDEX IF NOT EXISTS exemplars_crop_idx
    ON face_media.identity_exemplars (crop_id);

COMMIT;
