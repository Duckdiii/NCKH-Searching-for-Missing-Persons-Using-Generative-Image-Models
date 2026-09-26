-- PostgreSQL. Run once against an empty face_media schema.
-- UUIDs are assigned by the application (uuid.uuid4()).
BEGIN;
CREATE SCHEMA face_media;

CREATE TABLE face_media.assets (
    id uuid PRIMARY KEY,
    storage_key text NOT NULL UNIQUE,
    media_type text NOT NULL CHECK (media_type IN ('image', 'video')),
    mime_type text NOT NULL,
    sha256 varchar(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    byte_size bigint NOT NULL CHECK (byte_size > 0),
    width integer NOT NULL CHECK (width > 0),
    height integer NOT NULL CHECK (height > 0),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX assets_sha256_idx ON face_media.assets (sha256);

CREATE TABLE face_media.cameras (
    id uuid PRIMARY KEY,
    name text NOT NULL,
    location text,
    connection_secret_ref text,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- One source per uploaded image/video or camera capture session.
CREATE TABLE face_media.sources (
    id uuid PRIMARY KEY,
    purpose text NOT NULL CHECK (purpose IN ('reference', 'search')),
    UNIQUE (id, purpose),
    CHECK (purpose <> 'reference' OR kind = 'image'),
    kind text NOT NULL CHECK (kind IN ('image', 'video', 'camera')),
    original_asset_id uuid REFERENCES face_media.assets(id),
    camera_id uuid REFERENCES face_media.cameras(id),
    started_at timestamptz,
    ended_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((kind IN ('image', 'video') AND original_asset_id IS NOT NULL AND camera_id IS NULL)
        OR (kind = 'camera' AND camera_id IS NOT NULL AND started_at IS NOT NULL)),
    CHECK (ended_at IS NULL OR (started_at IS NOT NULL AND ended_at >= started_at))
);
CREATE INDEX sources_camera_idx ON face_media.sources (camera_id, started_at);
CREATE INDEX sources_purpose_idx ON face_media.sources (purpose, created_at);
CREATE INDEX sources_asset_idx ON face_media.sources (original_asset_id);

-- Still images have frame_index=0, offset_ms=0. Preserve original frame pixels.
CREATE TABLE face_media.frames (
    id uuid PRIMARY KEY,
    purpose text NOT NULL CHECK (purpose IN ('reference', 'search')),
    UNIQUE (id, purpose),
    source_id uuid NOT NULL,
    FOREIGN KEY (source_id, purpose) REFERENCES face_media.sources(id, purpose),
    asset_id uuid NOT NULL REFERENCES face_media.assets(id),
    frame_index bigint NOT NULL CHECK (frame_index >= 0),
    offset_ms bigint NOT NULL CHECK (offset_ms >= 0),
    captured_at timestamptz,
    UNIQUE (source_id, frame_index)
);
CREATE INDEX frames_time_idx ON face_media.frames (source_id, offset_ms);
CREATE INDEX frames_asset_idx ON face_media.frames (asset_id);

-- An observation is a detected face, not a confirmed person's identity.
CREATE TABLE face_media.face_detections (
    id uuid PRIMARY KEY,
    purpose text NOT NULL CHECK (purpose IN ('reference', 'search')),
    UNIQUE (id, purpose),
    frame_id uuid NOT NULL,
    FOREIGN KEY (frame_id, purpose) REFERENCES face_media.frames(id, purpose),
    detector_name text NOT NULL,
    detector_version text NOT NULL,
    run_id uuid NOT NULL,
    face_index integer NOT NULL CHECK (face_index >= 0),
    x1 real NOT NULL CHECK (x1 >= 0),
    y1 real NOT NULL CHECK (y1 >= 0),
    x2 real NOT NULL,
    y2 real NOT NULL,
    confidence real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    landmarks jsonb,
    track_id text,
    quality jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (x2 > x1 AND y2 > y1),
    UNIQUE (frame_id, run_id, face_index)
);

-- Keep every crop/alignment/restoration revision; never overwrite its asset.
CREATE TABLE face_media.face_crops (
    id uuid PRIMARY KEY,
    purpose text NOT NULL CHECK (purpose IN ('reference', 'search')),
    UNIQUE (id, purpose),
    detection_id uuid NOT NULL,
    FOREIGN KEY (detection_id, purpose) REFERENCES face_media.face_detections(id, purpose),
    asset_id uuid NOT NULL UNIQUE REFERENCES face_media.assets(id),
    method text NOT NULL CHECK (method IN ('bbox', 'ffhq', 'restored')),
    preprocessing jsonb NOT NULL DEFAULT '{}'::jsonb,
    transform_to_source jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX crops_detection_idx ON face_media.face_crops (detection_id);

CREATE TABLE face_media.generation_jobs (
    id uuid PRIMARY KEY,
    input_crop_id uuid NOT NULL,
    input_purpose text NOT NULL DEFAULT 'reference' CHECK (input_purpose = 'reference'),
    FOREIGN KEY (input_crop_id, input_purpose) REFERENCES face_media.face_crops(id, purpose),
    session_id uuid,
    status text NOT NULL CHECK (status IN ('pending', 'running', 'done', 'error')),
    model_name text NOT NULL,
    model_version text NOT NULL,
    initial_age integer CHECK (initial_age BETWEEN 0 AND 120),
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    CHECK (finished_at IS NULL OR finished_at >= created_at)
);
CREATE INDEX generation_input_idx ON face_media.generation_jobs (input_crop_id);
CREATE INDEX generation_status_idx ON face_media.generation_jobs (status, created_at);

CREATE TABLE face_media.generated_images (
    id uuid PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES face_media.generation_jobs(id),
    asset_id uuid NOT NULL UNIQUE REFERENCES face_media.assets(id),
    target_age integer NOT NULL CHECK (target_age BETWEEN 0 AND 120),
    variant_index integer NOT NULL DEFAULT 0 CHECK (variant_index >= 0),
    seed bigint,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (job_id, target_age, variant_index)
);

-- Optional embedding persistence; FAISS remains a rebuildable search index.
CREATE TABLE face_media.face_embeddings (
    id uuid PRIMARY KEY,
    crop_id uuid REFERENCES face_media.face_crops(id),
    generated_image_id uuid REFERENCES face_media.generated_images(id),
    model_name text NOT NULL,
    model_version text NOT NULL,
    preprocessing_version text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions > 0),
    values real[] NOT NULL,
    normalized boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK ((crop_id IS NOT NULL)::integer + (generated_image_id IS NOT NULL)::integer = 1),
    CHECK (array_ndims(values) = 1 AND cardinality(values) = dimensions
        AND array_position(values, NULL) IS NULL)
);
CREATE UNIQUE INDEX embeddings_crop_unique ON face_media.face_embeddings
    (crop_id, model_name, model_version, preprocessing_version) WHERE crop_id IS NOT NULL;
CREATE UNIQUE INDEX embeddings_generated_unique ON face_media.face_embeddings
    (generated_image_id, model_name, model_version, preprocessing_version) WHERE generated_image_id IS NOT NULL;

CREATE VIEW face_media.generated_image_lineage AS
SELECT gi.id AS generated_image_id, gi.target_age, gi.variant_index,
       ga.storage_key AS generated_key, j.id AS job_id,
       c.id AS crop_id, ca.storage_key AS crop_key,
       d.id AS detection_id, d.x1, d.y1, d.x2, d.y2,
       f.frame_index, f.offset_ms, f.captured_at,
       fa.storage_key AS frame_key, s.id AS source_id, s.kind AS source_kind, s.purpose AS source_purpose,
       s.camera_id, oa.storage_key AS original_key
FROM face_media.generated_images gi
JOIN face_media.assets ga ON ga.id = gi.asset_id
JOIN face_media.generation_jobs j ON j.id = gi.job_id
JOIN face_media.face_crops c ON c.id = j.input_crop_id
JOIN face_media.assets ca ON ca.id = c.asset_id
JOIN face_media.face_detections d ON d.id = c.detection_id
JOIN face_media.frames f ON f.id = d.frame_id
JOIN face_media.assets fa ON fa.id = f.asset_id
JOIN face_media.sources s ON s.id = f.source_id
LEFT JOIN face_media.assets oa ON oa.id = s.original_asset_id;
-- Search gallery contains observed crops only, never synthetic outputs.
CREATE VIEW face_media.search_face_gallery AS
SELECT c.id AS crop_id, a.storage_key AS crop_key,
       d.id AS detection_id, d.x1, d.y1, d.x2, d.y2,
       f.frame_index, f.offset_ms, f.captured_at,
       s.id AS source_id, s.kind AS source_kind, s.camera_id
FROM face_media.face_crops c
JOIN face_media.assets a ON a.id = c.asset_id
JOIN face_media.face_detections d ON d.id = c.detection_id
JOIN face_media.frames f ON f.id = d.frame_id
JOIN face_media.sources s ON s.id = f.source_id
WHERE c.purpose = 'search';
COMMIT;
