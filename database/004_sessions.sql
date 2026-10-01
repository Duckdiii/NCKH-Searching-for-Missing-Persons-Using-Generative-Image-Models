-- T12 — sessions bền vững: metadata + crop đã chọn để tiếp tục sau restart.
-- RAM chỉ là cache; DB là nguồn sự thật. Không serialize numpy/Face object.
-- Chạy qua migration runner (python database/migrate.py migrate).
BEGIN;

-- Một session tương tác tham chiếu: gắn nguồn ảnh gốc + crop đang chọn.
-- chosen_face_index: vị trí mặt trong lần detect của run_id (để dựng lại
-- chosen_face bằng re-detect khi restore; bbox/detection_id là chốt lineage).
CREATE TABLE IF NOT EXISTS face_media.sessions (
    id uuid PRIMARY KEY,
    source_id uuid REFERENCES face_media.sources(id),
    frame_id uuid REFERENCES face_media.frames(id),
    chosen_face_index integer CHECK (chosen_face_index IS NULL OR chosen_face_index >= 0),
    chosen_detection_id uuid REFERENCES face_media.face_detections(id),
    current_crop_id uuid REFERENCES face_media.face_crops(id),
    gender_word text NOT NULL DEFAULT 'man' CHECK (gender_word IN ('man', 'woman')),
    initial_age integer CHECK (initial_age IS NULL OR (initial_age BETWEEN 0 AND 120)),
    photo_year integer CHECK (photo_year IS NULL OR photo_year >= 1900),
    file_name text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_source_idx
    ON face_media.sessions (source_id, updated_at);

COMMIT;
