export interface FaceBox {
  index: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  det_score: number;
}

export interface UploadResponse {
  session_id: string;
  faces: FaceBox[];
  // T04: lineage bền vững (null khi backend chưa có DB — dùng session_id).
  source_id?: string | null;
  detection_ids?: string[];
}

export interface SelectFaceRequest {
  selected_idx: number;
}

export interface SelectFaceResponse {
  warnings: string[];
  cropped_preview_url: string;
  // T05: revision crop đã áp dụng (null khi chưa có DB).
  crop_id?: string | null;
}

export interface ResolveAgeRequest {
  mode: 'manual' | 'mivolo';
  manual_age?: number | null;
  gender_word?: 'man' | 'woman';
  photo_year?: number | null;
}

export interface ResolveAgeResponse {
  initial_age: number;
  gender_word: string;
  warning_text?: string | null;
}

export interface RunPipelineRequest {
  gallery_dir?: string | null;
  photo_year?: number | null;
}

export interface JobStatus {
  job_id: string;
  status: 'running' | 'done' | 'error';
  stage: 'specialization' | 'inversion' | 'editing' | 'search' | 'complete' | 'failed' | string;
  error_message?: string | null;
  result?: JobResult;
}

export interface PipelineParams {
  num_inference_steps?: number;
  guidance_scale?: number;
  attention_control_ratio?: number;
  checkpoint_name?: string;
  embedding_model?: string;
  rejection_threshold?: number;
}

export interface GeneratedVariant {
  id: string;
  target_age: number;
  variant_index: number;
  seed?: number | null;
  image_url: string;
}

export interface JobResult {
  cropped_image?: string | null;
  job_id: string;
  status: 'done' | 'error';
  edited_images: Record<number, string>; // age -> image URL (adapter UI cũ)
  variants?: GeneratedVariant[]; // T06: biến thể có ID (nhiều variant cùng tuổi)
  age_scores?: Record<number, number>; // age -> ID score
  final_scores: Record<string, number>;
  accepted: boolean;
  top_identity: string;
  top_score: number;
  best_age?: number | null;
  matched_gallery_image?: string | null;
  pipeline_params?: PipelineParams;
  error_message?: string | null;
}

export interface CheckpointHealth {
  ready: boolean;
  missing: string[];
  checkpoint_name?: string;
  app_version?: string;
}

export interface JobHistoryItem {
  job_id: string;
  session_id: string;
  status: 'idle' | 'running' | 'done' | 'error';
  stage: string;
  top_identity?: string | null;
  top_score?: number | null;
  accepted?: boolean | null;
  result?: JobResult | null;
  error_message?: string | null;
  timestamp?: number;
  cropped_preview_url?: string | null;
  initial_age?: number | null;
  photo_year?: number | null;
  gender_word?: string | null;
}

export interface VideoFaceMatch {
  face_image_url: string;
  frame_index: number;
  timestamp_sec: number;
  bbox: [number, number, number, number];
  det_score: number;
  best_age: number;
  best_age_image_url: string;
  score: number;
}

export interface VideoVerifyResponse {
  job_id: string;
  frames_sampled: number;
  faces_found: number;
  best_match: VideoFaceMatch | null;
  matches: VideoFaceMatch[];
  conditions?: Record<string, number>;
  // T08/T11: nguồn độc lập + lượt search đã lưu + phạm vi xử lý.
  source_id?: string | null;
  search_run_id?: string | null;
  processing?: {
    frames_total?: number | null;
    duration_sec?: number;
    truncated?: boolean;
  };
}

// ---- T07/T08: gallery search ----

export interface SearchSourceItem {
  source_id: string;
  purpose: string;
  kind: string;
  original_asset_id?: string | null;
  camera_id?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
  created_at: string;
}

export interface SourceCropItem {
  crop_id: string;
  crop_key: string;
  crop_url?: string | null;
  method: string;
  bbox: [number, number, number, number];
  det_score: number;
  frame_index: number;
  offset_ms: number;
  captured_at?: string | null;
  frame_id: string;
}

export interface IngestionRunStatus {
  run_id: string;
  source_id: string;
  status: string;
  fps_target: number;
  max_frames: number;
  frames_sampled: number;
  faces_found: number;
  frames_total?: number | null;
  duration_sec?: number | null;
  truncated: boolean;
  error_message?: string | null;
}

// ---- T11: search runs ----

export interface SearchResultItem {
  result_id: string;
  candidate_crop_id: string;
  crop_key: string;
  crop_url?: string | null;
  best_generated_image_id?: string | null;
  score: number;
  rank: number;
  accepted_by_threshold: boolean;
  human_confirmed: boolean;
  frame_index: number;
  offset_ms: number;
  captured_at?: string | null;
  source_id: string;
  source_kind: string;
  camera_id?: string | null;
}

export interface SearchRunDetail {
  run_id: string;
  query_kind: string;
  query_crop_id?: string | null;
  scope_source_id?: string | null;
  generation_job_id?: string | null;
  threshold: number;
  created_at: string;
  results: SearchResultItem[];
}
