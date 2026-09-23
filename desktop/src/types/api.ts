export interface FaceBox {
  index: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  det_score: number;
}

export interface UploadResponse {
  session_id: string;
  faces: FaceBox[];
}

export interface SelectFaceRequest {
  selected_idx: number;
}

export interface SelectFaceResponse {
  warnings: string[];
  cropped_preview_url: string;
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

export interface JobResult {
  job_id: string;
  status: 'done' | 'error';
  edited_images: Record<number, string>; // age -> image URL
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
}
