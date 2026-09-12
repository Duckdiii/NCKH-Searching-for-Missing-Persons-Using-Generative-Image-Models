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
  gender_word: 'man' | 'woman';
}

export interface ResolveAgeResponse {
  initial_age: number;
  gender_word: string;
  warning_text?: string | null;
}

export interface RunPipelineRequest {
  gallery_dir?: string | null;
}

export interface JobStatus {
  job_id: string;
  status: 'running' | 'done' | 'error';
  stage: 'specialization' | 'inversion' | 'editing' | 'search' | 'complete' | 'failed' | string;
  error_message?: string | null;
  result?: JobResult;
}

export interface JobResult {
  job_id: string;
  status: 'done' | 'error';
  edited_images: Record<number, string>; // age -> image URL
  final_scores: Record<string, number>;
  accepted: boolean;
  top_identity: string;
  top_score: number;
  best_age?: number | null;
  matched_gallery_image?: string | null;
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
}
