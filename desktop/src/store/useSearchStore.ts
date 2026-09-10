import { create } from 'zustand';
import { FaceBox, JobResult } from '../types/api';

interface SearchState {
  // Backend config
  backendPort: number;
  checkpointReady: boolean;
  missingCheckpoints: string[];
  isCheckingHealth: boolean;

  // Session state
  sessionId: string | null;
  uploadedImageUrl: string | null;
  faces: FaceBox[];
  selectedFaceIdx: number | null;
  warnings: string[];
  croppedPreviewUrl: string | null;

  // Age & Gender
  genderWord: 'man' | 'woman';
  ageMode: 'manual' | 'mivolo';
  manualAge: number;
  initialAge: number | null;
  ageWarningText: string | null;
  isEstimatingAge: boolean;

  // Gallery
  galleryDir: string | null;

  // Job status
  jobId: string | null;
  jobStatus: 'idle' | 'running' | 'done' | 'error';
  jobStage: string;
  jobResult: JobResult | null;
  jobError: string | null;

  // Actions
  setBackendPort: (port: number) => void;
  setCheckpoints: (ready: boolean, missing: string[]) => void;
  setIsCheckingHealth: (checking: boolean) => void;
  setUploadResult: (sessionId: string, faces: FaceBox[], imageUrl: string) => void;
  setSelectedFace: (idx: number, warnings: string[], cropUrl: string) => void;
  setGenderWord: (gender: 'man' | 'woman') => void;
  setAgeMode: (mode: 'manual' | 'mivolo') => void;
  setManualAge: (age: number) => void;
  setIsEstimatingAge: (estimating: boolean) => void;
  setResolvedAge: (age: number, warningText?: string | null) => void;
  setGalleryDir: (dir: string | null) => void;
  startJob: (jobId: string) => void;
  updateJobProgress: (
    status: 'idle' | 'running' | 'done' | 'error',
    stage: string,
    result?: JobResult,
    error?: string | null
  ) => void;
  resetForNewUpload: () => void;
}

export const useSearchStore = create<SearchState>((set) => ({
  backendPort: 8000,
  checkpointReady: true,
  missingCheckpoints: [],
  isCheckingHealth: true,

  sessionId: null,
  uploadedImageUrl: null,
  faces: [],
  selectedFaceIdx: null,
  warnings: [],
  croppedPreviewUrl: null,

  genderWord: 'man',
  ageMode: 'manual',
  manualAge: 10,
  initialAge: null,
  ageWarningText: null,
  isEstimatingAge: false,

  galleryDir: null,

  jobId: null,
  jobStatus: 'idle',
  jobStage: 'specialization',
  jobResult: null,
  jobError: null,

  setBackendPort: (port) => set({ backendPort: port }),
  setCheckpoints: (ready, missing) => set({ checkpointReady: ready, missingCheckpoints: missing, isCheckingHealth: false }),
  setIsCheckingHealth: (checking) => set({ isCheckingHealth: checking }),
  setUploadResult: (sessionId, faces, imageUrl) => set({
    sessionId,
    faces,
    uploadedImageUrl: imageUrl,
    selectedFaceIdx: faces.length === 1 ? 0 : null,
    warnings: [],
    croppedPreviewUrl: null,
    initialAge: null,
    ageWarningText: null,
    jobId: null,
    jobStatus: 'idle',
    jobResult: null,
    jobError: null,
  }),
  setSelectedFace: (idx, warnings, cropUrl) => set({
    selectedFaceIdx: idx,
    warnings,
    croppedPreviewUrl: cropUrl,
  }),
  setGenderWord: (gender) => set({ genderWord: gender }),
  setAgeMode: (mode) => set({ ageMode: mode }),
  setManualAge: (age) => set({ manualAge: age }),
  setIsEstimatingAge: (estimating) => set({ isEstimatingAge: estimating }),
  setResolvedAge: (age, warningText) => set({
    initialAge: age,
    ageWarningText: warningText ?? null,
    isEstimatingAge: false,
  }),
  setGalleryDir: (dir) => set({ galleryDir: dir }),
  startJob: (jobId) => set({
    jobId,
    jobStatus: 'running',
    jobStage: 'specialization',
    jobResult: null,
    jobError: null,
  }),
  updateJobProgress: (status, stage, result, error) => set({
    jobStatus: status,
    jobStage: stage,
    jobResult: result || null,
    jobError: error || null,
  }),
  resetForNewUpload: () => set({
    sessionId: null,
    uploadedImageUrl: null,
    faces: [],
    selectedFaceIdx: null,
    warnings: [],
    croppedPreviewUrl: null,
    initialAge: null,
    ageWarningText: null,
    jobId: null,
    jobStatus: 'idle',
    jobResult: null,
    jobError: null,
  }),
}));
