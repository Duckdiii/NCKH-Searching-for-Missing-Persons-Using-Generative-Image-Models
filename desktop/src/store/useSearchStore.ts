import { create } from 'zustand';
import { FaceBox, JobHistoryItem, JobResult } from '../types/api';
import { ToastData } from '../components/Toast';

const HISTORY_STORAGE_KEY = 'fading_session_history';

function loadInitialHistory(): JobHistoryItem[] {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {
    console.error('Failed to load history from localStorage:', e);
  }
  return [];
}

function saveHistoryToStorage(history: JobHistoryItem[]) {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
  } catch (e) {
    console.error('Failed to save history to localStorage:', e);
  }
}

interface SearchState {
  // Backend config & metadata
  backendPort: number;
  checkpointReady: boolean;
  missingCheckpoints: string[];
  isCheckingHealth: boolean;
  checkpointName: string;
  appVersion: string;

  // Shell & Navigation state
  isAdvancedMode: boolean;
  isDemoMode: boolean;
  isSidebarCollapsed: boolean;
  sessionHistory: JobHistoryItem[];
  isHistoricalView: boolean;
  currentWizardStep: 'restore' | 'generate' | 'results';

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
  photoYear: number | null;
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

  // Toast notification
  toast: ToastData | null;
  setToast: (toast: ToastData | null) => void;

  // Actions
  setBackendPort: (port: number) => void;
  setCheckpoints: (ready: boolean, missing: string[], checkpointName?: string, appVersion?: string) => void;
  setIsCheckingHealth: (checking: boolean) => void;
  toggleAdvancedMode: () => void;
  setAdvancedMode: (val: boolean) => void;
  toggleDemoMode: () => void;
  setDemoMode: (val: boolean) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (val: boolean) => void;
  setCurrentWizardStep: (step: 'restore' | 'generate' | 'results') => void;
  addHistoryItem: (item: JobHistoryItem) => void;
  loadHistoricalJob: (item: JobHistoryItem) => void;
  startNewSearch: () => void;
  setUploadResult: (sessionId: string, faces: FaceBox[], imageUrl: string) => void;
  setSelectedFace: (idx: number, warnings: string[], cropUrl: string) => void;
  setGenderWord: (gender: 'man' | 'woman') => void;
  setAgeMode: (mode: 'manual' | 'mivolo') => void;
  setManualAge: (age: number) => void;
  setPhotoYear: (year: number | null) => void;
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

export const useSearchStore = create<SearchState>((set, get) => ({
  backendPort: 8000,
  checkpointReady: true,
  missingCheckpoints: [],
  isCheckingHealth: true,
  checkpointName: 'unknown',
  appVersion: '0.1.0',

  isAdvancedMode: false,
  isDemoMode: false,
  isSidebarCollapsed: false,
  sessionHistory: loadInitialHistory(),
  isHistoricalView: false,
  currentWizardStep: 'restore',

  sessionId: null,
  uploadedImageUrl: null,
  faces: [],
  selectedFaceIdx: null,
  warnings: [],
  croppedPreviewUrl: null,

  genderWord: 'man',
  ageMode: 'manual',
  manualAge: 10,
  photoYear: null,
  initialAge: null,
  ageWarningText: null,
  isEstimatingAge: false,

  galleryDir: null,

  jobId: null,
  jobStatus: 'idle',
  jobStage: 'specialization',
  jobResult: null,
  jobError: null,

  toast: null,
  setToast: (toast) => set({ toast }),

  setBackendPort: (port) => set({ backendPort: port }),
  setCheckpoints: (ready, missing, checkpointName, appVersion) => set({
    checkpointReady: ready,
    missingCheckpoints: missing,
    isCheckingHealth: false,
    ...(checkpointName ? { checkpointName } : {}),
    ...(appVersion ? { appVersion } : {}),
  }),
  setIsCheckingHealth: (checking) => set({ isCheckingHealth: checking }),

  toggleAdvancedMode: () => set((state) => ({ isAdvancedMode: !state.isAdvancedMode })),
  setAdvancedMode: (val) => set({ isAdvancedMode: val }),
  toggleDemoMode: () => set((state) => ({ isDemoMode: !state.isDemoMode })),
  setDemoMode: (val) => set({ isDemoMode: val }),

  toggleSidebar: () => set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed })),
  setSidebarCollapsed: (val) => set({ isSidebarCollapsed: val }),
  setCurrentWizardStep: (step) => set({ currentWizardStep: step }),

  addHistoryItem: (item) => {
    const existing = get().sessionHistory;
    const filtered = existing.filter((h) => h.job_id !== item.job_id);
    const updated = [item, ...filtered];
    saveHistoryToStorage(updated);
    set({ sessionHistory: updated });
  },

  loadHistoricalJob: (item) => {
    if (item.result) {
      set({
        jobId: item.job_id,
        sessionId: item.session_id,
        jobStatus: item.status,
        jobStage: item.stage,
        jobResult: item.result,
        croppedPreviewUrl: item.cropped_preview_url || get().croppedPreviewUrl,
        initialAge: item.initial_age ?? get().initialAge,
        photoYear: item.photo_year ?? null,
        genderWord: (item.gender_word as any) || get().genderWord,
        isHistoricalView: true,
        currentWizardStep: 'results',
      });
    }
  },

  startNewSearch: () => {
    set({
      sessionId: null,
      uploadedImageUrl: null,
      faces: [],
      selectedFaceIdx: null,
      warnings: [],
      croppedPreviewUrl: null,
      initialAge: null,
      photoYear: null,
      ageWarningText: null,
      jobId: null,
      jobStatus: 'idle',
      jobStage: 'specialization',
      jobResult: null,
      jobError: null,
      isHistoricalView: false,
      currentWizardStep: 'restore',
    });
  },

  setUploadResult: (sessionId, faces, imageUrl) => set({
    sessionId,
    faces,
    uploadedImageUrl: imageUrl,
    selectedFaceIdx: faces.length === 1 ? 0 : null,
    warnings: [],
    croppedPreviewUrl: null,
    initialAge: null,
    photoYear: null,
    ageWarningText: null,
    jobId: null,
    jobStatus: 'idle',
    jobResult: null,
    jobError: null,
    isHistoricalView: false,
  }),
  setSelectedFace: (idx, warnings, cropUrl) => set({
    selectedFaceIdx: idx,
    warnings,
    croppedPreviewUrl: cropUrl,
  }),
  setGenderWord: (gender) => set({ genderWord: gender }),
  setAgeMode: (mode) => set({ ageMode: mode }),
  setManualAge: (age) => set({ manualAge: age }),
  setPhotoYear: (year) => set({ photoYear: year }),
  setIsEstimatingAge: (estimating) => set({ isEstimatingAge: estimating }),
  setResolvedAge: (age, warningText) => set({
    initialAge: age,
    ageWarningText: warningText ?? null,
    isEstimatingAge: false,
  }),
  setGalleryDir: (dir) => set({ galleryDir: dir }),
  startJob: (jobId) => {
    const sessionId = get().sessionId || 'unknown';
    const newHistoryItem: JobHistoryItem = {
      job_id: jobId,
      session_id: sessionId,
      status: 'running',
      stage: 'specialization',
      timestamp: Date.now(),
      cropped_preview_url: get().croppedPreviewUrl,
      initial_age: get().initialAge ?? get().manualAge,
      photo_year: get().photoYear,
      gender_word: get().genderWord,
    };
    get().addHistoryItem(newHistoryItem);

    set({
      jobId,
      jobStatus: 'running',
      jobStage: 'specialization',
      jobResult: null,
      jobError: null,
      isHistoricalView: false,
    });
  },
  updateJobProgress: (status, stage, result, error) => {
    const prevStatus = get().jobStatus;
    const jobId = get().jobId;
    if (jobId) {
      const history = get().sessionHistory;
      const target = history.find((h) => h.job_id === jobId);
      if (target) {
        target.status = status;
        target.stage = stage;
        if (result) {
          target.result = result;
          target.top_identity = result.top_identity;
          target.top_score = result.top_score;
          target.accepted = result.accepted;
        }
        if (error) {
          target.error_message = error;
        }
        saveHistoryToStorage([...history]);
        set({ sessionHistory: [...history] });
      }
    }

    // Giai đoạn 3: Hiện Toast thông báo khi job hoàn tất lúc người dùng đang ở màn khác
    if (status === 'done' && prevStatus === 'running') {
      const isAway = get().isHistoricalView || get().currentWizardStep !== 'results';
      if (isAway) {
        const identityName = result?.top_identity || 'Đối tượng';
        const scorePct = typeof result?.top_score === 'number'
          ? `${(result.top_score * 100).toFixed(1)}%`
          : null;
        get().setToast({
          id: String(Date.now()),
          identityName,
          scorePct,
          onViewResult: () => {
            set({
              isHistoricalView: false,
              jobResult: result || null,
            });
            get().setCurrentWizardStep('results');
          },
        });
      }
    }

    set({
      jobStatus: status,
      jobStage: stage,
      jobResult: result !== undefined ? (result || null) : get().jobResult,
      jobError: error !== undefined ? (error || null) : null,
    });
  },
  resetForNewUpload: () => set({
    sessionId: null,
    uploadedImageUrl: null,
    faces: [],
    selectedFaceIdx: null,
    warnings: [],
    croppedPreviewUrl: null,
    initialAge: null,
    photoYear: null,
    ageWarningText: null,
    jobId: null,
    jobStatus: 'idle',
    jobResult: null,
  }),
}));

if (typeof window !== 'undefined') {
  (window as any).__SEARCH_STORE__ = useSearchStore;
}
