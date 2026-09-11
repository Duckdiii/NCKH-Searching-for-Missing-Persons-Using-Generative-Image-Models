import { useCallback } from 'react';
import { api, getWsUrl } from './client';
import { useSearchStore } from '../store/useSearchStore';
import {
  CheckpointHealth,
  ResolveAgeResponse,
  SelectFaceResponse,
  UploadResponse,
} from '../types/api';

export function useSearchApi() {
  const store = useSearchStore();

  const checkHealth = useCallback(async () => {
    try {
      const res = await api.get<CheckpointHealth>('/api/health/checkpoints');
      store.setCheckpoints(res.data.ready, res.data.missing);
      return res.data;
    } catch (err) {
      console.error('Failed to check health:', err);
      store.setCheckpoints(false, ['Không thể kết nối đến Backend API server (Port 8000). Hãy chạy lệnh: python -m backend.api.main']);
      store.setIsCheckingHealth(false);
      return { ready: false, missing: ['Không kết nối được backend'] };
    }
  }, [store]);

  const uploadImage = useCallback(async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post<UploadResponse>('/api/sessions', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    const previewUrl = URL.createObjectURL(file);
    store.setUploadResult(res.data.session_id, res.data.faces, previewUrl);
    return res.data;
  }, [store]);

  const selectFace = useCallback(async (sessionId: string, selectedIdx: number) => {
    const res = await api.post<SelectFaceResponse>(`/api/sessions/${sessionId}/select-face`, {
      selected_idx: selectedIdx,
    });
    store.setSelectedFace(selectedIdx, res.data.warnings, res.data.cropped_preview_url);
    return res.data;
  }, [store]);

  const resolveAge = useCallback(
    async (sessionId: string, mode: 'manual' | 'mivolo', manualAge?: number, genderWord: 'man' | 'woman' = 'man') => {
      if (mode === 'mivolo') {
        store.setIsEstimatingAge(true);
      }
      try {
        const res = await api.post<ResolveAgeResponse>(`/api/sessions/${sessionId}/resolve-age`, {
          mode,
          manual_age: mode === 'manual' ? manualAge : null,
          gender_word: genderWord,
        });
        store.setResolvedAge(res.data.initial_age, res.data.warning_text);
        return res.data;
      } finally {
        store.setIsEstimatingAge(false);
      }
    },
    [store]
  );

  const runPipeline = useCallback(async (sessionId: string, galleryDir?: string | null) => {
    const res = await api.post<{ job_id: string; status: string }>(
      `/api/sessions/${sessionId}/run`,
      { gallery_dir: galleryDir }
    );
    const jobId = res.data.job_id;
    store.startJob(jobId);

    // Mở WebSocket lắng nghe tiến độ
    const wsUrl = `${getWsUrl()}/api/jobs/${jobId}/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status === 'done') {
          store.updateJobProgress('done', 'complete', data.result || data);
          ws.close();
        } else if (data.status === 'error') {
          store.updateJobProgress('error', 'failed', undefined, data.error_message);
          ws.close();
        } else {
          store.updateJobProgress('running', data.stage || 'running');
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    return jobId;
  }, [store]);

  return {
    checkHealth,
    uploadImage,
    selectFace,
    resolveAge,
    runPipeline,
  };
}
