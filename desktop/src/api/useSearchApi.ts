import { useCallback } from 'react';
import { api, getWsUrl } from './client';
import { useSearchStore } from '../store/useSearchStore';
import {
  CheckpointHealth,
  JobHistoryItem,
  ResolveAgeResponse,
  SelectFaceResponse,
  UploadResponse,
} from '../types/api';

export function useSearchApi() {
  const checkHealth = useCallback(async () => {
    const s = useSearchStore.getState();
    try {
      const res = await api.get<CheckpointHealth>('/api/health/checkpoints');
      s.setCheckpoints(res.data.ready, res.data.missing, res.data.checkpoint_name, res.data.app_version);
      return res.data;
    } catch (err) {
      console.error('Failed to check health:', err);
      s.setCheckpoints(false, ['Không thể kết nối đến Backend API server (Port 8000). Hãy chạy lệnh: python -m backend.api.main']);
      s.setIsCheckingHealth(false);
      return { ready: false, missing: ['Không kết nối được backend'] };
    }
  }, []);

  const fetchJobsHistory = useCallback(async () => {
    const s = useSearchStore.getState();
    try {
      const res = await api.get<JobHistoryItem[]>('/api/jobs');
      if (Array.isArray(res.data)) {
        res.data.forEach((item) => s.addHistoryItem(item));
      }
      return res.data;
    } catch (err) {
      console.warn('Could not fetch jobs history from backend:', err);
      return [];
    }
  }, []);

  const uploadImage = useCallback(async (file: File) => {
    const s = useSearchStore.getState();
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post<UploadResponse>('/api/sessions', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    const previewUrl = URL.createObjectURL(file);
    s.setUploadResult(res.data.session_id, res.data.faces, previewUrl);
    return res.data;
  }, []);

  const selectFace = useCallback(async (sessionId: string, selectedIdx: number) => {
    const s = useSearchStore.getState();
    const res = await api.post<SelectFaceResponse>(`/api/sessions/${sessionId}/select-face`, {
      selected_idx: selectedIdx,
    });
    s.setSelectedFace(selectedIdx, res.data.warnings, res.data.cropped_preview_url);
    return res.data;
  }, []);

  const resolveAge = useCallback(
    async (sessionId: string, mode: 'manual' | 'mivolo', manualAge?: number, genderWord: 'man' | 'woman' = 'man') => {
      const s = useSearchStore.getState();
      if (mode === 'mivolo') {
        s.setIsEstimatingAge(true);
      }
      try {
        const res = await api.post<ResolveAgeResponse>(`/api/sessions/${sessionId}/resolve-age`, {
          mode,
          manual_age: mode === 'manual' ? manualAge : null,
          gender_word: genderWord,
        });
        s.setResolvedAge(res.data.initial_age, res.data.warning_text);
        return res.data;
      } finally {
        s.setIsEstimatingAge(false);
      }
    },
    []
  );

  const runPipeline = useCallback(async (sessionId: string, galleryDir?: string | null) => {
    const s = useSearchStore.getState();
    const res = await api.post<{ job_id: string; status: string }>(
      `/api/sessions/${sessionId}/run`,
      { gallery_dir: galleryDir }
    );
    const jobId = res.data.job_id;
    s.startJob(jobId);

    // Mở WebSocket lắng nghe tiến độ
    const wsUrl = `${getWsUrl()}/api/jobs/${jobId}/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status === 'done') {
          s.updateJobProgress('done', 'complete', data.result || data);
          ws.close();
        } else if (data.status === 'error') {
          s.updateJobProgress('error', 'failed', undefined, data.error_message);
          ws.close();
        } else {
          s.updateJobProgress('running', data.stage || 'running');
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    return jobId;
  }, []);

  return {
    checkHealth,
    fetchJobsHistory,
    uploadImage,
    selectFace,
    resolveAge,
    runPipeline,
  };
}
