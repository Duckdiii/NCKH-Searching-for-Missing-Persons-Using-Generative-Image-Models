import { api, getWsUrl } from './client';
import { useSearchStore } from '../store/useSearchStore';

// Connections belong to jobs, not to the currently selected screen.
const watchers = new Map<string, () => void>();

export function stopWatchingJob(jobId: string) {
  watchers.get(jobId)?.();
}

export function watchJob(jobId: string) {
  if (watchers.has(jobId)) return;
  let closed = false;
  let polling = false;
  let revision = 0;
  let ws: WebSocket | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;
  const dispose = () => {
    closed = true;
    if (timer) clearInterval(timer);
    ws?.close();
    watchers.delete(jobId);
  };
  watchers.set(jobId, dispose);
  const apply = (data: any) => {
    if (closed) return;
    const status = data.status;
    if (!['running', 'done', 'error'].includes(status)) return;
    useSearchStore.getState().updateJobProgress(
      status, data.stage || (status === 'done' ? 'complete' : status === 'error' ? 'failed' : 'running'),
      status === 'done' ? data.result || data : undefined,
      data.error_message, jobId,
    );
    if (status === 'done' || status === 'error') dispose();
  };
  const poll = async () => {
    if (polling || closed) return;
    polling = true;
    const atStart = revision;
    try {
      const res = await api.get(`/api/jobs/${jobId}`);
      if (atStart === revision) apply(res.data);
    } catch (err: any) {
      if (!closed && err.response?.status === 404) {
        apply({ status: 'error', error_message: 'Tác vụ không còn trên máy chủ. Hãy kiểm tra lại phiên.' });
      }
    } finally {
      polling = false;
    }
  };
  try {
    ws = new WebSocket(`${getWsUrl()}/api/jobs/${jobId}/ws`);
    ws.onmessage = event => {
      try { revision++; apply(JSON.parse(event.data)); } catch { /* polling recovers */ }
    };
    ws.onerror = () => { /* polling continues while the socket is unavailable */ };
  } catch { /* HTTP polling also works when WebSocket is unavailable */ }
  timer = setInterval(() => void poll(), 2000);
  void poll();
}
