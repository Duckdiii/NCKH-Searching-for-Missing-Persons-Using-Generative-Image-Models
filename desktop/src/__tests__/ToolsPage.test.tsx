import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

const post = vi.fn();
const get = vi.fn();
vi.mock('../api/client', () => ({
  api: { post: (...a: any[]) => post(...a), get: (...a: any[]) => get(...a) },
  getBaseUrl: () => 'http://127.0.0.1:8000',
  getWsUrl: () => 'ws://127.0.0.1:8000',
}));
vi.mock('../api/useSearchApi', () => ({
  useSearchApi: () => ({ checkHealth: vi.fn(), fetchJobsHistory: vi.fn() }),
}));

import { App } from '../App';
import { ToolsPage } from '../pages/ToolsPage';
import { useSearchStore } from '../store/useSearchStore';

beforeEach(() => {
  post.mockReset();
  get.mockReset();
  get.mockImplementation((url: string, cfg?: any) => {
    if (url === '/api/ops/forgetting/schedule') {
      const p = cfg.params;
      return Promise.resolve({ data: { sighting_days: p.sighting_days, quality: p.quality,
        stability_days: 3, thin_after_days: 2.1, forget_after_days: 7, usage_ratio: 0 } });
    }
    if (url === '/api/search-sources') {
      return Promise.resolve({ data: { items: [{ source_id: 'src-12345678', kind: 'camera',
        storage_policy: 'crop_only', created_at: '2026-10-29T00:00:00Z', purpose: 'search' }], total: 1 } });
    }
    if (url.endsWith('/crops')) {
      return Promise.resolve({ data: { items: [{ crop_id: 'c1', offset_ms: 1600, det_score: 0.9 }], total: 1 } });
    }
    return Promise.resolve({ data: {} });
  });
  useSearchStore.setState({ checkpointReady: true, isCheckingHealth: false, isDemoMode: false,
    jobResult: null, mainView: 'search', sessionHistory: [], toast: null });
});

describe('Tab Công cụ', () => {
  it('luôn mở được từ trang chính (không cần ảnh tạo sinh)', () => {
    render(<App />);
    fireEvent.click(screen.getByTestId('tab-tools'));
    expect(screen.getByTestId('tools-page')).toBeInTheDocument();
  });

  it('kế hoạch lấy mẫu và tối ưu gọi đúng API, hiển thị khuyến nghị', async () => {
    post.mockResolvedValueOnce({ data: { fps: 2.34, binding: 'coverage', face_px: 355, max_recognition_distance_m: 11.1,
      dwell_s: 6.85, fps_tracking: 1.13, fps_coverage: 2.34, samples_per_pass: 16, tracklets_per_pass: 1,
      tracker_timeout_sec: 3, warnings: ['Nhòe chuyển động'] } });
    render(<ToolsPage />);
    await act(async () => { fireEvent.click(screen.getByTestId('tool-plan-run')); });
    expect(post.mock.calls[0][0]).toBe('/api/cameras/sampling-plan');
    expect(post.mock.calls[0][1]).not.toHaveProperty('storage_bytes');
    expect(screen.getByTestId('tool-plan-result')).toHaveTextContent('2.34');

    const rec = { fps: 5, k: 2, q_min: 0.3, p_found: 0.989, crops_per_pass: 2, tracklets_per_pass: 1,
                  storage_bytes: 0.27e9, false_alarms_per_query: 1.5 };
    post.mockResolvedValueOnce({ data: { recommended: rec, pareto: [rec], best_p_found: 0.998,
      quality_model: 'FG-NET', retention_days: { thin: 2, forget: 7 }, warnings: [] } });
    await act(async () => { fireEvent.click(screen.getByTestId('tool-opt-run')); });
    expect(post.mock.calls[1][0]).toBe('/api/ops/pipeline/optimize');
    expect(screen.getByTestId('tool-opt-result')).toHaveTextContent('98.9%');
  });

  it('lịch quên tải bảng mẫu và chạy thử không xóa', async () => {
    render(<ToolsPage />);
    await act(async () => { fireEvent.click(screen.getByTestId('tools-tab-forget')); });
    expect(await screen.findByTestId('tool-forget-schedule')).toHaveTextContent('7 ngày');
    post.mockResolvedValueOnce({ data: { counts: { keep: 3, thin: 1, forget: 2, hold: 0 }, crops_thin: 1,
      crops_forget: 4, bytes_to_free: 1e6 } });
    await act(async () => { fireEvent.click(screen.getByTestId('tool-forget-plan')); });
    expect(post.mock.calls[0][0]).toBe('/api/ops/forgetting/plan');
    expect(screen.getByTestId('tool-forget-db')).toHaveTextContent('cần quên 2');
  });

  it('chọn ảnh tốt nhất gửi nhiều file và hiện xếp hạng', async () => {
    render(<ToolsPage />);
    fireEvent.click(screen.getByTestId('tools-tab-rank'));
    const files = [new File(['a'], 'a.jpg', { type: 'image/jpeg' }), new File(['b'], 'b.jpg', { type: 'image/jpeg' })];
    fireEvent.change(screen.getByTestId('tool-rank-input'), { target: { files } });
    post.mockResolvedValueOnce({ data: { ranked: [{ rank: 1, file: 'b.jpg', score: 0.75, quality: 0.75, consistency: 1,
      gate: null, components: { sharpness: 0.8 }, weakest: 'sharpness', thumbnail: 'data:image/jpeg;base64,x' }],
      no_face: [{ file: 'a.jpg', reason: 'không thấy khuôn mặt' }], mixed: false, warning: null } });
    await act(async () => { fireEvent.click(screen.getByTestId('tool-rank-run')); });
    const form = post.mock.calls[0][1] as FormData;
    expect(post.mock.calls[0][0]).toBe('/api/tools/rank-faces');
    expect(form.getAll('files')).toHaveLength(2);
    expect(screen.getByTestId('tool-rank')).toHaveTextContent('#1 b.jpg');
    expect(screen.getByTestId('tool-rank')).toHaveTextContent('không thấy khuôn mặt');
  });

  it('kho dữ liệu liệt kê nguồn và crop', async () => {
    render(<ToolsPage />);
    await act(async () => { fireEvent.click(screen.getByTestId('tools-tab-store')); });
    const btn = await screen.findByText(/camera · crop_only/);
    await act(async () => { fireEvent.click(btn); });
    expect(screen.getByTestId('tool-store')).toHaveTextContent('1 crop');
    expect(screen.getByAltText('crop')).toHaveAttribute('src', expect.stringContaining('/api/search-sources/crops/c1/thumb'));
  });
});
