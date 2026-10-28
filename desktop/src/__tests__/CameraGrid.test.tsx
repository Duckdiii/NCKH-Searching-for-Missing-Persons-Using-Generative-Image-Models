import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
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
  useSearchApi: () => ({
    checkHealth: vi.fn(), fetchJobsHistory: vi.fn(), uploadImage: vi.fn(),
    estimateAge: vi.fn(), runPipeline: vi.fn(),
  }),
}));

import { App } from '../App';
import { CameraGridPage, hasGeneratedImages } from '../pages/CameraGridPage';
import { useSearchStore } from '../store/useSearchStore';

const DONE_JOB = {
  job_id: 'job-1', status: 'done' as const, edited_images: { 25: '/outputs/jobs/job-1/age_25.png' },
  final_scores: {}, accepted: false, top_identity: '', top_score: 0,
};

beforeEach(() => {
  post.mockReset();
  get.mockReset();
  (globalThis as any).URL.createObjectURL = vi.fn(() => 'blob:preview');
  (globalThis as any).URL.revokeObjectURL = vi.fn();
  useSearchStore.setState({
    checkpointReady: true, isCheckingHealth: false, isDemoMode: false,
    currentWizardStep: 'restore', jobId: null, jobStatus: 'idle', jobResult: null,
    sessionHistory: [], isHistoricalView: false, toast: null, mainView: 'search',
  });
});

describe('Tab 8 camera', () => {
  it('chỉ mở khi đã có ảnh tạo sinh', () => {
    expect(hasGeneratedImages(null)).toBe(false);
    expect(hasGeneratedImages({ status: 'done', edited_images: {} })).toBe(false);
    expect(hasGeneratedImages(DONE_JOB)).toBe(true);

    const { unmount } = render(<App />);
    expect(screen.getByTestId('tab-cameras')).toBeDisabled();
    unmount();

    useSearchStore.setState({ jobResult: DONE_JOB, jobId: 'job-1' });
    render(<App />);
    const tab = screen.getByTestId('tab-cameras');
    expect(tab).toBeEnabled();
    fireEvent.click(tab);
    expect(useSearchStore.getState().mainView).toBe('cameras');
    expect(screen.getByTestId('camera-grid-page')).toBeInTheDocument();
    expect(screen.getAllByTestId(/^camera-slot-\d$/)).toHaveLength(8);
  });

  it('trang tự khóa nếu mở mà chưa có ảnh tạo sinh', () => {
    render(<CameraGridPage />);
    expect(screen.getByTestId('camera-grid-locked')).toBeInTheDocument();
  });

  it('chạy camera đã có video, theo dõi tiến độ rồi đối soát', async () => {
    useSearchStore.setState({ jobResult: DONE_JOB, jobId: 'job-1' });
    render(<CameraGridPage />);
    const runBtn = screen.getByTestId('replay-run-all');
    expect(runBtn).toBeDisabled();

    const file = new File(['x'], 'cam3.mp4', { type: 'video/mp4' });
    fireEvent.change(screen.getByTestId('camera-slot-3-input'), { target: { files: [file] } });
    expect(runBtn).toBeEnabled();

    post.mockResolvedValueOnce({ data: { source_id: 'src-3', slot: 3 } });
    get.mockResolvedValue({
      data: { source_id: 'src-3', camera_id: 'c3', status: 'running', frames_sampled: 30,
              faces_found: 3, tracks_closed: 1, crops_persisted: 3, rejected_gate: 0,
              rejected_low_quality: 0,
              replay: { position_sec: 6, duration_sec: 6, progress: 1, finished: true,
                        offset_sec: 0, truncated: false } },
    });
    await act(async () => { fireEvent.click(runBtn); });
    const [url, form] = post.mock.calls[0];
    expect(url).toBe('/api/replay/start');
    expect((form as FormData).get('slot')).toBe('3');
    expect((form as FormData).get('job_id')).toBe('job-1');

    await waitFor(() => expect(screen.getByTestId('replay-match')).toBeEnabled(), { timeout: 4000 });

    const match = {
      slot: 3, face_image_url: '/outputs/media/crop.jpg', frame_index: 0, timestamp_sec: 1.6,
      bbox: [0, 0, 1, 1], det_score: 0.9, best_age: 25, best_age_image_url: '/x.png',
      score: 0.84, accepted: true, camera_time_sec: 1.6,
    };
    post.mockResolvedValueOnce({ data: {
      job_id: 'job-1', threshold: 0.45, reference_ages: [25],
      cameras: [{ slot: 3, source_id: 'src-3', crops: 3, best_match: match, matches: [match],
                  accepted: 1, tracks_closed: 1, crops_persisted: 3 }],
      timeline: [match], found_on_cameras: [3],
    } });
    await act(async () => { fireEvent.click(screen.getByTestId('replay-match')); });
    expect(post.mock.calls[1][0]).toBe('/api/replay/match');
    expect(post.mock.calls[1][1].sources).toEqual([{ slot: 3, source_id: 'src-3', start_offset_sec: 0 }]);
    expect(await screen.findByTestId('replay-timeline')).toHaveTextContent('Khớp ở camera 3');
    expect(screen.getByTestId('camera-slot-3')).toHaveTextContent('84.0%');
  });
});
