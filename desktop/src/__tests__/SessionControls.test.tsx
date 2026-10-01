import { render, screen, fireEvent, waitFor, act, cleanup } from '@testing-library/react';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { useSearchStore } from '../store/useSearchStore';
import { Sidebar } from '../components/Sidebar';
import { SearchPage } from '../pages/SearchPage';
import { JobHistoryItem } from '../types/api';

const mocks = vi.hoisted(() => ({ stop: vi.fn(), remove: vi.fn(), watch: vi.fn() }));
vi.mock('../api/jobMonitor', () => ({ watchJob: mocks.watch }));
vi.mock('../api/useSearchApi', () => ({ useSearchApi: () => ({
  stopSession: mocks.stop, deleteSession: mocks.remove,
  uploadImage: vi.fn(), runPipeline: vi.fn(), applyRestoration: vi.fn(),
}) }));

const running: JobHistoryItem = { job_id: 'job-a', session_id: 'session-a', status: 'running', stage: 'inversion' };

beforeEach(() => {
  vi.clearAllMocks();
  useSearchStore.getState().startNewSearch();
  useSearchStore.setState({ sessionHistory: [], toast: null, isDemoMode: false });
});
afterEach(cleanup);

describe('Session controls and running history', () => {
  it('reopens a running session without a result after switching away', () => {
    useSearchStore.getState().addHistoryItem(running);
    render(<><Sidebar /><SearchPage /></>);
    fireEvent.click(screen.getByTestId('history-item'));
    expect(useSearchStore.getState().jobId).toBe('job-a');
    expect(useSearchStore.getState().currentWizardStep).toBe('generate');
    expect(screen.getByRole('button', { name: 'Dừng phiên' })).toBeEnabled();
    expect(mocks.watch).toHaveBeenCalledWith('job-a');
    fireEvent.click(screen.getByTestId('sidebar-new-search-btn'));
    expect(useSearchStore.getState().sessionId).toBeNull();
    fireEvent.click(screen.getByTestId('history-item'));
    expect(useSearchStore.getState().jobStatus).toBe('running');
    expect(useSearchStore.getState().currentWizardStep).toBe('generate');
  });

  it('updates the originating job without overwriting the selected session', () => {
    const s = useSearchStore.getState();
    s.addHistoryItem(running);
    const other = { ...running, job_id: 'job-b', session_id: 'session-b', stage: 'editing' };
    s.addHistoryItem(other);
    s.loadHistoricalJob(other);
    s.updateJobProgress('error', 'failed', undefined, 'Stopped A', 'job-a');
    expect(useSearchStore.getState().jobId).toBe('job-b');
    expect(useSearchStore.getState().jobStatus).toBe('running');
    expect(useSearchStore.getState().jobStage).toBe('editing');
    expect(useSearchStore.getState().sessionHistory.find(h => h.job_id === 'job-a')?.error_message).toBe('Stopped A');
  });

  it('shows pending stop without falsely resetting the running job', async () => {
    let finish!: () => void;
    mocks.stop.mockImplementation(() => new Promise<void>(resolve => { finish = resolve; }));
    useSearchStore.getState().addHistoryItem(running);
    useSearchStore.getState().loadHistoricalJob(running);
    render(<SearchPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Dừng phiên' }));
    expect(mocks.stop).toHaveBeenCalledWith('session-a');
    expect(screen.getByRole('button', { name: 'Đang dừng…' })).toBeDisabled();
    expect(useSearchStore.getState().jobStatus).toBe('running');
    await act(async () => finish());
    expect(screen.getByText(/Các tác vụ của phiên đã dừng/)).toBeInTheDocument();
  });

  it('does not clear a different session when deletion completes', async () => {
    let finish!: () => void;
    mocks.remove.mockImplementation(() => new Promise<void>(resolve => { finish = resolve; }));
    useSearchStore.getState().addHistoryItem(running);
    useSearchStore.getState().loadHistoricalJob(running);
    render(<SearchPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Xóa phiên' }));
    expect(mocks.remove).toHaveBeenCalledWith('session-a');
    act(() => useSearchStore.getState().loadHistoricalJob({ ...running, job_id: 'job-b', session_id: 'session-b' }));
    await act(async () => finish());
    expect(useSearchStore.getState().sessionId).toBe('session-b');
  });

  it('keeps the session and displays deletion errors', async () => {
    mocks.remove.mockRejectedValue(new Error('Database unavailable'));
    useSearchStore.getState().addHistoryItem(running);
    useSearchStore.getState().loadHistoricalJob(running);
    render(<SearchPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Xóa phiên' }));
    await waitFor(() => expect(screen.getByText('Database unavailable')).toBeInTheDocument());
    expect(useSearchStore.getState().sessionId).toBe('session-a');
  });
});
