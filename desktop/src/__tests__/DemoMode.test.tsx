import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { WizardStepper } from '../components/WizardStepper';
import { App } from '../App';
import { SearchPage } from '../pages/SearchPage';
import { CommandPalette } from '../components/CommandPalette';
import { useSearchStore } from '../store/useSearchStore';

// Mock API hook so App / SearchPage don't make real network requests
vi.mock('../api/useSearchApi', () => ({
  useSearchApi: () => ({
    checkHealth: vi.fn(),
    fetchJobsHistory: vi.fn(),
    uploadImage: vi.fn(),
    estimateAge: vi.fn(),
    runPipeline: vi.fn(),
  }),
}));

describe('Demo Mode (Chế độ Demo) Tests', () => {
  beforeEach(() => {
    // Reset Zustand store state to pristine defaults
    useSearchStore.setState({
      checkpointReady: true,
      isCheckingHealth: false,
      isDemoMode: false,
      currentWizardStep: 'restore',
      sessionId: null,
      uploadedImageUrl: null,
      faces: [],
      selectedFaceIdx: null,
      croppedPreviewUrl: null,
      jobId: null,
      jobStatus: 'idle',
      jobResult: null,
      sessionHistory: [],
      isHistoricalView: false,
      toast: null,
    });
  });

  describe('1. WizardStepper Navigation Unlocking', () => {
    it('khi isDemoMode = false: khóa cứng bước 2 và 3 khi chưa completed', () => {
      const handleStepClick = vi.fn();
      render(
        <WizardStepper
          currentStep="restore"
          completedSteps={[]}
          onStepClick={handleStepClick}
          isDemoMode={false}
        />
      );

      const generateBtn = screen.getByTestId('wizard-step-generate');
      const resultsBtn = screen.getByTestId('wizard-step-results');

      expect(generateBtn).toBeDisabled();
      expect(resultsBtn).toBeDisabled();

      fireEvent.click(generateBtn);
      fireEvent.click(resultsBtn);

      expect(handleStepClick).not.toHaveBeenCalled();
    });

    it('khi isDemoMode = true: mở khóa cho phép click bước 2 và 3 ngay cả khi completedSteps rỗng', () => {
      const handleStepClick = vi.fn();
      render(
        <WizardStepper
          currentStep="restore"
          completedSteps={[]}
          onStepClick={handleStepClick}
          isDemoMode={true}
        />
      );

      const generateBtn = screen.getByTestId('wizard-step-generate');
      const resultsBtn = screen.getByTestId('wizard-step-results');

      expect(generateBtn).not.toBeDisabled();
      expect(resultsBtn).not.toBeDisabled();

      fireEvent.click(generateBtn);
      expect(handleStepClick).toHaveBeenCalledWith('generate');

      fireEvent.click(resultsBtn);
      expect(handleStepClick).toHaveBeenCalledWith('results');
    });
  });

  describe('2. Visual Warning Banner', () => {
    it('banner cảnh báo KHÔNG xuất hiện khi isDemoMode = false', () => {
      useSearchStore.setState({ isDemoMode: false });
      render(<App />);

      const banner = screen.queryByTestId('demo-mode-banner');
      expect(banner).not.toBeInTheDocument();
    });

    it('banner cảnh báo màu vàng xuất hiện khi isDemoMode = true', () => {
      useSearchStore.setState({ isDemoMode: true });
      render(<App />);

      const banner = screen.getByTestId('demo-mode-banner');
      expect(banner).toBeInTheDocument();
      expect(banner).toHaveTextContent(
        '⚠️ ĐANG Ở CHẾ ĐỘ DEMO — điều hướng không bị khóa, không phản ánh trạng thái xử lý thật'
      );
    });
  });

  describe('3. Step 3 (Results) in Demo Mode — Strict No Fake Data Guarantee', () => {
    it('khi sessionHistory rỗng: hiển thị thông báo yêu cầu chạy ít nhất 1 job thật, TUYỆT ĐỐI không có số liệu giả', () => {
      useSearchStore.setState({
        isDemoMode: true,
        sessionHistory: [],
        jobResult: null,
      });

      render(<SearchPage />);

      // Click vào bước 3 trong chế độ Demo
      const resultsStepBtn = screen.getByTestId('wizard-step-results');
      fireEvent.click(resultsStepBtn);

      // Phải hiển thị thông báo rỗng chuẩn
      const warningCard = screen.getByTestId('demo-empty-results-warning');
      expect(warningCard).toBeInTheDocument();
      expect(warningCard).toHaveTextContent(
        'Chưa có kết quả nào để xem — cần chạy ít nhất 1 job thật trước khi demo bước này'
      );

      // Không có bất kỳ card hay ID Score giả mạo nào (ví dụ 0.85 hay session_mock_demo)
      expect(screen.queryByText(/session_mock_demo/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/Độ tương đồng cao nhất/i)).not.toBeInTheDocument();
    });

    it('khi sessionHistory có ít nhất 1 job completed: tự động nạp kết quả của job lịch sử thật đó', () => {
      const mockHistoricalJob = {
        job_id: 'job_real_history_777',
        session_id: 'sess_real_777',
        status: 'done' as const,
        stage: 'complete',
        top_identity: 'REAL_SUBJECT_001.jpg',
        top_score: 0.789,
        accepted: true,
        timestamp: Date.now() - 10000,
        result: {
          job_id: 'job_real_history_777',
          status: 'done' as const,
          edited_images: { 20: '/img20.png', 40: '/img40.png' },
          final_scores: { 'REAL_SUBJECT_001.jpg': 0.789 },
          accepted: true,
          top_identity: 'REAL_SUBJECT_001.jpg',
          top_score: 0.789,
          best_age: 40,
          matched_gallery_image: '/matched_gallery.png',
        },
      };

      useSearchStore.setState({
        isDemoMode: true,
        sessionHistory: [mockHistoricalJob],
        jobResult: null,
      });

      render(<SearchPage />);

      // Click vào bước 3 trong chế độ Demo
      const resultsStepBtn = screen.getByTestId('wizard-step-results');
      fireEvent.click(resultsStepBtn);

      // Kết quả lịch sử thật được nạp tự động vào store
      expect(useSearchStore.getState().jobId).toBe('job_real_history_777');
      expect(useSearchStore.getState().jobResult).toEqual(mockHistoricalJob.result);

      // Thông báo rỗng không còn hiển thị, thay vào đó là kết quả thật
      expect(screen.queryByTestId('demo-empty-results-warning')).not.toBeInTheDocument();
      expect(screen.getByText('REAL_SUBJECT_001.jpg')).toBeInTheDocument();
    });
  });

  describe('4. Command Palette Demo Mode Toggle', () => {
    it('cho phép bật/tắt Chế độ Demo qua Command Palette', () => {
      render(<CommandPalette isOpen={true} onClose={vi.fn()} />);

      const toggleAction = screen.getByText('Bật Chế độ Demo');
      expect(toggleAction).toBeInTheDocument();

      fireEvent.click(toggleAction);
      expect(useSearchStore.getState().isDemoMode).toBe(true);
    });

    it('cho phép chọn Đi tới bước Kết quả khi đang ở Chế độ Demo', () => {
      useSearchStore.setState({ isDemoMode: true, jobResult: null });
      const handleGoToResults = vi.fn();

      render(
        <CommandPalette
          isOpen={true}
          onClose={vi.fn()}
          onSelectGoToResults={handleGoToResults}
        />
      );

      const goToResultsAction = screen.getByText('Đi tới bước Kết quả đối soát');
      expect(goToResultsAction).toBeInTheDocument();

      fireEvent.click(goToResultsAction);
      expect(handleGoToResults).toHaveBeenCalled();
    });
  });
});
