import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { WizardStepper } from '../components/WizardStepper';
import { useSearchStore } from '../store/useSearchStore';

describe('WizardStepper - Strict Step-Locking & Navigation', () => {
  it('Yêu cầu 1: onStepClick("results") KHÔNG có tác dụng khi completedSteps chưa chứa "results"', () => {
    const handleStepClick = vi.fn();
    render(
      <WizardStepper
        currentStep="restore"
        completedSteps={[]}
        onStepClick={handleStepClick}
      />
    );

    const resultsBtn = screen.getByTestId('wizard-step-results');

    // Nút phải bị disabled và có class pointer-events-none / cursor-not-allowed
    expect(resultsBtn).toBeDisabled();
    expect(resultsBtn.className).toContain('pointer-events-none');

    // Thử click
    fireEvent.click(resultsBtn);
    expect(handleStepClick).not.toHaveBeenCalled();
  });

  it('Yêu cầu 3A: Bước chưa tới bị khóa hoàn toàn, không thể click', () => {
    const handleStepClick = vi.fn();
    render(
      <WizardStepper
        currentStep="restore"
        completedSteps={[]}
        onStepClick={handleStepClick}
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

  it('Yêu cầu 3B: Bước đã hoàn thành trong completedSteps thì click lại được', () => {
    const handleStepClick = vi.fn();
    render(
      <WizardStepper
        currentStep="generate"
        completedSteps={['restore']}
        onStepClick={handleStepClick}
      />
    );

    const restoreBtn = screen.getByTestId('wizard-step-restore');

    // Bước 1 đã xong nên không bị disabled và click được
    expect(restoreBtn).not.toBeDisabled();
    expect(restoreBtn.className).not.toContain('pointer-events-none');

    fireEvent.click(restoreBtn);
    expect(handleStepClick).toHaveBeenCalledWith('restore');
    expect(handleStepClick).toHaveBeenCalledTimes(1);
  });

  it('Yêu cầu 4.2: Điều hướng xem lại bước đã hoàn thành không làm mất dữ liệu (State Persistence)', () => {
    const s = useSearchStore.getState();
    // Giả lập dữ liệu đầy đủ từ bước 1 và bước 2
    s.setUploadResult('sess-test-persist-123', [{ index: 0, bbox: [10, 10, 100, 100], det_score: 0.98 }], '/img_preview.png');
    s.setSelectedFace(0, ['Độ phân giải thấp'], '/crop_face.png');
    s.setResolvedAge(24);
    s.setGalleryDir('D:/Data/test_gallery');
    s.setGenderWord('man');

    // Chuyển sang bước results
    useSearchStore.getState().setCurrentWizardStep('results');
    expect(useSearchStore.getState().currentWizardStep).toBe('results');

    // Quay lại bước 1 (restore)
    useSearchStore.getState().setCurrentWizardStep('restore');
    expect(useSearchStore.getState().sessionId).toBe('sess-test-persist-123');
    expect(useSearchStore.getState().selectedFaceIdx).toBe(0);
    expect(useSearchStore.getState().croppedPreviewUrl).toBe('/crop_face.png');
    expect(useSearchStore.getState().uploadedImageUrl).toBe('/img_preview.png');

    // Quay lại bước 2 (generate)
    useSearchStore.getState().setCurrentWizardStep('generate');
    expect(useSearchStore.getState().initialAge).toBe(24);
    expect(useSearchStore.getState().galleryDir).toBe('D:/Data/test_gallery');
    expect(useSearchStore.getState().genderWord).toBe('man');
  });
});
