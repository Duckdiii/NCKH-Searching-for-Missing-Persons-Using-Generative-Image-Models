import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { WizardStepper } from '../components/WizardStepper';

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
});
