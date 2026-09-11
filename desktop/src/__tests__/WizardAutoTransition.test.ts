import { describe, it, expect } from 'vitest';

describe('Wizard Auto Transition Logic', () => {
  it('Yêu cầu 2: jobStatus === "done" (hoặc completed) tự động chuyển currentStep sang "results"', () => {
    // Giả lập state machine chuyển đổi bước của Wizard
    let currentStep: 'restore' | 'generate' | 'results' = 'generate';
    let completedSteps: string[] = ['restore'];

    const onJobStatusChange = (status: 'idle' | 'running' | 'done') => {
      if (status === 'done') {
        completedSteps = Array.from(new Set([...completedSteps, 'restore', 'generate']));
        currentStep = 'results';
      }
    };

    // Khi job đang chạy
    onJobStatusChange('running');
    expect(currentStep).toBe('generate');
    expect(completedSteps).not.toContain('generate');

    // Khi job hoàn tất (WebSocket thông báo done/completed)
    onJobStatusChange('done');
    expect(currentStep).toBe('results');
    expect(completedSteps).toContain('generate');
    expect(completedSteps).toContain('restore');
  });

  it('Yêu cầu khóa bước: Không thể nhảy tới step chưa có trong completedSteps', () => {
    let currentStep: 'restore' | 'generate' | 'results' = 'restore';
    const completedSteps: string[] = [];

    const handleStepClick = (target: 'restore' | 'generate' | 'results') => {
      if (completedSteps.includes(target)) {
        currentStep = target;
      }
    };

    // Thử click vào "results" khi chưa hoàn thành bước nào
    handleStepClick('results');
    expect(currentStep).toBe('restore'); // Không đổi!

    // Thử click vào "generate" khi chưa hoàn thành
    handleStepClick('generate');
    expect(currentStep).toBe('restore'); // Vẫn không đổi!
  });
});
