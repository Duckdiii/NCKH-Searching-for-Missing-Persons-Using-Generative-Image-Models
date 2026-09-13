import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { CountUpNumber } from '../components/CountUpNumber';
import { JobProgressBlock } from '../components/JobProgressBlock';

describe('Phase 4 Animations - CountUpNumber & CSS Transitions', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('CountUpNumber renders formatted value with decimals and suffix', () => {
    render(
      <CountUpNumber
        targetValue={87.4}
        decimals={1}
        suffix="%"
        duration={100}
      />
    );

    // Initial render displays formatted number (0.0% initially)
    expect(screen.getByText(/%/)).toBeInTheDocument();
  });

  it('CountUpNumber reaches target value after animation duration', async () => {
    render(
      <CountUpNumber
        targetValue={92.5}
        decimals={1}
        suffix="%"
        duration={200}
      />
    );

    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(screen.getByText('92.5%')).toBeInTheDocument();
  });

  it('JobProgressBlock contains progress-bar-smooth on progress bar and hover-lift on cancel button', () => {
    const { container } = render(
      <JobProgressBlock
        jobStage="editing"
        jobStatus="running"
        jobError={null}
        onCancel={() => {}}
      />
    );

    // Progress bar smooth class
    const progressBar = container.querySelector('.progress-bar-smooth');
    expect(progressBar).toBeInTheDocument();

    // Cancel button hover-lift
    const cancelBtn = screen.getByRole('button', { name: /Hủy tiến trình/i });
    expect(cancelBtn).toHaveClass('hover-lift');
  });
});
