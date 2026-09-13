import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { Skeleton } from '../components/Skeleton';
import { ImageWithSkeleton } from '../components/ImageWithSkeleton';
import { Toast, ToastData } from '../components/Toast';
import { Sidebar } from '../components/Sidebar';
import { useSearchStore } from '../store/useSearchStore';

describe('Phase 3 Components - Skeleton, Toast, Empty State', () => {
  it('Skeleton component renders with skeleton-box and custom classes', () => {
    const { container } = render(<Skeleton className="w-24 h-24 rounded-xl test-custom" />);
    const el = container.firstChild as HTMLElement;
    expect(el).toHaveClass('skeleton-box');
    expect(el).toHaveClass('w-24');
    expect(el).toHaveClass('h-24');
    expect(el).toHaveClass('rounded-xl');
    expect(el).toHaveClass('test-custom');
  });

  it('ImageWithSkeleton renders skeleton when loading or no src', () => {
    const { container } = render(
      <ImageWithSkeleton src="/test_image.png" alt="Test Face" />
    );
    // Skeleton should be present before load
    const skeleton = container.querySelector('.skeleton-box');
    expect(skeleton).toBeInTheDocument();
  });

  it('Toast renders title, identityName, score, and triggers onViewResult and onClose', () => {
    const handleClose = vi.fn();
    const handleView = vi.fn();

    const sampleToast: ToastData = {
      id: 'toast-123',
      identityName: 'MP_2023_0982',
      scorePct: '84.7%',
      onViewResult: handleView,
    };

    render(<Toast toast={sampleToast} onClose={handleClose} />);

    expect(screen.getByText(/Job hoàn tất/i)).toBeInTheDocument();
    expect(screen.getByText(/MP_2023_0982/i)).toBeInTheDocument();
    expect(screen.getByText(/84.7%/i)).toBeInTheDocument();

    const viewBtn = screen.getByRole('button', { name: /Xem kết quả/i });
    fireEvent.click(viewBtn);
    expect(handleView).toHaveBeenCalledTimes(1);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('Sidebar renders Empty State with exact required text when sessionHistory is empty', () => {
    const s = useSearchStore.getState();
    s.sessionHistory = [];

    render(<Sidebar />);

    expect(screen.getByText(/Chưa có phiên tìm kiếm nào\. Bấm 'Tìm kiếm mới' để bắt đầu\./i)).toBeInTheDocument();
  });
});
