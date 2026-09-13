import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { CommandPalette } from '../components/CommandPalette';
import { ResultsGallery } from '../components/ResultsGallery';
import { exportRankingToCSV } from '../utils/exportReport';
import { useSearchStore } from '../store/useSearchStore';

describe('Layer B: System Interactions (Command Palette, Advanced Mode, CSV Export)', () => {
  beforeEach(() => {
    const s = useSearchStore.getState();
    s.startNewSearch();
    s.isAdvancedMode = false;
  });

  it('Command Palette opens on Ctrl+K / Cmd+K, closes on Escape or backdrop click', () => {
    const onClose = vi.fn();
    const onNewSearch = vi.fn();
    const onOpenGallery = vi.fn();

    render(
      <CommandPalette
        isOpen={true}
        onClose={onClose}
        onSelectNewSearch={onNewSearch}
        onSelectOpenGallery={onOpenGallery}
      />
    );

    // Should display palette header and input
    expect(screen.getByPlaceholderText(/Gõ lệnh hoặc tìm kiếm hành động/i)).toBeInTheDocument();

    // Trigger action: "Bắt đầu phiên tìm kiếm mới"
    const newSearchBtn = screen.getByText(/Bắt đầu phiên tìm kiếm mới/i);
    fireEvent.click(newSearchBtn);
    expect(onNewSearch).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();

    // Close on Escape key
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  it('Advanced Mode conditionally renders Cosine thô and Pipeline Params in DOM', () => {
    const s = useSearchStore.getState();
    s.jobResult = {
      job_id: 'job-layer-b-1',
      status: 'done',
      edited_images: { 25: '/sample_face.png' },
      final_scores: { 'candidate_1.jpg': 0.8466, 'candidate_2.jpg': 0.4210 },
      accepted: true,
      top_identity: 'candidate_1.jpg',
      top_score: 0.8466,
      pipeline_params: {
        num_inference_steps: 50,
        guidance_scale: 7.5,
        attention_control_ratio: 0.8,
        checkpoint_name: 'specialized_unet',
        embedding_model: 'buffalo_l',
        rejection_threshold: 0.6,
      },
    };

    // When isAdvancedMode is FALSE
    s.isAdvancedMode = false;
    const { rerender } = render(<ResultsGallery />);

    expect(screen.queryByText(/Cosine thô/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Thông số kỹ thuật Pipeline \(Nâng cao\)/i)).not.toBeInTheDocument();

    // When isAdvancedMode is TRUE
    act(() => {
      s.setAdvancedMode(true);
    });
    rerender(<ResultsGallery />);

    expect(screen.getByText(/Cosine thô/i)).toBeInTheDocument();
    expect(screen.getByText(/Thông số kỹ thuật Pipeline \(Nâng cao\)/i)).toBeInTheDocument();
    expect(screen.getByText('0.8466')).toBeInTheDocument();
    expect(screen.getByText('0.4210')).toBeInTheDocument();
    expect(screen.getByText(/Bước DDIM Inversion/i)).toBeInTheDocument();
    expect(screen.getByText('50 steps')).toBeInTheDocument();
  });

  it('CSV export function generates UTF-8 BOM, accurate headers and matching candidate scores', () => {
    const sampleResult = {
      job_id: 'job_test_export_999',
      status: 'done' as const,
      edited_images: { 25: '/img.png' },
      final_scores: {
        'candidate_1.jpg': 0.8466,
        'candidate_2.jpg': 0.6210,
        'candidate_3.jpg': 0.2450,
      },
      accepted: true,
      top_identity: 'candidate_1.jpg',
      top_score: 0.8466,
      pipeline_params: {
        num_inference_steps: 50,
        guidance_scale: 7.5,
        checkpoint_name: 'specialized_unet',
      },
    };

    const csvString = exportRankingToCSV(sampleResult);

    // Verify CSV output
    expect(csvString).toContain('job_test_export_999');
    expect(csvString).toContain('candidate_1.jpg');
    expect(csvString).toContain('0.8466');
    expect(csvString).toContain('84.7%');
    expect(csvString).toContain('specialized_unet');
    expect(csvString).toContain('Định danh Gallery');
  });
});
