import { describe, it, expect, beforeEach } from 'vitest';
import { useSearchStore } from '../store/useSearchStore';

describe('Layer C: Session Isolation & Dynamic Job IDs', () => {
  beforeEach(() => {
    const s = useSearchStore.getState();
    s.startNewSearch();
    s.sessionHistory = [];
  });

  it('generates distinct job IDs and isolates historical session data', () => {
    const s = useSearchStore.getState();

    // 1. Session 1
    s.setUploadResult('sess-001', [], '/face1_original.png');
    s.setSelectedFace(0, [], '/face1_crop.png');
    s.setResolvedAge(15);
    s.setGenderWord('man');

    // Run Job 1 with unique ID
    s.startJob('job-uuid-001');
    s.updateJobProgress('done', 'complete', {
      job_id: 'job-uuid-001',
      status: 'done',
      edited_images: { 20: '/edited1.png' },
      final_scores: { 'candidate_A': 0.82 },
      accepted: true,
      top_identity: 'candidate_A',
      top_score: 0.82,
    });

    expect(useSearchStore.getState().sessionHistory.length).toBe(1);
    expect(useSearchStore.getState().sessionHistory[0].job_id).toBe('job-uuid-001');
    expect(useSearchStore.getState().sessionHistory[0].cropped_preview_url).toBe('/face1_crop.png');
    expect(useSearchStore.getState().sessionHistory[0].initial_age).toBe(15);

    // 2. Session 2 (New search)
    useSearchStore.getState().startNewSearch();
    useSearchStore.getState().setUploadResult('sess-002', [], '/face2_original.png');
    useSearchStore.getState().setSelectedFace(0, [], '/face2_crop.png');
    useSearchStore.getState().setResolvedAge(35);
    useSearchStore.getState().setGenderWord('woman');

    // Run Job 2 with completely different unique ID
    useSearchStore.getState().startJob('job-uuid-002');
    useSearchStore.getState().updateJobProgress('done', 'complete', {
      job_id: 'job-uuid-002',
      status: 'done',
      edited_images: { 40: '/edited2.png' },
      final_scores: { 'candidate_B': 0.65 },
      accepted: true,
      top_identity: 'candidate_B',
      top_score: 0.65,
    });

    const history = useSearchStore.getState().sessionHistory;
    expect(history.length).toBe(2);
    expect(history[0].job_id).toBe('job-uuid-002');
    expect(history[1].job_id).toBe('job-uuid-001');

    // Currently active is Job 2
    expect(useSearchStore.getState().jobId).toBe('job-uuid-002');
    expect(useSearchStore.getState().croppedPreviewUrl).toBe('/face2_crop.png');
    expect(useSearchStore.getState().initialAge).toBe(35);

    // 3. User clicks on Session 1 in sidebar history
    useSearchStore.getState().loadHistoricalJob(history[1]);

    // Verify it restored Session 1 data, NOT Session 2
    const s1 = useSearchStore.getState();
    expect(s1.jobId).toBe('job-uuid-001');
    expect(s1.croppedPreviewUrl).toBe('/face1_crop.png');
    expect(s1.initialAge).toBe(15);
    expect(s1.genderWord).toBe('man');
    expect(s1.jobResult?.top_identity).toBe('candidate_A');
    expect(s1.jobResult?.top_score).toBe(0.82);

    // 4. User clicks back to Session 2 in sidebar history
    useSearchStore.getState().loadHistoricalJob(history[0]);

    // Verify it restored Session 2 data
    const s2 = useSearchStore.getState();
    expect(s2.jobId).toBe('job-uuid-002');
    expect(s2.croppedPreviewUrl).toBe('/face2_crop.png');
    expect(s2.initialAge).toBe(35);
    expect(s2.genderWord).toBe('woman');
    expect(s2.jobResult?.top_identity).toBe('candidate_B');
    expect(s2.jobResult?.top_score).toBe(0.65);
  });
});
