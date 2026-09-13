import React, { useEffect, useState } from 'react';
import { useSearchStore } from './store/useSearchStore';
import { useSearchApi } from './api/useSearchApi';
import { SearchPage } from './pages/SearchPage';
import { FirstRunSetup } from './components/FirstRunSetup';
import { Sidebar } from './components/Sidebar';
import { StatusBar } from './components/StatusBar';
import { CommandPalette } from './components/CommandPalette';
import { Toast } from './components/Toast';
import { Loader2 } from 'lucide-react';

export const App: React.FC = () => {
  const { checkpointReady, isCheckingHealth, currentWizardStep, toast, setToast } = useSearchStore();
  const { checkHealth, fetchJobsHistory } = useSearchApi();
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

  useEffect(() => {
    // Chỉ kích hoạt preview mode trong môi trường phát triển (DEV), tuyệt đối không chạy ở production
    if (import.meta.env.DEV) {
      const params = new URLSearchParams(window.location.search);
      if (params.has('preview')) {
        const s = useSearchStore.getState();
        s.setCheckpoints(true, [], 'specialized_unet (stable-diffusion-v1-5)', '1.0.0');
        s.setIsCheckingHealth(false);

      if (params.get('history') === 'empty') {
        useSearchStore.setState({ sessionHistory: [] });
        return;
      }

      // Seed mock history in preview mode if empty
      if (s.sessionHistory.length === 0) {
        s.addHistoryItem({
          job_id: 'job_cctv_binhthanh_041',
          session_id: 'sess_cctv_041',
          status: 'done',
          stage: 'complete',
          top_identity: 'IMG_CCTV_BinhThanh_041.jpg',
          top_score: 0.621,
          accepted: true,
          timestamp: Date.now() - 3600000 * 3,
        });
        s.addHistoryItem({
          job_id: 'job_mp_2023_0982',
          session_id: 'sess_mp_0982',
          status: 'done',
          stage: 'complete',
          top_identity: 'MP_2023_0982.jpg',
          top_score: 0.8466,
          accepted: true,
          timestamp: Date.now() - 3600000 * 24,
        });
      }

      // Trigger sample toast in preview mode if requested
      if (params.get('preview') === 'toast' || params.has('toast')) {
        setTimeout(() => {
          s.setToast({
            id: 'toast_preview_1',
            identityName: 'MP_2023_0982',
            scorePct: '84.7%',
            onViewResult: () => {
              s.setCurrentWizardStep('results');
            },
          });
        }, 100);
      }

        return;
      }
    }
    checkHealth();
    fetchJobsHistory();
  }, []);

  // Global shortcut Ctrl+K / Cmd+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const stepLabels: Record<string, string> = {
    restore: 'Bước 1: Khôi phục ảnh',
    generate: 'Bước 2: Sinh ảnh & Đối soát',
    results: 'Bước 3: Kết quả đối soát',
  };

  if (isCheckingHealth) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-[#F5F6F8] text-[#111827]">
        <Loader2 className="w-8 h-8 text-[#E8804A] animate-spin mb-3" />
        <p className="text-xs font-medium text-[#6B7280]">Đang kiểm tra môi trường và trọng số...</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#F5F6F8] text-[#111827] overflow-hidden select-none">
      {/* Cột 1: Sidebar thanh điều hướng & lịch sử phiên */}
      <Sidebar />

      {/* Cột 2: Nội dung chính bao gồm StatusBar ở trên cùng */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        <StatusBar
          currentStepLabel={stepLabels[currentWizardStep] || 'Bước 1: Khôi phục ảnh'}
          onOpenCommandPalette={() => setIsCommandPaletteOpen(true)}
        />

        <main className="flex-1 overflow-y-auto">
          {!checkpointReady ? <FirstRunSetup /> : <SearchPage />}
        </main>
      </div>

      {/* Command Palette (Cmd+K / Ctrl+K) */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        onSelectGoToResults={() => {
          useSearchStore.getState().setCurrentWizardStep('results');
        }}
      />

      {/* Toast Notification (Phase 3) */}
      <Toast toast={toast} onClose={() => setToast(null)} />
    </div>
  );
};

export default App;

