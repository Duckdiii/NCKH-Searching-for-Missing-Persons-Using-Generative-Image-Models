import React, { useEffect } from 'react';
import { useSearchStore } from './store/useSearchStore';
import { useSearchApi } from './api/useSearchApi';
import { SearchPage } from './pages/SearchPage';
import { FirstRunSetup } from './components/FirstRunSetup';
import { Loader2 } from 'lucide-react';

export const App: React.FC = () => {
  const { checkpointReady, isCheckingHealth } = useSearchStore();
  const { checkHealth } = useSearchApi();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.has('preview')) {
      useSearchStore.setState({
        checkpointReady: true,
        isCheckingHealth: false,
      });
      return;
    }
    checkHealth();
  }, []);

  if (isCheckingHealth) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-[#12161C] text-[#E8E6E0]">
        <Loader2 className="w-8 h-8 text-[#C97B4A] animate-spin mb-3" />
        <p className="text-xs font-medium text-[#8E98A5]">Đang kiểm tra môi trường và trọng số...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#12161C] text-[#E8E6E0] selection:bg-[#C97B4A] selection:text-white">
      {!checkpointReady ? <FirstRunSetup /> : <SearchPage />}
    </div>
  );
};

export default App;
