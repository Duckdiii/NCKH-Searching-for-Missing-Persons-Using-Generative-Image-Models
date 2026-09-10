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
    checkHealth();
  }, [checkHealth]);

  if (isCheckingHealth) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-slate-950 text-slate-200">
        <Loader2 className="w-8 h-8 text-indigo-500 animate-spin mb-3" />
        <p className="text-sm font-medium">Đang kiểm tra môi trường và trọng số...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-indigo-500 selection:text-white">
      {!checkpointReady ? <FirstRunSetup /> : <SearchPage />}
    </div>
  );
};

export default App;
