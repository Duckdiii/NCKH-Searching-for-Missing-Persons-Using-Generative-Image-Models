import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { Sparkles, Loader2, AlertCircle } from 'lucide-react';

export const AgeInputForm: React.FC = () => {
  const store = useSearchStore();
  const { resolveAge } = useSearchApi();

  const handleGenderChange = (gender: 'man' | 'woman') => {
    store.setGenderWord(gender);
    if (store.sessionId && store.croppedPreviewUrl) {
      resolveAge(store.sessionId, store.ageMode, store.manualAge, gender);
    }
  };

  const handleModeChange = (mode: 'manual' | 'mivolo') => {
    store.setAgeMode(mode);
    if (store.sessionId && store.croppedPreviewUrl) {
      resolveAge(store.sessionId, mode, store.manualAge, store.genderWord);
    }
  };

  const handleManualAgeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const age = parseInt(e.target.value) || 0;
    store.setManualAge(age);
    if (store.sessionId && store.croppedPreviewUrl) {
      resolveAge(store.sessionId, 'manual', age, store.genderWord);
    }
  };

  return (
    <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-6 shadow-xl space-y-5">
      <h3 className="text-lg font-semibold text-slate-100">Thông tin đối tượng tìm kiếm</h3>

      {/* Chọn giới tính */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Giới tính của người trong ảnh:
        </label>
        <div className="flex gap-4">
          <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-200 bg-slate-900/60 px-4 py-2.5 rounded-lg border border-slate-700 hover:border-indigo-500 transition-colors">
            <input
              type="radio"
              name="gender"
              value="man"
              checked={store.genderWord === 'man'}
              onChange={() => handleGenderChange('man')}
              className="text-indigo-600 focus:ring-indigo-500"
            />
            <span>Nam (Man / Boy)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-200 bg-slate-900/60 px-4 py-2.5 rounded-lg border border-slate-700 hover:border-indigo-500 transition-colors">
            <input
              type="radio"
              name="gender"
              value="woman"
              checked={store.genderWord === 'woman'}
              onChange={() => handleGenderChange('woman')}
              className="text-indigo-600 focus:ring-indigo-500"
            />
            <span>Nữ (Woman / Girl)</span>
          </label>
        </div>
      </div>

      {/* Lựa chọn nguồn tuổi */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Bạn có biết chính xác tuổi của người trong ảnh lúc chụp không?
        </label>
        <div className="flex flex-col gap-2.5">
          <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-200 bg-slate-900/60 p-3 rounded-lg border border-slate-700 hover:border-indigo-500">
            <input
              type="radio"
              name="ageMode"
              value="manual"
              checked={store.ageMode === 'manual'}
              onChange={() => handleModeChange('manual')}
              className="text-indigo-600 focus:ring-indigo-500"
            />
            <span>Có, tôi biết chính xác tuổi (khuyến nghị, độ chính xác cao nhất)</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-200 bg-slate-900/60 p-3 rounded-lg border border-slate-700 hover:border-indigo-500">
            <input
              type="radio"
              name="ageMode"
              value="mivolo"
              checked={store.ageMode === 'mivolo'}
              onChange={() => handleModeChange('mivolo')}
              className="text-indigo-600 focus:ring-indigo-500"
            />
            <span>Không, nhờ hệ thống ước tính tự động (MiVOLO AI)</span>
          </label>
        </div>
      </div>

      {/* Nhánh nhập tay */}
      {store.ageMode === 'manual' && (
        <div className="bg-slate-900/40 p-4 rounded-lg border border-slate-700/60">
          <label className="block text-sm font-medium text-slate-300 mb-1.5">
            Nhập số tuổi lúc chụp ảnh:
          </label>
          <div className="flex items-center gap-3">
            <input
              type="number"
              min={0}
              max={120}
              value={store.manualAge}
              onChange={handleManualAgeChange}
              className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-white w-28 focus:outline-none focus:border-indigo-500"
            />
            <span className="text-sm text-slate-400">tuổi</span>
          </div>
        </div>
      )}

      {/* Nhánh MiVOLO ước tính */}
      {store.ageMode === 'mivolo' && (
        <div className="bg-indigo-950/30 p-4 rounded-lg border border-indigo-500/30 space-y-2">
          {store.isEstimatingAge ? (
            <div className="flex items-center gap-2 text-sm text-indigo-300">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Đang ước tính tuổi bằng mô hình MiVOLO...</span>
            </div>
          ) : store.initialAge !== null ? (
            <div>
              <div className="flex items-center gap-2 text-emerald-400 font-medium text-sm">
                <Sparkles className="w-4 h-4" />
                <span>Hệ thống ước tính: khoảng {store.initialAge} tuổi</span>
              </div>
              {store.ageWarningText && (
                <div className="mt-2 text-xs text-amber-300/90 flex items-start gap-1.5 bg-amber-950/40 p-2.5 rounded border border-amber-500/30">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5 text-amber-400" />
                  <span>{store.ageWarningText}</span>
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={() =>
                store.sessionId &&
                resolveAge(store.sessionId, 'mivolo', undefined, store.genderWord)
              }
              className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded"
            >
              Ước tính tuổi ngay
            </button>
          )}
        </div>
      )}
    </div>
  );
};
