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
    <div className="bg-[#1B2129] border border-[#262E38] rounded-xl p-5 shadow-lg space-y-4">
      <h3 className="text-base font-semibold text-[#E8E6E0]">Thông tin đối tượng tìm kiếm</h3>

      {/* Chọn giới tính */}
      <div>
        <label className="block text-xs font-semibold text-[#8E98A5] mb-2 uppercase tracking-wide">
          Giới tính của người trong ảnh:
        </label>
        <div className="flex gap-3">
          <label
            className={`flex-1 flex items-center justify-center gap-2 cursor-pointer text-xs font-medium py-2.5 px-3 rounded-lg border transition-all ${
              store.genderWord === 'man'
                ? 'bg-[#12161C] border-[#C97B4A] text-[#E8E6E0]'
                : 'bg-[#12161C] border-[#262E38] text-[#8E98A5] hover:border-[#2E3844]'
            }`}
          >
            <input
              type="radio"
              name="gender"
              value="man"
              checked={store.genderWord === 'man'}
              onChange={() => handleGenderChange('man')}
              className="accent-[#C97B4A]"
            />
            <span>Nam (Man / Boy)</span>
          </label>
          <label
            className={`flex-1 flex items-center justify-center gap-2 cursor-pointer text-xs font-medium py-2.5 px-3 rounded-lg border transition-all ${
              store.genderWord === 'woman'
                ? 'bg-[#12161C] border-[#C97B4A] text-[#E8E6E0]'
                : 'bg-[#12161C] border-[#262E38] text-[#8E98A5] hover:border-[#2E3844]'
            }`}
          >
            <input
              type="radio"
              name="gender"
              value="woman"
              checked={store.genderWord === 'woman'}
              onChange={() => handleGenderChange('woman')}
              className="accent-[#C97B4A]"
            />
            <span>Nữ (Woman / Girl)</span>
          </label>
        </div>
      </div>

      {/* Lựa chọn nguồn tuổi */}
      <div>
        <label className="block text-xs font-semibold text-[#8E98A5] mb-2 uppercase tracking-wide">
          Phương thức xác định tuổi lúc chụp ảnh:
        </label>
        <div className="flex flex-col gap-2">
          <label
            className={`flex items-center gap-2 cursor-pointer text-xs p-3 rounded-lg border transition-all ${
              store.ageMode === 'manual'
                ? 'bg-[#12161C] border-[#C97B4A] text-[#E8E6E0]'
                : 'bg-[#12161C] border-[#262E38] text-[#8E98A5] hover:border-[#2E3844]'
            }`}
          >
            <input
              type="radio"
              name="ageMode"
              value="manual"
              checked={store.ageMode === 'manual'}
              onChange={() => handleModeChange('manual')}
              className="accent-[#C97B4A]"
            />
            <span>Có, tôi biết chính xác tuổi (khuyến nghị, độ chính xác cao nhất)</span>
          </label>

          <label
            className={`flex items-center gap-2 cursor-pointer text-xs p-3 rounded-lg border transition-all ${
              store.ageMode === 'mivolo'
                ? 'bg-[#12161C] border-[#C97B4A] text-[#E8E6E0]'
                : 'bg-[#12161C] border-[#262E38] text-[#8E98A5] hover:border-[#2E3844]'
            }`}
          >
            <input
              type="radio"
              name="ageMode"
              value="mivolo"
              checked={store.ageMode === 'mivolo'}
              onChange={() => handleModeChange('mivolo')}
              className="accent-[#C97B4A]"
            />
            <span>Không, nhờ hệ thống ước tính tự động (MiVOLO AI)</span>
          </label>
        </div>
      </div>

      {/* Nhánh nhập tay */}
      {store.ageMode === 'manual' && (
        <div className="bg-[#12161C] p-3.5 rounded-lg border border-[#262E38]">
          <label className="block text-xs font-medium text-[#8E98A5] mb-1.5">
            Nhập số tuổi lúc chụp ảnh:
          </label>
          <div className="flex items-center gap-2.5">
            <input
              type="number"
              min={0}
              max={120}
              value={store.manualAge}
              onChange={handleManualAgeChange}
              className="bg-[#1B2129] border border-[#2E3844] rounded-lg px-3 py-1.5 text-[#E8E6E0] w-24 text-sm focus:outline-none focus:border-[#C97B4A]"
            />
            <span className="text-xs text-[#8E98A5]">tuổi</span>
          </div>
        </div>
      )}

      {/* Nhánh MiVOLO ước tính */}
      {store.ageMode === 'mivolo' && (
        <div className="bg-[#12161C] p-3.5 rounded-lg border border-[#4A8FA0]/40 space-y-2">
          {store.isEstimatingAge ? (
            <div className="flex items-center gap-2 text-xs text-[#4A8FA0]">
              <Loader2 className="w-4 h-4 animate-spin text-[#4A8FA0]" />
              <span>Đang ước tính tuổi bằng mô hình MiVOLO...</span>
            </div>
          ) : store.initialAge !== null ? (
            <div>
              <div className="flex items-center gap-2 text-[#4A8FA0] font-medium text-xs">
                <Sparkles className="w-4 h-4" />
                <span>Hệ thống ước tính: khoảng {store.initialAge} tuổi</span>
              </div>
              {store.ageWarningText && (
                <div className="mt-2 text-xs text-[#C9A24A] flex items-start gap-1.5 bg-[#1B2129] p-2.5 rounded border border-[#C9A24A]/40">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5 text-[#C9A24A]" />
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
              className="text-xs bg-[#C97B4A] hover:brightness-110 text-white px-3.5 py-1.5 rounded-lg font-medium"
            >
              Ước tính tuổi ngay
            </button>
          )}
        </div>
      )}
    </div>
  );
};
