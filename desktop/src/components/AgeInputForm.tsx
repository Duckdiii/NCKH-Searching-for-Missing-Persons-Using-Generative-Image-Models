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
    <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-xs space-y-4">
      <h3 className="text-base font-semibold text-[#111827]">Thông tin đối tượng tìm kiếm</h3>

      {/* Chọn giới tính */}
      <div>
        <label className="block text-xs font-semibold text-[#6B7280] mb-2 uppercase tracking-wide">
          Giới tính của người trong ảnh:
        </label>
        <div className="flex gap-3">
          <label
            className={`flex-1 flex items-center justify-center gap-2 cursor-pointer text-xs font-medium py-2.5 px-3 rounded-lg border transition-all hover-lift ${
              store.genderWord === 'man'
                ? 'bg-white border-[#E8804A] text-[#111827] shadow-xs ring-1 ring-[#E8804A]/30'
                : 'bg-[#F9FAFB] border-[#E5E7EB] text-[#6B7280] hover:border-[#D1D5DB]'
            }`}
          >
            <input
              type="radio"
              name="gender"
              value="man"
              checked={store.genderWord === 'man'}
              onChange={() => handleGenderChange('man')}
              className="accent-[#E8804A]"
            />
            <span>Nam (Man / Boy)</span>
          </label>
          <label
            className={`flex-1 flex items-center justify-center gap-2 cursor-pointer text-xs font-medium py-2.5 px-3 rounded-lg border transition-all hover-lift ${
              store.genderWord === 'woman'
                ? 'bg-white border-[#E8804A] text-[#111827] shadow-xs ring-1 ring-[#E8804A]/30'
                : 'bg-[#F9FAFB] border-[#E5E7EB] text-[#6B7280] hover:border-[#D1D5DB]'
            }`}
          >
            <input
              type="radio"
              name="gender"
              value="woman"
              checked={store.genderWord === 'woman'}
              onChange={() => handleGenderChange('woman')}
              className="accent-[#E8804A]"
            />
            <span>Nữ (Woman / Girl)</span>
          </label>
        </div>
      </div>

      {/* Lựa chọn nguồn tuổi */}
      <div>
        <label className="block text-xs font-semibold text-[#6B7280] mb-2 uppercase tracking-wide">
          Phương thức xác định tuổi lúc chụp ảnh:
        </label>
        <div className="flex flex-col gap-2">
          <label
            className={`flex items-center gap-2 cursor-pointer text-xs p-3 rounded-lg border transition-all hover-lift ${
              store.ageMode === 'manual'
                ? 'bg-white border-[#E8804A] text-[#111827] shadow-xs ring-1 ring-[#E8804A]/30'
                : 'bg-[#F9FAFB] border-[#E5E7EB] text-[#6B7280] hover:border-[#D1D5DB]'
            }`}
          >
            <input
              type="radio"
              name="ageMode"
              value="manual"
              checked={store.ageMode === 'manual'}
              onChange={() => handleModeChange('manual')}
              className="accent-[#E8804A]"
            />
            <span>Có, tôi biết chính xác tuổi (khuyến nghị, độ chính xác cao nhất)</span>
          </label>

          <label
            className={`flex items-center gap-2 cursor-pointer text-xs p-3 rounded-lg border transition-all hover-lift ${
              store.ageMode === 'mivolo'
                ? 'bg-white border-[#E8804A] text-[#111827] shadow-xs ring-1 ring-[#E8804A]/30'
                : 'bg-[#F9FAFB] border-[#E5E7EB] text-[#6B7280] hover:border-[#D1D5DB]'
            }`}
          >
            <input
              type="radio"
              name="ageMode"
              value="mivolo"
              checked={store.ageMode === 'mivolo'}
              onChange={() => handleModeChange('mivolo')}
              className="accent-[#E8804A]"
            />
            <span>Chưa rõ tuổi, dùng mô hình MiVOLO AI ước tính tự động</span>
          </label>
        </div>
      </div>

      {/* Nhánh nhập tay */}
      {store.ageMode === 'manual' && (
        <div className="bg-[#F9FAFB] p-3.5 rounded-lg border border-[#E5E7EB]">
          <label className="block text-xs font-medium text-[#6B7280] mb-1.5">
            Nhập số tuổi lúc chụp ảnh:
          </label>
          <div className="flex items-center gap-2.5">
            <input
              type="number"
              min={0}
              max={120}
              value={store.manualAge}
              onChange={handleManualAgeChange}
              className="bg-white border border-[#D1D5DB] rounded-lg px-3 py-1.5 text-[#111827] w-24 text-sm focus:outline-none focus:border-[#E8804A]"
            />
            <span className="text-xs text-[#6B7280]">tuổi</span>
          </div>
        </div>
      )}

      {/* Nhánh MiVOLO ước tính */}
      {store.ageMode === 'mivolo' && (
        <div className="bg-[#EFF6FF] p-3.5 rounded-lg border border-[#BFDBFE] space-y-2">
          {store.isEstimatingAge ? (
            <div className="flex items-center gap-2 text-xs text-[#3B82C7]">
              <Loader2 className="w-4 h-4 animate-spin text-[#3B82C7]" />
              <span>Đang ước tính tuổi bằng mô hình MiVOLO...</span>
            </div>
          ) : store.initialAge !== null ? (
            <div>
              <div className="flex items-center gap-2 text-[#3B82C7] font-medium text-xs">
                <Sparkles className="w-4 h-4" />
                <span>Hệ thống ước tính: khoảng {store.initialAge} tuổi</span>
              </div>
              {store.ageWarningText && (
                <div className="mt-2 text-xs text-[#D97706] flex items-start gap-1.5 bg-[#FEF3C7] p-2.5 rounded border border-[#FDE68A]">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5 text-[#D97706]" />
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
              className="text-xs bg-[#E8804A] hover:bg-[#D97706] text-white px-3.5 py-1.5 rounded-lg font-medium shadow-xs"
            >
              Ước tính tuổi ngay
            </button>
          )}
        </div>
      )}
    </div>
  );
};
