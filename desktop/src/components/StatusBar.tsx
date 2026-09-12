import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { Cpu, Command, Sliders, ChevronRight } from 'lucide-react';

interface StatusBarProps {
  currentStepLabel?: string;
  onOpenCommandPalette?: () => void;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  currentStepLabel = 'Bước 1: Khôi phục ảnh',
  onOpenCommandPalette,
}) => {
  const { jobStatus, isAdvancedMode, toggleAdvancedMode, isHistoricalView } = useSearchStore();

  const isGpuBusy = jobStatus === 'running';

  return (
    <header className="h-12 border-b border-[#E5E7EB] bg-white px-4 sm:px-6 flex items-center justify-between z-10 select-none">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-xs text-[#6B7280]">
        <span className="font-medium hover:text-[#111827] transition-colors">
          {isHistoricalView ? 'Xem lại phiên' : 'Tìm kiếm mới'}
        </span>
        <ChevronRight className="w-3.5 h-3.5 text-[#9CA3AF]" />
        <span className="font-semibold text-[#111827]">{currentStepLabel}</span>
      </div>

      {/* Center & Right Status Elements */}
      <div className="flex items-center gap-3 sm:gap-4">
        {/* GPU Status Pill */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
            isGpuBusy
              ? 'bg-[#FEF3C7] text-[#D97706] border-[#FDE68A]'
              : 'bg-[#F0FDF4] text-[#16A34A] border-[#BBF7D0]'
          }`}
          title={
            isGpuBusy
              ? 'GPU đang xử lý pipeline FADING (Inversion / Editing / FAISS)'
              : 'GPU đang ở trạng thái rảnh rỗi, sẵn sàng nhận tác vụ'
          }
        >
          <span className="relative flex h-2 w-2">
            {isGpuBusy && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#D97706] opacity-75"></span>
            )}
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                isGpuBusy ? 'bg-[#D97706]' : 'bg-[#16A34A]'
              }`}
            ></span>
          </span>
          <Cpu className="w-3.5 h-3.5" />
          <span>{isGpuBusy ? 'GPU: Đang xử lý' : 'GPU: Rảnh'}</span>
        </div>

        {/* Command Palette Hint Button */}
        <button
          onClick={onOpenCommandPalette}
          className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-[#6B7280] hover:text-[#111827] hover:bg-[#F3F4F6] border border-[#E5E7EB] transition-colors"
          title="Mở Command Palette (Ctrl+K hoặc ⌘K)"
        >
          <Command className="w-3 h-3 text-[#9CA3AF]" />
          <span>Tìm nhanh</span>
          <kbd className="text-[10px] font-mono bg-[#F9FAFB] px-1 py-0.5 rounded border border-[#E5E7EB] text-[#6B7280]">
            ⌘K
          </kbd>
        </button>

        {/* Advanced Mode Toggle Switch */}
        <div className="flex items-center gap-2 pl-2 border-l border-[#E5E7EB]">
          <span className="text-xs text-[#6B7280] hidden md:inline font-medium flex items-center gap-1">
            <Sliders className="w-3 h-3" />
            Nâng cao
          </span>

          <button
            type="button"
            role="switch"
            aria-checked={isAdvancedMode}
            onClick={toggleAdvancedMode}
            title={isAdvancedMode ? 'Tắt chế độ nâng cao' : 'Bật chế độ nâng cao'}
            className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
              isAdvancedMode ? 'bg-[#E8804A]' : 'bg-[#D1D5DB]'
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-sm ring-0 transition duration-200 ease-in-out ${
                isAdvancedMode ? 'translate-x-4' : 'translate-x-0'
              }`}
            />
          </button>
        </div>
      </div>
    </header>
  );
};
