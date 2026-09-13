import React, { useEffect, useState } from 'react';
import { CheckCircle2, X, ArrowRight } from 'lucide-react';

export interface ToastData {
  id: string;
  identityName: string;
  scorePct?: string | null;
  onViewResult?: () => void;
}

interface ToastProps {
  toast: ToastData | null;
  onClose: () => void;
}

export const Toast: React.FC<ToastProps> = ({ toast, onClose }) => {
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    if (!toast) return;

    setProgress(100);
    const startTime = Date.now();
    const duration = 5000;

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, 100 - (elapsed / duration) * 100);
      setProgress(remaining);
      if (elapsed >= duration) {
        clearInterval(interval);
        onClose();
      }
    }, 50);

    return () => clearInterval(interval);
  }, [toast, onClose]);

  if (!toast) return null;

  return (
    <div
      role="alert"
      className="fixed bottom-5 right-5 z-50 bg-white border border-[#E5E7EB] rounded-xl shadow-2xl p-4 flex flex-col gap-2 max-w-sm w-full animate-in slide-in-from-bottom-5 fade-in duration-300"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-[#EFF6FF] border border-[#BFDBFE] text-[#3B82C7] rounded-lg flex-shrink-0">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <h4 className="text-xs font-bold text-[#111827] uppercase tracking-wide">
              Job hoàn tất
            </h4>
            <p className="text-sm font-medium text-[#374151] mt-0.5">
              {toast.identityName} {toast.scorePct ? `· ${toast.scorePct}` : ''}
            </p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="text-[#9CA3AF] hover:text-[#111827] p-1 rounded-md transition-colors"
          title="Đóng thông báo"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex items-center justify-between pt-1 mt-1 border-t border-[#F3F4F6]">
        {toast.onViewResult ? (
          <button
            onClick={() => {
              toast.onViewResult?.();
              onClose();
            }}
            className="flex items-center gap-1.5 text-xs font-semibold text-[#E8804A] hover:text-[#C96B37] transition-colors py-1 px-1"
          >
            <span>Xem kết quả</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        ) : (
          <span className="text-[11px] text-[#6B7280]">Đã lưu vào lịch sử</span>
        )}
        <span className="text-[10px] text-[#9CA3AF]">Tự ẩn sau 5s</span>
      </div>

      {/* Progress bar countdown */}
      <div className="w-full bg-[#F3F4F6] h-1 rounded-full overflow-hidden mt-1">
        <div
          className="bg-[#3B82C7] h-full transition-all duration-75 ease-linear"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
};
