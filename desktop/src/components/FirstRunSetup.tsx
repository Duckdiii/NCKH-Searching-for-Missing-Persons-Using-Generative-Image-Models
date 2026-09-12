import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { AlertOctagon, RefreshCw, FolderGit2 } from 'lucide-react';

export const FirstRunSetup: React.FC = () => {
  const { missingCheckpoints, isCheckingHealth } = useSearchStore();
  const { checkHealth } = useSearchApi();

  return (
    <div className="max-w-[680px] mx-auto my-12 bg-white border border-[#D97706]/40 rounded-2xl p-7 shadow-lg space-y-5">
      <div className="flex items-center gap-3 text-[#D97706]">
        <AlertOctagon className="w-7 h-7 flex-shrink-0 text-[#D97706]" />
        <div>
          <h3 className="text-lg font-bold text-[#111827]">Chưa tìm thấy trọng số mô hình (Checkpoints)</h3>
          <p className="text-xs text-[#6B7280] mt-0.5">
            Ứng dụng cần các file checkpoint trọng số để khởi động pipeline FADING.
          </p>
        </div>
      </div>

      <div className="bg-[#F9FAFB] p-4 rounded-xl border border-[#E5E7EB] space-y-2">
        <p className="text-xs font-semibold text-[#6B7280] uppercase tracking-wider">
          Các file / thư mục đang thiếu:
        </p>
        <ul className="list-disc pl-5 text-xs text-[#DC2626] space-y-1">
          {missingCheckpoints.map((item, idx) => (
            <li key={idx} className="font-mono">
              {item}
            </li>
          ))}
        </ul>
      </div>

      <div className="bg-[#F9FAFB] border border-[#E5E7EB] p-4 rounded-xl text-xs text-[#374151] space-y-2">
        <div className="flex items-center gap-2 font-semibold text-[#E8804A]">
          <FolderGit2 className="w-4 h-4 text-[#E8804A]" />
          <span>Hướng dẫn chuẩn bị trọng số:</span>
        </div>
        <p>
          1. Đảm bảo thư mục <code className="bg-white px-1.5 py-0.5 rounded border border-[#E5E7EB] text-[#111827]">checkpoints/</code> nằm cạnh thư mục chạy ứng dụng.
        </p>
        <p>
          2. Đặt checkpoint Specialized UNet vào: <code className="bg-white px-1.5 py-0.5 rounded border border-[#E5E7EB] text-[#111827]">checkpoints/specialized_unet/</code>.
        </p>
        <p>
          3. Đặt các mô hình MiVOLO vào: <code className="bg-white px-1.5 py-0.5 rounded border border-[#E5E7EB] text-[#111827]">checkpoints/mivolo/</code>.
        </p>
      </div>

      <div className="flex justify-end pt-2">
        <button
          onClick={() => checkHealth()}
          disabled={isCheckingHealth}
          className="flex items-center gap-2 bg-[#E8804A] hover:bg-[#D97706] text-white font-medium text-xs px-5 py-2.5 rounded-xl transition-all shadow-sm disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isCheckingHealth ? 'animate-spin' : ''}`} />
          <span>Kiểm tra lại trạng thái</span>
        </button>
      </div>
    </div>
  );
};

