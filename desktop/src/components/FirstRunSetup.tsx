import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { AlertOctagon, RefreshCw, FolderGit2 } from 'lucide-react';

export const FirstRunSetup: React.FC = () => {
  const { missingCheckpoints, isCheckingHealth } = useSearchStore();
  const { checkHealth } = useSearchApi();

  return (
    <div className="max-w-[680px] mx-auto my-12 bg-[#1B2129] border border-[#C9A24A]/40 rounded-2xl p-7 shadow-2xl space-y-5">
      <div className="flex items-center gap-3 text-[#C9A24A]">
        <AlertOctagon className="w-7 h-7 flex-shrink-0 text-[#C9A24A]" />
        <div>
          <h3 className="text-lg font-bold text-[#E8E6E0]">Chưa tìm thấy trọng số mô hình (Checkpoints)</h3>
          <p className="text-xs text-[#8E98A5] mt-0.5">
            Ứng dụng cần các file checkpoint trọng số để khởi động pipeline FADING.
          </p>
        </div>
      </div>

      <div className="bg-[#12161C] p-4 rounded-xl border border-[#262E38] space-y-2">
        <p className="text-xs font-semibold text-[#8E98A5] uppercase tracking-wider">
          Các file / thư mục đang thiếu:
        </p>
        <ul className="list-disc pl-5 text-xs text-[#B8564A] space-y-1">
          {missingCheckpoints.map((item, idx) => (
            <li key={idx} className="font-mono">
              {item}
            </li>
          ))}
        </ul>
      </div>

      <div className="bg-[#12161C] border border-[#262E38] p-4 rounded-xl text-xs text-[#E8E6E0]/90 space-y-2">
        <div className="flex items-center gap-2 font-semibold text-[#C97B4A]">
          <FolderGit2 className="w-4 h-4 text-[#C97B4A]" />
          <span>Hướng dẫn chuẩn bị trọng số:</span>
        </div>
        <p>
          1. Đảm bảo thư mục <code className="bg-[#1B2129] px-1 py-0.5 rounded border border-[#2E3844]">checkpoints/</code> nằm cạnh thư mục chạy ứng dụng.
        </p>
        <p>
          2. Đặt checkpoint Specialized UNet vào: <code className="bg-[#1B2129] px-1 py-0.5 rounded border border-[#2E3844]">checkpoints/specialized_unet/</code>.
        </p>
        <p>
          3. Đặt các mô hình MiVOLO vào: <code className="bg-[#1B2129] px-1 py-0.5 rounded border border-[#2E3844]">checkpoints/mivolo/</code>.
        </p>
      </div>

      <div className="flex justify-end pt-2">
        <button
          onClick={() => checkHealth()}
          disabled={isCheckingHealth}
          className="flex items-center gap-2 bg-[#C97B4A] hover:brightness-110 text-white font-medium text-xs px-5 py-2.5 rounded-xl transition-all shadow-md disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isCheckingHealth ? 'animate-spin' : ''}`} />
          <span>Kiểm tra lại trạng thái</span>
        </button>
      </div>
    </div>
  );
};

