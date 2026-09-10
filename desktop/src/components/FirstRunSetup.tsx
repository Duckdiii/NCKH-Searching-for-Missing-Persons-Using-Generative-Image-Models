import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { AlertOctagon, RefreshCw, FolderGit2 } from 'lucide-react';

export const FirstRunSetup: React.FC = () => {
  const { missingCheckpoints, isCheckingHealth } = useSearchStore();
  const { checkHealth } = useSearchApi();

  return (
    <div className="max-w-2xl mx-auto my-12 bg-slate-800/90 border border-amber-500/40 rounded-2xl p-8 shadow-2xl space-y-6">
      <div className="flex items-center gap-3 text-amber-400">
        <AlertOctagon className="w-8 h-8 flex-shrink-0" />
        <div>
          <h3 className="text-xl font-bold">Chưa tìm thấy trọng số mô hình (Checkpoints)</h3>
          <p className="text-xs text-slate-400 mt-1">
            Ứng dụng cần các file checkpoint trọng số để khởi động pipeline FADING.
          </p>
        </div>
      </div>

      <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-700 space-y-2">
        <p className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
          Các file / thư mục đang thiếu:
        </p>
        <ul className="list-disc pl-5 text-sm text-rose-300 space-y-1">
          {missingCheckpoints.map((item, idx) => (
            <li key={idx} className="font-mono text-xs">
              {item}
            </li>
          ))}
        </ul>
      </div>

      <div className="bg-indigo-950/30 border border-indigo-500/30 p-4 rounded-xl text-xs text-indigo-200/90 space-y-2">
        <div className="flex items-center gap-2 font-semibold text-indigo-300">
          <FolderGit2 className="w-4 h-4" />
          <span>Hướng dẫn chuẩn bị trọng số:</span>
        </div>
        <p>
          1. Đảm bảo thư mục <code className="bg-indigo-900/50 px-1 py-0.5 rounded">checkpoints/</code> nằm cạnh thư mục chạy ứng dụng.
        </p>
        <p>
          2. Đặt checkpoint Specialized UNet vào: <code className="bg-indigo-900/50 px-1 py-0.5 rounded">checkpoints/specialized_unet/</code>.
        </p>
        <p>
          3. Đặt các mô hình MiVOLO vào: <code className="bg-indigo-900/50 px-1 py-0.5 rounded">checkpoints/mivolo/</code>.
        </p>
      </div>

      <div className="flex justify-end">
        <button
          onClick={() => checkHealth()}
          disabled={isCheckingHealth}
          className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm px-5 py-2.5 rounded-xl transition-all shadow-lg hover:shadow-indigo-500/25 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isCheckingHealth ? 'animate-spin' : ''}`} />
          <span>Kiểm tra lại trạng thái</span>
        </button>
      </div>
    </div>
  );
};
