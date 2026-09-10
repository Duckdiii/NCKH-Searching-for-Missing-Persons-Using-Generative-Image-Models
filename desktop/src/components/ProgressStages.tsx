import React, { useEffect, useState } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { Check, Loader2, Clock, AlertCircle } from 'lucide-react';

const STAGES = [
  { id: 'specialization', title: '1. Chuẩn bị Checkpoint', desc: 'Kiểm tra và tải mô hình UNet đã tinh chỉnh' },
  { id: 'inversion', title: '2. Null-text Inversion', desc: 'Nghịch đảo DDIM và tối ưu chuỗi embedding (50 steps)' },
  { id: 'editing', title: '3. Age Editing', desc: 'Sinh khuôn mặt qua Cross-Attention Injection' },
  { id: 'search', title: '4. Embedding & Search', desc: 'InsightFace trích xuất vector và FAISS đối soát' },
];

export const ProgressStages: React.FC = () => {
  const { jobStatus, jobStage, jobError } = useSearchStore();
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let interval: any = null;
    if (jobStatus === 'running') {
      const start = Date.now();
      interval = setInterval(() => {
        setElapsed(Math.floor((Date.now() - start) / 1000));
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobStatus]);

  if (jobStatus === 'idle') {
    return null;
  }

  const getStageIndex = (stage: string) => {
    switch (stage) {
      case 'specialization': return 0;
      case 'inversion': return 1;
      case 'editing': return 2;
      case 'search': return 3;
      case 'complete': return 4;
      default: return 0;
    }
  };

  const currentIdx = getStageIndex(jobStage);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m > 0 ? `${m}m ` : ''}${s}s`;
  };

  return (
    <div className="bg-slate-800/90 border border-slate-700 rounded-xl p-6 shadow-2xl space-y-6">
      <div className="flex items-center justify-between border-b border-slate-700 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            {jobStatus === 'running' && <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />}
            {jobStatus === 'done' && <Check className="w-5 h-5 text-emerald-400" />}
            {jobStatus === 'error' && <AlertCircle className="w-5 h-5 text-rose-400" />}
            <span>
              {jobStatus === 'running' && 'Pipeline đang thực thi trên GPU...'}
              {jobStatus === 'done' && 'Pipeline đã hoàn tất thành công!'}
              {jobStatus === 'error' && 'Đã xảy ra lỗi trong quá trình thực thi'}
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Quá trình xử lý Null-text Inversion và Editing có thể kéo dài vài phút tùy GPU.
          </p>
        </div>

        {jobStatus === 'running' && (
          <div className="flex items-center gap-1.5 text-xs text-indigo-300 font-mono bg-indigo-950/60 px-3 py-1.5 rounded-full border border-indigo-500/30">
            <Clock className="w-3.5 h-3.5" />
            <span>Thời gian: {formatTime(elapsed)}</span>
          </div>
        )}
      </div>

      {jobStatus === 'error' && jobError && (
        <div className="bg-rose-950/50 border border-rose-500/50 p-4 rounded-lg text-rose-200 text-sm">
          <p className="font-semibold mb-1">Chi tiết lỗi:</p>
          <code className="text-xs font-mono">{jobError}</code>
        </div>
      )}

      {/* Stepper danh sách 4 giai đoạn */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {STAGES.map((s, idx) => {
          const isDone = jobStatus === 'done' || currentIdx > idx;
          const isCurrent = jobStatus === 'running' && currentIdx === idx;
          const isPending = !isDone && !isCurrent;

          return (
            <div
              key={s.id}
              className={`p-4 rounded-xl border transition-all duration-300 ${
                isDone
                  ? 'bg-emerald-950/20 border-emerald-500/40 text-emerald-200'
                  : isCurrent
                  ? 'bg-indigo-950/40 border-indigo-500 shadow-lg shadow-indigo-500/10 text-white'
                  : 'bg-slate-900/40 border-slate-700/60 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">
                  Giai đoạn {idx + 1}
                </span>
                <div className="w-6 h-6 rounded-full flex items-center justify-center">
                  {isDone && <Check className="w-4 h-4 text-emerald-400" />}
                  {isCurrent && <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />}
                  {isPending && <span className="text-xs text-slate-500">{idx + 1}</span>}
                </div>
              </div>

              <p className="font-semibold text-sm mb-1 text-slate-100">{s.title}</p>
              <p className="text-xs text-slate-400 line-clamp-2">{s.desc}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
