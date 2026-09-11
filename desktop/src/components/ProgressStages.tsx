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
    <div className="bg-[#1B2129] border border-[#262E38] rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-[#262E38] pb-3">
        <div>
          <h3 className="text-base font-bold text-[#E8E6E0] flex items-center gap-2">
            {jobStatus === 'running' && <Loader2 className="w-5 h-5 text-[#C97B4A] animate-spin" />}
            {jobStatus === 'done' && <Check className="w-5 h-5 text-[#4A8FA0]" />}
            {jobStatus === 'error' && <AlertCircle className="w-5 h-5 text-[#B8564A]" />}
            <span>
              {jobStatus === 'running' && 'Pipeline đang thực thi trên GPU...'}
              {jobStatus === 'done' && 'Pipeline đã hoàn tất thành công!'}
              {jobStatus === 'error' && 'Đã xảy ra lỗi trong quá trình thực thi'}
            </span>
          </h3>
          <p className="text-xs text-[#8E98A5] mt-0.5">
            Quá trình xử lý Null-text Inversion và Editing có thể kéo dài vài phút tùy GPU.
          </p>
        </div>

        {jobStatus === 'running' && (
          <div className="flex items-center gap-1.5 text-xs text-[#C97B4A] font-mono bg-[#12161C] px-3 py-1.5 rounded-full border border-[#262E38]">
            <Clock className="w-3.5 h-3.5" />
            <span>Thời gian: {formatTime(elapsed)}</span>
          </div>
        )}
      </div>

      {jobStatus === 'error' && jobError && (
        <div className="bg-[#B8564A]/15 border border-[#B8564A]/40 p-3.5 rounded-lg text-[#E8E6E0] text-xs">
          <p className="font-semibold mb-1 text-[#B8564A]">Chi tiết lỗi:</p>
          <code className="text-xs font-mono">{jobError}</code>
        </div>
      )}

      {/* Stepper danh sách 4 giai đoạn */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        {STAGES.map((s, idx) => {
          const isDone = jobStatus === 'done' || currentIdx > idx;
          const isCurrent = jobStatus === 'running' && currentIdx === idx;
          const isPending = !isDone && !isCurrent;

          return (
            <div
              key={s.id}
              className={`p-3 rounded-lg border transition-all duration-200 ${
                isDone
                  ? 'bg-[#4A8FA0]/15 border-[#4A8FA0]/40 text-[#E8E6E0]'
                  : isCurrent
                  ? 'bg-[#C97B4A]/15 border-[#C97B4A] text-[#E8E6E0]'
                  : 'bg-[#12161C] border-[#262E38] text-[#8E98A5]'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider">
                  Giai đoạn {idx + 1}
                </span>
                <div className="w-5 h-5 rounded-full flex items-center justify-center">
                  {isDone && <Check className="w-4 h-4 text-[#4A8FA0]" />}
                  {isCurrent && <Loader2 className="w-4 h-4 text-[#C97B4A] animate-spin" />}
                  {isPending && <span className="text-[11px] text-[#8E98A5]">{idx + 1}</span>}
                </div>
              </div>

              <p className="font-semibold text-xs mb-1 text-[#E8E6E0]">{s.title}</p>
              <p className="text-[11px] text-[#8E98A5] line-clamp-2 leading-relaxed">{s.desc}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
