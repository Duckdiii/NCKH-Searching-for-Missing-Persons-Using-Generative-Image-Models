import React, { useEffect, useState } from 'react';
import { Loader2, Check, Lock, XCircle, Clock } from 'lucide-react';

interface JobProgressBlockProps {
  jobStage: string;
  jobStatus: 'idle' | 'running' | 'done' | 'error';
  jobError?: string | null;
  onCancel: () => void;
}

interface SubStage {
  id: string;
  title: string;
  desc: string;
}

const SUB_STAGES: SubStage[] = [
  { id: 'inversion', title: '1. Null-text Inversion', desc: 'Nghịch đảo DDIM 50 bước tối ưu latent vector' },
  { id: 'editing', title: '2. Attention Editing', desc: 'Tiêm cross-attention biến đổi lứa tuổi FADING' },
  { id: 'search', title: '3. Rendering & FAISS', desc: 'Trích xuất đặc trưng InsightFace & đối soát Gallery' },
];

export const JobProgressBlock: React.FC<JobProgressBlockProps> = ({
  jobStage,
  jobStatus,
  jobError,
  onCancel,
}) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let timer: any = null;
    if (jobStatus === 'running') {
      const start = Date.now();
      timer = setInterval(() => {
        setElapsed(Math.floor((Date.now() - start) / 1000));
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [jobStatus]);

  // Xác định tiến độ tổng theo %
  const getProgressPercentage = () => {
    switch (jobStage) {
      case 'specialization': return 15;
      case 'inversion': return 45;
      case 'editing': return 75;
      case 'search': return 92;
      case 'complete': return 100;
      default: return 10;
    }
  };

  const getSubStageState = (stageId: string): 'done' | 'running' | 'pending' => {
    if (jobStatus === 'done' || jobStage === 'complete') return 'done';

    if (stageId === 'inversion') {
      if (jobStage === 'specialization' || jobStage === 'inversion') return 'running';
      if (['editing', 'search', 'complete'].includes(jobStage)) return 'done';
      return 'pending';
    }

    if (stageId === 'editing') {
      if (jobStage === 'editing') return 'running';
      if (['search', 'complete'].includes(jobStage)) return 'done';
      return 'pending';
    }

    if (stageId === 'search') {
      if (jobStage === 'search') return 'running';
      if (jobStage === 'complete') return 'done';
      return 'pending';
    }

    return 'pending';
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m > 0 ? `${m}m ` : ''}${s}s`;
  };

  const pct = getProgressPercentage();

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-xs space-y-5 animate-in fade-in duration-300">
      {/* Tiêu đề & Đồng hồ bấm giờ */}
      <div className="flex items-center justify-between border-b border-[#E5E7EB] pb-3.5">
        <div className="flex items-center gap-2.5">
          <Loader2 className="w-5 h-5 text-[#E8804A] animate-spin flex-shrink-0" />
          <div>
            <h4 className="text-sm font-bold text-[#111827]">
              Pipeline AI FADING đang thực thi trên GPU...
            </h4>
            <p className="text-xs text-[#6B7280] mt-0.5">
              Đang tính toán các latent vectors và sinh ảnh độ tuổi theo luồng
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-xs text-[#E8804A] font-mono bg-[#F9FAFB] px-3 py-1.5 rounded-full border border-[#E5E7EB]">
          <Clock className="w-3.5 h-3.5" />
          <span>{formatTime(elapsed)}</span>
        </div>
      </div>

      {/* Thanh Progress bar tổng theo % */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs font-semibold">
          <span className="text-[#6B7280]">Tiến độ tổng thể:</span>
          <span className="font-mono text-[#E8804A]">{pct}%</span>
        </div>
        <div className="w-full h-2.5 bg-[#F3F4F6] rounded-full overflow-hidden border border-[#E5E7EB]">
          <div
            className="h-full bg-[#E8804A] progress-bar-smooth rounded-full"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Stepper con 3 ô: Inversion -> Editing -> Rendering */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {SUB_STAGES.map((st) => {
          const state = getSubStageState(st.id);
          return (
            <div
              key={st.id}
              className={`p-3.5 rounded-lg border transition-all duration-200 ${
                state === 'done'
                  ? 'bg-[#EFF6FF] border-[#3B82C7] text-[#111827]'
                  : state === 'running'
                  ? 'bg-[#FFF7ED] border-[#E8804A] text-[#111827] shadow-xs ring-1 ring-[#E8804A]/40'
                  : 'bg-[#F9FAFB] border-[#E5E7EB] text-[#9CA3AF] opacity-60'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-bold uppercase tracking-wider">
                  {state === 'done' ? 'Hoàn thành' : state === 'running' ? 'Đang chạy' : 'Chờ xử lý'}
                </span>
                <div className="w-5 h-5 rounded-full flex items-center justify-center">
                  {state === 'done' && <Check className="w-4 h-4 text-[#3B82C7] stroke-[2.5]" />}
                  {state === 'running' && <Loader2 className="w-4 h-4 text-[#E8804A] animate-spin" />}
                  {state === 'pending' && <Lock className="w-3.5 h-3.5 text-[#9CA3AF]" />}
                </div>
              </div>

              <p className="font-bold text-xs text-[#111827]">{st.title}</p>
              <p className="text-[10px] text-[#6B7280] mt-0.5 line-clamp-2">{st.desc}</p>
            </div>
          );
        })}
      </div>

      {/* Thông báo lỗi nếu có */}
      {jobStatus === 'error' && jobError && (
        <div className="bg-[#FEF2F2] border border-[#FECACA] p-3 rounded-lg text-xs text-[#DC2626]">
          <span className="font-bold text-[#DC2626]">Lỗi thực thi:</span> {jobError}
        </div>
      )}

      {/* Nút Hủy tiến trình viền --accent-error */}
      <div className="flex justify-end pt-1">
        <button
          type="button"
          onClick={onCancel}
          className="flex items-center gap-1.5 text-xs text-[#DC2626] hover:text-white hover:bg-[#DC2626] bg-white border border-[#DC2626] px-4 py-2 rounded-lg transition-all font-semibold shadow-xs hover-lift"
        >
          <XCircle className="w-3.5 h-3.5" />
          <span>Hủy tiến trình</span>
        </button>
      </div>
    </div>
  );
};
