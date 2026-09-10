import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { AlertTriangle, Info } from 'lucide-react';

export const QualityWarnings: React.FC = () => {
  const warnings = useSearchStore((s) => s.warnings);

  if (!warnings || warnings.length === 0) {
    return null;
  }

  return (
    <div className="bg-amber-950/40 border border-amber-500/50 rounded-xl p-5 shadow-lg space-y-3">
      <div className="flex items-center gap-2 text-amber-400 font-semibold text-sm">
        <AlertTriangle className="w-5 h-5 flex-shrink-0" />
        <span>Cảnh báo chất lượng ảnh đầu vào:</span>
      </div>

      <ul className="space-y-1 pl-7 list-disc text-sm text-amber-200/90">
        {warnings.map((w, idx) => (
          <li key={idx}>{w}</li>
        ))}
      </ul>

      <div className="flex items-center gap-2 text-xs text-amber-300/80 bg-amber-900/30 px-3 py-2 rounded-lg border border-amber-700/30">
        <Info className="w-4 h-4 flex-shrink-0" />
        <span>Vẫn có thể tiếp tục chạy pipeline, nhưng kết quả nhận diện sẽ tối ưu hơn nếu chọn ảnh chính diện, rõ nét.</span>
      </div>
    </div>
  );
};
