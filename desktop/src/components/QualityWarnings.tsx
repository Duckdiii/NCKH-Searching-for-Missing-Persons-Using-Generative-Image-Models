import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { AlertTriangle, Info } from 'lucide-react';

export const QualityWarnings: React.FC = () => {
  const warnings = useSearchStore((s) => s.warnings);

  if (!warnings || warnings.length === 0) {
    return null;
  }

  return (
    <div className="bg-[#1B2129] border border-[#C9A24A]/40 rounded-xl p-4 shadow-md space-y-2.5">
      <div className="flex items-center gap-2 text-[#C9A24A] font-semibold text-xs uppercase tracking-wide">
        <AlertTriangle className="w-4 h-4 flex-shrink-0 text-[#C9A24A]" />
        <span>Cảnh báo chất lượng ảnh đầu vào:</span>
      </div>

      <ul className="space-y-1 pl-6 list-disc text-xs text-[#E8E6E0]/90">
        {warnings.map((w, idx) => (
          <li key={idx}>{w}</li>
        ))}
      </ul>

      <div className="flex items-center gap-2 text-xs text-[#8E98A5] bg-[#12161C] px-3 py-2 rounded-lg border border-[#262E38]">
        <Info className="w-3.5 h-3.5 flex-shrink-0 text-[#C9A24A]" />
        <span>Vẫn có thể tiếp tục chạy, nhưng kết quả nhận diện sẽ tối ưu nhất với ảnh chính diện, rõ nét.</span>
      </div>
    </div>
  );
};

