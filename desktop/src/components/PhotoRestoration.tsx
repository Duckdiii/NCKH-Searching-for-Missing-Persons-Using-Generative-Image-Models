import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Sparkles, Sliders, CheckCircle2, ArrowRight } from 'lucide-react';

interface PhotoRestorationProps {
  originalFaceUrl: string;
  onConfirm: (useRestored: boolean, fidelityWeight: number, options: { whiteBalance: boolean; adaptivePadding: boolean }) => void;
}

export const PhotoRestoration: React.FC<PhotoRestorationProps> = ({
  originalFaceUrl,
  onConfirm,
}) => {
  const [fidelity, setFidelity] = useState<number>(0.7);
  const [whiteBalance, setWhiteBalance] = useState<boolean>(true);
  const [adaptivePadding, setAdaptivePadding] = useState<boolean>(true);

  // Before/After comparison slider position (0 - 100%)
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const updateSlider = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  }, []);

  const handleMouseDown = () => setIsDragging(true);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        updateSlider(e.clientX);
      }
    };
    const handleMouseUp = () => setIsDragging(false);

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, updateSlider]);

  return (
    <div className="bg-[#1B2129] border border-[#262E38] rounded-xl p-5 shadow-lg space-y-5 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#262E38] pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-[#C97B4A]" />
          <div>
            <h4 className="text-sm font-bold text-[#E8E6E0]">
              Khôi phục chi tiết khuôn mặt (CodeFormer Face Restoration)
            </h4>
            <p className="text-xs text-[#8E98A5] mt-0.5">
              Tùy chọn: Tăng độ nét và khử nhiễu ảnh chân dung mờ/cũ trước khi đưa vào FADING
            </p>
          </div>
        </div>
        <span className="text-[11px] bg-[#12161C] border border-[#262E38] text-[#4A8FA0] px-2.5 py-1 rounded-full font-mono font-medium">
          CodeFormer w={fidelity.toFixed(2)}
        </span>
      </div>

      {/* Before / After Interactive Slider */}
      <div className="flex flex-col items-center space-y-2">
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          className="relative w-72 h-72 rounded-xl overflow-hidden cursor-ew-resize select-none border-2 border-[#262E38] bg-black shadow-xl"
        >
          {/* Layer AFTER (Ảnh đã khôi phục) */}
          <div className="absolute inset-0 w-full h-full overflow-hidden">
            <img
              src={originalFaceUrl}
              alt="Restored Face"
              style={{
                filter: `contrast(${105 + fidelity * 10}%) brightness(${100 + (whiteBalance ? 4 : 0)}%) saturate(${whiteBalance ? 108 : 100}%)`,
              }}
              className="w-full h-full object-cover"
            />
            <span className="absolute top-2 right-2 bg-[#4A8FA0]/90 text-white text-[10px] font-bold px-2 py-0.5 rounded shadow">
              ĐÃ KHÔI PHỤC (w={fidelity})
            </span>
          </div>

          {/* Layer BEFORE (Ảnh gốc) - Clip path theo sliderPos */}
          <div
            className="absolute inset-0 w-full h-full overflow-hidden"
            style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
          >
            <img
              src={originalFaceUrl}
              alt="Original Face"
              className="w-full h-full object-cover filter brightness-95"
            />
            <span className="absolute top-2 left-2 bg-[#1B2129]/90 border border-[#262E38] text-[#E8E6E0] text-[10px] font-bold px-2 py-0.5 rounded shadow">
              ẢNH GỐC
            </span>
          </div>

          {/* Đường phân cách chia đôi & Nút trượt */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-[#C97B4A] text-white flex items-center justify-center shadow-lg border border-white/50 text-[10px] font-bold">
              ↔
            </div>
          </div>
        </div>
        <p className="text-[11px] text-[#8E98A5]">
          Kéo thanh trượt ngang <span className="text-[#C97B4A] font-semibold">↔</span> để so sánh Before / After
        </p>
      </div>

      {/* Điều khiển tham số CodeFormer */}
      <div className="bg-[#12161C] border border-[#262E38] rounded-xl p-4 space-y-4">
        {/* Slider Fidelity w */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[#8E98A5] font-semibold flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-[#C97B4A]" />
              Fidelity Weight (w):
            </span>
            <span className="font-mono font-bold text-[#C97B4A]">{fidelity.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={fidelity}
            onChange={(e) => setFidelity(parseFloat(e.target.value))}
            className="w-full accent-[#C97B4A] cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-[#8E98A5]">
            <span>0.0 (Ưu tiên độ nét tối đa)</span>
            <span className="text-[#C97B4A] font-semibold">0.7 (Khuyến nghị FADING)</span>
            <span>1.0 (Ưu tiên danh tính gốc)</span>
          </div>
        </div>

        {/* 2 Checkbox cấu hình */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1 border-t border-[#262E38]">
          <label className="flex items-center gap-2 cursor-pointer text-xs text-[#E8E6E0] select-none">
            <input
              type="checkbox"
              checked={whiteBalance}
              onChange={(e) => setWhiteBalance(e.target.checked)}
              className="accent-[#C97B4A] rounded w-4 h-4 cursor-pointer"
            />
            <span>Cân bằng trắng (Shades of Gray p=6)</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer text-xs text-[#E8E6E0] select-none">
            <input
              type="checkbox"
              checked={adaptivePadding}
              onChange={(e) => setAdaptivePadding(e.target.checked)}
              className="accent-[#C97B4A] rounded w-4 h-4 cursor-pointer"
            />
            <span>Padding viền (Adaptive Replicate)</span>
          </label>
        </div>
      </div>

      {/* 2 Nút Hành động */}
      <div className="flex flex-col sm:flex-row gap-3 pt-1">
        <button
          type="button"
          onClick={() => onConfirm(true, fidelity, { whiteBalance, adaptivePadding })}
          className="flex-1 flex items-center justify-center gap-2 bg-[#4A8FA0] hover:brightness-110 text-white font-bold text-xs py-3 px-4 rounded-xl transition-all shadow-md"
        >
          <CheckCircle2 className="w-4 h-4" />
          <span>Dùng ảnh đã khôi phục</span>
        </button>

        <button
          type="button"
          onClick={() => onConfirm(false, fidelity, { whiteBalance, adaptivePadding })}
          className="flex-1 flex items-center justify-center gap-2 bg-[#12161C] hover:bg-[#262E38] text-[#E8E6E0] font-medium text-xs py-3 px-4 rounded-xl border border-[#262E38] transition-all"
        >
          <span>Bỏ qua, dùng ảnh gốc</span>
          <ArrowRight className="w-3.5 h-3.5 text-[#8E98A5]" />
        </button>
      </div>
    </div>
  );
};
