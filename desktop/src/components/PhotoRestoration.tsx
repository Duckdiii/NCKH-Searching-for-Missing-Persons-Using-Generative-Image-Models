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
    <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-xs space-y-5 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#E5E7EB] pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-[#E8804A]" />
          <div>
            <h4 className="text-sm font-bold text-[#111827]">
              Khôi phục chi tiết khuôn mặt (CodeFormer Face Restoration)
            </h4>
            <p className="text-xs text-[#6B7280] mt-0.5">
              Tùy chọn: Tăng độ nét và khử nhiễu ảnh chân dung mờ/cũ trước khi đưa vào FADING
            </p>
          </div>
        </div>
        <span className="text-[11px] bg-[#F9FAFB] border border-[#E5E7EB] text-[#3B82C7] px-2.5 py-1 rounded-full font-mono font-medium">
          CodeFormer w={fidelity.toFixed(2)}
        </span>
      </div>

      {/* Before / After Interactive Slider */}
      <div className="flex flex-col items-center space-y-2">
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          className="relative w-72 h-72 rounded-xl overflow-hidden cursor-ew-resize select-none border-2 border-[#E5E7EB] bg-[#F9FAFB] shadow-md"
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
            <span className="absolute top-2 right-2 bg-[#3B82C7] text-white text-[10px] font-bold px-2 py-0.5 rounded shadow-sm">
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
            <span className="absolute top-2 left-2 bg-white/95 border border-[#E5E7EB] text-[#111827] text-[10px] font-bold px-2 py-0.5 rounded shadow-sm">
              ẢNH GỐC
            </span>
          </div>

          {/* Đường phân cách chia đôi & Nút trượt */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-[#E8804A] text-white flex items-center justify-center shadow-md border-2 border-white text-[10px] font-bold">
              ↔
            </div>
          </div>
        </div>
        <p className="text-[11px] text-[#6B7280]">
          Kéo thanh trượt ngang <span className="text-[#E8804A] font-semibold">↔</span> để so sánh Before / After
        </p>
      </div>

      {/* Điều khiển tham số CodeFormer */}
      <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl p-4 space-y-4">
        {/* Slider Fidelity w */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[#6B7280] font-semibold flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-[#E8804A]" />
              Fidelity Weight (w):
            </span>
            <span className="font-mono font-bold text-[#E8804A]">{fidelity.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={fidelity}
            onChange={(e) => setFidelity(parseFloat(e.target.value))}
            className="w-full accent-[#E8804A] cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-[#6B7280]">
            <span>0.0 (Ưu tiên độ nét tối đa)</span>
            <span className="text-[#E8804A] font-semibold">0.7 (Khuyến nghị FADING)</span>
            <span>1.0 (Ưu tiên danh tính gốc)</span>
          </div>
        </div>

        {/* 2 Checkbox cấu hình */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1 border-t border-[#E5E7EB]">
          <label className="flex items-center gap-2 cursor-pointer text-xs text-[#111827] select-none">
            <input
              type="checkbox"
              checked={whiteBalance}
              onChange={(e) => setWhiteBalance(e.target.checked)}
              className="accent-[#E8804A] rounded w-4 h-4 cursor-pointer"
            />
            <span>Cân bằng trắng (Shades of Gray p=6)</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer text-xs text-[#111827] select-none">
            <input
              type="checkbox"
              checked={adaptivePadding}
              onChange={(e) => setAdaptivePadding(e.target.checked)}
              className="accent-[#E8804A] rounded w-4 h-4 cursor-pointer"
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
          className="flex-1 flex items-center justify-center gap-2 bg-[#E8804A] hover:bg-[#D97706] text-white font-bold text-xs py-3 px-4 rounded-xl transition-all shadow-xs hover:shadow"
        >
          <CheckCircle2 className="w-4 h-4" />
          <span>Dùng ảnh đã khôi phục</span>
        </button>

        <button
          type="button"
          onClick={() => onConfirm(false, fidelity, { whiteBalance, adaptivePadding })}
          className="flex-1 flex items-center justify-center gap-2 bg-white hover:bg-[#F9FAFB] text-[#111827] font-medium text-xs py-3 px-4 rounded-xl border border-[#E5E7EB] transition-all shadow-xs"
        >
          <span>Bỏ qua, dùng ảnh gốc</span>
          <ArrowRight className="w-3.5 h-3.5 text-[#6B7280]" />
        </button>
      </div>
    </div>
  );
};
