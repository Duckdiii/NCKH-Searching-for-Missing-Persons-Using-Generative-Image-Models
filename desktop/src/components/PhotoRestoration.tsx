import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Sparkles, Sliders, CheckCircle2, ArrowRight, Pipette, X } from 'lucide-react';
import { Skeleton } from './Skeleton';

export interface PhotoRestorationOptions {
  mode: 'auto' | 'manual';
  whiteBalance: boolean;
  adaptivePadding: boolean;
  refPoint?: { x: number; y: number } | null;
  wbGains?: [number, number, number] | null;
}

export interface PhotoRestorationProps {
  originalFaceUrl: string;
  onConfirm: (
    useRestored: boolean,
    fidelityWeight: number,
    options: PhotoRestorationOptions
  ) => void;
  sessionId?: string | null;
}

export const PhotoRestoration: React.FC<PhotoRestorationProps> = ({
  originalFaceUrl,
  onConfirm,
}) => {
  // Mode selection: 'auto' (default) vs 'manual'
  const [mode, setMode] = useState<'auto' | 'manual'>('auto');

  // Parameters
  const [fidelity, setFidelity] = useState<number>(0.7);
  const [whiteBalance, setWhiteBalance] = useState<boolean>(true);
  const [adaptivePadding, setAdaptivePadding] = useState<boolean>(true);
  const [isImgLoaded, setIsImgLoaded] = useState<boolean>(false);

  // Manual Mode: Eyedropper reference point & gains
  const [isEyedropperActive, setIsEyedropperActive] = useState<boolean>(false);
  const [refPoint, setRefPoint] = useState<{ x: number; y: number } | null>(null);
  const [wbGains, setWbGains] = useState<{ r: number; g: number; b: number } | null>(null);
  const [sampledRgb, setSampledRgb] = useState<{ r: number; g: number; b: number } | null>(null);

  // Before/After comparison slider position (0 - 100%)
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const updateSlider = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  }, []);

  const handleMouseDown = () => {
    if (isEyedropperActive) return;
    setIsDragging(true);
  };

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

  // Click on image when Eyedropper mode is active
  const handleContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isEyedropperActive || !containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const clickRelX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const clickRelY = Math.max(0, Math.min(e.clientY - rect.top, rect.height));

    const naturalW = imageRef.current?.naturalWidth || 512;
    const naturalH = imageRef.current?.naturalHeight || 512;

    const px = Math.round((clickRelX / rect.width) * naturalW);
    const py = Math.round((clickRelY / rect.height) * naturalH);

    // Read pixel RGB from offscreen canvas
    try {
      const canvas = document.createElement('canvas');
      canvas.width = naturalW;
      canvas.height = naturalH;
      const ctx = canvas.getContext('2d');
      if (ctx && imageRef.current) {
        ctx.drawImage(imageRef.current, 0, 0, naturalW, naturalH);
        const pixelData = ctx.getImageData(px, py, 1, 1).data;
        const r = pixelData[0];
        const g = pixelData[1];
        const b = pixelData[2];

        const gray = (r + g + b) / 3.0;
        const rawR = gray / Math.max(r, 1);
        const rawG = gray / Math.max(g, 1);
        const rawB = gray / Math.max(b, 1);

        const clampedR = Math.min(1.30, Math.max(0.75, rawR));
        const clampedG = Math.min(1.30, Math.max(0.75, rawG));
        const clampedB = Math.min(1.30, Math.max(0.75, rawB));

        setSampledRgb({ r, g, b });
        setWbGains({ r: clampedR, g: clampedG, b: clampedB });
      } else {
        setWbGains({ r: 0.95, g: 1.02, b: 1.05 });
      }
    } catch {
      setWbGains({ r: 0.95, g: 1.02, b: 1.05 });
    }

    setRefPoint({ x: px, y: py });
    setIsEyedropperActive(false);
  };

  // Real-time filter for Layer AFTER
  const getAfterFilter = () => {
    if (mode === 'auto') {
      return `contrast(${105 + fidelity * 10}%) brightness(${100 + (whiteBalance ? 4 : 0)}%) saturate(${whiteBalance ? 108 : 100}%)`;
    }

    // Manual mode
    const baseContrast = 100 + fidelity * 15;
    // Khi ở chế độ Thủ công nhưng chưa chọn điểm (wbGains === null) -> không áp filter màu nào (bypass)
    const baseBrightness = (whiteBalance && wbGains) ? 102 : 100;
    const baseSaturate = (whiteBalance && wbGains) ? 104 : 100;
    return `contrast(${baseContrast}%) brightness(${baseBrightness}%) saturate(${baseSaturate}%)`;
  };

  const naturalW = imageRef.current?.naturalWidth || 512;
  const naturalH = imageRef.current?.naturalHeight || 512;

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-xs space-y-4 animate-in fade-in duration-300">
      {/* SVG Filter for manual color matrix */}
      {wbGains && (
        <svg className="hidden" aria-hidden="true">
          <filter id="manual-point-wb-filter">
            <feColorMatrix
              type="matrix"
              values={`
                ${wbGains.r} 0 0 0 0
                0 ${wbGains.g} 0 0 0
                0 0 ${wbGains.b} 0 0
                0 0 0 1 0
              `}
            />
          </filter>
        </svg>
      )}

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
          {mode === 'auto' ? `Tự động • w=${fidelity.toFixed(2)}` : `Thủ công • w=${fidelity.toFixed(2)}`}
        </span>
      </div>

      {/* 1. Bộ chọn Chế độ: Tự động (khuyến nghị) / Thủ công (Chuẩn Style Bước 2) */}
      <div>
        <label className="block text-xs font-semibold text-[#6B7280] mb-2 uppercase tracking-wide">
          Phương thức khôi phục:
        </label>
        <div className="flex flex-col sm:flex-row gap-2.5">
          <label
            data-testid="mode-auto"
            className={`flex-1 flex items-center gap-2.5 cursor-pointer text-xs p-3 rounded-lg border transition-all hover-lift ${
              mode === 'auto'
                ? 'bg-white border-[#E8804A] text-[#111827] shadow-xs ring-1 ring-[#E8804A]/30'
                : 'bg-[#F9FAFB] border-[#E5E7EB] text-[#6B7280] hover:border-[#D1D5DB]'
            }`}
          >
            <input
              type="radio"
              name="restoreMode"
              value="auto"
              checked={mode === 'auto'}
              onChange={() => {
                setMode('auto');
                setIsEyedropperActive(false);
              }}
              className="accent-[#E8804A]"
            />
            <div>
              <span className="font-semibold text-xs text-[#111827]">Tự động (khuyến nghị)</span>
              <p className="text-[10px] text-[#6B7280] mt-0.5">Padding + Shades of Gray WB + CodeFormer</p>
            </div>
          </label>

          <label
            data-testid="mode-manual"
            className={`flex-1 flex items-center gap-2.5 cursor-pointer text-xs p-3 rounded-lg border transition-all hover-lift ${
              mode === 'manual'
                ? 'bg-white border-[#E8804A] text-[#111827] shadow-xs ring-1 ring-[#E8804A]/30'
                : 'bg-[#F9FAFB] border-[#E5E7EB] text-[#6B7280] hover:border-[#D1D5DB]'
            }`}
          >
            <input
              type="radio"
              name="restoreMode"
              value="manual"
              checked={mode === 'manual'}
              onChange={() => setMode('manual')}
              className="accent-[#E8804A]"
            />
            <div>
              <span className="font-semibold text-xs text-[#111827]">Thủ công</span>
              <p className="text-[10px] text-[#6B7280] mt-0.5">Tùy chỉnh riêng 3 bước & chọn điểm tham chiếu màu</p>
            </div>
          </label>
        </div>
      </div>

      {/* Before / After Interactive Slider */}
      <div className="flex flex-col items-center space-y-2">
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          onClick={handleContainerClick}
          className={`relative w-72 h-72 rounded-xl overflow-hidden select-none border-2 border-[#E5E7EB] bg-[#F9FAFB] shadow-md ${
            isEyedropperActive ? 'cursor-crosshair ring-2 ring-[#E8804A]' : 'cursor-ew-resize'
          }`}
        >
          {/* Skeleton Shimmer while loading */}
          {!isImgLoaded && (
            <Skeleton className="absolute inset-0 w-full h-full z-20" />
          )}

          {/* Eyedropper Instruction Overlay */}
          {isEyedropperActive && (
            <div className="absolute top-2 left-2 right-2 bg-black/85 text-white text-[11px] font-semibold py-1.5 px-3 rounded-lg flex items-center justify-between z-30 animate-pulse shadow-md">
              <span className="flex items-center gap-1.5">
                <Pipette className="w-3.5 h-3.5 text-[#E8804A]" />
                Click vào điểm trắng/xám trên ảnh gốc
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsEyedropperActive(false);
                }}
                className="text-white hover:text-gray-300 ml-2"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {/* Layer AFTER (Ảnh đã khôi phục) */}
          <div className="absolute inset-0 w-full h-full overflow-hidden">
            <img
              src={originalFaceUrl}
              alt="Restored Face"
              crossOrigin="anonymous"
              onLoad={() => setIsImgLoaded(true)}
              style={{
                filter: mode === 'manual' && whiteBalance && wbGains
                  ? `url(#manual-point-wb-filter) ${getAfterFilter()}`
                  : getAfterFilter(),
              }}
              className="w-full h-full object-cover transition-all duration-150"
            />
            <span className="absolute top-2 right-2 bg-[#3B82C7] text-white text-[10px] font-bold px-2 py-0.5 rounded shadow-sm z-10">
              ĐÃ KHÔI PHỤC ({mode === 'manual' ? 'Thủ công' : `w=${fidelity}`})
            </span>
          </div>

          {/* Layer BEFORE (Ảnh gốc) - Clip path theo sliderPos */}
          <div
            className="absolute inset-0 w-full h-full overflow-hidden"
            style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
          >
            <img
              ref={imageRef}
              src={originalFaceUrl}
              alt="Original Face"
              crossOrigin="anonymous"
              className="w-full h-full object-cover filter brightness-95"
            />
            <span className="absolute top-2 left-2 bg-white/95 border border-[#E5E7EB] text-[#111827] text-[10px] font-bold px-2 py-0.5 rounded shadow-sm z-10">
              ẢNH GỐC
            </span>
          </div>

          {/* Reference point marker on image */}
          {refPoint && (
            <div
              className="absolute w-4 h-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-[#16A34A] bg-[#22C55E]/40 pointer-events-none z-25 shadow-sm"
              style={{
                left: `${(refPoint.x / naturalW) * 100}%`,
                top: `${(refPoint.y / naturalH) * 100}%`,
              }}
            />
          )}

          {/* Đường phân cách chia đôi & Nút trượt */}
          {!isEyedropperActive && (
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none z-20"
              style={{ left: `${sliderPos}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-[#E8804A] text-white flex items-center justify-center shadow-md border-2 border-white text-[10px] font-bold">
                ↔
              </div>
            </div>
          )}
        </div>
        <p className="text-[11px] text-[#6B7280]">
          {isEyedropperActive
            ? 'Đang ở chế độ Eyedropper: Nhấp chuột lên ảnh để chọn điểm cân bằng trắng'
            : 'Kéo thanh trượt ngang ↔ để so sánh Trước / Sau'}
        </p>
      </div>

      {/* 2. Điều khiển tham số: Khác biệt giữa Tự động và Thủ công */}
      {mode === 'manual' ? (
        /* CHẾ ĐỘ THỦ CÔNG: 3 Khối bước riêng biệt */
        <div className="space-y-3 pt-2 border-t border-[#E5E7EB]">
          {/* Khối 1: Padding viền */}
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl p-3.5 flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-[#111827]">Khối 1: Padding viền</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                    adaptivePadding
                      ? 'bg-[#EFF6FF] text-[#3B82C7] border border-[#BFDBFE]'
                      : 'bg-gray-100 text-gray-500'
                  }`}
                >
                  {adaptivePadding ? 'BẬT (Adaptive Replicate)' : 'TẮT'}
                </span>
              </div>
              <p className="text-[11px] text-[#6B7280] mt-0.5">
                Bù biên viền tự động tránh mất cằm/tai khi căn chỉnh góc nghiêng FFHQ
              </p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                data-testid="toggle-padding"
                checked={adaptivePadding}
                onChange={(e) => setAdaptivePadding(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-[#E5E7EB] peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[#E8804A]"></div>
            </label>
          </div>

          {/* Khối 2: Cân bằng trắng */}
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl p-3.5 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-[#111827]">Khối 2: Cân bằng trắng</span>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                      whiteBalance
                        ? 'bg-[#EFF6FF] text-[#3B82C7] border border-[#BFDBFE]'
                        : 'bg-gray-100 text-gray-500'
                    }`}
                  >
                    {whiteBalance ? (refPoint ? 'Đã chọn điểm' : 'BẬT (Chờ chọn điểm)') : 'TẮT'}
                  </span>
                </div>
                <p className="text-[11px] text-[#6B7280] mt-0.5">
                  Khử ám sắc cục bộ theo điểm tham chiếu trắng/xám người dùng tự chọn
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  data-testid="toggle-whitebalance"
                  checked={whiteBalance}
                  onChange={(e) => {
                    setWhiteBalance(e.target.checked);
                    if (!e.target.checked) setIsEyedropperActive(false);
                  }}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-[#E5E7EB] peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[#E8804A]"></div>
              </label>
            </div>

            {whiteBalance && (
              <div className="pt-2 border-t border-[#E5E7EB] space-y-2.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <button
                    type="button"
                    data-testid="eyedropper-btn"
                    onClick={() => setIsEyedropperActive((prev) => !prev)}
                    className={`flex items-center gap-1.5 text-xs font-semibold px-3 py-2 rounded-lg border transition-all cursor-pointer shadow-2xs hover-lift ${
                      isEyedropperActive
                        ? 'bg-[#E8804A] text-white border-[#E8804A] ring-2 ring-[#E8804A]/30'
                        : 'bg-white hover:bg-[#F9FAFB] text-[#111827] border-[#E5E7EB]'
                    }`}
                  >
                    <Pipette className="w-3.5 h-3.5 text-[#D97706]" />
                    <span>
                      {isEyedropperActive
                        ? 'Đang chờ click trên ảnh gốc...'
                        : 'Chọn điểm tham chiếu trắng/xám trên ảnh'}
                    </span>
                  </button>
                  {refPoint && (
                    <button
                      type="button"
                      data-testid="reselect-point-btn"
                      onClick={() => setIsEyedropperActive(true)}
                      className="text-xs text-[#E8804A] hover:underline font-semibold cursor-pointer"
                    >
                      Chọn lại
                    </button>
                  )}
                </div>

                {refPoint ? (
                  <div className="flex items-center gap-3 bg-white border border-[#E5E7EB] rounded-lg p-2 text-xs">
                    {sampledRgb && (
                      <div
                        className="w-4 h-4 rounded border border-gray-300 shadow-2xs flex-shrink-0"
                        style={{ backgroundColor: `rgb(${sampledRgb.r}, ${sampledRgb.g}, ${sampledRgb.b})` }}
                        title={`Sampled RGB: (${sampledRgb.r}, ${sampledRgb.g}, ${sampledRgb.b})`}
                      />
                    )}
                    <span className="text-[#111827] font-medium" data-testid="ref-point-label">
                      Điểm tham chiếu: <strong className="font-mono">({refPoint.x}, {refPoint.y})</strong>
                    </span>
                    {wbGains && (
                      <span className="text-[#6B7280] text-[11px] font-mono ml-auto" data-testid="wb-gains-label">
                        Gains: R:{wbGains.r.toFixed(2)} G:{wbGains.g.toFixed(2)} B:{wbGains.b.toFixed(2)}
                      </span>
                    )}
                  </div>
                ) : (
                  <p className="text-[11px] text-[#6B7280]">
                    * Gợi ý: Bấm nút trên rồi click vào cổ áo trắng, răng hoặc vùng xám trung tính trên ảnh.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Khối 3: Độ nét CodeFormer */}
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[#111827] font-bold flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-[#E8804A]" />
                Khối 3: Độ nét CodeFormer (Fidelity Weight w):
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
        </div>
      ) : (
        /* CHẾ ĐỘ TỰ ĐỘNG: Giữ nguyên 100% giao diện slider + 2 checkbox cũ */
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
      )}

      {/* 2 Nút Hành động */}
      <div className="flex flex-col sm:flex-row gap-3 pt-1">
        <button
          type="button"
          data-testid="confirm-restore-btn"
          onClick={() =>
            onConfirm(true, fidelity, {
              mode,
              whiteBalance,
              adaptivePadding,
              refPoint,
              wbGains: wbGains ? [wbGains.r, wbGains.g, wbGains.b] : null,
            })
          }
          className="flex-1 flex items-center justify-center gap-2 bg-[#E8804A] hover:bg-[#D97706] text-white font-bold text-xs py-3 px-4 rounded-xl transition-all shadow-xs hover:shadow hover-lift cursor-pointer"
        >
          <CheckCircle2 className="w-4 h-4" />
          <span>Dùng ảnh đã khôi phục</span>
        </button>

        <button
          type="button"
          onClick={() =>
            onConfirm(false, fidelity, {
              mode,
              whiteBalance,
              adaptivePadding,
              refPoint,
              wbGains: wbGains ? [wbGains.r, wbGains.g, wbGains.b] : null,
            })
          }
          className="flex-1 flex items-center justify-center gap-2 bg-white hover:bg-[#F9FAFB] text-[#111827] font-medium text-xs py-3 px-4 rounded-xl border border-[#E5E7EB] transition-all shadow-xs hover-lift cursor-pointer"
        >
          <span>Bỏ qua, dùng ảnh gốc</span>
          <ArrowRight className="w-4 h-4 text-[#6B7280]" />
        </button>
      </div>
    </div>
  );
};
