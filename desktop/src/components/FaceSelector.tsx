import React, { useRef, useState, useEffect } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { CheckCircle2, UserCheck, AlertTriangle } from 'lucide-react';

export const FaceSelector: React.FC = () => {
  const store = useSearchStore();
  const { selectFace } = useSearchApi();
  const imgRef = useRef<HTMLImageElement>(null);
  const [naturalDim, setNaturalDim] = useState<{ w: number; h: number } | null>(null);
  const [displayDim, setDisplayDim] = useState<{ w: number; h: number } | null>(null);

  // Tự động chọn nếu chỉ có đúng 1 khuôn mặt
  useEffect(() => {
    if (store.faces.length === 1 && store.sessionId && !store.croppedPreviewUrl) {
      selectFace(store.sessionId, 0);
    }
  }, [store.faces.length, store.sessionId, store.croppedPreviewUrl, selectFace]);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setNaturalDim({ w: img.naturalWidth, h: img.naturalHeight });
    setDisplayDim({ w: img.clientWidth, h: img.clientHeight });

    if (store.faces.length === 1 && store.sessionId && !store.croppedPreviewUrl) {
      selectFace(store.sessionId, 0);
    }
  };

  useEffect(() => {
    const handleResize = () => {
      if (imgRef.current) {
        setDisplayDim({
          w: imgRef.current.clientWidth,
          h: imgRef.current.clientHeight,
        });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const scaleX = naturalDim && displayDim ? displayDim.w / naturalDim.w : 1;
  const scaleY = naturalDim && displayDim ? displayDim.h / naturalDim.h : 1;

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-semibold text-[#111827] flex items-center gap-2">
            <UserCheck className="w-5 h-5 text-[#E8804A]" />
            {store.faces.length > 1
              ? `Phát hiện ${store.faces.length} khuôn mặt - Vui lòng click chọn người cần tìm`
              : 'Đã phát hiện 1 khuôn mặt'}
          </h3>
          <p className="text-xs text-[#6B7280] mt-0.5">
            {store.faces.length > 1
              ? 'Click trực tiếp vào khung khuôn mặt để chọn đối tượng tìm kiếm.'
              : 'Hệ thống tự động chọn khuôn mặt duy nhất trong ảnh.'}
          </p>
        </div>
      </div>

      <div className="flex flex-col md:flex-row gap-5 items-start">
        {/* Ảnh gốc với SVG Overlay Bounding Boxes */}
        <div className="relative inline-block border border-[#E5E7EB] rounded-lg overflow-hidden max-w-full bg-[#F9FAFB]">
          <img
            ref={imgRef}
            src={store.uploadedImageUrl || ''}
            alt="Uploaded"
            onLoad={handleImageLoad}
            className="max-h-[380px] object-contain block"
          />

          {naturalDim && displayDim && (
            <svg
              className="absolute top-0 left-0 w-full h-full pointer-events-auto"
              viewBox={`0 0 ${displayDim.w} ${displayDim.h}`}
            >
              {store.faces.map((f) => {
                const [x1, y1, x2, y2] = f.bbox;
                const sx1 = x1 * scaleX;
                const sy1 = y1 * scaleY;
                const sw = (x2 - x1) * scaleX;
                const sh = (y2 - y1) * scaleY;
                const isSelected = store.selectedFaceIdx === f.index;

                return (
                  <g
                    key={f.index}
                    onClick={() => store.sessionId && selectFace(store.sessionId, f.index)}
                    className="cursor-pointer group"
                  >
                    <rect
                      x={sx1}
                      y={sy1}
                      width={sw}
                      height={sh}
                      fill={isSelected ? 'rgba(59, 130, 199, 0.2)' : 'rgba(220, 38, 38, 0.15)'}
                      stroke={isSelected ? '#3B82C7' : '#DC2626'}
                      strokeWidth={isSelected ? 3 : 2}
                      className="transition-all duration-200 group-hover:stroke-[#D97706] group-hover:fill-[#D97706]/20"
                    />
                    {/* Badge số thứ tự */}
                    <rect
                      x={sx1}
                      y={Math.max(0, sy1 - 22)}
                      width={44}
                      height={20}
                      fill={isSelected ? '#3B82C7' : '#DC2626'}
                      rx={4}
                    />
                    <text
                      x={sx1 + 22}
                      y={Math.max(0, sy1 - 22) + 14}
                      fill="#ffffff"
                      fontSize="12"
                      fontWeight="bold"
                      textAnchor="middle"
                    >
                      #{f.index + 1}
                    </text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>

        {/* Cột Preview ảnh mặt đã căn chỉnh FFHQ */}
        {store.croppedPreviewUrl && (
          <div className="bg-[#F9FAFB] p-4 rounded-lg border border-[#E5E7EB] flex flex-col items-center">
            <span className="text-xs uppercase tracking-wider text-[#3B82C7] font-semibold mb-2 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#3B82C7]" />
              Mặt căn chỉnh FFHQ (256x256)
            </span>
            <img
              src={
                store.croppedPreviewUrl.startsWith('/') || store.croppedPreviewUrl.startsWith('data:')
                  ? store.croppedPreviewUrl
                  : `http://127.0.0.1:${store.backendPort}${store.croppedPreviewUrl}`
              }
              alt="Cropped Face"
              className="w-36 h-36 rounded-lg border-2 border-[#3B82C7] shadow-sm object-cover bg-white"
            />
            <p className="text-xs text-[#6B7280] mt-2 text-center">
              Khuôn mặt #{(store.selectedFaceIdx ?? 0) + 1} (Tin cậy: {(
                (store.faces[store.selectedFaceIdx ?? 0]?.det_score || 0) * 100
              ).toFixed(1)}%)
            </p>
          </div>
        )}
      </div>

      {/* Banner cảnh báo chất lượng nhẹ nhàng tích hợp ngay trong card */}
      {store.warnings && store.warnings.length > 0 && (
        <div className="mt-3.5 pt-3 border-t border-[#E5E7EB] space-y-1.5">
          <div className="flex items-center gap-1.5 text-[#D97706] text-xs font-medium">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 text-[#D97706]" />
            <span>Gợi ý tối ưu chất lượng ảnh:</span>
          </div>
          <ul className="space-y-0.5 pl-5 list-disc text-xs text-[#6B7280]">
            {store.warnings.map((w, idx) => (
              <li key={idx}>{w}</li>
            ))}
          </ul>
          <p className="text-[11px] text-[#6B7280] italic pt-0.5">
            Khuyến nghị: Ảnh chính diện, góc nghiêng ≤ 15° và rõ nét sẽ cho kết quả nhận diện FADING tối ưu nhất.
          </p>
        </div>
      )}
    </div>
  );
};
