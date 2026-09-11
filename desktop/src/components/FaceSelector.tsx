import React, { useRef, useState, useEffect } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { CheckCircle2, UserCheck } from 'lucide-react';

export const FaceSelector: React.FC = () => {
  const store = useSearchStore();
  const { selectFace } = useSearchApi();
  const imgRef = useRef<HTMLImageElement>(null);
  const [naturalDim, setNaturalDim] = useState<{ w: number; h: number } | null>(null);
  const [displayDim, setDisplayDim] = useState<{ w: number; h: number } | null>(null);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setNaturalDim({ w: img.naturalWidth, h: img.naturalHeight });
    setDisplayDim({ w: img.clientWidth, h: img.clientHeight });

    // Tự động chọn nếu chỉ có đúng 1 khuôn mặt
    if (store.faces.length === 1 && store.sessionId && store.selectedFaceIdx === null) {
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
    <div className="bg-[#1B2129] border border-[#262E38] rounded-xl p-5 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-semibold text-[#E8E6E0] flex items-center gap-2">
            <UserCheck className="w-5 h-5 text-[#C97B4A]" />
            {store.faces.length > 1
              ? `Phát hiện ${store.faces.length} khuôn mặt - Vui lòng click chọn người cần tìm`
              : 'Đã phát hiện 1 khuôn mặt'}
          </h3>
          <p className="text-xs text-[#8E98A5] mt-0.5">
            {store.faces.length > 1
              ? 'Click trực tiếp vào khung khuôn mặt để chọn đối tượng tìm kiếm.'
              : 'Hệ thống tự động chọn khuôn mặt duy nhất trong ảnh.'}
          </p>
        </div>
      </div>

      <div className="flex flex-col md:flex-row gap-5 items-start">
        {/* Ảnh gốc với SVG Overlay Bounding Boxes */}
        <div className="relative inline-block border border-[#262E38] rounded-lg overflow-hidden max-w-full bg-black/40">
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
                      fill={isSelected ? 'rgba(74, 143, 160, 0.25)' : 'rgba(184, 86, 74, 0.2)'}
                      stroke={isSelected ? '#4A8FA0' : '#B8564A'}
                      strokeWidth={isSelected ? 3 : 2}
                      className="transition-all duration-200 group-hover:stroke-[#C9A24A] group-hover:fill-[#C9A24A]/20"
                    />
                    {/* Badge số thứ tự */}
                    <rect
                      x={sx1}
                      y={Math.max(0, sy1 - 22)}
                      width={44}
                      height={20}
                      fill={isSelected ? '#4A8FA0' : '#B8564A'}
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
          <div className="bg-[#12161C] p-4 rounded-lg border border-[#262E38] flex flex-col items-center">
            <span className="text-xs uppercase tracking-wider text-[#4A8FA0] font-semibold mb-2 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#4A8FA0]" />
              Mặt căn chỉnh FFHQ (256x256)
            </span>
            <img
              src={
                store.croppedPreviewUrl.startsWith('/') || store.croppedPreviewUrl.startsWith('data:')
                  ? store.croppedPreviewUrl
                  : `http://127.0.0.1:${store.backendPort}${store.croppedPreviewUrl}`
              }
              alt="Cropped Face"
              className="w-36 h-36 rounded-lg border-2 border-[#4A8FA0] shadow object-cover bg-black"
            />
            <p className="text-xs text-[#8E98A5] mt-2 text-center">
              Khuôn mặt #{(store.selectedFaceIdx ?? 0) + 1} (Tin cậy: {(
                (store.faces[store.selectedFaceIdx ?? 0]?.det_score || 0) * 100
              ).toFixed(1)}%)
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
