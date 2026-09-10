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
    <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
            <UserCheck className="w-5 h-5 text-indigo-400" />
            {store.faces.length > 1
              ? `Phát hiện ${store.faces.length} khuôn mặt - Vui lòng click chọn người cần tìm`
              : 'Đã phát hiện 1 khuôn mặt'}
          </h3>
          <p className="text-sm text-slate-400">
            {store.faces.length > 1
              ? 'Click trực tiếp vào khung đỏ của khuôn mặt để chọn đối tượng tìm kiếm.'
              : 'Hệ thống tự động chọn khuôn mặt duy nhất trong ảnh.'}
          </p>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Ảnh gốc với SVG Overlay Bounding Boxes */}
        <div className="relative inline-block border border-slate-700 rounded-lg overflow-hidden max-w-full bg-black/40">
          <img
            ref={imgRef}
            src={store.uploadedImageUrl || ''}
            alt="Uploaded"
            onLoad={handleImageLoad}
            className="max-h-[420px] object-contain block"
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
                      fill={isSelected ? 'rgba(34, 197, 94, 0.25)' : 'rgba(239, 68, 68, 0.15)'}
                      stroke={isSelected ? '#22c55e' : '#ef4444'}
                      strokeWidth={isSelected ? 3 : 2}
                      className="transition-all duration-200 group-hover:stroke-yellow-400 group-hover:fill-yellow-400/20"
                    />
                    {/* Badge số thứ tự */}
                    <rect
                      x={sx1}
                      y={Math.max(0, sy1 - 22)}
                      width={44}
                      height={20}
                      fill={isSelected ? '#22c55e' : '#ef4444'}
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
          <div className="flex-1 bg-slate-900/60 p-4 rounded-lg border border-slate-700/80 flex flex-col items-center">
            <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2 flex items-center gap-1">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Mặt đã căn chỉnh FFHQ (256x256)
            </span>
            <img
              src={`http://127.0.0.1:${store.backendPort}${store.croppedPreviewUrl}`}
              alt="Cropped Face"
              className="w-40 h-40 rounded-lg border-2 border-emerald-500 shadow-md object-cover bg-black"
            />
            <p className="text-xs text-slate-400 mt-2 text-center">
              Khuôn mặt #{ (store.selectedFaceIdx ?? 0) + 1 } (Độ tin cậy: {(
                (store.faces[store.selectedFaceIdx ?? 0]?.det_score || 0) * 100
              ).toFixed(1)}%)
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
