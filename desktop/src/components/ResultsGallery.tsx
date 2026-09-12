import React, { useState, useEffect } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { CheckCircle2, XCircle, Award, ZoomIn, X, SplitSquareVertical } from 'lucide-react';

interface LightboxState {
  url: string;
  title: string;
  subtitle?: string;
}

export const ResultsGallery: React.FC = () => {
  const { jobResult, backendPort, croppedPreviewUrl, uploadedImageUrl } = useSearchStore();
  const [lightboxImg, setLightboxImg] = useState<LightboxState | null>(null);

  // Lắng nghe phím Esc để đóng Lightbox
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setLightboxImg(null);
      }
    };
    if (lightboxImg) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [lightboxImg]);

  if (!jobResult || jobResult.status !== 'done') {
    return null;
  }

  const {
    edited_images,
    final_scores,
    accepted,
    top_identity,
    top_score,
    best_age,
    matched_gallery_image,
  } = jobResult;

  const baseUrl = `http://127.0.0.1:${backendPort}`;

  const resolveUrl = (path?: string | null) => {
    if (!path) return '';
    if (path.startsWith('/') || path.startsWith('data:') || path.startsWith('http://') || path.startsWith('https://')) {
      return path;
    }
    return `${baseUrl}${path}`;
  };

  // 1. Ảnh gốc lúc nhỏ
  const originalFaceUrl = resolveUrl(croppedPreviewUrl) || uploadedImageUrl || '';

  // 2. Mốc tuổi và ảnh FADING khớp nhất
  const ages = Object.keys(edited_images).map((a) => parseInt(a)).sort((a, b) => a - b);
  const matchedAge = best_age ?? (ages.length > 0 ? ages[Math.floor(ages.length / 2)] : 50);
  const bestEditedUrl = resolveUrl(edited_images[matchedAge]) || resolveUrl(Object.values(edited_images)[0]) || '';

  // 3. Ảnh đối soát trong Gallery
  const galleryMatchUrl = resolveUrl(matched_gallery_image) ||
    (top_identity ? `${baseUrl}/data/test_gallery/${top_identity}.png` : '');

  // Badge ID Score theo 3 ngưỡng: trust (>=70%), warning (30-70%), error (<30%)
  // Badge ID Score theo 3 ngưỡng: trust (>=60%), warning (30-60%), error (<30%)
  const getIdScoreBadge = (score: number) => {
    const pct = score * 100;
    if (pct >= 60) {
      return {
        label: `ID Score: ${pct.toFixed(1)}%`,
        scorePct: `${pct.toFixed(1)}%`,
        statusText: 'Khớp cao',
        colorClass: 'bg-[#EFF6FF] text-[#3B82C7] border border-[#BFDBFE]',
      };
    } else if (pct >= 30) {
      return {
        label: `ID Score: ${pct.toFixed(1)}%`,
        scorePct: `${pct.toFixed(1)}%`,
        statusText: 'Cần thẩm tra',
        colorClass: 'bg-[#FEF3C7] text-[#D97706] border border-[#FDE68A]',
      };
    } else {
      return {
        label: `ID Score: ${pct.toFixed(1)}%`,
        scorePct: `${pct.toFixed(1)}%`,
        statusText: 'Không khớp',
        colorClass: 'bg-[#FEF2F2] text-[#DC2626] border border-[#FECACA]',
      };
    }
  };

  const idScoreBadge = getIdScoreBadge(top_score);

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 sm:p-6 shadow-xs space-y-6 animate-in fade-in duration-300">
      {/* 1. Banner kết quả chấp nhận / từ chối */}
      <div
        className={`p-4 sm:p-5 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
          accepted
            ? 'bg-[#EFF6FF] border-[#BFDBFE] text-[#111827]'
            : 'bg-[#FEF2F2] border-[#FECACA] text-[#111827]'
        }`}
      >
        <div className="flex items-center gap-3">
          {accepted ? (
            <CheckCircle2 className="w-7 h-7 text-[#3B82C7] flex-shrink-0" />
          ) : (
            <XCircle className="w-7 h-7 text-[#DC2626] flex-shrink-0" />
          )}
          <div>
            <h4 className="text-base sm:text-lg font-bold">
              {accepted
                ? `Tìm thấy đối tượng phù hợp: "${top_identity}"`
                : 'Không tìm thấy kết quả đủ tin cậy trong Gallery'}
            </h4>
            <p className="text-xs text-[#6B7280] mt-0.5">
              {accepted
                ? `Độ tương đồng cao nhất: ${(top_score * 100).toFixed(1)}% (Vượt ngưỡng chấp nhận 60%)`
                : `Điểm cao nhất: ${(top_score * 100).toFixed(1)}% (Thấp hơn ngưỡng tin cậy 60%)`}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Badge ID Score theo 3 ngưỡng token */}
          <div className={`px-3 py-1.5 rounded-full text-xs font-bold ${idScoreBadge.colorClass}`}>
            {idScoreBadge.label} • {idScoreBadge.statusText}
          </div>

          {accepted && (
            <div className="flex items-center gap-1.5 bg-[#EFF6FF] border border-[#BFDBFE] px-3 py-1.5 rounded-full text-xs font-semibold text-[#3B82C7]">
              <Award className="w-4 h-4" />
              <span>Xác thực thành công</span>
            </div>
          )}
        </div>
      </div>

      {/* Lưới 2 cột cho Bước 3: Cột trái (Hero Card + Lưới ảnh già hóa), Cột phải (Bảng xếp hạng) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Cột trái (~65%): Hero Card 3 ảnh + Lưới mốc tuổi khác */}
        <div className="lg:col-span-8 space-y-6">
          {/* 2. Hero Card: So sánh trực quan 3 ảnh theo hàng ngang */}
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl p-4 sm:p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between border-b border-[#E5E7EB] pb-3">
              <div className="flex items-center gap-2 text-sm font-bold text-[#111827]">
                <SplitSquareVertical className="w-4 h-4 text-[#E8804A]" />
                <span>Bằng chứng đối soát trực quan (Side-by-Side Comparison)</span>
              </div>
              <span className="text-xs text-[#6B7280] hidden sm:inline">Click ảnh để phóng to</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
              {/* Cột 1: Ảnh gốc */}
              <div className="bg-white border border-[#E5E7EB] rounded-lg p-3 flex flex-col items-center">
                <span className="text-xs font-semibold text-[#6B7280] mb-2 uppercase tracking-wide">
                  1. Ảnh gốc lúc nhỏ
                </span>
                <div
                  onClick={() =>
                    originalFaceUrl &&
                    setLightboxImg({
                      url: originalFaceUrl,
                      title: 'Ảnh gốc / Căn chỉnh khuôn mặt',
                      subtitle: 'Khuôn mặt đối tượng khi mất tích (chuẩn hóa FFHQ 256x256)',
                    })
                  }
                  className="relative w-full aspect-square bg-gray-100 rounded-lg overflow-hidden cursor-pointer group border border-[#E5E7EB] hover:border-[#E8804A] transition-all"
                >
                  {originalFaceUrl ? (
                    <img
                      src={originalFaceUrl}
                      alt="Ảnh gốc"
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-xs text-[#6B7280]">
                      Chưa có ảnh
                    </div>
                  )}
                  <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white">
                    <ZoomIn className="w-6 h-6" />
                  </div>
                </div>
                <p className="text-xs text-[#6B7280] mt-2 text-center">Đối tượng ban đầu</p>
              </div>

              {/* Cột 2: Ảnh FADING dự đoán khớp nhất */}
              <div className="bg-white border border-[#E5E7EB] rounded-lg p-3 flex flex-col items-center">
                <span className="text-xs font-semibold text-[#E8804A] mb-2 uppercase tracking-wide">
                  2. FADING dự đoán ({matchedAge} tuổi)
                </span>
                <div
                  onClick={() =>
                    bestEditedUrl &&
                    setLightboxImg({
                      url: bestEditedUrl,
                      title: `Ảnh FADING già hóa dự đoán (${matchedAge} tuổi)`,
                      subtitle: 'Sinh bởi mạng khuếch tán Dual-Attention bảo toàn danh tính',
                    })
                  }
                  className="relative w-full aspect-square bg-gray-100 rounded-lg overflow-hidden cursor-pointer group border border-[#E5E7EB] hover:border-[#E8804A] transition-all"
                >
                  {bestEditedUrl ? (
                    <img
                      src={bestEditedUrl}
                      alt={`Dự đoán ${matchedAge} tuổi`}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-xs text-[#6B7280]">
                      Đang xử lý
                    </div>
                  )}
                  <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white">
                    <ZoomIn className="w-6 h-6" />
                  </div>
                </div>
                <p className="text-xs text-[#E8804A] font-medium mt-2 text-center">
                  Mốc tuổi tương đồng cao nhất
                </p>
              </div>

              {/* Cột 3: Ảnh trong Gallery */}
              <div className="bg-white border border-[#E5E7EB] rounded-lg p-3 flex flex-col items-center">
                <span className="text-xs font-semibold text-[#3B82C7] mb-2 uppercase tracking-wide">
                  3. Ảnh Gallery ({top_identity || 'Đối soát'})
                </span>
                <div
                  onClick={() =>
                    galleryMatchUrl &&
                    setLightboxImg({
                      url: galleryMatchUrl,
                      title: `Ảnh đối soát trong Gallery (${top_identity})`,
                      subtitle: `Độ tương đồng cosine: ${(top_score * 100).toFixed(1)}%`,
                    })
                  }
                  className="relative w-full aspect-square bg-gray-100 rounded-lg overflow-hidden cursor-pointer group border border-[#E5E7EB] hover:border-[#3B82C7] transition-all"
                >
                  {galleryMatchUrl ? (
                    <img
                      src={galleryMatchUrl}
                      alt={`Gallery ${top_identity}`}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = 'none';
                      }}
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-xs text-[#6B7280]">
                      Chưa tải được
                    </div>
                  )}
                  <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white">
                    <ZoomIn className="w-6 h-6" />
                  </div>
                </div>
                {/* Badge ID Score */}
                <div className={`mt-2 px-3 py-1 rounded text-xs font-bold text-center ${idScoreBadge.colorClass}`}>
                  {idScoreBadge.label}
                </div>
              </div>
            </div>
          </div>

          {/* 3. Khối ảnh già hóa duy nhất (Gộp 2 khối cũ thành 1) */}
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl p-4 sm:p-5 shadow-xs space-y-3.5">
            <div className="flex items-center justify-between border-b border-[#E5E7EB] pb-3">
              <h4 className="text-xs font-bold text-[#111827] uppercase tracking-wider flex items-center gap-2">
                <span>Khuôn mặt dự đoán qua các độ tuổi (FADING Dual-Attention):</span>
              </h4>
              <span className="text-[11px] text-[#6B7280]">Click để xem lớn</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(edited_images).map(([ageStr, relPath]) => {
                const ageNum = parseInt(ageStr);
                const isBest = ageNum === matchedAge;
                const badge = getIdScoreBadge(top_score);
                return (
                  <div
                    key={ageStr}
                    onClick={() =>
                      setLightboxImg({
                        url: resolveUrl(relPath),
                        title: `Khuôn mặt dự đoán mốc ${ageStr} tuổi`,
                        subtitle: isBest
                          ? 'Mốc tuổi có độ tương đồng cao nhất'
                          : 'Sinh bởi mạng khuếch tán Dual-Attention',
                      })
                    }
                    className={`bg-white border rounded-xl overflow-hidden shadow-xs group transition-all cursor-pointer ${
                      isBest
                        ? 'border-[#E8804A] ring-2 ring-[#E8804A]/30'
                        : 'border-[#E5E7EB] hover:border-[#E8804A]'
                    }`}
                  >
                    <div className="relative aspect-square overflow-hidden bg-gray-100">
                      <img
                        src={resolveUrl(relPath)}
                        alt={`${ageStr} tuổi`}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                      />
                      {/* Badge tuổi + % góc dưới trái với màu theo ngưỡng */}
                      <div
                        className={`absolute bottom-1.5 left-1.5 px-2 py-0.5 rounded text-[10px] font-bold shadow-xs ${
                          isBest
                            ? badge.colorClass
                            : 'bg-white/95 text-[#111827] border border-[#E5E7EB]'
                        }`}
                      >
                        {ageStr}t • {(top_score * 100).toFixed(0)}%
                      </div>

                      <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white">
                        <ZoomIn className="w-5 h-5" />
                      </div>
                    </div>
                    <div className="p-2 text-center bg-white border-t border-[#E5E7EB]">
                      <span
                        className={`text-xs font-semibold ${
                          isBest ? 'text-[#E8804A]' : 'text-[#111827]'
                        }`}
                      >
                        {ageStr} tuổi {isBest && '★'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Cột phải (~35%): Bảng xếp hạng độ tương đồng nhận diện */}
        <div className="lg:col-span-4 lg:sticky lg:top-6 space-y-4">
          <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-xl p-4 sm:p-5 shadow-xs space-y-3.5">
            <div>
              <h4 className="text-xs font-bold text-[#111827] uppercase tracking-wider">
                Bảng xếp hạng độ tương đồng
              </h4>
              <p className="text-[11px] text-[#6B7280] mt-0.5">
                Đối soát không gian vector FAISS
              </p>
            </div>

            <div className="overflow-x-auto rounded-lg border border-[#E5E7EB] bg-white">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-[#F9FAFB] text-[#6B7280] border-b border-[#E5E7EB]">
                    <th className="py-2.5 px-3 font-semibold">#</th>
                    <th className="py-2.5 px-3 font-semibold">Định danh Gallery</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Điểm khớp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E5E7EB] bg-white">
                  {Object.entries(final_scores)
                    .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
                    .map(([id, score], idx) => {
                      const isMatched = id === top_identity && accepted;
                      const rowBadge = getIdScoreBadge(score);
                      return (
                        <tr
                          key={id}
                          className={`hover:bg-[#F9FAFB] transition-colors ${
                            isMatched ? 'bg-[#EFF6FF]' : ''
                          }`}
                        >
                          <td className="py-2.5 px-3 font-medium text-[#6B7280]">
                            #{idx + 1}
                          </td>
                          <td
                            className="py-2.5 px-3 font-mono text-[#111827] truncate max-w-[130px]"
                            title={id}
                          >
                            {id}
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            <span
                              className={`inline-block px-2 py-0.5 rounded text-[11px] font-mono font-semibold ${rowBadge.colorClass}`}
                            >
                              {(score * 100).toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* 5. Lightbox Modal Phóng To Ảnh */}
      {lightboxImg && (
        <div
          onClick={() => setLightboxImg(null)}
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4 animate-in fade-in duration-200"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative bg-white border border-[#E5E7EB] rounded-2xl p-4 max-w-2xl w-full max-h-[90vh] flex flex-col items-center shadow-2xl space-y-3"
          >
            {/* Nút đóng */}
            <button
              onClick={() => setLightboxImg(null)}
              aria-label="Đóng"
              className="absolute top-3 right-3 text-[#6B7280] hover:text-[#111827] bg-[#F3F4F6] p-2 rounded-full border border-[#E5E7EB] transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Khung ảnh phóng to */}
            <div className="w-full flex items-center justify-center overflow-hidden rounded-lg bg-gray-100 max-h-[70vh]">
              <img
                src={lightboxImg.url}
                alt={lightboxImg.title}
                className="max-h-[68vh] max-w-full object-contain rounded"
              />
            </div>

            {/* Chú thích ảnh */}
            <div className="text-center">
              <h5 className="text-sm font-bold text-[#111827]">{lightboxImg.title}</h5>
              {lightboxImg.subtitle && (
                <p className="text-xs text-[#6B7280] mt-0.5">{lightboxImg.subtitle}</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

