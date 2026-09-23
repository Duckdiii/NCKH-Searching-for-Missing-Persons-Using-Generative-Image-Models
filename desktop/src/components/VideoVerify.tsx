import React, { useState } from 'react';
import { Clapperboard, UploadCloud, Trophy, Loader2 } from 'lucide-react';
import { api } from '../api/client';
import { useSearchStore } from '../store/useSearchStore';
import { VideoFaceMatch, VideoVerifyResponse } from '../types/api';
import { ImageWithSkeleton } from './ImageWithSkeleton';
import { CountUpNumber } from './CountUpNumber';

interface Props {
  jobId: string;
}

export const VideoVerify: React.FC<Props> = ({ jobId }) => {
  const { backendPort } = useSearchStore();
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VideoVerifyResponse | null>(null);

  const baseUrl = `http://127.0.0.1:${backendPort}`;
  const resolveUrl = (path?: string | null) => {
    if (!path) return '';
    if (path.startsWith('data:') || path.startsWith('http://') || path.startsWith('https://')) return path;
    return `${baseUrl}${path.startsWith('/') ? path : '/' + path}`;
  };

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await api.post<VideoVerifyResponse>(`/api/jobs/${jobId}/video-verify`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000,
      });
      setResult(res.data);
    } catch (err: any) {
      const status = err.response?.status;
      let detail: unknown = err.response?.data?.detail;
      if (Array.isArray(detail)) {
        detail = detail.map((d: any) => d?.msg || JSON.stringify(d)).join('; ');
      }
      if (status === 404) {
        setError(
          'Không tìm thấy ảnh FADING của job này trên backend (job demo, hoặc backend vừa khởi động lại mất phiên cũ). Hãy chạy lại pipeline FADING rồi thử lại.'
        );
      } else if (err.code === 'ECONNABORTED') {
        setError(
          'Xử lý video quá lâu (hết 5 phút chờ). Hãy thử video ngắn hơn (dưới 30s, rõ mặt) rồi tải lại.'
        );
      } else {
        setError((detail as string) || 'Không thể xử lý video. Hãy thử file .mp4 (H.264) khác, ngắn dưới 30s.')
      }
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  const best: VideoFaceMatch | null | undefined = result?.best_match;
  const matches: VideoFaceMatch[] = result?.matches ?? [];

  return (
    <div data-testid="video-verify-card" className="bg-white border border-[#E5E7EB] rounded-xl p-5 sm:p-6 shadow-xs space-y-4">
      <div className="flex items-center gap-2 border-b border-[#E5E7EB] pb-3">
        <Clapperboard className="w-4 h-4 text-[#E8804A]" />
        <div>
          <h4 className="text-sm font-bold text-[#111827]">Đối soát bổ sung bằng video</h4>
          <p className="text-[11px] text-[#6B7280] mt-0.5">
            Tải video lên — hệ thống tự cắt các khuôn mặt trong video và so với ảnh FADING đã sinh, hiện mặt khớp nhất.
            Mỗi giây lấy 3 frame (tối đa ~30s video).
          </p>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        <label className="inline-flex items-center gap-2 bg-[#E8804A] hover:bg-[#D97706] text-white text-xs font-bold py-2.5 px-4 rounded-xl cursor-pointer transition-colors shadow-xs">
          {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <UploadCloud className="w-4 h-4" />}
          <span>{isUploading ? 'Đang trích frame & so khớp...' : 'Tải video lên (.mp4, .avi, .mov, ≤200MB)'}</span>
          <input type="file" accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm" onChange={handleFile} className="hidden" disabled={isUploading} />
        </label>
        {result && (
          <p className="text-[11px] text-[#6B7280]">
            Đã quét {result.frames_sampled} frame • phát hiện {result.faces_found} khuôn mặt
          </p>
        )}
      </div>

      {error && <div className="bg-[#FEF2F2] border border-[#FECACA] text-[#DC2626] text-xs px-3.5 py-2 rounded-lg">{error}</div>}

      {result && !best && (
        <p className="text-xs text-[#D97706] bg-[#FEF3C7] border border-[#FDE68A] rounded-lg px-3 py-2">
          Không phát hiện khuôn mặt nào trong video (đã quét {result.frames_sampled} frame). Hãy thử video rõ mặt, chính diện hơn.
        </p>
      )}

      {best && (
        <div className="space-y-3">
          <div className="bg-[#EFF6FF] border border-[#BFDBFE] rounded-xl p-4 grid grid-cols-1 sm:grid-cols-3 gap-4 items-center">
            <div className="text-center">
              <p className="text-[11px] font-semibold text-[#6B7280] uppercase mb-2">Mặt trong video (khớp nhất)</p>
              <ImageWithSkeleton src={resolveUrl(best.face_image_url)} alt="Best video face" className="w-32 h-32 object-cover rounded-lg mx-auto border border-[#BFDBFE]" fallbackText="Mặt video" />
              <p className="text-[11px] text-[#6B7280] mt-1 font-mono">frame #{best.frame_index} • {best.timestamp_sec}s</p>
            </div>
            <div className="text-center">
              <Trophy className="w-6 h-6 text-[#E8804A] mx-auto mb-1" />
              <p className="text-sm font-bold text-[#111827]">
                <CountUpNumber targetValue={best.score * 100} decimals={1} suffix="%" />
              </p>
              <p className="text-[11px] text-[#6B7280]">độ tương đồng với ảnh sinh {best.best_age} tuổi</p>
            </div>
            <div className="text-center">
              <p className="text-[11px] font-semibold text-[#6B7280] uppercase mb-2">Ảnh FADING {best.best_age} tuổi</p>
              <ImageWithSkeleton src={resolveUrl(best.best_age_image_url)} alt={`FADING ${best.best_age}`} className="w-32 h-32 object-cover rounded-lg mx-auto border border-[#E8804A]" fallbackText={`${best.best_age} tuổi`} />
            </div>
          </div>

          {matches.length > 1 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 gap-2.5">
              {matches.slice(0, 10).map((m, i) => (
                <div key={`${m.frame_index}-${i}`} className={`bg-[#F9FAFB] border rounded-lg overflow-hidden ${i === 0 ? 'border-[#E8804A] ring-2 ring-[#E8804A]/30' : 'border-[#E5E7EB]'}`}>
                  <ImageWithSkeleton src={resolveUrl(m.face_image_url)} alt={`face ${i + 1}`} className="w-full aspect-square object-cover" fallbackText={`#${i + 1}`} />
                  <p className="text-[10px] text-center py-1 font-mono text-[#111827]">
                    {(m.score * 100).toFixed(1)}% • {m.best_age}t{i === 0 ? ' ★' : ''}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
