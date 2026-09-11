import React, { useState } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { FaceSelector } from '../components/FaceSelector';
import { QualityWarnings } from '../components/QualityWarnings';
import { AgeInputForm } from '../components/AgeInputForm';
import { GalleryPicker } from '../components/GalleryPicker';
import { ProgressStages } from '../components/ProgressStages';
import { ResultsGallery } from '../components/ResultsGallery';
import { UploadCloud, Play, RotateCcw, ShieldCheck, Loader2 } from 'lucide-react';

export const SearchPage: React.FC = () => {
  const store = useSearchStore();
  const { uploadImage, runPipeline } = useSearchApi();
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    try {
      await uploadImage(file);
    } catch (err: any) {
      console.error(err);
      const isNetworkError = !err.response || err.code === 'ERR_NETWORK';
      const msg = isNetworkError
        ? 'Không thể kết nối đến Backend API server (Port 8000). Vui lòng đảm bảo bạn đã khởi động backend bằng lệnh: python -m backend.api.main'
        : (err.response?.data?.detail || 'Không thể xử lý ảnh tải lên.');
      setUploadError(msg);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRun = async () => {
    if (!store.sessionId) return;
    try {
      await runPipeline(store.sessionId, store.galleryDir);
    } catch (err: any) {
      console.error(err);
      const msg = err.response?.data?.detail || 'Không thể bắt đầu pipeline.';
      alert(msg);
    }
  };

  const isReadyToRun =
    store.sessionId &&
    store.croppedPreviewUrl &&
    store.initialAge !== null &&
    store.jobStatus !== 'running';

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-8">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-sky-300 to-emerald-400">
              Missing Person Search via FADING
            </h1>
            <span className="bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs px-2.5 py-0.5 rounded-full font-mono font-semibold">
              v2.0 Desktop
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Hệ thống nhận diện & tìm kiếm người mất tích sử dụng mô hình tạo ảnh khuếch tán FADING (Dual-Attention)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-full text-xs font-mono text-emerald-400">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Port: {store.backendPort}</span>
          </div>

          {store.sessionId && (
            <button
              onClick={() => store.resetForNewUpload()}
              className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3.5 py-2 rounded-xl border border-slate-700 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Làm mới</span>
            </button>
          )}
        </div>
      </header>

      {/* Bước 1: Upload ảnh (nếu chưa upload) */}
      {!store.sessionId && (
        <div className="bg-slate-800/60 border-2 border-dashed border-slate-700 hover:border-indigo-500/80 rounded-2xl p-12 text-center transition-all duration-300 group shadow-xl">
          <input
            type="file"
            id="file-upload"
            accept="image/png, image/jpeg, image/jpg"
            onChange={handleFileChange}
            className="hidden"
            disabled={isUploading}
          />
          <label htmlFor="file-upload" className="cursor-pointer block">
            <div className="w-16 h-16 bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
              {isUploading ? (
                <Loader2 className="w-8 h-8 animate-spin" />
              ) : (
                <UploadCloud className="w-8 h-8" />
              )}
            </div>
            <h3 className="text-lg font-bold text-slate-200 mb-1">
              {isUploading ? 'Đang tải và phát hiện khuôn mặt...' : 'Chọn hoặc kéo thả ảnh cần tìm'}
            </h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Hỗ trợ PNG, JPG, JPEG. Hệ thống sẽ tự động phát hiện khuôn mặt và kiểm tra chất lượng góc nghiêng.
            </p>
          </label>

          {uploadError && (
            <div className="mt-4 inline-block bg-rose-950/60 border border-rose-500/50 text-rose-300 text-xs px-4 py-2 rounded-lg">
              {uploadError}
            </div>
          )}
        </div>
      )}

      {/* Bước 2: Wizard cấu hình khi đã có ảnh */}
      {store.sessionId && (
        <div className="space-y-6">
          <FaceSelector />
          <QualityWarnings />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
            <AgeInputForm />
            <div className="space-y-6">
              <GalleryPicker />

              {/* Nút hành động Chạy Pipeline */}
              <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-6 shadow-xl space-y-4">
                <div>
                  <h4 className="text-sm font-semibold text-slate-200">Sẵn sàng khởi động quy trình</h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Pipeline sẽ chạy tuần tự qua Inversion, Editing độ tuổi và FAISS Search.
                  </p>
                </div>

                <button
                  onClick={handleRun}
                  disabled={!isReadyToRun}
                  className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-indigo-600 via-indigo-500 to-emerald-600 hover:from-indigo-500 hover:to-emerald-500 text-white font-bold py-3.5 px-6 rounded-xl transition-all shadow-lg hover:shadow-indigo-500/25 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {store.jobStatus === 'running' ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Pipeline Đang Chạy...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 fill-current" />
                      <span>Chạy Pipeline FADING</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          <ProgressStages />
          <ResultsGallery />
        </div>
      )}
    </div>
  );
};
