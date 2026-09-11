import React, { useState, useEffect } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { FaceSelector } from '../components/FaceSelector';
import { QualityWarnings } from '../components/QualityWarnings';
import { AgeInputForm } from '../components/AgeInputForm';
import { GalleryPicker } from '../components/GalleryPicker';
import { ProgressStages } from '../components/ProgressStages';
import { ResultsGallery } from '../components/ResultsGallery';
import {
  UploadCloud,
  Play,
  RotateCcw,
  ShieldCheck,
  Loader2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  UserCheck,
  Settings2,
} from 'lucide-react';

export const SearchPage: React.FC = () => {
  const store = useSearchStore();
  const { uploadImage, runPipeline } = useSearchApi();
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Trạng thái đóng/mở Accordion cho các bước đã hoàn thành
  const [isFaceSelectorOpen, setIsFaceSelectorOpen] = useState(true);
  const [isConfigOpen, setIsConfigOpen] = useState(true);

  // Khi chọn mặt xong, tự động thu gọn FaceSelector để người dùng làm bước tiếp
  useEffect(() => {
    if (store.selectedFaceIdx !== null) {
      setIsFaceSelectorOpen(false);
    } else {
      setIsFaceSelectorOpen(true);
    }
  }, [store.selectedFaceIdx]);

  // Khi bấm chạy pipeline hoặc đã xong, tự động thu gọn khối cấu hình
  useEffect(() => {
    if (store.jobStatus !== 'idle') {
      setIsConfigOpen(false);
      setIsFaceSelectorOpen(false);
    }
  }, [store.jobStatus]);

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
        ? 'Không thể kết nối đến Backend API server (Port 8000). Vui lòng đảm bảo backend đang chạy.'
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
    Boolean(store.sessionId &&
    store.croppedPreviewUrl &&
    store.initialAge !== null &&
    store.jobStatus !== 'running');

  return (
    <div className="max-w-[760px] mx-auto py-7 px-4 space-y-5">
      {/* Header phẳng, không gradient */}
      <header className="flex items-center justify-between border-b border-[#262E38] pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-bold text-[#E8E6E0]">
              Missing Person Search via <span className="text-[#C97B4A]">FADING</span>
            </h1>
            <span className="bg-[#12161C] border border-[#262E38] text-[#8E98A5] text-[11px] px-2 py-0.5 rounded-full font-mono font-medium">
              v2.0 Desktop
            </span>
          </div>
          <p className="text-xs text-[#8E98A5] mt-1">
            Hệ thống nhận diện & tìm kiếm người mất tích qua mô hình khuếch tán Dual-Attention
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5 bg-[#12161C] border border-[#262E38] px-3 py-1.5 rounded-full text-xs font-mono text-[#4A8FA0]">
            <ShieldCheck className="w-3.5 h-3.5 text-[#4A8FA0]" />
            <span>Port: {store.backendPort}</span>
          </div>

          {store.sessionId && (
            <button
              onClick={() => store.resetForNewUpload()}
              className="flex items-center gap-1.5 text-xs bg-[#1B2129] hover:bg-[#262E38] text-[#E8E6E0] px-3 py-1.5 rounded-lg border border-[#262E38] transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5 text-[#8E98A5]" />
              <span>Làm mới</span>
            </button>
          )}
        </div>
      </header>

      {/* BƯỚC 1: UPLOAD ẢNH */}
      {!store.sessionId ? (
        <div className="bg-[#1B2129] border-2 border-dashed border-[#262E38] hover:border-[#C97B4A] rounded-2xl p-10 text-center transition-all duration-200 group shadow-lg">
          <input
            type="file"
            id="file-upload"
            accept="image/png, image/jpeg, image/jpg"
            onChange={handleFileChange}
            className="hidden"
            disabled={isUploading}
          />
          <label htmlFor="file-upload" className="cursor-pointer block">
            <div className="w-14 h-14 bg-[#12161C] border border-[#262E38] text-[#C97B4A] rounded-2xl flex items-center justify-center mx-auto mb-3.5 group-hover:scale-105 transition-transform">
              {isUploading ? (
                <Loader2 className="w-7 h-7 animate-spin text-[#C97B4A]" />
              ) : (
                <UploadCloud className="w-7 h-7" />
              )}
            </div>
            <h3 className="text-base font-bold text-[#E8E6E0] mb-1">
              {isUploading ? 'Đang tải và phát hiện khuôn mặt...' : 'Chọn hoặc kéo thả ảnh cần tìm'}
            </h3>
            <p className="text-xs text-[#8E98A5] max-w-sm mx-auto">
              Hỗ trợ PNG, JPG, JPEG. Tự động phát hiện khuôn mặt và kiểm tra chất lượng góc nghiêng.
            </p>
          </label>

          {uploadError && (
            <div className="mt-4 inline-block bg-[#B8564A]/15 border border-[#B8564A]/40 text-[#E8E6E0] text-xs px-3.5 py-2 rounded-lg">
              {uploadError}
            </div>
          )}
        </div>
      ) : (
        /* Accordion Bước 1 đã hoàn thành */
        <div className="bg-[#1B2129] border border-[#262E38] rounded-xl px-4 py-3 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-[#4A8FA0] flex-shrink-0" />
            <span className="font-semibold text-[#E8E6E0]">Bước 1: Đã tải ảnh đầu vào thành công</span>
            <span className="text-[#8E98A5] text-[11px]">({store.faces.length} khuôn mặt phát hiện)</span>
          </div>
          <button
            onClick={() => store.resetForNewUpload()}
            className="text-[11px] text-[#C97B4A] hover:underline flex items-center gap-1 font-medium"
          >
            Tải ảnh khác
          </button>
        </div>
      )}

      {/* BƯỚC 2: CHỌN KHUÔN MẶT & CẢNH BÁO */}
      {store.sessionId && (
        <div className="space-y-4">
          {/* Accordion Header / Summary khi đã chọn mặt */}
          {store.selectedFaceIdx !== null && !isFaceSelectorOpen ? (
            <div
              onClick={() => setIsFaceSelectorOpen(true)}
              className="bg-[#1B2129] border border-[#262E38] hover:border-[#C97B4A]/50 rounded-xl px-4 py-3 flex items-center justify-between text-xs cursor-pointer transition-colors"
            >
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-4 h-4 text-[#4A8FA0] flex-shrink-0" />
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-[#E8E6E0]">
                    Bước 2: Khuôn mặt đối tượng #{store.selectedFaceIdx + 1}
                  </span>
                  {store.croppedPreviewUrl && (
                    <img
                      src={`http://127.0.0.1:${store.backendPort}${store.croppedPreviewUrl}`}
                      alt="Face thumbnail"
                      className="w-6 h-6 rounded object-cover border border-[#4A8FA0]"
                    />
                  )}
                  <span className="text-[#8E98A5] text-[11px]">
                    (Độ tin cậy: {((store.faces[store.selectedFaceIdx]?.det_score || 0) * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-1 text-[#8E98A5] hover:text-[#E8E6E0]">
                <span>Xem & đổi mặt</span>
                <ChevronDown className="w-3.5 h-3.5" />
              </div>
            </div>
          ) : (
            /* Nội dung đầy đủ khi đang chọn mặt */
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[#8E98A5] uppercase tracking-wide flex items-center gap-1.5">
                  <UserCheck className="w-4 h-4 text-[#C97B4A]" />
                  Bước 2: Chọn khuôn mặt cần tìm
                </span>
                {store.selectedFaceIdx !== null && (
                  <button
                    onClick={() => setIsFaceSelectorOpen(false)}
                    className="text-xs text-[#8E98A5] hover:text-[#E8E6E0] flex items-center gap-1"
                  >
                    <span>Thu gọn</span>
                    <ChevronUp className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              <FaceSelector />
              <QualityWarnings />
            </div>
          )}

          {/* BƯỚC 3: CẤU HÌNH THÔNG TIN & GALLERY */}
          {store.selectedFaceIdx !== null && (
            <div>
              {/* Accordion Header / Summary khi đã xong bước cấu hình và đang chạy / hoàn tất */}
              {store.initialAge !== null && !isConfigOpen ? (
                <div
                  onClick={() => setIsConfigOpen(true)}
                  className="bg-[#1B2129] border border-[#262E38] hover:border-[#C97B4A]/50 rounded-xl px-4 py-3 flex items-center justify-between text-xs cursor-pointer transition-colors"
                >
                  <div className="flex items-center gap-2.5">
                    <CheckCircle2 className="w-4 h-4 text-[#4A8FA0] flex-shrink-0" />
                    <span className="font-semibold text-[#E8E6E0]">
                      Bước 3: {store.genderWord === 'man' ? 'Nam' : 'Nữ'} • {store.initialAge} tuổi
                    </span>
                    <span className="text-[#8E98A5] truncate max-w-[260px]">
                      • Gallery: {store.galleryDir ? store.galleryDir : 'Mặc định'}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 text-[#8E98A5] hover:text-[#E8E6E0]">
                    <span>Chỉnh sửa</span>
                    <ChevronDown className="w-3.5 h-3.5" />
                  </div>
                </div>
              ) : (
                /* Nội dung đầy đủ khi đang cấu hình thông tin */
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-[#8E98A5] uppercase tracking-wide flex items-center gap-1.5">
                      <Settings2 className="w-4 h-4 text-[#C97B4A]" />
                      Bước 3: Thông tin đối tượng & Thư viện đối soát
                    </span>
                    {store.initialAge !== null && store.jobStatus !== 'idle' && (
                      <button
                        onClick={() => setIsConfigOpen(false)}
                        className="text-xs text-[#8E98A5] hover:text-[#E8E6E0] flex items-center gap-1"
                      >
                        <span>Thu gọn</span>
                        <ChevronUp className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>

                  <div className="space-y-4">
                    <AgeInputForm />
                    <GalleryPicker />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* BƯỚC 4: NÚT HÀNH ĐỘNG CHẠY PIPELINE (Màu phẳng #C97B4A) */}
          {store.selectedFaceIdx !== null && (
            <div className="bg-[#1B2129] border border-[#262E38] rounded-xl p-5 shadow-lg space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-semibold text-[#E8E6E0]">Sẵn sàng khởi động quy trình</h4>
                  <p className="text-xs text-[#8E98A5] mt-0.5">
                    Pipeline tuần tự qua Inversion, Cross-Attention Editing và FAISS Search.
                  </p>
                </div>
              </div>

              <button
                onClick={handleRun}
                disabled={!isReadyToRun}
                className="w-full flex items-center justify-center gap-2 bg-[#C97B4A] hover:brightness-110 text-white font-bold py-3 px-5 rounded-xl transition-all shadow-md disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {store.jobStatus === 'running' ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-white" />
                    <span>Pipeline Đang Chạy Trên GPU...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Chạy Pipeline FADING</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* TIẾN TRÌNH 4 GIAI ĐOẠN */}
          <ProgressStages />

          {/* BẢNG KẾT QUẢ, HERO CARD 3 ẢNH & LIGHTBOX */}
          <ResultsGallery />
        </div>
      )}
    </div>
  );
};

