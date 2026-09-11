import React, { useState, useEffect } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { WizardStepper, WizardStep } from '../components/WizardStepper';
import { FaceSelector } from '../components/FaceSelector';
import { QualityWarnings } from '../components/QualityWarnings';
import { PhotoRestoration } from '../components/PhotoRestoration';
import { AgeInputForm } from '../components/AgeInputForm';
import { GalleryPicker } from '../components/GalleryPicker';
import { JobProgressBlock } from '../components/JobProgressBlock';
import { ResultsGallery } from '../components/ResultsGallery';
import {
  UploadCloud,
  Play,
  RotateCcw,
  ShieldCheck,
  Loader2,
  ArrowLeft,
  RefreshCw,
} from 'lucide-react';

export const SearchPage: React.FC = () => {
  const store = useSearchStore();
  const { uploadImage, runPipeline } = useSearchApi();
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Quản lý trạng thái Wizard 3 bước (khóa bước tuyến tính)
  const [currentStep, setCurrentStep] = useState<WizardStep>(() => {
    const params = new URLSearchParams(window.location.search);
    const pStep = params.get('step') as WizardStep;
    if (pStep === 'generate' || pStep === 'results') return pStep;
    return 'restore';
  });
  const [completedSteps, setCompletedSteps] = useState<WizardStep[]>(() => {
    const params = new URLSearchParams(window.location.search);
    const pStep = params.get('step') as WizardStep;
    if (pStep === 'results') return ['restore', 'generate'];
    if (pStep === 'generate') return ['restore'];
    return [];
  });

  // Khởi tạo dữ liệu mô phỏng trực quan khi chạy ở chế độ preview (?preview=1)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.has('preview')) {
      const pStep = params.get('step') || 'restore';
      const pStage = params.get('stage') || '';
      const s = useSearchStore.getState();

      s.setCheckpoints(true, []);
      s.setIsCheckingHealth(false);
      s.setUploadResult(
        'session_mock_demo',
        [{ index: 0, bbox: [80, 60, 240, 220], det_score: 0.99 }],
        '/sample_face.png'
      );
      s.setSelectedFace(
        0,
        ['Độ phân giải khuôn mặt thấp (110x110px)', 'Khuyên dùng CodeFormer để tăng độ nét'],
        '/restored_face.png'
      );
      s.setResolvedAge(12);
      s.setGalleryDir('D:/Data/missing_persons_gallery');

      if (pStep === 'generate' && pStage === 'running') {
        s.updateJobProgress('running', 'editing');
      } else if (pStep === 'results') {
        s.updateJobProgress('done', 'complete', {
          job_id: 'job_mock_demo_982',
          status: 'done',
          edited_images: {
            15: '/sample_face.png',
            20: '/gallery_match.png',
            25: '/restored_face.png',
            30: '/sample_face.png',
            40: '/gallery_match.png',
            50: '/restored_face.png',
            60: '/sample_face.png',
            70: '/gallery_match.png',
          },
          final_scores: {
            'MP_2023_0982.jpg': 0.8466,
            'IMG_CCTV_BinhThanh_041.jpg': 0.6210,
            'ID_CARD_772189.png': 0.2450,
          },
          accepted: true,
          top_identity: 'MP_2023_0982',
          top_score: 0.8466,
          best_age: 25,
          matched_gallery_image: '/gallery_match.png',
        });
      }
    }
  }, []);

  // Tự động chuyển sang bước 3 (Kết quả) khi pipeline hoàn tất
  useEffect(() => {
    if (store.jobStatus === 'done' && currentStep !== 'results') {
      setCompletedSteps((prev) => Array.from(new Set([...prev, 'restore', 'generate'])));
      setCurrentStep('results');
    }
  }, [store.jobStatus, currentStep]);

  // Điều hướng stepper: chỉ cho phép click nếu bước đã nằm trong completedSteps
  const handleStepClick = (step: WizardStep) => {
    if (completedSteps.includes(step)) {
      setCurrentStep(step);
    }
  };

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

  // Xác nhận bước 1 (Khôi phục ảnh) -> Chuyển sang Bước 2
  const handleConfirmRestore = (_useRestored: boolean, _fidelity: number, _opts: any) => {
    setCompletedSteps((prev) => Array.from(new Set([...prev, 'restore'])));
    setCurrentStep('generate');
  };

  // Khởi động pipeline FADING ở Bước 2
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

  // Hủy tiến trình
  const handleCancelJob = () => {
    store.updateJobProgress('idle', 'specialization', undefined, 'Tiến trình đã được người dùng dừng lại');
  };

  // Làm mới toàn bộ phiên tìm kiếm
  const handleResetAll = () => {
    store.resetForNewUpload();
    setCompletedSteps([]);
    setCurrentStep('restore');
  };

  const isReadyToRun = Boolean(
    store.sessionId &&
    store.croppedPreviewUrl &&
    store.initialAge !== null &&
    store.jobStatus !== 'running'
  );

  const croppedFaceFullUrl = store.croppedPreviewUrl?.startsWith('/') || store.croppedPreviewUrl?.startsWith('data:')
    ? store.croppedPreviewUrl
    : store.croppedPreviewUrl
    ? `http://127.0.0.1:${store.backendPort}${store.croppedPreviewUrl}`
    : (store.uploadedImageUrl || '');

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
              v2.0 Desktop Wizard
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
              onClick={handleResetAll}
              className="flex items-center gap-1.5 text-xs bg-[#1B2129] hover:bg-[#262E38] text-[#E8E6E0] px-3 py-1.5 rounded-lg border border-[#262E38] transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5 text-[#8E98A5]" />
              <span>Làm mới</span>
            </button>
          )}
        </div>
      </header>

      {/* THANH STEPPER ĐIỀU HƯỚNG CỐ ĐỊNH TRÊN ĐẦU MỌI MÀN HÌNH */}
      <WizardStepper
        currentStep={currentStep}
        completedSteps={completedSteps}
        onStepClick={handleStepClick}
      />

      {/* ============================================================ */}
      {/* BƯỚC 1: KHÔI PHỤC ẢNH CŨ (TÙY CHỌN) */}
      {/* ============================================================ */}
      {currentStep === 'restore' && (
        <div className="space-y-4 animate-in fade-in duration-200">
          {!store.sessionId ? (
            /* Upload dropzone */
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
            /* Khi đã có ảnh: Hiển thị FaceSelector và PhotoRestoration */
            <div className="space-y-4">
              <FaceSelector />
              <QualityWarnings />

              {/* Khối Khôi phục ảnh CodeFormer với Before/After slider */}
              {store.croppedPreviewUrl && (
                <PhotoRestoration
                  originalFaceUrl={croppedFaceFullUrl}
                  onConfirm={handleConfirmRestore}
                />
              )}
            </div>
          )}
        </div>
      )}

      {/* ============================================================ */}
      {/* BƯỚC 2: SINH ẢNH & ĐỐI SOÁT */}
      {/* ============================================================ */}
      {currentStep === 'generate' && (
        <div className="space-y-4 animate-in fade-in duration-200">
          {store.jobStatus === 'running' ? (
            /* Khi bấm chạy: Khối tiến trình in-place thay thế Form ngay tại chỗ */
            <JobProgressBlock
              jobStage={store.jobStage}
              jobStatus={store.jobStatus}
              jobError={store.jobError}
              onCancel={handleCancelJob}
            />
          ) : (
            /* Form cấu hình đối tượng & Gallery đối soát */
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => setCurrentStep('restore')}
                  className="text-xs text-[#8E98A5] hover:text-[#E8E6E0] flex items-center gap-1.5 transition-colors"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>Quay lại bước khôi phục ảnh</span>
                </button>
              </div>

              <AgeInputForm />
              <GalleryPicker />

              {/* Nút hành động Chạy Pipeline */}
              <div className="bg-[#1B2129] border border-[#262E38] rounded-xl p-5 shadow-lg space-y-3">
                <div>
                  <h4 className="text-sm font-semibold text-[#E8E6E0]">Sẵn sàng khởi động quy trình FADING</h4>
                  <p className="text-xs text-[#8E98A5] mt-0.5">
                    Pipeline sẽ chạy tuần tự qua Inversion, Attention Editing và FAISS Search.
                  </p>
                </div>

                <button
                  type="button"
                  data-testid="run-pipeline-btn"
                  onClick={handleRun}
                  disabled={!isReadyToRun}
                  className="w-full flex items-center justify-center gap-2 bg-[#C97B4A] hover:brightness-110 text-white font-bold py-3.5 px-5 rounded-xl transition-all shadow-md disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Play className="w-4 h-4 fill-current" />
                  <span>Chạy Pipeline FADING</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ============================================================ */}
      {/* BƯỚC 3: KẾT QUẢ ĐỐI SOÁT */}
      {/* ============================================================ */}
      {currentStep === 'results' && (
        <div className="space-y-5 animate-in fade-in duration-200">
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => setCurrentStep('generate')}
              className="text-xs text-[#8E98A5] hover:text-[#E8E6E0] flex items-center gap-1.5 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Xem lại thông tin cấu hình đối soát</span>
            </button>

            <button
              type="button"
              onClick={handleResetAll}
              className="text-xs text-[#C97B4A] hover:underline flex items-center gap-1 font-semibold"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Bắt đầu phiên tìm kiếm mới</span>
            </button>
          </div>

          <ResultsGallery />
        </div>
      )}
    </div>
  );
};
