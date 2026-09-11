import React, { useState, useEffect } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { useSearchApi } from '../api/useSearchApi';
import { WizardStepper, WizardStep } from '../components/WizardStepper';
import { FaceSelector } from '../components/FaceSelector';
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
  CheckCircle2,
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

      if (params.get('empty') === '1') {
        return;
      }

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
    <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 space-y-6">
      {/* Header phẳng, không gradient */}
      <header className="flex items-center justify-between border-b border-[#262E38] pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-bold text-[#E8E6E0]">
              Missing Person Search via <span className="text-[#C97B4A]">FADING</span>
            </h1>
            <span className="bg-[#12161C] border border-[#262E38] text-[#8E98A5] text-[11px] px-2.5 py-0.5 rounded-full font-mono font-medium">
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
        <div className="space-y-6 animate-in fade-in duration-200">
          {!store.sessionId ? (
            /* Trạng thái trống (chưa có ảnh): 2 cột (lg: 1024px) */
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Cột trái (~58% / lg:col-span-7): Ô upload / kéo thả */}
              <div className="lg:col-span-7">
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
                    <div className="w-16 h-16 bg-[#12161C] border border-[#262E38] text-[#C97B4A] rounded-2xl flex items-center justify-center mx-auto mb-4 group-hover:scale-105 transition-transform shadow-md">
                      {isUploading ? (
                        <Loader2 className="w-8 h-8 animate-spin text-[#C97B4A]" />
                      ) : (
                        <UploadCloud className="w-8 h-8" />
                      )}
                    </div>
                    <h3 className="text-base font-bold text-[#E8E6E0] mb-1.5">
                      {isUploading ? 'Đang tải và phát hiện khuôn mặt...' : 'Chọn hoặc kéo thả ảnh cần tìm'}
                    </h3>
                    <p className="text-xs text-[#8E98A5] max-w-md mx-auto">
                      Hỗ trợ định dạng PNG, JPG, JPEG. Hệ thống sẽ tự động nhận diện và căn chỉnh khuôn mặt chuẩn FFHQ 256×256.
                    </p>
                  </label>

                  {uploadError && (
                    <div className="mt-4 inline-block bg-[#B8564A]/15 border border-[#B8564A]/40 text-[#E8E6E0] text-xs px-3.5 py-2 rounded-lg">
                      {uploadError}
                    </div>
                  )}
                </div>
              </div>

              {/* Cột phải (~42% / lg:col-span-5): Card tĩnh "Ảnh đầu vào tốt cần gì?" */}
              <div className="lg:col-span-5 bg-[#1B2129] border border-[#262E38] rounded-2xl p-5 sm:p-6 shadow-lg space-y-4">
                <div className="flex items-center gap-2 border-b border-[#262E38] pb-3">
                  <CheckCircle2 className="w-4 h-4 text-[#4A8FA0]" />
                  <h4 className="text-sm font-bold text-[#E8E6E0]">Ảnh đầu vào tốt cần gì?</h4>
                </div>
                <p className="text-xs text-[#8E98A5]">
                  Để mô hình khuếch tán FADING sinh các lứa tuổi đạt độ nhận diện danh tính cao nhất:
                </p>

                <div className="space-y-3.5 text-xs">
                  <div className="flex items-start gap-3">
                    <div className="w-5 h-5 rounded-full bg-[#4A8FA0]/20 text-[#4A8FA0] flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                      1
                    </div>
                    <div>
                      <p className="font-semibold text-[#E8E6E0]">Chính diện, rõ nét</p>
                      <p className="text-[#8E98A5] mt-0.5">
                        Khuôn mặt nhìn thẳng, góc quay đầu nghiêng không quá 15° so với trục ống kính.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="w-5 h-5 rounded-full bg-[#4A8FA0]/20 text-[#4A8FA0] flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                      2
                    </div>
                    <div>
                      <p className="font-semibold text-[#E8E6E0]">Chỉ 1 khuôn mặt</p>
                      <p className="text-[#8E98A5] mt-0.5">
                        Ảnh chân dung chụp đơn lẻ một người, không bị che khuất bởi người đứng cạnh.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="w-5 h-5 rounded-full bg-[#4A8FA0]/20 text-[#4A8FA0] flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                      3
                    </div>
                    <div>
                      <p className="font-semibold text-[#E8E6E0]">Độ phân giải khuôn mặt ≥ 128×128px</p>
                      <p className="text-[#8E98A5] mt-0.5">
                        Vùng mặt càng sắc nét thì vector đặc trưng khuôn mặt trích xuất càng chuẩn xác.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="w-5 h-5 rounded-full bg-[#4A8FA0]/20 text-[#4A8FA0] flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                      4
                    </div>
                    <div>
                      <p className="font-semibold text-[#E8E6E0]">Không bị che khuất ngũ quan</p>
                      <p className="text-[#8E98A5] mt-0.5">
                        Tránh ảnh đeo kính râm tối màu, khẩu trang hoặc tóc xõa phủ kín mắt và mũi.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Khi đã có ảnh: Bố cục 2 cột song song (FaceSelector + PhotoRestoration) */
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Cột trái (lg:col-span-6): Card khuôn mặt đã phát hiện + Cảnh báo tích hợp */}
              <div className="lg:col-span-6">
                <FaceSelector />
              </div>

              {/* Cột phải (lg:col-span-6): Khôi phục ảnh CodeFormer */}
              <div className="lg:col-span-6">
                {store.croppedPreviewUrl && (
                  <PhotoRestoration
                    originalFaceUrl={croppedFaceFullUrl}
                    onConfirm={handleConfirmRestore}
                  />
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ============================================================ */}
      {/* BƯỚC 2: SINH ẢNH & ĐỐI SOÁT */}
      {/* ============================================================ */}
      {currentStep === 'generate' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          {store.jobStatus === 'running' ? (
            /* Khi bấm chạy: Tiến trình 2 cột */
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Cột trái (~60% / lg:col-span-7): Khối tiến trình 3 bước con in-place */}
              <div className="lg:col-span-7">
                <JobProgressBlock
                  jobStage={store.jobStage}
                  jobStatus={store.jobStatus}
                  jobError={store.jobError}
                  onCancel={handleCancelJob}
                />
              </div>

              {/* Cột phải (~40% / lg:col-span-5): Card Thông tin phiên chạy */}
              <div className="lg:col-span-5 bg-[#1B2129] border border-[#262E38] rounded-xl p-5 shadow-lg space-y-4">
                <div className="border-b border-[#262E38] pb-3">
                  <h4 className="text-sm font-bold text-[#E8E6E0]">Thông tin phiên chạy</h4>
                  <p className="text-xs text-[#8E98A5] mt-0.5">Tiến trình AI đang thực thi trên phần cứng GPU</p>
                </div>

                <div className="flex items-center gap-3.5 p-3.5 bg-[#12161C] border border-[#262E38] rounded-xl">
                  {croppedFaceFullUrl ? (
                    <img
                      src={croppedFaceFullUrl}
                      alt="Cropped target"
                      className="w-14 h-14 rounded-lg object-cover border border-[#C97B4A] shadow bg-black flex-shrink-0"
                    />
                  ) : (
                    <div className="w-14 h-14 rounded-lg bg-black/40 flex items-center justify-center text-xs text-[#8E98A5]">
                      Mặt gốc
                    </div>
                  )}
                  <div className="text-xs space-y-1 overflow-hidden">
                    <p className="font-semibold text-[#E8E6E0]">
                      Đối tượng: {store.genderWord === 'man' ? 'Nam' : 'Nữ'} • {store.initialAge ?? store.manualAge} tuổi
                    </p>
                    <p className="text-[11px] text-[#4A8FA0] font-mono truncate">
                      Job ID: {store.jobId || store.sessionId || 'job_fading_active'}
                    </p>
                  </div>
                </div>

                <div className="space-y-2.5 text-xs">
                  <div className="flex justify-between py-1.5 border-b border-[#262E38]">
                    <span className="text-[#8E98A5]">Dải tuổi dự đoán:</span>
                    <span className="text-[#E8E6E0] font-mono">15 - 70 tuổi (8 mốc)</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-[#262E38]">
                    <span className="text-[#8E98A5]">Phần cứng xử lý:</span>
                    <span className="text-[#4A8FA0] font-mono">NVIDIA CUDA / TensorRT</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-[#262E38]">
                    <span className="text-[#8E98A5]">Giai đoạn hiện tại:</span>
                    <span className="text-[#C97B4A] font-semibold uppercase">{store.jobStage}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Trạng thái nhập liệu: 2 cột */
            <div className="space-y-4">
              <button
                type="button"
                onClick={() => setCurrentStep('restore')}
                className="text-xs text-[#8E98A5] hover:text-[#E8E6E0] flex items-center gap-1.5 transition-colors"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                <span>Quay lại bước khôi phục ảnh</span>
              </button>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                {/* Cột trái (~60% / lg:col-span-7): Form nhập liệu */}
                <div className="lg:col-span-7 space-y-5">
                  <AgeInputForm />
                  <GalleryPicker />
                </div>

                {/* Cột phải (~40% / lg:col-span-5): Card Tóm tắt trước khi chạy + Nút CTA chính */}
                <div className="lg:col-span-5 bg-[#1B2129] border border-[#262E38] rounded-xl p-5 sm:p-6 shadow-lg space-y-4 lg:sticky lg:top-6">
                  <div className="border-b border-[#262E38] pb-3">
                    <h4 className="text-sm font-bold text-[#E8E6E0]">Tóm tắt trước khi chạy</h4>
                    <p className="text-xs text-[#8E98A5] mt-0.5">Xác nhận cấu hình trước khi khởi động pipeline FADING</p>
                  </div>

                  <div className="flex items-center gap-3.5 p-3.5 bg-[#12161C] border border-[#262E38] rounded-xl">
                    {croppedFaceFullUrl ? (
                      <img
                        src={croppedFaceFullUrl}
                        alt="Selected Target Face"
                        className="w-16 h-16 rounded-lg object-cover border-2 border-[#4A8FA0] shadow bg-black flex-shrink-0"
                      />
                    ) : (
                      <div className="w-16 h-16 rounded-lg bg-black/40 border border-[#262E38] flex items-center justify-center text-[10px] text-[#8E98A5]">
                        Chưa có ảnh
                      </div>
                    )}
                    <div className="text-xs space-y-1 overflow-hidden">
                      <p className="font-bold text-[#E8E6E0]">
                        Đối tượng: {store.genderWord === 'man' ? 'Nam (Man / Boy)' : 'Nữ (Woman / Girl)'}
                      </p>
                      <p className="text-[#8E98A5]">
                        Tuổi lúc chụp:{' '}
                        <span className="font-semibold text-[#C97B4A]">
                          {store.initialAge ?? store.manualAge} tuổi
                        </span>
                      </p>
                      <p className="text-[11px] text-[#8E98A5]">
                        Xác định:{' '}
                        {store.ageMode === 'manual' ? 'Nhập chính xác' : 'MiVOLO AI ước tính'}
                      </p>
                    </div>
                  </div>

                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1.5 border-b border-[#262E38]">
                      <span className="text-[#8E98A5]">Thư viện đối soát:</span>
                      <span
                        className="font-mono text-[#E8E6E0] text-right truncate max-w-[170px]"
                        title={store.galleryDir || 'Mặc định (test_gallery)'}
                      >
                        {store.galleryDir
                          ? store.galleryDir.split(/[\\/]/).pop() || store.galleryDir
                          : 'Mặc định (test_gallery)'}
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-[#262E38]">
                      <span className="text-[#8E98A5]">Dải tuổi sẽ sinh:</span>
                      <span className="text-[#E8E6E0] font-mono">15 - 70 tuổi (8 mốc)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-[#262E38]">
                      <span className="text-[#8E98A5]">Quy trình xử lý:</span>
                      <span className="text-[#4A8FA0] font-mono">Inversion → Edit → FAISS</span>
                    </div>
                  </div>

                  {/* Nút CTA chính nằm ngay trong card tóm tắt */}
                  <button
                    type="button"
                    data-testid="run-pipeline-btn"
                    onClick={handleRun}
                    disabled={!isReadyToRun}
                    className="w-full flex items-center justify-center gap-2 bg-[#C97B4A] hover:bg-[#B56D40] text-white font-bold py-3.5 px-5 rounded-xl transition-all shadow-md disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <Play className="w-4 h-4 fill-current" />
                    <span>Chạy Pipeline FADING</span>
                  </button>
                  <p className="text-[11px] text-[#8E98A5] text-center">
                    Mỗi lần chạy sẽ tính toán và sinh ảnh qua GPU (ước tính ~30s)
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ============================================================ */}
      {/* BƯỚC 3: KẾT QUẢ ĐỐI SOÁT */}
      {/* ============================================================ */}
      {currentStep === 'results' && (
        <div className="space-y-6 animate-in fade-in duration-200">
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

