import React, { useState } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { FolderCheck, FolderOpen, RotateCcw, Edit3 } from 'lucide-react';
import { open } from '@tauri-apps/plugin-dialog';

export const GalleryPicker: React.FC = () => {
  const store = useSearchStore();
  const [isManualInput, setIsManualInput] = useState(false);
  const [tempDir, setTempDir] = useState(store.galleryDir || '');

  const handleNativeBrowse = async () => {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Chọn thư mục Gallery đối soát (chứa ảnh .png, .jpg)',
      });
      if (selected) {
        store.setGalleryDir(selected as string);
      }
    } catch (err) {
      console.warn('Native dialog unavailable or canceled, falling back to manual edit:', err);
      setIsManualInput(true);
    }
  };

  const handleSaveManual = () => {
    store.setGalleryDir(tempDir.trim() ? tempDir.trim() : null);
    setIsManualInput(false);
  };

  const handleReset = () => {
    store.setGalleryDir(null);
    setTempDir('');
    setIsManualInput(false);
  };

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-xs space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-[#111827] font-semibold text-sm">
          <FolderCheck className="w-5 h-5 text-[#E8804A]" />
          <span>Thư mục Gallery đối soát (FAISS Face Search):</span>
        </div>

        <div className="flex items-center gap-2">
          {!isManualInput && (
            <>
              <button
                type="button"
                onClick={handleNativeBrowse}
                className="text-xs bg-[#E8804A] hover:bg-[#D97706] text-white font-medium px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all shadow-xs"
              >
                <FolderOpen className="w-3.5 h-3.5" />
                <span>Chọn thư mục</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setTempDir(store.galleryDir || '');
                  setIsManualInput(true);
                }}
                title="Nhập đường dẫn trực tiếp"
                className="text-xs text-[#6B7280] hover:text-[#111827] p-1.5 rounded-lg hover:bg-[#F3F4F6] transition-colors"
              >
                <Edit3 className="w-3.5 h-3.5" />
              </button>
            </>
          )}
        </div>
      </div>

      {!isManualInput ? (
        <div className="flex items-center justify-between bg-[#F9FAFB] px-3.5 py-2.5 rounded-lg border border-[#E5E7EB]">
          <code className="text-xs text-[#111827] font-mono truncate max-w-lg">
            {store.galleryDir ? store.galleryDir : './data/test_gallery (Mặc định dự án)'}
          </code>
          {store.galleryDir && (
            <button
              onClick={handleReset}
              title="Đặt lại về mặc định"
              className="text-[#6B7280] hover:text-[#111827] ml-2 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          )}
        </div>
      ) : (
        <div className="flex gap-2">
          <input
            type="text"
            value={tempDir}
            onChange={(e) => setTempDir(e.target.value)}
            placeholder="Nhập đường dẫn thư mục ảnh đối soát (vd: D:\Data\gallery...)"
            className="flex-1 bg-white border border-[#D1D5DB] rounded-lg px-3 py-2 text-xs text-[#111827] focus:outline-none focus:border-[#E8804A] font-mono"
          />
          <button
            onClick={handleSaveManual}
            className="bg-[#E8804A] hover:bg-[#D97706] text-white text-xs px-3.5 py-2 rounded-lg font-medium shadow-xs"
          >
            Lưu
          </button>
          <button
            onClick={() => setIsManualInput(false)}
            className="bg-[#F3F4F6] hover:bg-[#E5E7EB] text-[#374151] text-xs px-3.5 py-2 rounded-lg font-medium transition-colors"
          >
            Hủy
          </button>
        </div>
      )}

      <p className="text-[11px] text-[#6B7280]">
        Chứa ảnh chân dung các ứng viên cần đối soát. Hệ thống sẽ trích xuất vector ArcFace 512D và so khớp qua Cosine Similarity.
      </p>
    </div>
  );
};
