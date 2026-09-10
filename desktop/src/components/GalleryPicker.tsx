import React, { useState } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { FolderCheck, FolderOpen, RotateCcw } from 'lucide-react';

export const GalleryPicker: React.FC = () => {
  const store = useSearchStore();
  const [isEditing, setIsEditing] = useState(false);
  const [tempDir, setTempDir] = useState(store.galleryDir || '');

  const handleSave = () => {
    store.setGalleryDir(tempDir.trim() ? tempDir.trim() : null);
    setIsEditing(false);
  };

  const handleReset = () => {
    store.setGalleryDir(null);
    setTempDir('');
    setIsEditing(false);
  };

  return (
    <div className="bg-slate-800/80 border border-slate-700 rounded-xl p-5 shadow-xl space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-slate-200 font-semibold text-sm">
          <FolderCheck className="w-5 h-5 text-indigo-400" />
          <span>Thư mục Gallery đối soát (FAISS Face Search):</span>
        </div>
        {!isEditing && (
          <button
            onClick={() => {
              setTempDir(store.galleryDir || '');
              setIsEditing(true);
            }}
            className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium"
          >
            <FolderOpen className="w-3.5 h-3.5" />
            Thay đổi thư mục
          </button>
        )}
      </div>

      {!isEditing ? (
        <div className="flex items-center justify-between bg-slate-900/60 px-4 py-2.5 rounded-lg border border-slate-700/80">
          <code className="text-xs text-slate-300 font-mono truncate max-w-xl">
            {store.galleryDir ? store.galleryDir : './data/test_gallery (Mặc định dự án)'}
          </code>
          {store.galleryDir && (
            <button
              onClick={handleReset}
              title="Đặt lại về mặc định"
              className="text-slate-400 hover:text-slate-200 ml-2"
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
            className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono"
          />
          <button
            onClick={handleSave}
            className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3.5 py-2 rounded-lg font-medium"
          >
            Lưu
          </button>
          <button
            onClick={() => setIsEditing(false)}
            className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs px-3 py-2 rounded-lg"
          >
            Hủy
          </button>
        </div>
      )}
    </div>
  );
};
