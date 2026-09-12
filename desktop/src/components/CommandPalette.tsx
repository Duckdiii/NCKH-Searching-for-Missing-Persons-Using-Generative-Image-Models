import React, { useState, useEffect, useRef } from 'react';
import { useSearchStore } from '../store/useSearchStore';
import {
  Search,
  Plus,
  FolderOpen,
  Sliders,
  Award,
  ArrowRight,
  Command,
  X,
} from 'lucide-react';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectNewSearch?: () => void;
  onSelectOpenGallery?: () => void;
  onSelectGoToResults?: () => void;
}

interface ActionItem {
  id: string;
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  badge?: string;
  action: () => void;
  available: boolean;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onSelectNewSearch,
  onSelectOpenGallery,
  onSelectGoToResults,
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const { isAdvancedMode, toggleAdvancedMode, startNewSearch, jobResult } = useSearchStore();

  const hasResults = Boolean(jobResult && jobResult.status === 'done');

  // List of real actionable commands
  const actions: ActionItem[] = [
    {
      id: 'new_search',
      title: 'Bắt đầu phiên tìm kiếm mới',
      subtitle: 'Xóa phiên hiện tại và quay về Bước 1 (Khôi phục ảnh)',
      icon: <Plus className="w-4 h-4 text-[#E8804A]" />,
      action: () => {
        startNewSearch();
        if (onSelectNewSearch) onSelectNewSearch();
        onClose();
      },
      available: true,
    },
    {
      id: 'open_gallery',
      title: 'Mở thư mục gallery đối soát',
      subtitle: 'Chọn đường dẫn thư mục ảnh đối chứng FAISS trên máy',
      icon: <FolderOpen className="w-4 h-4 text-[#3B82C7]" />,
      action: () => {
        if (onSelectOpenGallery) onSelectOpenGallery();
        onClose();
      },
      available: true,
    },
    {
      id: 'toggle_advanced',
      title: isAdvancedMode ? 'Tắt chế độ Nâng cao' : 'Bật chế độ Nâng cao',
      subtitle: isAdvancedMode
        ? 'Ẩn các thông số kỹ thuật (Cosine thô, tham số pipeline)'
        : 'Hiển thị điểm số Cosine thô và chi tiết tham số FADING',
      icon: <Sliders className="w-4 h-4 text-[#D97706]" />,
      badge: isAdvancedMode ? 'Đang BẬT' : 'Đang TẮT',
      action: () => {
        toggleAdvancedMode();
        onClose();
      },
      available: true,
    },
    {
      id: 'go_to_results',
      title: 'Đi tới bước Kết quả đối soát',
      subtitle: 'Xem bảng xếp hạng và hình ảnh nhận diện của phiên này',
      icon: <Award className="w-4 h-4 text-[#16A34A]" />,
      action: () => {
        if (onSelectGoToResults) onSelectGoToResults();
        onClose();
      },
      available: hasResults,
    },
  ];

  const filteredActions = actions.filter(
    (item) =>
      item.available &&
      (item.title.toLowerCase().includes(query.toLowerCase()) ||
        item.subtitle.toLowerCase().includes(query.toLowerCase()))
  );

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % Math.max(1, filteredActions.length));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev === 0 ? Math.max(0, filteredActions.length - 1) : prev - 1
        );
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredActions[selectedIndex]) {
          filteredActions[selectedIndex].action();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filteredActions, selectedIndex, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 sm:pt-28 px-4 bg-black/40 backdrop-blur-[2px] transition-all"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg bg-white rounded-xl shadow-2xl border border-[#E5E7EB] overflow-hidden animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input Bar */}
        <div className="flex items-center px-4 py-3 border-b border-[#E5E7EB] gap-2.5">
          <Search className="w-4 h-4 text-[#9CA3AF] flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Gõ lệnh hoặc tìm kiếm hành động..."
            className="w-full text-sm text-[#111827] placeholder-[#9CA3AF] outline-none bg-transparent"
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              className="p-1 rounded text-[#9CA3AF] hover:text-[#111827]"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
          <kbd className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-[#F3F4F6] text-[#6B7280] border border-[#E5E7EB]">
            ESC
          </kbd>
        </div>

        {/* Action list */}
        <div className="max-h-72 overflow-y-auto p-2 space-y-1">
          {filteredActions.length === 0 ? (
            <div className="py-8 text-center text-[#9CA3AF] text-xs">
              Không tìm thấy hành động phù hợp cho "{query}"
            </div>
          ) : (
            filteredActions.map((item, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={item.id}
                  onClick={item.action}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full flex items-center justify-between p-2.5 rounded-lg text-left transition-colors ${
                    isSelected ? 'bg-[#F5F6F8] text-[#111827]' : 'text-[#374151] hover:bg-[#F9FAFB]'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="p-1.5 rounded-md bg-white border border-[#E5E7EB] shadow-xs flex-shrink-0">
                      {item.icon}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold truncate">{item.title}</span>
                        {item.badge && (
                          <span className="text-[10px] px-1.5 py-0.2 rounded font-medium bg-[#F3F4F6] text-[#4B5563]">
                            {item.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-[#6B7280] truncate">{item.subtitle}</p>
                    </div>
                  </div>

                  <ArrowRight
                    className={`w-3.5 h-3.5 text-[#9CA3AF] transition-transform ${
                      isSelected ? 'translate-x-0.5 text-[#111827]' : 'opacity-0'
                    }`}
                  />
                </button>
              );
            })
          )}
        </div>

        {/* Footer shortcuts hint */}
        <div className="px-4 py-2 bg-[#F9FAFB] border-t border-[#E5E7EB] flex items-center justify-between text-[11px] text-[#6B7280]">
          <div className="flex items-center gap-3">
            <span>
              <kbd className="px-1 py-0.5 rounded bg-white border border-[#D1D5DB] font-mono text-[10px]">
                ↑
              </kbd>{' '}
              <kbd className="px-1 py-0.5 rounded bg-white border border-[#D1D5DB] font-mono text-[10px]">
                ↓
              </kbd>{' '}
              Điều hướng
            </span>
            <span>
              <kbd className="px-1.5 py-0.5 rounded bg-white border border-[#D1D5DB] font-mono text-[10px]">
                ↵
              </kbd>{' '}
              Chọn
            </span>
          </div>
          <span className="flex items-center gap-1 font-medium">
            <Command className="w-3 h-3" /> FADING Shell
          </span>
        </div>
      </div>
    </div>
  );
};
