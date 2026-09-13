import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { JobHistoryItem } from '../types/api';
import {
  Plus,
  Clock,
  ChevronLeft,
  ChevronRight,
  Shield,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Loader2,
  History,
  Layers,
} from 'lucide-react';

interface SidebarProps {
  onNewSearch?: () => void;
  onSelectHistory?: (item: JobHistoryItem) => void;
}

function formatTime(timestamp?: number): string {
  if (!timestamp) return 'Gần đây';
  const diff = Date.now() - timestamp;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Vừa xong';
  if (mins < 60) return `${mins}m trước`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h trước`;
  return new Date(timestamp).toLocaleDateString('vi-VN', {
    month: 'numeric',
    day: 'numeric',
  });
}

export const Sidebar: React.FC<SidebarProps> = ({ onNewSearch, onSelectHistory }) => {
  const {
    isSidebarCollapsed,
    toggleSidebar,
    sessionHistory,
    jobId: currentJobId,
    checkpointName,
    appVersion,
    startNewSearch,
    loadHistoricalJob,
  } = useSearchStore();

  const handleNewSearch = () => {
    startNewSearch();
    if (onNewSearch) onNewSearch();
  };

  const handleItemClick = (item: JobHistoryItem) => {
    loadHistoricalJob(item);
    if (onSelectHistory) onSelectHistory(item);
  };

  return (
    <aside
      className={`relative flex flex-col bg-white border-r border-[#E5E7EB] transition-all duration-200 z-20 select-none ${
        isSidebarCollapsed ? 'w-16' : 'w-60 sm:w-64'
      } h-screen flex-shrink-0`}
    >
      {/* Header Brand */}
      <div className="h-14 flex items-center justify-between px-3 border-b border-[#E5E7EB]">
        <div className="flex items-center gap-2.5 overflow-hidden">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#E8804A] to-[#D97706] flex items-center justify-center text-white shadow-sm flex-shrink-0">
            <Sparkles className="w-4 h-4" />
          </div>
          {!isSidebarCollapsed && (
            <div className="flex flex-col min-w-0">
              <span className="font-semibold text-sm text-[#111827] tracking-tight truncate">
                FADING Studio
              </span>
              <span className="text-[10px] text-[#6B7280] truncate font-medium">
                Tìm kiếm người mất tích
              </span>
            </div>
          )}
        </div>

        <button
          onClick={toggleSidebar}
          title={isSidebarCollapsed ? 'Mở rộng sidebar' : 'Thu gọn sidebar'}
          className="p-1 rounded-md text-[#6B7280] hover:text-[#111827] hover:bg-[#F3F4F6] transition-colors"
        >
          {isSidebarCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* New Search Action Button */}
      <div className="p-3 border-b border-[#E5E7EB]">
        <button
          onClick={handleNewSearch}
          className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg font-medium text-xs text-white bg-[#E8804A] hover:bg-[#D97706] shadow-sm hover:shadow transition-all hover-lift cursor-pointer ${
            isSidebarCollapsed ? 'px-0' : ''
          }`}
          title="Bắt đầu phiên tìm kiếm mới"
        >
          <Plus className="w-4 h-4 flex-shrink-0" />
          {!isSidebarCollapsed && <span>Tìm kiếm mới</span>}
        </button>
      </div>

      {/* Session History List */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="px-3 py-2 flex items-center justify-between">
          {!isSidebarCollapsed && (
            <span className="text-[11px] font-semibold tracking-wider uppercase text-[#6B7280]">
              Lịch sử phiên ({sessionHistory.length})
            </span>
          )}
          <Clock className={`w-3.5 h-3.5 text-[#9CA3AF] ${isSidebarCollapsed ? 'mx-auto' : ''}`} />
        </div>

        <div className="flex-1 overflow-y-auto px-2 space-y-1">
          {sessionHistory.length === 0 ? (
            !isSidebarCollapsed && (
              <div className="py-10 px-3 text-center">
                <div className="w-11 h-11 rounded-full bg-[#F3F4F6] border border-[#E5E7EB] flex items-center justify-center mx-auto text-[#9CA3AF] mb-3 shadow-2xs">
                  <History className="w-5 h-5 opacity-70" />
                </div>
                <p className="text-xs font-semibold text-[#374151]">Chưa có phiên nào</p>
                <p className="text-[11px] text-[#6B7280] mt-1.5 leading-relaxed">
                  Chưa có phiên tìm kiếm nào. Bấm 'Tìm kiếm mới' để bắt đầu.
                </p>
                <button
                  onClick={handleNewSearch}
                  className="mt-3.5 inline-flex items-center gap-1 text-[11px] font-semibold text-[#E8804A] hover:text-[#C96B37] hover:underline cursor-pointer"
                >
                  <Plus className="w-3 h-3" />
                  <span>Bắt đầu ngay</span>
                </button>
              </div>
            )
          ) : (
            sessionHistory.map((item) => {
              const isSelected = item.job_id === currentJobId;
              const hasScore = typeof item.top_score === 'number';
              const pct = hasScore ? (item.top_score! * 100).toFixed(1) : null;

              // Color badge logic
              let badgeColor = 'bg-[#F3F4F6] text-[#6B7280]';
              if (item.status === 'running') {
                badgeColor = 'bg-[#FEF3C7] text-[#D97706] animate-pulse';
              } else if (item.status === 'error') {
                badgeColor = 'bg-[#FEE2E2] text-[#DC2626]';
              } else if (hasScore) {
                if (item.top_score! >= 0.6) badgeColor = 'bg-[#EFF6FF] text-[#3B82C7] border border-[#BFDBFE]';
                else if (item.top_score! >= 0.3) badgeColor = 'bg-[#FEF3C7] text-[#D97706] border border-[#FDE68A]';
                else badgeColor = 'bg-[#FEE2E2] text-[#DC2626] border border-[#FECACA]';
              }

              return (
                <button
                  key={item.job_id}
                  onClick={() => handleItemClick(item)}
                  title={`Xem lại phiên: ${item.top_identity || item.job_id}`}
                  className={`w-full text-left rounded-lg transition-all flex items-center gap-2 p-2 hover-lift cursor-pointer ${
                    isSelected
                      ? 'bg-[#F5F6F8] ring-1 ring-[#E8804A]/50 text-[#111827]'
                      : 'hover:bg-[#F9FAFB] text-[#374151]'
                  } ${isSidebarCollapsed ? 'justify-center px-1' : ''}`}
                >
                  <div className="flex-shrink-0">
                    {item.status === 'running' ? (
                      <Loader2 className="w-4 h-4 text-[#D97706] animate-spin" />
                    ) : item.status === 'error' ? (
                      <AlertCircle className="w-4 h-4 text-[#DC2626]" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-[#3B82C7]" />
                    )}
                  </div>

                  {!isSidebarCollapsed && (
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-medium text-[#111827] truncate">
                          {item.top_identity || `Job ${item.job_id.slice(0, 8)}`}
                        </span>
                        {pct && (
                          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${badgeColor}`}>
                            {pct}%
                          </span>
                        )}
                        {item.status === 'running' && (
                          <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-[#FEF3C7] text-[#D97706]">
                            Đang chạy
                          </span>
                        )}
                      </div>

                      <div className="flex items-center justify-between text-[10px] text-[#6B7280] mt-0.5">
                        <span className="truncate">{item.stage || 'Hoàn tất'}</span>
                        <span className="flex-shrink-0">{formatTime(item.timestamp)}</span>
                      </div>
                    </div>
                  )}
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Footer Info */}
      <div className="p-3 border-t border-[#E5E7EB] bg-[#F9FAFB]">
        {!isSidebarCollapsed ? (
          <div className="space-y-1.5 text-[11px]">
            <div className="flex items-center justify-between text-[#6B7280]">
              <span className="flex items-center gap-1.5">
                <Shield className="w-3.5 h-3.5 text-[#3B82C7]" />
                Phiên bản
              </span>
              <span className="font-medium text-[#111827]">v{appVersion}</span>
            </div>

            <div className="flex items-center justify-between text-[#6B7280]">
              <span className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-[#E8804A]" />
                Model
              </span>
              <span
                className="font-medium text-[#111827] truncate max-w-[110px]"
                title={checkpointName}
              >
                {checkpointName}
              </span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-1 text-[#9CA3AF]" title={`v${appVersion} • ${checkpointName}`}>
            <span className="text-[9px] font-bold">v{appVersion}</span>
          </div>
        )}
      </div>
    </aside>
  );
};
