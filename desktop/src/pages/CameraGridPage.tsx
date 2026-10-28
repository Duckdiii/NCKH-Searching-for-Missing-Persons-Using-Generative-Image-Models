import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle, Camera, CheckCircle2, Clock, Loader2, Play, RotateCcw, Search, UploadCloud,
} from 'lucide-react';
import { api, getBaseUrl } from '../api/client';
import { useSearchStore } from '../store/useSearchStore';
import {
  ReplayCameraResult, ReplayMatchResponse, ReplayStartResponse, ReplayStatus,
} from '../types/api';

const SLOT_COUNT = 8;
const POLL_MS = 2000;
const ACCEPT = 'video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm';

type Phase = 'empty' | 'ready' | 'uploading' | 'processing' | 'done' | 'error';

interface SlotState {
  slot: number;
  file: File | null;
  previewUrl: string | null;
  offset: number;          // giây lệch so với mốc chung (camera đồng bộ = 0)
  phase: Phase;
  sourceId?: string;
  status?: ReplayStatus;
  error?: string;
  result?: ReplayCameraResult;
}

const emptySlots = (): SlotState[] =>
  Array.from({ length: SLOT_COUNT }, (_, i) => ({
    slot: i + 1, file: null, previewUrl: null, offset: 0, phase: 'empty' as Phase,
  }));

const detailOf = (err: any, fallback: string): string => {
  let d: unknown = err?.response?.data?.detail;
  if (Array.isArray(d)) d = d.map((x: any) => x?.msg || JSON.stringify(x)).join('; ');
  return (d as string) || fallback;
};

export const resolveMediaUrl = (path?: string | null): string => {
  if (!path) return '';
  if (/^(data:|https?:\/\/|blob:)/.test(path)) return path;
  return `${getBaseUrl()}${path.startsWith('/') ? path : '/' + path}`;
};

export const hasGeneratedImages = (jobResult: { status?: string; edited_images?: Record<number, string> } | null): boolean =>
  !!jobResult && jobResult.status === 'done' && Object.keys(jobResult.edited_images || {}).length > 0;

export const CameraGridPage: React.FC = () => {
  const { jobId, jobResult } = useSearchStore();
  const [slots, setSlots] = useState<SlotState[]>(emptySlots);
  const [fps, setFps] = useState(5);
  const [maxCandidates, setMaxCandidates] = useState(3);
  const [minQuality, setMinQuality] = useState(0);
  const [threshold, setThreshold] = useState(0.45);
  const [isMatching, setIsMatching] = useState(false);
  const [matchError, setMatchError] = useState<string | null>(null);
  const [matchResult, setMatchResult] = useState<ReplayMatchResponse | null>(null);
  const slotsRef = useRef(slots);
  slotsRef.current = slots;

  const effectiveJobId = jobResult?.job_id || jobId;
  const refImages = useMemo(
    () => Object.entries(jobResult?.edited_images || {})
      .map(([age, url]) => ({ age: Number(age), url: resolveMediaUrl(url) }))
      .sort((a, b) => a.age - b.age),
    [jobResult],
  );

  const update = useCallback((slot: number, patch: Partial<SlotState>) => {
    setSlots((prev) => prev.map((s) => (s.slot === slot ? { ...s, ...patch } : s)));
  }, []);

  // Giải phóng URL xem trước khi rời trang.
  useEffect(() => () => {
    slotsRef.current.forEach((s) => s.previewUrl && URL.revokeObjectURL(s.previewUrl));
  }, []);

  const busy = slots.some((s) => s.phase === 'uploading' || s.phase === 'processing');
  const doneSlots = slots.filter((s) => s.phase === 'done' && s.sourceId);
  const readySlots = slots.filter((s) => s.file && (s.phase === 'ready' || s.phase === 'error'));

  const pickFile = (slot: number, file: File | null) => {
    const cur = slots.find((s) => s.slot === slot);
    if (cur?.previewUrl) URL.revokeObjectURL(cur.previewUrl);
    update(slot, {
      file, previewUrl: file ? URL.createObjectURL(file) : null,
      phase: file ? 'ready' : 'empty', sourceId: undefined, status: undefined,
      error: undefined, result: undefined,
    });
    setMatchResult(null);
  };

  const runAll = async () => {
    if (!effectiveJobId) return;
    setMatchResult(null);
    setMatchError(null);
    // Mốc chung cho cả lượt: camera đồng bộ giờ, lệch riêng qua "offset".
    const baseEpoch = Date.now() / 1000;
    for (const s of readySlots) {
      update(s.slot, { phase: 'uploading', error: undefined, result: undefined });
      try {
        const form = new FormData();
        form.append('job_id', effectiveJobId);
        form.append('slot', String(s.slot));
        form.append('file', s.file as File);
        form.append('start_offset_sec', String(s.offset));
        form.append('base_epoch', String(baseEpoch));
        form.append('sample_fps', String(fps));
        form.append('max_candidates', String(maxCandidates));
        form.append('min_quality', String(minQuality));
        const res = await api.post<ReplayStartResponse>('/api/replay/start', form, {
          headers: { 'Content-Type': 'multipart/form-data' }, timeout: 600000,
        });
        update(s.slot, { phase: 'processing', sourceId: res.data.source_id });
      } catch (err: any) {
        update(s.slot, { phase: 'error', error: detailOf(err, 'Không tải được video lên backend.') });
      }
    }
  };

  // Theo dõi tiến độ các camera đang xử lý (chỉ dựng lại khi bật/tắt trạng thái xử lý).
  const anyProcessing = slots.some((s) => s.phase === 'processing');
  useEffect(() => {
    if (!anyProcessing) return undefined;
    const timer = window.setInterval(async () => {
      for (const s of slotsRef.current.filter((x) => x.phase === 'processing' && x.sourceId)) {
        try {
          const res = await api.get<ReplayStatus>(`/api/replay/status/${s.sourceId}`);
          const st = res.data;
          if (st.error_message) {
            update(s.slot, { status: st, phase: 'error', error: st.error_message });
          } else {
            update(s.slot, { status: st, phase: st.replay?.finished ? 'done' : 'processing' });
          }
        } catch (err: any) {
          update(s.slot, { phase: 'error', error: detailOf(err, 'Mất trạng thái phiên phát lại.') });
        }
      }
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [anyProcessing, update]);

  const runMatch = async () => {
    if (!effectiveJobId || doneSlots.length === 0) return;
    setIsMatching(true);
    setMatchError(null);
    try {
      const res = await api.post<ReplayMatchResponse>('/api/replay/match', {
        job_id: effectiveJobId, threshold,
        sources: doneSlots.map((s) => ({
          slot: s.slot, source_id: s.sourceId, start_offset_sec: s.offset,
        })),
      }, { timeout: 600000 });
      setMatchResult(res.data);
      setSlots((prev) => prev.map((s) => ({
        ...s, result: res.data.cameras.find((c) => c.slot === s.slot) || s.result,
      })));
    } catch (err: any) {
      setMatchError(detailOf(err, 'Đối soát thất bại.'));
    } finally {
      setIsMatching(false);
    }
  };

  const resetAll = () => {
    slots.forEach((s) => s.previewUrl && URL.revokeObjectURL(s.previewUrl));
    setSlots(emptySlots());
    setMatchResult(null);
    setMatchError(null);
  };

  if (!hasGeneratedImages(jobResult)) {
    return (
      <div data-testid="camera-grid-locked" className="p-8 max-w-xl mx-auto text-center space-y-2">
        <Camera className="w-8 h-8 mx-auto text-[#9CA3AF]" />
        <h2 className="text-sm font-bold text-[#111827]">Chưa có ảnh tạo sinh</h2>
        <p className="text-xs text-[#6B7280]">
          Màn hình 8 camera chỉ hoạt động sau khi đã chạy xong bước sinh ảnh FADING.
          Hãy quay lại tab Tìm kiếm, chạy pipeline, rồi mở lại tab này.
        </p>
      </div>
    );
  }

  return (
    <div data-testid="camera-grid-page" className="p-4 sm:p-6 space-y-5 max-w-[1400px] mx-auto">
      {/* Ảnh tạo sinh dùng để đối soát + cấu hình */}
      <section className="bg-white border border-[#E5E7EB] rounded-xl p-4 shadow-xs flex flex-col lg:flex-row gap-4 lg:items-center">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex -space-x-2">
            {refImages.map((r) => (
              <img key={r.age} src={r.url} alt={`Ảnh sinh ${r.age} tuổi`} title={`${r.age} tuổi`}
                   className="w-11 h-11 rounded-lg object-cover border-2 border-white shadow-xs" />
            ))}
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-bold text-[#111827]">8 camera — video thay thế</h2>
            <p className="text-[11px] text-[#6B7280] truncate">
              Đối soát với {refImages.length} ảnh tạo sinh ({refImages.map((r) => r.age).join(', ')} tuổi) · job {effectiveJobId?.slice(0, 8)}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-end gap-3 lg:ml-auto text-[11px] text-[#6B7280]">
          <label className="flex flex-col gap-1">Khung/giây
            <input type="number" min={0.5} max={10} step={0.5} value={fps} disabled={busy}
                   onChange={(e) => setFps(Number(e.target.value))}
                   className="w-20 border border-[#E5E7EB] rounded-lg px-2 py-1.5 text-xs text-[#111827]" />
          </label>
          <label className="flex flex-col gap-1">Crop/tracklet (K)
            <input type="number" min={1} max={5} value={maxCandidates} disabled={busy}
                   onChange={(e) => setMaxCandidates(Number(e.target.value))}
                   className="w-20 border border-[#E5E7EB] rounded-lg px-2 py-1.5 text-xs text-[#111827]" />
          </label>
          <label className="flex flex-col gap-1">Chất lượng tối thiểu
            <input type="number" min={0} max={0.9} step={0.05} value={minQuality} disabled={busy}
                   onChange={(e) => setMinQuality(Number(e.target.value))}
                   className="w-20 border border-[#E5E7EB] rounded-lg px-2 py-1.5 text-xs text-[#111827]" />
          </label>
          <label className="flex flex-col gap-1" title="0.45 giữ ~90% cặp cùng người gần tuổi (FG-NET); 0.6 bỏ sót ~40%">
            Ngưỡng khớp
            <input type="number" min={0.2} max={0.9} step={0.05} value={threshold}
                   onChange={(e) => setThreshold(Number(e.target.value))}
                   className="w-20 border border-[#E5E7EB] rounded-lg px-2 py-1.5 text-xs text-[#111827]" />
          </label>
          <button type="button" onClick={runAll} disabled={busy || readySlots.length === 0}
                  data-testid="replay-run-all"
                  className="inline-flex items-center gap-1.5 bg-[#E8804A] hover:bg-[#D97706] disabled:opacity-50 text-white text-xs font-bold py-2 px-3 rounded-lg">
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            Chạy {readySlots.length || ''} camera
          </button>
          <button type="button" onClick={runMatch} disabled={busy || isMatching || doneSlots.length === 0}
                  data-testid="replay-match"
                  className="inline-flex items-center gap-1.5 bg-[#111827] hover:bg-black disabled:opacity-50 text-white text-xs font-bold py-2 px-3 rounded-lg">
            {isMatching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
            Đối soát {doneSlots.length || ''} camera
          </button>
          <button type="button" onClick={resetAll} disabled={busy} title="Xóa toàn bộ video"
                  className="inline-flex items-center gap-1 text-xs text-[#6B7280] hover:text-[#111827] py-2 px-2 disabled:opacity-50">
            <RotateCcw className="w-3.5 h-3.5" /> Làm lại
          </button>
        </div>
      </section>

      {matchError && (
        <div className="text-xs text-[#B91C1C] bg-[#FEF2F2] border border-[#FECACA] rounded-lg px-3 py-2">{matchError}</div>
      )}

      {/* Lưới 8 camera */}
      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {slots.map((s) => (
          <CameraSlot key={s.slot} state={s} disabled={busy}
                      onPick={(f) => pickFile(s.slot, f)}
                      onOffset={(v) => update(s.slot, { offset: v })} />
        ))}
      </section>

      {/* Dòng thời gian xuất hiện qua các camera */}
      {matchResult && (
        <section data-testid="replay-timeline" className="bg-white border border-[#E5E7EB] rounded-xl p-4 shadow-xs space-y-3">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-[#E8804A]" />
            <h3 className="text-sm font-bold text-[#111827]">Dòng thời gian xuất hiện</h3>
            <span className="text-[11px] text-[#6B7280]">
              {matchResult.found_on_cameras.length
                ? `Khớp ở camera ${matchResult.found_on_cameras.join(', ')} (ngưỡng ${matchResult.threshold})`
                : `Không camera nào vượt ngưỡng ${matchResult.threshold}`}
            </span>
          </div>
          {matchResult.timeline.length > 0 && (
            <ol className="flex gap-3 overflow-x-auto pb-1">
              {matchResult.timeline.map((t, i) => (
                <li key={`${t.slot}-${t.camera_time_sec}-${i}`} className="shrink-0 w-32 text-center">
                  <img src={resolveMediaUrl(t.face_image_url)} alt="crop khớp"
                       className="w-32 h-32 object-cover rounded-lg border border-[#E5E7EB]" />
                  <div className="text-[11px] font-semibold text-[#111827] mt-1">Camera {t.slot}</div>
                  <div className="text-[11px] text-[#6B7280]">{formatTime(t.camera_time_sec)} · {(t.score * 100).toFixed(1)}%</div>
                </li>
              ))}
            </ol>
          )}
          <p className="text-[10px] text-[#9CA3AF]">
            Kết quả đã lưu vào search_runs (mỗi camera một lượt). Đây là gợi ý cho người kiểm tra — cần xác nhận bằng mắt.
          </p>
        </section>
      )}
    </div>
  );
};

const formatTime = (sec: number): string => {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
};

interface SlotProps {
  state: SlotState;
  disabled: boolean;
  onPick: (file: File | null) => void;
  onOffset: (v: number) => void;
}

const CameraSlot: React.FC<SlotProps> = ({ state, disabled, onPick, onOffset }) => {
  const { slot, previewUrl, phase, status, error, result, file, offset } = state;
  const progress = status?.replay?.progress ?? (phase === 'done' ? 1 : 0);
  const best = result?.best_match;
  const locked = disabled || phase === 'uploading' || phase === 'processing';
  return (
    <div data-testid={`camera-slot-${slot}`}
         className={`bg-white border rounded-xl p-3 shadow-xs flex flex-col gap-2 ${
           best?.accepted ? 'border-[#16A34A] ring-1 ring-[#16A34A]' : 'border-[#E5E7EB]'}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold text-[#111827] flex items-center gap-1.5">
          <Camera className="w-3.5 h-3.5 text-[#E8804A]" /> Camera {slot}
        </span>
        <PhaseBadge phase={phase} accepted={!!best?.accepted} />
      </div>

      <div className="relative aspect-video bg-[#111827] rounded-lg overflow-hidden">
        {previewUrl ? (
          <video src={previewUrl} muted controls className="w-full h-full object-contain" />
        ) : (
          <label className={`absolute inset-0 flex flex-col items-center justify-center gap-1 text-[11px] text-[#9CA3AF] ${
            locked ? '' : 'cursor-pointer hover:text-white'}`}>
            <UploadCloud className="w-6 h-6" />
            Tải video thay camera
            <input type="file" accept={ACCEPT} className="hidden" disabled={locked}
                   data-testid={`camera-slot-${slot}-input`}
                   onChange={(e) => { onPick(e.target.files?.[0] || null); e.target.value = ''; }} />
          </label>
        )}
        {best && (
          <img src={resolveMediaUrl(best.face_image_url)} alt="crop khớp nhất"
               className="absolute right-1.5 bottom-1.5 w-14 h-14 object-cover rounded-md border-2 border-white shadow" />
        )}
      </div>

      {(phase === 'processing' || phase === 'done') && (
        <div className="h-1.5 bg-[#F3F4F6] rounded-full overflow-hidden">
          <div className="h-full bg-[#E8804A] transition-all" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      )}

      <div className="flex items-center gap-2 text-[11px] text-[#6B7280]">
        <span className="truncate flex-1" title={file?.name}>{file ? file.name : 'Chưa có video'}</span>
        {file && (
          <label className={`shrink-0 font-semibold ${locked ? 'opacity-40' : 'cursor-pointer text-[#E8804A] hover:underline'}`}>
            Đổi
            <input type="file" accept={ACCEPT} className="hidden" disabled={locked}
                   onChange={(e) => { onPick(e.target.files?.[0] || null); e.target.value = ''; }} />
          </label>
        )}
      </div>

      <label className="flex items-center justify-between text-[11px] text-[#6B7280]"
             title="Giây lệch so với camera khác (video quay cùng lúc = 0)">
        Lệch giờ (giây)
        <input type="number" min={0} step={1} value={offset} disabled={locked}
               onChange={(e) => onOffset(Math.max(0, Number(e.target.value)))}
               className="w-20 border border-[#E5E7EB] rounded-md px-2 py-1 text-xs text-[#111827]" />
      </label>

      {status && (
        <div className="grid grid-cols-4 gap-1 text-center text-[10px] text-[#6B7280]">
          <Stat label="khung" value={status.frames_sampled} />
          <Stat label="mặt" value={status.faces_found} />
          <Stat label="tracklet" value={result?.tracks_closed ?? status.tracks_closed} />
          <Stat label="crop" value={result?.crops ?? status.crops_persisted} />
        </div>
      )}

      {best && (
        <div className={`text-[11px] rounded-md px-2 py-1.5 ${best.accepted ? 'bg-[#F0FDF4] text-[#166534]' : 'bg-[#F9FAFB] text-[#6B7280]'}`}>
          Khớp nhất {(best.score * 100).toFixed(1)}% với ảnh {best.best_age} tuổi lúc {formatTime(best.camera_time_sec)}
          {result && result.accepted > 0 ? ` · ${result.accepted} crop vượt ngưỡng` : ''}
        </div>
      )}
      {error && (
        <div className="text-[11px] text-[#B91C1C] flex gap-1"><AlertTriangle className="w-3.5 h-3.5 shrink-0" />{error}</div>
      )}
    </div>
  );
};

const Stat: React.FC<{ label: string; value?: number | null }> = ({ label, value }) => (
  <div className="bg-[#F9FAFB] rounded-md py-1">
    <div className="text-xs font-bold text-[#111827]">{value ?? '–'}</div>
    {label}
  </div>
);

const PhaseBadge: React.FC<{ phase: Phase; accepted: boolean }> = ({ phase, accepted }) => {
  if (accepted) {
    return <span className="text-[10px] font-bold text-[#16A34A] flex items-center gap-1"><CheckCircle2 className="w-3 h-3" />Khớp</span>;
  }
  const map: Record<Phase, [string, string]> = {
    empty: ['Trống', 'text-[#9CA3AF]'],
    ready: ['Sẵn sàng', 'text-[#2563EB]'],
    uploading: ['Đang tải lên', 'text-[#D97706]'],
    processing: ['Đang xử lý', 'text-[#D97706]'],
    done: ['Xong', 'text-[#16A34A]'],
    error: ['Lỗi', 'text-[#B91C1C]'],
  };
  const [text, cls] = map[phase];
  return <span className={`text-[10px] font-bold ${cls}`}>{text}</span>;
};

export default CameraGridPage;
