import React, { useEffect, useState } from 'react';
import { CalendarClock, Database, Gauge, ImageUp, Loader2, AlertTriangle } from 'lucide-react';
import { api, getBaseUrl } from '../api/client';

type ToolTab = 'plan' | 'forget' | 'rank' | 'store';

const detailOf = (err: any, fallback: string): string => {
  let d: unknown = err?.response?.data?.detail;
  if (Array.isArray(d)) d = d.map((x: any) => x?.msg || JSON.stringify(x)).join('; ');
  return (d as string) || fallback;
};

const pct = (v?: number | null) => (v == null ? '–' : `${(v * 100).toFixed(1)}%`);
const gb = (b?: number | null) => (b == null ? '–' : `${(b / 1e9).toFixed(2)} GB`);

const Card: React.FC<{ title: string; icon?: React.ReactNode; children: React.ReactNode; testId?: string }> = ({ title, icon, children, testId }) => (
  <section data-testid={testId} className="bg-white border border-[#E5E7EB] rounded-xl p-4 shadow-xs space-y-3">
    <h3 className="text-sm font-bold text-[#111827] flex items-center gap-2">{icon}{title}</h3>
    {children}
  </section>
);

const Num: React.FC<{ label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; title?: string }> = ({ label, value, onChange, step = 1, min, max, title }) => (
  <label className="flex flex-col gap-1 text-[11px] text-[#6B7280]" title={title}>
    {label}
    <input type="number" value={value} step={step} min={min} max={max}
           onChange={(e) => onChange(Number(e.target.value))}
           className="w-28 border border-[#E5E7EB] rounded-lg px-2 py-1.5 text-xs text-[#111827]" />
  </label>
);

const Btn: React.FC<{ onClick: () => void; busy?: boolean; disabled?: boolean; children: React.ReactNode; testId?: string; dark?: boolean }> = ({ onClick, busy, disabled, children, testId, dark }) => (
  <button type="button" onClick={onClick} disabled={busy || disabled} data-testid={testId}
          className={`inline-flex items-center gap-1.5 text-white text-xs font-bold py-2 px-3 rounded-lg disabled:opacity-50 ${
            dark ? 'bg-[#111827] hover:bg-black' : 'bg-[#E8804A] hover:bg-[#D97706]'}`}>
    {busy && <Loader2 className="w-3.5 h-3.5 animate-spin" />}{children}
  </button>
);

const ErrorLine: React.FC<{ msg: string | null }> = ({ msg }) =>
  msg ? <div className="text-xs text-[#B91C1C] flex gap-1"><AlertTriangle className="w-3.5 h-3.5 shrink-0" />{msg}</div> : null;

// ---------------------------------------------------------------- 1. Kế hoạch camera
const PlanTool: React.FC = () => {
  const [g, setG] = useState({
    motion: 'approaching', hfov_deg: 60, image_width_px: 3840, distance_m: 4, near_m: 1.5, far_m: 12,
    walk_speed_mps: 1.4, camera_fps: 25, fps_budget: 5, people_per_hour: 120, storage_gb: 0,
  });
  const [plan, setPlan] = useState<any>(null);
  const [opt, setOpt] = useState<any>(null);
  const [busy, setBusy] = useState<'plan' | 'opt' | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const set = (k: string) => (v: number | string) => setG((p) => ({ ...p, [k]: v }));
  const body = () => {
    const { storage_gb, ...rest } = g;
    return { ...rest, ...(storage_gb > 0 ? { storage_bytes: storage_gb * 1e9 } : {}) };
  };
  const run = async (kind: 'plan' | 'opt') => {
    setBusy(kind); setErr(null);
    try {
      if (kind === 'plan') {
        const { storage_bytes, ...b } = body() as any;
        setPlan((await api.post('/api/cameras/sampling-plan', b)).data);
      } else {
        setOpt((await api.post('/api/ops/pipeline/optimize', body(), { timeout: 120000 })).data);
      }
    } catch (e: any) { setErr(detailOf(e, 'Không tính được kế hoạch.')); }
    finally { setBusy(null); }
  };
  const rec = opt?.recommended;
  return (
    <div className="space-y-4">
      <Card title="Thông số camera và lưu lượng" icon={<Gauge className="w-4 h-4 text-[#E8804A]" />}>
        <div className="flex flex-wrap gap-3 items-end">
          <label className="flex flex-col gap-1 text-[11px] text-[#6B7280]">Hướng đi
            <select value={g.motion} onChange={(e) => set('motion')(e.target.value)}
                    className="border border-[#E5E7EB] rounded-lg px-2 py-1.5 text-xs text-[#111827]">
              <option value="approaching">Đi về phía camera</option>
              <option value="crossing">Đi ngang qua camera</option>
            </select>
          </label>
          <Num label="Góc nhìn ngang (°)" value={g.hfov_deg} onChange={set('hfov_deg')} min={10} max={170} />
          <Num label="Độ rộng khung (px)" value={g.image_width_px} onChange={set('image_width_px')} step={160} />
          {g.motion === 'crossing'
            ? <Num label="Cách lối đi (m)" value={g.distance_m} onChange={set('distance_m')} step={0.5} />
            : <>
                <Num label="Điểm gần (m)" value={g.near_m} onChange={set('near_m')} step={0.5} />
                <Num label="Điểm xa (m)" value={g.far_m} onChange={set('far_m')} step={0.5} />
              </>}
          <Num label="Tốc độ đi (m/s)" value={g.walk_speed_mps} onChange={set('walk_speed_mps')} step={0.1} />
          <Num label="Fps camera" value={g.camera_fps} onChange={set('camera_fps')} />
          <Num label="Ngân sách GPU (fps)" value={g.fps_budget} onChange={set('fps_budget')} step={0.5} />
          <Num label="Người/giờ" value={g.people_per_hour} onChange={set('people_per_hour')} step={10} />
          <Num label="Ổ đĩa tối đa (GB, 0 = không giới hạn)" value={g.storage_gb} onChange={set('storage_gb')} step={0.1} />
        </div>
        <div className="flex gap-2">
          <Btn onClick={() => run('plan')} busy={busy === 'plan'} testId="tool-plan-run">Tính kế hoạch lấy mẫu</Btn>
          <Btn onClick={() => run('opt')} busy={busy === 'opt'} dark testId="tool-opt-run">Tối ưu fps × số crop</Btn>
        </div>
        <ErrorLine msg={err} />
      </Card>

      {plan && (
        <Card title="Kế hoạch lấy mẫu" testId="tool-plan-result">
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2 text-center">
            {[
              ['fps khuyến nghị', plan.fps], ['ràng buộc quyết định', plan.binding],
              ['mặt (px)', plan.face_px], ['xa nhất nhận diện (m)', plan.max_recognition_distance_m],
              ['ở trong vùng (s)', plan.dwell_s], ['fps để bám vết', plan.fps_tracking],
              ['fps để đủ ảnh', plan.fps_coverage ?? '–'], ['khung / lượt', plan.samples_per_pass],
              ['tracklet / lượt', plan.tracklets_per_pass], ['timeout tracker (s)', plan.tracker_timeout_sec],
            ].map(([k, v]) => (
              <div key={String(k)} className="bg-[#F9FAFB] rounded-lg p-2">
                <div className="text-sm font-bold text-[#111827]">{String(v)}</div>
                <div className="text-[10px] text-[#6B7280]">{k}</div>
              </div>
            ))}
          </div>
          {plan.warnings?.map((w: string) => (
            <div key={w} className="text-[11px] text-[#92400E] bg-[#FFFBEB] border border-[#FDE68A] rounded-md px-2 py-1">{w}</div>
          ))}
        </Card>
      )}

      {opt && (
        <Card title="Cấu hình tối ưu" testId="tool-opt-result">
          {rec && (
            <div className="text-xs text-[#111827] bg-[#F0FDF4] border border-[#BBF7D0] rounded-lg p-3">
              Khuyến nghị: <b>{rec.fps} fps</b>, <b>{rec.k ?? '— (bị tách tracklet)'} crop/tracklet</b>,
              chất lượng tối thiểu <b>{rec.q_min}</b> → tìm thấy <b>{pct(rec.p_found)}</b>,
              {' '}{rec.crops_per_pass} crop & {rec.tracklets_per_pass} tracklet mỗi lượt, lưu {gb(rec.storage_bytes)},
              {' '}{rec.false_alarms_per_query} báo động giả/truy vấn.
              <div className="text-[10px] text-[#6B7280] mt-1">
                Mô hình chất lượng: {opt.quality_model} · tốt nhất có thể {pct(opt.best_p_found)} · lưu {opt.retention_days?.thin} ngày đủ ảnh, quên sau {opt.retention_days?.forget} ngày
              </div>
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead><tr className="text-[#6B7280] text-left">
                {['fps', 'K', 'q_min', 'Tìm thấy', 'crop/lượt', 'tracklet/lượt', 'Lưu', 'Báo động giả'].map((h) => <th key={h} className="py-1 pr-3">{h}</th>)}
              </tr></thead>
              <tbody>
                {(opt.pareto || []).slice(0, 15).map((r: any, i: number) => (
                  <tr key={i} className={`border-t border-[#F3F4F6] ${r === rec || (r.fps === rec?.fps && r.k === rec?.k && r.q_min === rec?.q_min) ? 'bg-[#F0FDF4] font-semibold' : ''}`}>
                    <td className="py-1 pr-3">{r.fps}</td><td className="pr-3">{r.k ?? 'tách'}</td><td className="pr-3">{r.q_min}</td>
                    <td className="pr-3">{pct(r.p_found)}</td><td className="pr-3">{r.crops_per_pass}</td>
                    <td className="pr-3">{r.tracklets_per_pass}</td><td className="pr-3">{gb(r.storage_bytes)}</td>
                    <td className="pr-3">{r.false_alarms_per_query}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[10px] text-[#9CA3AF] mt-1">Tập Pareto: không cấu hình nào vừa rẻ hơn vừa tốt hơn. Dòng xanh là khuyến nghị.</p>
          </div>
          {opt.warnings?.map((w: string) => (
            <div key={w} className="text-[11px] text-[#92400E] bg-[#FFFBEB] border border-[#FDE68A] rounded-md px-2 py-1">{w}</div>
          ))}
        </Card>
      )}
    </div>
  );
};

// ---------------------------------------------------------------- 2. Lịch quên
const ForgetTool: React.FC = () => {
  const presets = [[1, 0.5], [1, 0.1], [5, 0.9], [20, 1.0]];
  const [rows, setRows] = useState<any[]>([]);
  const [days, setDays] = useState(3);
  const [quality, setQuality] = useState(0.7);
  const [custom, setCustom] = useState<any>(null);
  const [plan, setPlan] = useState<any>(null);
  const [dry, setDry] = useState<any>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const sched = (d: number, q: number) => api.get('/api/ops/forgetting/schedule', { params: { sighting_days: d, quality: q } }).then((r) => r.data);
  useEffect(() => {
    Promise.all(presets.map(([d, q]) => sched(d, q))).then(setRows).catch((e) => setErr(detailOf(e, 'Không đọc được lịch quên.')));
  }, []);
  const call = async (kind: string, fn: () => Promise<any>, setter: (v: any) => void) => {
    setBusy(kind); setErr(null);
    try { setter(await fn()); } catch (e: any) { setErr(detailOf(e, 'Lỗi gọi API.')); } finally { setBusy(null); }
  };
  return (
    <div className="space-y-4">
      <Card title="Sau bao lâu thì máy quên một người?" icon={<CalendarClock className="w-4 h-4 text-[#E8804A]" />} testId="tool-forget-schedule">
        <p className="text-[11px] text-[#6B7280]">
          Độ nhớ R = exp(−Δt / S). S tăng khi người đó xuất hiện lại vào nhiều ngày và khi ảnh tốt; giảm khi ổ gần đầy.
          Lược = chỉ giữ crop tốt nhất. Ảnh từng khớp truy vấn tìm người không bao giờ bị quên tự động.
        </p>
        <table className="w-full text-[11px]">
          <thead><tr className="text-left text-[#6B7280]">{['Số ngày xuất hiện', 'Chất lượng', 'Độ bền S (ngày)', 'Lược sau', 'Quên sau'].map((h) => <th key={h} className="py-1">{h}</th>)}</tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.sighting_days}-${r.quality}`} className="border-t border-[#F3F4F6]">
                <td className="py-1">{r.sighting_days}</td><td>{r.quality}</td><td>{r.stability_days}</td>
                <td>{r.thin_after_days} ngày</td><td className="font-semibold">{r.forget_after_days} ngày</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="flex flex-wrap gap-3 items-end">
          <Num label="Số ngày xuất hiện" value={days} onChange={setDays} min={1} />
          <Num label="Chất lượng (0–1)" value={quality} onChange={setQuality} step={0.05} min={0} max={1} />
          <Btn onClick={() => call('custom', () => sched(days, quality), setCustom)} busy={busy === 'custom'}>Tính</Btn>
          {custom && <span className="text-xs text-[#111827]">Lược sau <b>{custom.thin_after_days}</b> ngày, quên sau <b>{custom.forget_after_days}</b> ngày (ổ đang {pct(custom.usage_ratio)})</span>}
        </div>
      </Card>
      <Card title="Áp dụng lên dữ liệu trong DB (chỉ chạy thử)" icon={<Database className="w-4 h-4 text-[#E8804A]" />} testId="tool-forget-db">
        <p className="text-[11px] text-[#6B7280]">
          Giao diện chỉ chạy thử — không xóa gì. Quên thật cần FORGETTING_ENABLED=true và gọi API vận hành.
        </p>
        <div className="flex gap-2">
          <Btn onClick={() => call('plan', () => api.post('/api/ops/forgetting/plan').then((r) => r.data), setPlan)} busy={busy === 'plan'} testId="tool-forget-plan">Lập kế hoạch quên</Btn>
          <Btn onClick={() => call('dry', () => api.post('/api/ops/forgetting/run', { dry_run: true }).then((r) => r.data), setDry)} busy={busy === 'dry'} dark>Chạy thử</Btn>
        </div>
        {plan && (
          <div className="text-xs text-[#111827]">
            Còn nhớ {plan.counts?.keep} · cần lược {plan.counts?.thin} · cần quên {plan.counts?.forget} · đang giữ vì đã khớp {plan.counts?.hold}
            {' '}→ {plan.crops_thin} crop lược, {plan.crops_forget} crop quên, giải phóng {gb(plan.bytes_to_free)}
          </div>
        )}
        {dry && <div className="text-xs text-[#6B7280]">Chạy thử: sẽ lược {dry.thinned}, quên {dry.forgotten} crop, hết hạn {dry.tracklets_expired} tracklet (chưa xóa gì).</div>}
      </Card>
      <ErrorLine msg={err} />
    </div>
  );
};

// ---------------------------------------------------------------- 3. Chọn ảnh tốt nhất
const RankTool: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [mixed, setMixed] = useState(false);
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const form = new FormData();
      files.forEach((f) => form.append('files', f));
      form.append('mixed', String(mixed));
      setRes((await api.post('/api/tools/rank-faces', form, { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 300000 })).data);
    } catch (e: any) { setErr(detailOf(e, 'Không xếp hạng được ảnh.')); }
    finally { setBusy(false); }
  };
  const labels: Record<string, string> = {
    recognizability: 'nhận dạng', sharpness: 'độ nét', resolution: 'phân giải', exposure: 'phơi sáng',
    completeness: 'đầy đủ', pose: 'góc mặt', detection: 'phát hiện',
  };
  return (
    <Card title="Chọn ảnh mặt tốt nhất" icon={<ImageUp className="w-4 h-4 text-[#E8804A]" />} testId="tool-rank">
      <p className="text-[11px] text-[#6B7280]">
        Tải nhiều ảnh của cùng một người (ví dụ ảnh người nhà gửi) — hệ thống chấm 7 thành phần chất lượng và gợi ý ảnh nên dùng làm truy vấn.
      </p>
      <div className="flex flex-wrap gap-3 items-center">
        <input type="file" multiple accept="image/*" data-testid="tool-rank-input"
               onChange={(e) => setFiles(Array.from(e.target.files || []))} className="text-xs" />
        <label className="text-xs text-[#6B7280] flex items-center gap-1">
          <input type="checkbox" checked={mixed} onChange={(e) => setMixed(e.target.checked)} /> Nhiều người (bỏ kiểm tra cùng người)
        </label>
        <Btn onClick={run} busy={busy} disabled={files.length === 0} testId="tool-rank-run">Xếp hạng {files.length || ''} ảnh</Btn>
      </div>
      <ErrorLine msg={err} />
      {res?.warning && <div className="text-[11px] text-[#92400E] bg-[#FFFBEB] border border-[#FDE68A] rounded-md px-2 py-1">{res.warning}</div>}
      {res && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {res.ranked.map((r: any) => (
            <div key={r.file} className={`flex gap-3 border rounded-lg p-2 ${r.rank === 1 ? 'border-[#16A34A] ring-1 ring-[#16A34A]' : 'border-[#E5E7EB]'}`}>
              <img src={r.thumbnail} alt={r.file} className="w-20 h-20 object-cover rounded-md" />
              <div className="min-w-0 flex-1 text-[11px]">
                <div className="font-bold text-[#111827] truncate">#{r.rank} {r.file}</div>
                <div className="text-[#6B7280]">Điểm {r.score.toFixed(3)} · chất lượng {r.quality.toFixed(3)} · cùng người {r.consistency.toFixed(2)}</div>
                {r.gate
                  ? <div className="text-[#B91C1C]">Bị loại: {r.gate}</div>
                  : <div className="text-[#6B7280]">Yếu nhất: {labels[r.weakest] || r.weakest}</div>}
                <div className="mt-1 space-y-0.5">
                  {Object.entries(r.components).map(([k, v]: any) => (
                    <div key={k} className="flex items-center gap-1">
                      <span className="w-16 shrink-0 text-[10px] text-[#9CA3AF]">{labels[k] || k}</span>
                      <div className="flex-1 h-1 bg-[#F3F4F6] rounded"><div className="h-1 bg-[#E8804A] rounded" style={{ width: `${Math.round(v * 100)}%` }} /></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
          {res.no_face.map((n: any) => (
            <div key={n.file} className="border border-dashed border-[#E5E7EB] rounded-lg p-2 text-[11px] text-[#9CA3AF]">{n.file}: {n.reason}</div>
          ))}
        </div>
      )}
    </Card>
  );
};

// ---------------------------------------------------------------- 4. Kho crop
const StoreTool: React.FC = () => {
  const [sources, setSources] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [sel, setSel] = useState<string | null>(null);
  const [crops, setCrops] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const load = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.get('/api/search-sources', { params: { purpose: 'search', limit: 50 } });
      setSources(r.data.items); setTotal(r.data.total);
    } catch (e: any) { setErr(detailOf(e, 'Không đọc được kho dữ liệu (cần DB).')); }
    finally { setBusy(false); }
  };
  useEffect(() => { load(); }, []);
  const open = async (id: string) => {
    setSel(id); setCrops(null);
    try { setCrops((await api.get(`/api/search-sources/${id}/crops`, { params: { limit: 60 } })).data); }
    catch (e: any) { setErr(detailOf(e, 'Không đọc được crop.')); }
  };
  return (
    <Card title={`Kho dữ liệu crop (${total} nguồn)`} icon={<Database className="w-4 h-4 text-[#E8804A]" />} testId="tool-store">
      <div className="flex gap-2 items-center">
        <Btn onClick={load} busy={busy} dark>Tải lại</Btn>
        <span className="text-[11px] text-[#6B7280]">Nguồn camera lưu theo chính sách crop-only: chỉ crop mặt + embedding, không lưu khung hình.</span>
      </div>
      <ErrorLine msg={err} />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <ul className="space-y-1 max-h-[420px] overflow-y-auto">
          {sources.map((s) => (
            <li key={s.source_id}>
              <button type="button" onClick={() => open(s.source_id)}
                      className={`w-full text-left text-[11px] border rounded-lg px-2 py-1.5 ${sel === s.source_id ? 'border-[#E8804A] bg-[#FFF7ED]' : 'border-[#E5E7EB] hover:bg-[#F9FAFB]'}`}>
                <div className="font-semibold text-[#111827]">{s.kind} · {s.storage_policy}</div>
                <div className="text-[#6B7280] font-mono">{s.source_id.slice(0, 8)} · {new Date(s.created_at).toLocaleString('vi-VN')}</div>
              </button>
            </li>
          ))}
        </ul>
        <div className="lg:col-span-2">
          {crops && (
            <>
              <div className="text-[11px] text-[#6B7280] mb-2">{crops.total} crop</div>
              <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
                {crops.items.map((c: any) => (
                  <figure key={c.crop_id} className="text-center">
                    <img src={`${getBaseUrl()}/api/search-sources/crops/${c.crop_id}/thumb?edge=128`} alt="crop"
                         className="w-full aspect-square object-cover rounded-md border border-[#E5E7EB]" />
                    <figcaption className="text-[10px] text-[#6B7280]">{(c.offset_ms / 1000).toFixed(1)}s · {(c.det_score * 100).toFixed(0)}%</figcaption>
                  </figure>
                ))}
              </div>
            </>
          )}
          {!crops && <div className="text-[11px] text-[#9CA3AF]">Chọn một nguồn để xem crop.</div>}
        </div>
      </div>
    </Card>
  );
};

export const ToolsPage: React.FC = () => {
  const [tab, setTab] = useState<ToolTab>('plan');
  const tabs: [ToolTab, string][] = [
    ['plan', 'Kế hoạch camera & tối ưu'], ['forget', 'Lịch quên dữ liệu'],
    ['rank', 'Chọn ảnh tốt nhất'], ['store', 'Kho dữ liệu'],
  ];
  return (
    <div data-testid="tools-page" className="p-4 sm:p-6 space-y-4 max-w-[1400px] mx-auto">
      <div className="flex flex-wrap gap-2">
        {tabs.map(([k, label]) => (
          <button key={k} type="button" onClick={() => setTab(k)} data-testid={`tools-tab-${k}`}
                  className={`text-xs font-semibold px-3 py-1.5 rounded-full border ${
                    tab === k ? 'bg-[#111827] text-white border-[#111827]' : 'bg-white text-[#6B7280] border-[#E5E7EB] hover:text-[#111827]'}`}>
            {label}
          </button>
        ))}
      </div>
      {tab === 'plan' && <PlanTool />}
      {tab === 'forget' && <ForgetTool />}
      {tab === 'rank' && <RankTool />}
      {tab === 'store' && <StoreTool />}
    </div>
  );
};

export default ToolsPage;
