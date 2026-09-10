import React from 'react';
import { useSearchStore } from '../store/useSearchStore';
import { CheckCircle2, XCircle, Award } from 'lucide-react';

export const ResultsGallery: React.FC = () => {
  const { jobResult, backendPort } = useSearchStore();

  if (!jobResult || jobResult.status !== 'done') {
    return null;
  }

  const { edited_images, final_scores, accepted, top_identity, top_score } = jobResult;
  const baseUrl = `http://127.0.0.1:${backendPort}`;

  return (
    <div className="bg-slate-800/90 border border-slate-700 rounded-xl p-6 shadow-2xl space-y-8 animate-in fade-in duration-500">
      {/* Banner kết quả chấp nhận / từ chối */}
      <div
        className={`p-5 rounded-xl border flex items-center justify-between ${
          accepted
            ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-200'
            : 'bg-rose-950/40 border-rose-500/50 text-rose-200'
        }`}
      >
        <div className="flex items-center gap-3">
          {accepted ? (
            <CheckCircle2 className="w-8 h-8 text-emerald-400 flex-shrink-0" />
          ) : (
            <XCircle className="w-8 h-8 text-rose-400 flex-shrink-0" />
          )}
          <div>
            <h4 className="text-lg font-bold">
              {accepted
                ? `Tìm thấy đối tượng phù hợp: "${top_identity}"`
                : 'Không tìm thấy kết quả đủ tin cậy trong Gallery'}
            </h4>
            <p className="text-xs opacity-90 mt-0.5">
              {accepted
                ? `Độ tương đồng cao nhất: ${(top_score * 100).toFixed(2)}% (Vượt ngưỡng chấp nhận 60%)`
                : `Điểm cao nhất: ${(top_score * 100).toFixed(2)}% (Thấp hơn ngưỡng tin cậy 60%)`}
            </p>
          </div>
        </div>

        {accepted && (
          <div className="flex items-center gap-1.5 bg-emerald-500/20 border border-emerald-500/40 px-3.5 py-1.5 rounded-full text-xs font-semibold text-emerald-300">
            <Award className="w-4 h-4" />
            <span>Xác thực thành công</span>
          </div>
        )}
      </div>

      {/* Lưới ảnh sinh ra theo từng độ tuổi */}
      <div>
        <h4 className="text-base font-semibold text-slate-100 mb-4 flex items-center gap-2">
          <span>Khuôn mặt dự đoán qua các độ tuổi (FADING Dual-Attention):</span>
        </h4>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
          {Object.entries(edited_images).map(([age, relPath]) => (
            <div
              key={age}
              className="bg-slate-900 border border-slate-700/80 rounded-xl overflow-hidden shadow-lg group hover:border-indigo-500 transition-all"
            >
              <div className="aspect-square overflow-hidden bg-black">
                <img
                  src={`${baseUrl}${relPath}`}
                  alt={`${age} tuổi`}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
              </div>
              <div className="p-3 text-center bg-slate-900/90 border-t border-slate-800">
                <span className="text-sm font-bold text-slate-200">{age} tuổi</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Bảng điểm xếp hạng FAISS đối soát */}
      <div>
        <h4 className="text-base font-semibold text-slate-100 mb-3">
          Bảng xếp hạng độ tương đồng nhận diện (Đã qua Ensemble & Rejection):
        </h4>

        <div className="overflow-x-auto rounded-lg border border-slate-700">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="bg-slate-900/80 text-slate-300 border-b border-slate-700">
                <th className="py-3 px-4 font-semibold">Xếp hạng</th>
                <th className="py-3 px-4 font-semibold">Định danh trong Gallery (ID)</th>
                <th className="py-3 px-4 font-semibold">Điểm tương đồng (Cosine Score)</th>
                <th className="py-3 px-4 font-semibold">Đánh giá</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 bg-slate-900/40">
              {Object.entries(final_scores).map(([id, score], idx) => {
                const isTop1 = idx === 0;
                return (
                  <tr
                    key={id}
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isTop1 && accepted ? 'bg-emerald-950/20' : ''
                    }`}
                  >
                    <td className="py-2.5 px-4 font-medium text-slate-400">#{idx + 1}</td>
                    <td className="py-2.5 px-4 font-mono font-semibold text-slate-200">{id}</td>
                    <td className="py-2.5 px-4 font-mono text-indigo-300">
                      {(score * 100).toFixed(2)}% ({score.toFixed(4)})
                    </td>
                    <td className="py-2.5 px-4">
                      {isTop1 && accepted ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                          Khớp danh tính
                        </span>
                      ) : (
                        <span className="text-xs text-slate-500">Ứng viên</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
