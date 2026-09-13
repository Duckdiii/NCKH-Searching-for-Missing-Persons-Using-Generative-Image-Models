import { JobResult } from '../types/api';

/**
 * Xuất dữ liệu bảng xếp hạng đối soát sang file CSV (UTF-8 có BOM để mở trực tiếp trong Excel)
 */
export function exportRankingToCSV(jobResult: JobResult): string {
  if (!jobResult || !jobResult.final_scores) {
    throw new Error('Không có dữ liệu kết quả để xuất CSV.');
  }

  const sortedScores = Object.entries(jobResult.final_scores).sort(
    ([, scoreA], [, scoreB]) => scoreB - scoreA
  );

  const timestamp = new Date().toLocaleString('vi-VN');
  const topScorePct = (jobResult.top_score * 100).toFixed(1);
  const acceptedText = jobResult.accepted ? 'ĐẠT NGƯỠNG XÁC THỰC (>=60%)' : 'DƯỚI NGƯỠNG CHẤP NHẬN (<60%)';
  const checkpoint = jobResult.pipeline_params?.checkpoint_name || 'specialized_unet (stable-diffusion-v1-5)';

  const lines: string[] = [
    '=== BÁO CÁO KẾT QUẢ ĐỐI SOÁT NHẬN DIỆN FADING PIPELINE ===',
    `"Mã phiên làm việc (Job ID):","${jobResult.job_id}"`,
    `"Thời gian trích xuất:","${timestamp}"`,
    `"Mô hình khuếch tán:","${checkpoint}"`,
    `"Đối tượng khớp cao nhất:","${jobResult.top_identity || 'N/A'}"`,
    `"Điểm tương đồng cao nhất:","${topScorePct}%"`,
    `"Mốc tuổi khớp dự đoán:","${jobResult.best_age ? `${jobResult.best_age} tuổi` : 'N/A'}"`,
    `"Kết luận hệ thống:","${acceptedText}"`,
    '',
    '"Hạng","Định danh Gallery (File)","Điểm tương đồng (ID Score)","Cosine Similarity thô","Đánh giá mức độ khớp","Kết luận xác thực"',
  ];

  sortedScores.forEach(([id, score], idx) => {
    const pct = (score * 100).toFixed(1) + '%';
    const rawCosine = score.toFixed(4);
    let evalText = 'Không khớp';
    if (score >= 0.6) evalText = 'Khớp cao (Đạt ngưỡng)';
    else if (score >= 0.3) evalText = 'Cần thẩm tra (Trung bình)';

    const idBase = id.replace(/\.[^/.]+$/, '');
    const topBase = (jobResult.top_identity || '').replace(/\.[^/.]+$/, '');
    const isMatch = (id === jobResult.top_identity || idBase === topBase) && jobResult.accepted;
    const matchStatus = isMatch
      ? 'Xác thực thành công (Khớp mục tiêu)'
      : score >= 0.6
      ? 'Đạt ngưỡng khớp'
      : 'Dưới ngưỡng';

    lines.push(`"${idx + 1}","${id}","${pct}","${rawCosine}","${evalText}","${matchStatus}"`);
  });

  const csvContent = lines.join('\r\n');

  // Kích hoạt download client-side với UTF-8 BOM (\uFEFF)
  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `FADING_DoiSoat_${jobResult.job_id.slice(0, 8)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  return csvContent;
}

/**
 * Mở cửa sổ in ấn chuyên nghiệp / xuất PDF chuẩn tài liệu biên bản đối soát A4
 */
export function exportReportPrint(jobResult: JobResult, originalFaceUrl?: string, bestEditedUrl?: string, galleryMatchUrl?: string): void {
  if (typeof window === 'undefined') return;

  const sortedScores = Object.entries(jobResult.final_scores || {}).sort(
    ([, scoreA], [, scoreB]) => scoreB - scoreA
  );

  const timestamp = new Date().toLocaleString('vi-VN');
  const topScorePct = (jobResult.top_score * 100).toFixed(1);
  const acceptedStatus = jobResult.accepted
    ? '<span style="color: #15803d; font-weight: bold;">XÁC THỰC THÀNH CÔNG (Khớp danh tính)</span>'
    : '<span style="color: #b91c1c; font-weight: bold;">DƯỚI NGƯỠNG CHẤP NHẬN (Cần giám định thêm)</span>';

  const checkpoint = jobResult.pipeline_params?.checkpoint_name || 'specialized_unet (stable-diffusion-v1-5)';
  const steps = jobResult.pipeline_params?.num_inference_steps || 50;
  const guidance = jobResult.pipeline_params?.guidance_scale || 7.5;

  const printWindow = window.open('', '_blank', 'width=900,height=800');
  if (!printWindow) {
    alert('Trình duyệt đã chặn cửa sổ pop-up in ấn. Hãy cho phép pop-up để xuất PDF.');
    return;
  }

  const html = `<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>Biên bản đối soát nhận diện FADING - ${jobResult.job_id.slice(0, 8)}</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #111827; line-height: 1.4; font-size: 13px; }
    h1 { font-size: 18px; text-transform: uppercase; margin: 0 0 4px 0; text-align: center; color: #0f172a; }
    h2 { font-size: 13px; text-align: center; color: #64748b; margin: 0 0 20px 0; font-weight: normal; }
    .meta-box { border: 1px solid #e2e8f0; background: #f8fafc; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .meta-item { display: flex; gap: 6px; font-size: 12px; }
    .meta-label { color: #64748b; font-weight: 500; min-width: 140px; }
    .meta-value { color: #0f172a; font-weight: 600; }
    .images-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 24px; text-align: center; }
    .image-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; background: #ffffff; }
    .image-card img { width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 6px; border: 1px solid #cbd5e1; }
    .image-title { font-size: 11px; font-weight: bold; margin-bottom: 6px; text-transform: uppercase; color: #334155; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
    th { background: #f1f5f9; color: #475569; padding: 8px 12px; text-align: left; border: 1px solid #e2e8f0; font-weight: 600; }
    td { padding: 8px 12px; border: 1px solid #e2e8f0; }
    .text-right { text-align: right; }
    .font-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .signature-grid { display: grid; grid-template-columns: 1fr 1fr; margin-top: 40px; text-align: center; }
    .signature-title { font-weight: bold; margin-bottom: 50px; }
    @media print {
      body { margin: 15mm; }
      .no-print { display: none; }
    }
  </style>
</head>
<body>
  <div class="no-print" style="margin-bottom: 16px; text-align: right;">
    <button onclick="window.print()" style="padding: 8px 16px; background: #e8804a; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 13px;">
      🖨️ In hoặc Lưu thành PDF
    </button>
  </div>

  <h1>Báo cáo Kết quả Đối soát Nhận diện Người mất tích</h1>
  <h2>Hệ thống FADING (Diffusion-based Dual-Attention Age Progression & FAISS)</h2>

  <div class="meta-box">
    <div class="meta-item"><span class="meta-label">Mã phiên (Job ID):</span><span class="meta-value font-mono">${jobResult.job_id}</span></div>
    <div class="meta-item"><span class="meta-label">Thời gian thực hiện:</span><span class="meta-value">${timestamp}</span></div>
    <div class="meta-item"><span class="meta-label">Mô hình khuếch tán:</span><span class="meta-value">${checkpoint}</span></div>
    <div class="meta-item"><span class="meta-label">Tham số lấy mẫu:</span><span class="meta-value">${steps} steps (Guidance: ${guidance})</span></div>
    <div class="meta-item"><span class="meta-label">Đối tượng khớp nhất:</span><span class="meta-value font-mono">${jobResult.top_identity || 'N/A'}</span></div>
    <div class="meta-item"><span class="meta-label">Điểm số cao nhất:</span><span class="meta-value">${topScorePct}%</span></div>
    <div class="meta-item" style="grid-column: span 2;"><span class="meta-label">Kết luận giám định:</span><span class="meta-value">${acceptedStatus}</span></div>
  </div>

  <div class="images-grid">
    <div class="image-card">
      <div class="image-title">1. Ảnh gốc lúc nhỏ</div>
      ${originalFaceUrl ? `<img src="${originalFaceUrl}" alt="Ảnh gốc" />` : '<div style="height: 180px; display: flex; align-items: center; justify-content: center; background: #f1f5f9;">Không có ảnh</div>'}
    </div>
    <div class="image-card">
      <div class="image-title">2. FADING dự đoán (${jobResult.best_age || 'Mốc'} tuổi)</div>
      ${bestEditedUrl ? `<img src="${bestEditedUrl}" alt="FADING dự đoán" />` : '<div style="height: 180px; display: flex; align-items: center; justify-content: center; background: #f1f5f9;">Không có ảnh</div>'}
    </div>
    <div class="image-card">
      <div class="image-title">3. Ảnh Gallery đối chứng</div>
      ${galleryMatchUrl ? `<img src="${galleryMatchUrl}" alt="Gallery" />` : '<div style="height: 180px; display: flex; align-items: center; justify-content: center; background: #f1f5f9;">Không có ảnh</div>'}
    </div>
  </div>

  <h3 style="font-size: 14px; margin-bottom: 6px; text-transform: uppercase;">Bảng xếp hạng độ tương đồng nhận diện</h3>
  <table>
    <thead>
      <tr>
        <th style="width: 40px;">#</th>
        <th>Định danh Gallery (File)</th>
        <th class="text-right" style="width: 120px;">ID Score (%)</th>
        <th class="text-right" style="width: 120px;">Cosine thô</th>
        <th style="width: 140px;">Đánh giá</th>
      </tr>
    </thead>
    <tbody>
      ${sortedScores
        .map(
          ([id, score], idx) => `
        <tr>
          <td>${idx + 1}</td>
          <td class="font-mono">${id}</td>
          <td class="text-right font-mono" style="font-weight: bold;">${(score * 100).toFixed(1)}%</td>
          <td class="text-right font-mono">${score.toFixed(4)}</td>
          <td>${score >= 0.6 ? '<span style="color: #15803d; font-weight: bold;">Khớp cao (≥60%)</span>' : score >= 0.3 ? '<span style="color: #b45309;">Cần thẩm tra</span>' : '<span style="color: #b91c1c;">Không khớp</span>'}</td>
        </tr>
      `
        )
        .join('')}
    </tbody>
  </table>

  <div class="signature-grid">
    <div>
      <div class="signature-title">CÁN BỘ GIÁM ĐỊNH</div>
      <div>(Ký và ghi rõ họ tên)</div>
    </div>
    <div>
      <div class="signature-title">THỦ TRƯỞNG ĐƠN VỊ</div>
      <div>(Ký tên và đóng dấu)</div>
    </div>
  </div>
</body>
</html>`;

  printWindow.document.open();
  printWindow.document.write(html);
  printWindow.document.close();
}
