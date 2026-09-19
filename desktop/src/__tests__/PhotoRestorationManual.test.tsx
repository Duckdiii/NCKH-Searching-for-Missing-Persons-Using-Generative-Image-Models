import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { PhotoRestoration } from '../components/PhotoRestoration';

describe('PhotoRestoration - Chế độ Thủ công (Manual Mode)', () => {
  const sampleUrl = '/sample_face.png';

  it('1. Mặc định chọn Chế độ Tự động (khuyến nghị) và hiển thị giao diện cũ', () => {
    const handleConfirm = vi.fn();
    render(<PhotoRestoration originalFaceUrl={sampleUrl} onConfirm={handleConfirm} />);

    // Kiểm tra Radio
    const autoRadio = screen.getByRole('radio', { name: /Tự động \(khuyến nghị\)/i });
    const manualRadio = screen.getByRole('radio', { name: /Thủ công/i });

    expect(autoRadio).toBeChecked();
    expect(manualRadio).not.toBeChecked();

    // Trong chế độ Tự động, không có 3 tiêu đề "Khối 1", "Khối 2", "Khối 3"
    expect(screen.queryByText(/Khối 1: Padding viền/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Khối 2: Cân bằng trắng/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Khối 3: Độ nét CodeFormer/i)).not.toBeInTheDocument();

    // Có 2 checkbox cấu hình tự động
    expect(screen.getByText(/Cân bằng trắng \(Shades of Gray p=6\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Padding viền \(Adaptive Replicate\)/i)).toBeInTheDocument();
  });

  it('2. Khi chọn Chế độ Thủ công: hiển thị 3 khối bước độc lập', () => {
    const handleConfirm = vi.fn();
    render(<PhotoRestoration originalFaceUrl={sampleUrl} onConfirm={handleConfirm} />);

    const manualRadio = screen.getByRole('radio', { name: /Thủ công/i });
    fireEvent.click(manualRadio);

    expect(manualRadio).toBeChecked();

    // 3 khối bước xuất hiện
    expect(screen.getByText(/Khối 1: Padding viền/i)).toBeInTheDocument();
    expect(screen.getByText(/Khối 2: Cân bằng trắng/i)).toBeInTheDocument();
    expect(screen.getByText(/Khối 3: Độ nét CodeFormer/i)).toBeInTheDocument();

    // Có công tắc toggle độc lập cho Padding và White Balance
    expect(screen.getByTestId('toggle-padding')).toBeInTheDocument();
    expect(screen.getByTestId('toggle-whitebalance')).toBeInTheDocument();

    // Nút Eyedropper xuất hiện khi White Balance bật
    expect(screen.getByTestId('eyedropper-btn')).toBeInTheDocument();
  });

  it('3. Thao tác bật/tắt các công tắc độc lập trong Chế độ Thủ công', () => {
    const handleConfirm = vi.fn();
    render(<PhotoRestoration originalFaceUrl={sampleUrl} onConfirm={handleConfirm} />);

    fireEvent.click(screen.getByRole('radio', { name: /Thủ công/i }));

    const togglePadding = screen.getByTestId('toggle-padding');
    const toggleWb = screen.getByTestId('toggle-whitebalance');

    expect(togglePadding).toBeChecked();
    expect(toggleWb).toBeChecked();

    // Tắt padding
    fireEvent.click(togglePadding);
    expect(togglePadding).not.toBeChecked();

    // Tắt White Balance -> Nút Eyedropper bị ẩn
    fireEvent.click(toggleWb);
    expect(toggleWb).not.toBeChecked();
    expect(screen.queryByTestId('eyedropper-btn')).not.toBeInTheDocument();

    // Bật lại White Balance -> Nút Eyedropper hiển thị lại
    fireEvent.click(toggleWb);
    expect(toggleWb).toBeChecked();
    expect(screen.getByTestId('eyedropper-btn')).toBeInTheDocument();
  });

  it('4. Bấm nút Eyedropper và click chọn điểm tham chiếu trên ảnh', () => {
    const handleConfirm = vi.fn();
    render(<PhotoRestoration originalFaceUrl={sampleUrl} onConfirm={handleConfirm} />);

    fireEvent.click(screen.getByRole('radio', { name: /Thủ công/i }));

    const eyedropperBtn = screen.getByTestId('eyedropper-btn');
    expect(eyedropperBtn).toHaveTextContent(/Chọn điểm tham chiếu/i);

    // Kích hoạt Eyedropper
    fireEvent.click(eyedropperBtn);
    expect(eyedropperBtn).toHaveTextContent(/Đang chờ click/i);
    expect(screen.getByText(/Click vào điểm trắng\/xám trên ảnh gốc/i)).toBeInTheDocument();

    // Giả lập click lên ảnh gốc
    const imgContainer = screen.getByAltText('Original Face').parentElement?.parentElement;
    expect(imgContainer).toBeTruthy();

    if (imgContainer) {
      fireEvent.click(imgContainer, { clientX: 100, clientY: 150 });
    }

    // Đã chọn điểm tham chiếu thành công
    expect(screen.getByTestId('ref-point-label')).toBeInTheDocument();
    expect(screen.getByTestId('ref-point-label')).toHaveTextContent(/Điểm tham chiếu:/i);
    expect(screen.getByTestId('reselect-point-btn')).toBeInTheDocument();
    expect(screen.getByTestId('wb-gains-label')).toBeInTheDocument();
  });

  it('5. Xác nhận dùng ảnh đã khôi phục ở Chế độ Thủ công truyền đúng options', () => {
    const handleConfirm = vi.fn();
    render(<PhotoRestoration originalFaceUrl={sampleUrl} onConfirm={handleConfirm} />);

    // Đổi sang thủ công
    fireEvent.click(screen.getByRole('radio', { name: /Thủ công/i }));

    // Bấm dùng ảnh đã khôi phục
    const confirmBtn = screen.getByRole('button', { name: /Dùng ảnh đã khôi phục/i });
    fireEvent.click(confirmBtn);

    expect(handleConfirm).toHaveBeenCalledWith(
      true,
      0.7,
      expect.objectContaining({
        mode: 'manual',
        adaptivePadding: true,
        whiteBalance: true,
      })
    );
  });

  it('6. Chuyển đổi qua lại giữa Tự động và Thủ công: giữ nguyên các tham số đã chỉnh', () => {
    const handleConfirm = vi.fn();
    render(<PhotoRestoration originalFaceUrl={sampleUrl} onConfirm={handleConfirm} />);

    const autoRadio = screen.getByRole('radio', { name: /Tự động \(khuyến nghị\)/i });
    const manualRadio = screen.getByRole('radio', { name: /Thủ công/i });

    // 1. Vào Thủ công, tắt Padding
    fireEvent.click(manualRadio);
    const togglePadding = screen.getByTestId('toggle-padding');
    fireEvent.click(togglePadding);
    expect(togglePadding).not.toBeChecked();

    // 2. Chọn điểm tham chiếu
    const eyedropperBtn = screen.getByTestId('eyedropper-btn');
    fireEvent.click(eyedropperBtn);
    const imgContainer = screen.getByAltText('Original Face').parentElement?.parentElement;
    if (imgContainer) {
      fireEvent.click(imgContainer, { clientX: 80, clientY: 120 });
    }
    expect(screen.getByTestId('ref-point-label')).toBeInTheDocument();

    // 3. Chuyển về Tự động
    fireEvent.click(autoRadio);
    expect(autoRadio).toBeChecked();
    expect(screen.queryByTestId('toggle-padding')).not.toBeInTheDocument();

    // 4. Chuyển lại Thủ công -> xác nhận Padding vẫn tắt và điểm tham chiếu vẫn còn
    fireEvent.click(manualRadio);
    expect(manualRadio).toBeChecked();
    const togglePaddingAfter = screen.getByTestId('toggle-padding');
    expect(togglePaddingAfter).not.toBeChecked();
    expect(screen.getByTestId('ref-point-label')).toBeInTheDocument();
  });
});

