"""Tien xu ly dau vao cho video/camera (nhiem vu rieng, tach khoi src/utils/).

Goi nay dam bao frame dua vao detect/embedding on dinh duoi dieu kien xau
ngoai thuc te: troi mua, troi toi/ban dem, nang choi (glare), suong mu,
nguoc sang, anh mo/nhoe. Chi dung OpenCV + numpy (khong them model nang)
de chay kip toc do video tren CPU.

Cac module:
- conditions: do metric + phan loai dieu kien tren 1 frame.
- enhance: tung buoc tang cuong nhe theo tung dieu kien.
- pipeline: ghep thanh 1 ham preprocess_video_frame() duy nhat cho
  backend (video_verify) va sau nay cho nhanh camera truc tiep.
"""

from src.preprocessing.pipeline import preprocess_video_frame

__all__ = ["preprocess_video_frame"]
