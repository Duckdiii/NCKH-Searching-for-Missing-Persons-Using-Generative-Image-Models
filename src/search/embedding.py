"""
Module 4 - InsightFace Embedding

Trích xuất vector embedding 512-chiều (đã chuẩn hoá L2) cho từng khuôn mặt, dùng
insightface.app.FaceAnalysis(buffalo_l). Dùng làm đầu vào cho Module 5 (FAISS search).
"""

import glob
import os
from typing import List, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


class FaceEmbedder:
    """Bọc insightface.app.FaceAnalysis: trích xuất embedding 512-chiều cho 1 ảnh, và build
    gallery embedding cho toàn bộ ảnh trong 1 folder."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,
        det_size: Tuple[int, int] = (256, 256),
    ):
        """Lưu config (model_name, ctx_id: -1=CPU mặc định để tránh tranh chấp VRAM với các
        module diffusion trên GPU 6GB, 0=GPU nếu đo thấy còn dư VRAM).

        FIX (phát hiện qua thử nghiệm thực tế): det_size mặc định = (256, 256), KHÔNG dùng
        (640, 640) mặc định của insightface - vì mọi ảnh trong pipeline (gallery lẫn ảnh FADING
        sinh ra) đều là 256x256; dùng 640x640 gây tỷ lệ fail detect rất cao (~65% ảnh bị bỏ qua
        trong thử nghiệm thực tế, bao gồm cả ảnh chính diện rõ nét).

        Model CHƯA được load ở đây, sẽ load lazily trong _load_model()."""
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.app = None

    def _load_model(self) -> None:
        """Load FaceAnalysis(buffalo_l) và prepare() theo đúng ctx_id/det_size đã cấu hình."""
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=self.model_name)
        self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

    def embed(self, image_path: str) -> np.ndarray:
        """Đọc ảnh từ image_path bằng cv2.imread (BGR - đúng chuẩn insightface mong đợi,
        KHÔNG dùng PIL vì PIL đọc RGB sẽ cho ra embedding sai lệch mà không báo lỗi gì), chạy
        face detection + embedding, trả về normed_embedding (512-dim, đã L2-normalize sẵn).

        Happy path: giả định ảnh chính diện, chất lượng tốt, chỉ có 1 khuôn mặt. Nếu không
        detect được mặt nào, raise ValueError rõ ràng (không trả về vector rỗng/None) - đây là
        lỗi dữ liệu cần người gọi biết rõ, không được âm thầm bỏ qua.

        Nếu ảnh có nhiều mặt: lấy faces[0] (thứ tự insightface trả về), KHÔNG có UI/logic chọn
        mặt - đúng yêu cầu happy path."""
        if self.app is None:
            self._load_model()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError(f"Không phát hiện được khuôn mặt nào trong ảnh: {image_path}")

        return faces[0].normed_embedding

    def detect_faces(self, image_path: str) -> list:
        """Add-on cho UI tương tác (app.py) - chạy detection ĐÚNG 1 LẦN, trả về NGUYÊN list
        đối tượng Face thô của insightface (mỗi Face có .bbox, .kps 5 điểm mốc, .det_score,
        .normed_embedding), thay vì chỉ lấy embedding của faces[0] như embed().

        Lý do tách riêng thay vì sửa embed(): 1 lần detect này cần dùng lại cho CẢ 3 việc -
        (1) UI cho người dùng chọn đúng mặt khi ảnh có nhiều người, (2) cảnh báo chất lượng
        ảnh (head-pose, dùng .kps + .det_score), (3) embedding cuối cùng (dùng .normed_embedding
        của mặt đã chọn) - nếu gọi lại self.app.get() riêng cho từng việc sẽ chạy detect 2-3
        lần thừa trên cùng 1 ảnh. embed()/build_gallery() giữ nguyên không đổi, chỉ phục vụ
        pipeline tự động (Module 4 gốc).

        KHÁC embed() có chủ đích: KHÔNG raise nếu 0 mặt, trả về [] - vì đây là hàm phục vụ UI
        tương tác (Streamlit), nơi người gọi cần tự quyết định cách báo cho người dùng biết
        (vd "vui lòng chọn ảnh khác"), không phải để pipeline tự động dừng đột ngột.

        Vẫn raise ValueError nếu không đọc được ảnh (lỗi file, không phải lỗi detect)."""
        if self.app is None:
            self._load_model()

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        return self.app.get(img)

    def build_gallery(self, folder_path: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Chạy embedding cho toàn bộ ảnh (.png/.jpg/.jpeg) trong folder_path (sorted cho
        deterministic). Với ảnh lỗi (không đọc được / không detect được mặt) - KHÔNG dừng cả
        hàm lại, nhưng cũng KHÔNG âm thầm bỏ qua: gom lại vào failed_files để người gọi biết rõ
        gallery đang thiếu identity nào (tránh false negative âm thầm khi tìm kiếm sau này).

        Trả về (embeddings, identity_labels, failed_files):
          - embeddings: list vector 512-dim, chỉ gồm ảnh thành công
          - identity_labels: tên file (không đuôi), tương ứng 1-1 với embeddings
          - failed_files: đường dẫn các ảnh lỗi, KHÔNG có trong embeddings/identity_labels
        """
        image_paths = sorted(
            f
            for f in glob.glob(os.path.join(folder_path, "*"))
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )

        embeddings: List[np.ndarray] = []
        identity_labels: List[str] = []
        failed_files: List[str] = []

        for image_path in image_paths:
            try:
                embedding = self.embed(image_path)
            except ValueError as e:
                print(f"[FaceEmbedder] BỎ QUA ảnh lỗi: {e}")
                failed_files.append(image_path)
                continue
            embeddings.append(embedding)
            identity_labels.append(os.path.splitext(os.path.basename(image_path))[0])

        if failed_files:
            print(
                f"[FaceEmbedder] Gallery '{folder_path}' thiếu {len(failed_files)}/"
                f"{len(image_paths)} identity do lỗi đọc/detect ảnh. Xem failed_files để biết chi tiết."
            )

        return embeddings, identity_labels, failed_files
