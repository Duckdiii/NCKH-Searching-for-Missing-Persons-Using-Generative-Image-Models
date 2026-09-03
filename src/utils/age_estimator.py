"""
Age Estimator - ước lượng tuổi cho ảnh BẤT KỲ bằng MiVOLO (github.com/WildChlamydia/MiVOLO).

Vì sao cần module này: cách cũ trong main.py lấy INITIAL_AGE bằng cách tra cứu
sampled_labels.csv - chỉ chạy được với 140 ảnh FFHQ đã có nhãn sẵn. Ảnh thật của người
mất tích (đường dùng chính khi triển khai) KHÔNG có CSV đi kèm, nên cần 1 model tự đoán
tuổi. Cách tra CSV vẫn được giữ trong main.py để test/so sánh, KHÔNG bị thay thế.

Theo đúng pattern lazy-load + "lỗi rõ, không âm thầm sai" đã dùng cho FaceEmbedder (Module 4):
model chưa load ở __init__, chỉ load 1 lần trong _load_model(); nếu không đọc được ảnh hoặc
không đoán được tuổi thì raise ValueError rõ ràng (không trả về giá trị mặc định).
"""

from typing import Optional

import cv2
import pandas as pd

from src.utils.prompts import age_group_to_age


class AgeEstimator:
    """Bọc mivolo.predictor.Predictor: đọc 1 ảnh, trả về tuổi ước tính (int).

    API đã verify với mivolo 0.6.0.dev0 (clone từ main, 2026-09):
      - Predictor(config, verbose) trong đó config là 1 object có các thuộc tính:
        detector_weights, checkpoint, device, with_persons, disable_faces, draw
      - Predictor.recognize(img_bgr) -> (PersonAndFaceResult, Optional[np.ndarray])
      - result.ages: list[Optional[float]] dài bằng số object YOLO detect được; phần tử
        là None nếu object đó không được gán tuổi -> phải lọc bỏ None (bản draft trong
        notebook chỉ check len(...)==0 là CHƯA đủ, list luôn có sẵn phần tử None).
    """

    def __init__(self, detector_checkpoint: str, age_checkpoint: str, device: str = "cuda"):
        """Lưu config, KHÔNG load model ngay (lazy load, giống FaceEmbedder).

        detector_checkpoint: đường dẫn yolov8x_person_face.pt (detector người + mặt của MiVOLO)
        age_checkpoint:      đường dẫn mivolo_imdb.pth.tar (model age/gender)
        device:              "cuda" hoặc "cpu"
        """
        self.detector_checkpoint = detector_checkpoint
        self.age_checkpoint = age_checkpoint
        self.device = device
        self.predictor = None

    def _load_model(self) -> None:
        """Load mivolo.predictor.Predictor 1 lần duy nhất.

        with_persons=True + disable_faces=False: dùng cả thân người lẫn mặt để đoán tuổi
        (chính xác hơn khi ảnh chỉ có 1 người), MiVOLO tự fallback về chỉ-mặt nếu không
        detect được thân.
        """
        import argparse

        import torch
        from mivolo.predictor import Predictor

        config = argparse.Namespace(
            detector_weights=self.detector_checkpoint,
            checkpoint=self.age_checkpoint,
            device=self.device,
            with_persons=True,
            disable_faces=False,
            draw=False,
        )

        # Checkpoint YOLO của MiVOLO (2023) là định dạng cũ; torch >= 2.6 mặc định
        # torch.load(weights_only=True) và ultralytics đời mới không tự bỏ allowlist cho
        # nó -> UnpicklingError. Ta tin nguồn (release chính thức của MiVOLO) nên tạm ép
        # weights_only=False chỉ trong lúc dựng Predictor, rồi khôi phục ngay.
        _orig_load = torch.load

        def _load_trusted(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _load_trusted
        try:
            self.predictor = Predictor(config, verbose=False)
        finally:
            torch.load = _orig_load

    def estimate(self, image_path: str) -> int:
        """Đọc ảnh bằng cv2.imread (BGR - đúng chuẩn MiVOLO/YOLO mong đợi), chạy MiVOLO,
        trả về 1 số nguyên = tuổi ước tính (làm tròn).

        Nếu ảnh có nhiều người: lấy tuổi hợp lệ đầu tiên (theo thứ tự MiVOLO trả về) -
        không có UI/logic chọn người, đúng tinh thần happy path như FaceEmbedder.

        Raise ValueError rõ ràng khi:
          - không đọc được ảnh (đường dẫn sai / file hỏng)
          - MiVOLO không phát hiện người/mặt nào, hoặc không gán được tuổi
        """
        if self.predictor is None:
            self._load_model()

        # Lưu ý: sau khi import ultralytics (trong _load_model), cv2.imread bị nó
        # monkeypatch để RAISE FileNotFoundError khi file không tồn tại, thay vì trả None
        # như OpenCV gốc. Bắt cả 2 trường hợp rồi quy về ValueError cho nhất quán.
        try:
            img = cv2.imread(image_path)
        except (FileNotFoundError, OSError) as e:
            raise ValueError(f"Không đọc được ảnh: {image_path} ({e})") from e
        if img is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        try:
            detected_objects, _ = self.predictor.recognize(img)
        except Exception as e:
            raise ValueError(f"Lỗi khi chạy MiVOLO trên ảnh {image_path}: {e}") from e

        ages = getattr(detected_objects, "ages", None) or []
        valid_ages = [a for a in ages if a is not None]
        if not valid_ages:
            raise ValueError(
                f"MiVOLO không đoán được tuổi cho ảnh: {image_path} "
                f"(không phát hiện người/mặt phù hợp)"
            )

        return round(valid_ages[0])


def resolve_initial_age(
    mode: str,
    image_path: Optional[str] = None,
    manual_age: Optional[int] = None,
    labels_csv: Optional[str] = None,
    image_number: Optional[int] = None,
    age_estimator: Optional[AgeEstimator] = None,
) -> int:
    """Lấy Initial Age (IA) từ ĐÚNG 1 trong 3 nguồn tách biệt - KHÔNG trộn lẫn, vì mỗi nguồn
    có mức độ tin cậy khác nhau và phục vụ ngữ cảnh khác nhau:

      mode="csv":    tra cứu labels_csv theo image_number. CHỈ dùng nội bộ khi test bằng ảnh
                     FFHQ có sẵn nhãn (sampled_labels.csv) - KHÔNG hiển thị lựa chọn này cho
                     người dùng thật trong app.py, vì ảnh thật không có CSV đi kèm.
      mode="manual": dùng con số gia đình/người dùng tự nhập tay (manual_age) - ĐỘ TIN CẬY
                     NGANG GROUND TRUTH nếu gia đình nhớ đúng tuổi lúc chụp, ưu tiên dùng khi
                     có thể, vì loại bỏ hoàn toàn sai số ước tính.
      mode="mivolo": gọi age_estimator.estimate(image_path) (MiVOLO) - CHỈ dùng khi người
                     dùng KHÔNG biết/không chắc tuổi trong ảnh. Có sai số vốn có ~4-5 năm
                     (đã đo thật, xem docstring module) - không phải lỗi, là hạn chế cố hữu
                     của mọi age estimator, cần hiển thị cảnh báo rõ ràng ở UI khi dùng.

    Raise ValueError NGAY nếu thiếu tham số bắt buộc cho mode đã chọn, giá trị không hợp lệ
    (vd manual_age ngoài [0,120]), hoặc mode không tồn tại - KHÔNG được âm thầm trả về giá trị
    mặc định nào (0, None, hay giá trị đoán) trong bất kỳ trường hợp lỗi nào."""
    if mode == "csv":
        if labels_csv is None or image_number is None:
            raise ValueError("mode='csv' cần cả labels_csv và image_number")
        df = pd.read_csv(labels_csv)
        matched = df[df["image_number"] == image_number]
        if matched.empty:
            raise ValueError(f"Không tìm thấy image_number={image_number} trong {labels_csv}")
        row = matched.iloc[0]
        return age_group_to_age(row["age_group"])

    elif mode == "manual":
        if manual_age is None:
            raise ValueError("mode='manual' cần manual_age")
        if not (0 <= manual_age <= 120):
            raise ValueError(f"manual_age không hợp lệ: {manual_age} (phải từ 0 đến 120)")
        return int(manual_age)

    elif mode == "mivolo":
        if image_path is None or age_estimator is None:
            raise ValueError("mode='mivolo' cần cả image_path và age_estimator")
        return age_estimator.estimate(image_path)

    else:
        raise ValueError(f"mode không hợp lệ: '{mode}' (phải là 'csv', 'manual', hoặc 'mivolo')")
