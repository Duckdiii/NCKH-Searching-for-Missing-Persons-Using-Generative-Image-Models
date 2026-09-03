"""
Script rời (chạy tay, KHÔNG phải pytest) - đo độ chính xác thật của estimate_head_pose()
bằng cách so với ground-truth head_yaw/head_pitch/head_roll đã có sẵn trong
sampled_labels.csv (140 ảnh FFHQ). Đây là số đo độ chính xác của 1 phép xấp xỉ hình học
(solvePnP + camera pinhole giả định) đối chiếu ground-truth từ 1 pipeline ước lượng khác
(công cụ gán nhãn gốc của FFHQ-Aging) - không phải hợp đồng đúng/sai cố định của code, nên
KHÔNG viết thành pytest assert (xem giải thích trong tests/test_head_pose.py).

Chạy: python scripts/validate_head_pose.py
"""

import os

import numpy as np
import pandas as pd

from src.search.embedding import FaceEmbedder
from src.utils.head_pose import (
    PITCH_THRESHOLD_DEFAULT,
    YAW_THRESHOLD_DEFAULT,
    estimate_head_pose,
)

FFHQ_DIR = "D:/Data/project/nckh/ffhq_aging_150_samples"
LABELS_CSV = "D:/Data/project/nckh/ffhq_aging_150_samples/sampled_labels.csv"


def main() -> None:
    df = pd.read_csv(LABELS_CSV)
    embedder = FaceEmbedder(ctx_id=-1)

    yaw_errors, pitch_errors, roll_errors = [], [], []
    n_skipped = 0

    # Dem so anh CSV coi la "nghieng" (theo ground truth) va so anh code cua ta coi la
    # "nghieng" (theo uoc luong) - de tinh false positive/negative cua nguong canh bao.
    n_gt_yaw_extreme = 0
    n_pred_yaw_extreme = 0
    n_gt_pitch_extreme = 0
    n_pred_pitch_extreme = 0

    for _, row in df.iterrows():
        image_path = os.path.join(FFHQ_DIR, f"{int(row['image_number']):05d}.png")
        if not os.path.isfile(image_path):
            continue

        try:
            faces = embedder.detect_faces(image_path)
        except ValueError:
            n_skipped += 1
            continue

        if not faces:
            n_skipped += 1
            continue

        face = faces[0]
        image_shape = (256, 256, 3)  # tat ca anh FFHQ da align san, co dinh 256x256
        pose = estimate_head_pose(face.kps, image_shape)
        if pose is None:
            n_skipped += 1
            continue

        pred_yaw, pred_pitch, pred_roll = pose
        gt_yaw, gt_pitch, gt_roll = row["head_yaw"], row["head_pitch"], row["head_roll"]

        yaw_errors.append(abs(pred_yaw - gt_yaw))
        pitch_errors.append(abs(pred_pitch - gt_pitch))
        roll_errors.append(abs(pred_roll - gt_roll))

        gt_yaw_extreme = abs(gt_yaw) > YAW_THRESHOLD_DEFAULT
        pred_yaw_extreme = abs(pred_yaw) > YAW_THRESHOLD_DEFAULT
        gt_pitch_extreme = abs(gt_pitch) > PITCH_THRESHOLD_DEFAULT
        pred_pitch_extreme = abs(pred_pitch) > PITCH_THRESHOLD_DEFAULT

        n_gt_yaw_extreme += gt_yaw_extreme
        n_pred_yaw_extreme += pred_yaw_extreme
        n_gt_pitch_extreme += gt_pitch_extreme
        n_pred_pitch_extreme += pred_pitch_extreme

    n = len(yaw_errors)
    print(f"So anh danh gia duoc: {n} (bo qua {n_skipped} anh khong detect duoc / khong uoc luong duoc pose)")
    print()
    print(f"MAE yaw:   {np.mean(yaw_errors):.2f} do  (median {np.median(yaw_errors):.2f}, max {np.max(yaw_errors):.2f})")
    print(f"MAE pitch: {np.mean(pitch_errors):.2f} do  (median {np.median(pitch_errors):.2f}, max {np.max(pitch_errors):.2f})")
    print(f"MAE roll:  {np.mean(roll_errors):.2f} do  (median {np.median(roll_errors):.2f}, max {np.max(roll_errors):.2f})")
    print()
    print(f"So anh CSV coi la yaw nghieng (>|{YAW_THRESHOLD_DEFAULT}|do): {n_gt_yaw_extreme}/{n}")
    print(f"So anh code ta canh bao yaw nghieng: {n_pred_yaw_extreme}/{n}")
    print(f"So anh CSV coi la pitch nghieng (>|{PITCH_THRESHOLD_DEFAULT}|do): {n_gt_pitch_extreme}/{n}")
    print(f"So anh code ta canh bao pitch nghieng: {n_pred_pitch_extreme}/{n}")


if __name__ == "__main__":
    main()
