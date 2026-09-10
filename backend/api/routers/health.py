import os
from fastapi import APIRouter
from backend.api.dependencies import get_config

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
def health_check():
    return {"status": "ok"}


@router.get("/checkpoints")
def check_checkpoints():
    config = get_config()
    missing = []

    # 1. Specialized UNet Checkpoint
    unet_ckpt = config["paths"]["specialized_unet_ckpt"]
    if not os.path.isdir(unet_ckpt) or not os.listdir(unet_ckpt):
        missing.append(f"Specialized UNet checkpoint ({unet_ckpt})")

    # 2. MiVOLO Checkpoints
    mivolo_cfg = config.get("age_estimator", {})
    det_ckpt = mivolo_cfg.get("detector_checkpoint", "")
    age_ckpt = mivolo_cfg.get("age_checkpoint", "")

    if det_ckpt and not os.path.isfile(det_ckpt):
        missing.append(f"MiVOLO YOLO detector ({det_ckpt})")
    if age_ckpt and not os.path.isfile(age_ckpt):
        missing.append(f"MiVOLO Age model ({age_ckpt})")

    return {
        "ready": len(missing) == 0,
        "missing": missing
    }
