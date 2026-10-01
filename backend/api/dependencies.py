from functools import lru_cache
import os
import yaml
from src.search.embedding import FaceEmbedder
from src.utils.age_estimator import AgeEstimator

CONFIG_PATH = "configs/config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_EMBEDDER_INSTANCE = None
_ESTIMATOR_INSTANCE = None
_VIDEO_EMBEDDER_INSTANCE = None
_VIDEO_EMBEDDER_CTX = None


def get_embedder() -> FaceEmbedder:
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        cfg = get_config()
        _EMBEDDER_INSTANCE = FaceEmbedder(
            model_name=cfg["embedding"]["model_name"],
            ctx_id=cfg["embedding"]["ctx_id"],
            det_size=tuple(cfg["embedding"]["det_size"]),
        )
    return _EMBEDDER_INSTANCE


def get_video_embedder() -> FaceEmbedder:
    """Embedder riêng cho đối soát video: ưu tiên GPU vì lúc này pipeline diffusion
    (Module 1-3) đã chạy xong, VRAM đã rảnh — detect/embedding trên GPU nhanh gấp
    nhiều lần CPU, nhất là với frame video lớn. Tự rơi về embedder CPU chung nếu
    GPU khởi tạo thất bại (máy không có CUDA, hết VRAM...)."""
    global _VIDEO_EMBEDDER_INSTANCE, _VIDEO_EMBEDDER_CTX
    cfg = get_config()
    want_ctx = int(cfg["embedding"].get("video_ctx_id", 0))
    if want_ctx == int(cfg["embedding"].get("ctx_id", -1)):
        return get_embedder()
    if _VIDEO_EMBEDDER_INSTANCE is None or _VIDEO_EMBEDDER_CTX != want_ctx:
        try:
            emb = FaceEmbedder(
                model_name=cfg["embedding"]["model_name"],
                ctx_id=want_ctx,
                det_size=tuple(cfg["embedding"]["det_size"]),
            )
            emb._load_model()  # fail fast để rơi về CPU ngay, trước khi xử lý video
            _VIDEO_EMBEDDER_INSTANCE = emb
            _VIDEO_EMBEDDER_CTX = want_ctx
        except Exception as e:
            print(f"[dependencies] GPU embedder (ctx_id={want_ctx}) lỗi, dùng CPU: {e}")
            return get_embedder()
    return _VIDEO_EMBEDDER_INSTANCE


def get_age_estimator() -> AgeEstimator:
    global _ESTIMATOR_INSTANCE
    if _ESTIMATOR_INSTANCE is None:
        cfg = get_config()
        _ESTIMATOR_INSTANCE = AgeEstimator(
            detector_checkpoint=cfg["age_estimator"]["detector_checkpoint"],
            age_checkpoint=cfg["age_estimator"]["age_checkpoint"],
            device=cfg["age_estimator"].get("device", "cuda"),
        )
    return _ESTIMATOR_INSTANCE
