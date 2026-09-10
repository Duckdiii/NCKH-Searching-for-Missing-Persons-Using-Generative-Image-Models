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
