from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.ml.inference import InferenceEngine


@lru_cache
def get_inference_engine() -> InferenceEngine:
    settings = get_settings()
    return InferenceEngine(Path(settings.ml_model_dir))
