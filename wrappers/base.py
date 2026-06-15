from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import numpy as np


class BaseModel(ABC):
    """Abstract base model wrapper with a standard interface."""

    model_type: str = "base"       # "classical" or "quantum"
    model_name: str = "Unknown"
    description: str = ""
    metrics: Dict = {}
    _loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights/state into memory."""
        ...

    @abstractmethod
    def predict(self, input_data: Any) -> Dict:
        """
        Run inference and return a dict with keys:
          prediction, confidence, inference_time_ms, extras
        """
        ...

    def get_metrics(self) -> Dict:
        return self.metrics

    def _timed_predict(self, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, round(elapsed_ms, 2)

    def ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True
