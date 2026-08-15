"""
src/models/base_model.py
─────────────────────────
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import joblib

from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseModel(ABC):
    def __init__(self, name: str, params: dict):
        self.name   = name
        self.params = params
        self._model: Any = None

    @abstractmethod
    def _build(self) -> Any:
        """Повертає об'єкт sklearn/xgboost моделі."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        if self._model is None:
            self._model = self._build()
        log.info("%s: навчання на %d рядках", self.name, len(X))
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check()
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        log.info("%s збережено → %s", self.name, path)

    def load(self, path: Path) -> "BaseModel":
        self._model = joblib.load(path)
        log.info("%s завантажено ← %s", self.name, path)
        return self

    def _check(self) -> None:
        if self._model is None:
            raise RuntimeError(f"{self.name}: спочатку виклич .fit()")

    @property
    def sklearn_model(self) -> Any:
        return self._model