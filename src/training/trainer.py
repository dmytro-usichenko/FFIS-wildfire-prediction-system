"""
src/training/trainer.py
────────────────────────
"""

from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import joblib

from src.utils.logger import get_logger
from src.models.base_model import BaseModel
from src.features.engineering import (
    FeatureEngineer,
    MissingValueImputer,
    Winsorizer,
    Log1pTransformer,
    ColumnSelector,
    FeatureScaler,
)
import config.settings as s

log = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        feature_cols: List[str],
        apply_scaling: bool = False,
    ) -> None:
        self.model         = model
        self.feature_cols  = feature_cols
        self.apply_scaling = apply_scaling
        self._train_time   = 0.0

        # Створюємо трансформери окремо — не через sklearn Pipeline
        self._engineer  = FeatureEngineer()
        self._imputer   = MissingValueImputer()
        self._winsorizer= Winsorizer()
        self._log1p     = Log1pTransformer()
        self._selector  = ColumnSelector(keep=feature_cols)
        self._scaler    = FeatureScaler() if apply_scaling else None
        self._fitted    = False

    def _transform_steps(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._engineer.transform(X)
        X = self._imputer.transform(X)
        X = self._winsorizer.transform(X)
        X = self._log1p.transform(X)
        X = self._selector.transform(X)
        if self._scaler:
            X = self._scaler.transform(X)
        return X

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "Trainer":
        log.info("=== Навчання: %s ===", self.model.name)
        t0 = time.time()

        # Fit всіх трансформерів на train
        X = self._engineer.fit(X_train, y_train).transform(X_train)
        X = self._imputer.fit(X).transform(X)
        X = self._winsorizer.fit(X).transform(X)
        X = self._log1p.fit(X).transform(X)
        X = self._selector.fit(X).transform(X)
        if self._scaler:
            X = self._scaler.fit(X).transform(X)

        log.info("Ознак після pipeline: %d", X.shape[1])

        # Навчання моделі
        self.model.fit(X, y_train)
        self._train_time = time.time() - t0
        self._fitted = True
        log.info("Час навчання: %.1f сек", self._train_time)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Спочатку виклич .fit()")
        return self._transform_steps(X)

    def save_artifacts(self, version: str = "v1") -> Dict[str, Path]:
        name = self.model.name.lower().replace(" ", "_")
        paths = {
            "model": s.MODEL_DIR / f"{name}_{version}.joblib",
            "meta":  s.MODEL_DIR / f"{name}_{version}_meta.json",
        }

        self.model.save(paths["model"])

        meta = {
            "model_name":   self.model.name,
            "version":      version,
            "trained_at":   datetime.now().isoformat(),
            "train_time_s": round(self._train_time, 2),
            "n_features":   len(self.feature_cols),
        }
        with open(paths["meta"], "w") as f:
            json.dump(meta, f, indent=2)

        log.info("Артефакти збережено")
        return paths