"""
src/features/engineering.py
────────────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger
import config.settings as s

log = get_logger(__name__)


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self._fit(X, y)

    def transform(self, X):
        return self._transform(X)

    def _fit(self, X, y=None):
        return self

    def _transform(self, X):
        return X


class MissingValueImputer(BaseTransformer):
    """Заповнює пропуски медіаною."""

    def __init__(self):
        self._medians: dict = {}

    def _fit(self, X: pd.DataFrame, y=None):
        num = X.select_dtypes(include="number").columns
        self._medians = X[num].median().to_dict()
        log.info("Imputer: %d колонок", len(self._medians))
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.fillna(self._medians, inplace=True)
        if X.isnull().sum().sum():
            X.fillna(0, inplace=True)
        return X


class Winsorizer(BaseTransformer):
    """Обрізає екстремальні значення."""

    def __init__(
        self,
        lower: float = s.WINSOR_LOWER,
        upper: float = s.WINSOR_UPPER,
        exclude: Optional[List[str]] = None,
    ):
        self.lower   = lower
        self.upper   = upper
        self.exclude = exclude or []
        self._bounds: dict = {}

    def _fit(self, X: pd.DataFrame, y=None):
        for col in X.select_dtypes(include="number").columns:
            if col in self.exclude:
                continue
            self._bounds[col] = (
                float(X[col].quantile(self.lower)),
                float(X[col].quantile(self.upper)),
            )
        log.info("Winsorizer: %d колонок", len(self._bounds))
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self._bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X


class Log1pTransformer(BaseTransformer):
    """Логарифмує скошені колонки."""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns or s.LOG_TRANSFORM_FEATURES
        self._shifts: dict = {}

    def _fit(self, X: pd.DataFrame, y=None):
        for col in self.columns:
            if col in X.columns:
                mn = float(X[col].min())
                self._shifts[col] = max(0.0, -mn)
        log.info("Log1p: %d колонок", len(self._shifts))
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, shift in self._shifts.items():
            if col in X.columns:
                X[col] = np.log1p(X[col] + shift)
        return X


class FeatureEngineer(BaseTransformer):
    """
    Створює нові ознаки:
    - temp_range_lag_1      — різниця max і min температур
    - temp_wind_interaction — температура × вітер
    - drought_index         — опади / температура
    - ndvi_anomaly_flag     — 1 якщо NDVI нижче 15-го перцентилю
    - fire_weather_score    — зважена комбінація ризик-факторів
    """

    def __init__(self):
        self._ndvi_p15: float = 0.0

    def _fit(self, X: pd.DataFrame, y=None):
        if "NDVI" in X.columns:
            self._ndvi_p15 = float(X["NDVI"].quantile(0.15))
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if all(c in X.columns for c in
               ["maximum_temperature_lag_1", "minimum_temperature_lag_1"]):
            X["temp_range_lag_1"] = (
                X["maximum_temperature_lag_1"] - X["minimum_temperature_lag_1"]
            )

        if all(c in X.columns for c in
               ["average_temperature_lag_1", "wind_speed_lag_1"]):
            X["temp_wind_interaction"] = (
                X["average_temperature_lag_1"] * X["wind_speed_lag_1"]
            )

        if all(c in X.columns for c in
               ["precipitation_lag_1", "average_temperature_lag_1"]):
            X["drought_index"] = (
                X["precipitation_lag_1"]
                / (X["average_temperature_lag_1"].abs() + 1.0)
            )

        if "NDVI" in X.columns:
            X["ndvi_anomaly_flag"] = (
                X["NDVI"] < self._ndvi_p15
            ).astype("int8")

        need = {"average_temperature_lag_1", "precipitation_lag_1",
                "wind_gust_lag_1", "NDVI"}
        if need.issubset(X.columns):
            X["fire_weather_score"] = (
                X["average_temperature_lag_1"] * 0.4
                - X["precipitation_lag_1"]      * 0.3
                + X["wind_gust_lag_1"]           * 0.2
                - X["NDVI"]                      * 0.1
            )

        return X


class ColumnSelector(BaseTransformer):
    """Залишає тільки потрібні колонки."""

    def __init__(self, keep: Optional[List[str]] = None):
        self.keep     = keep
        self._fitted: List[str] = []

    def _fit(self, X: pd.DataFrame, y=None):
        if self.keep:
            self._fitted = [c for c in self.keep if c in X.columns]
        else:
            self._fitted = list(X.columns)
        log.info("ColumnSelector: %d / %d", len(self._fitted), len(X.columns))
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self._fitted if c in X.columns]
        return X[available].copy()


class FeatureScaler(BaseTransformer):
    """StandardScaler який зберігає назви колонок."""

    def __init__(self):
        self._scaler    = StandardScaler()
        self._num_cols: List[str] = []

    def _fit(self, X: pd.DataFrame, y=None):
        self._num_cols = list(X.select_dtypes(include="number").columns)
        self._scaler.fit(X[self._num_cols])
        log.info("Scaler: %d колонок", len(self._num_cols))
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = [c for c in self._num_cols if c in X.columns]
        X[cols] = self._scaler.transform(X[cols])
        return X


def build_pipeline(
    feature_columns: List[str],
    apply_scaling: bool = False,
) -> Pipeline:
    """
    Збирає preprocessing pipeline.
    apply_scaling=True потрібен для SVM та Logistic Regression.
    """
    steps = [
        ("engineer",  FeatureEngineer()),
        ("imputer",   MissingValueImputer()),
        ("winsorize", Winsorizer()),
        ("log1p",     Log1pTransformer()),
        ("selector",  ColumnSelector(keep=feature_columns)),
    ]
    if apply_scaling:
        steps.append(("scaler", FeatureScaler()))

    return Pipeline(steps)