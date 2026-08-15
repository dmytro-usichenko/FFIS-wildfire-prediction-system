"""
src/explainability/shap_explainer.py
──────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)


class SHAPExplainer:
    """
    Обчислює реальні SHAP значення через shap.TreeExplainer.
    Використовує навчену XGBoost модель.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._explainer = None
        self._model     = None

    def load(self) -> "SHAPExplainer":
        try:
            import shap
            self._model     = joblib.load(self.model_path)
            self._explainer = shap.TreeExplainer(self._model)
            log.info("SHAP TreeExplainer завантажено")
        except Exception as e:
            log.warning("Помилка завантаження SHAP: %s", e)
        return self

    def explain_global(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> dict:
        """
        Глобальне пояснення — mean |SHAP| по всіх ознаках.
        Повертає словник {feature_name: importance}.
        """
        if self._explainer is None:
            return {}

        try:
            sample = X.sample(
                min(max_samples, len(X)), random_state=42
            )
            shap_values = self._explainer.shap_values(sample)

            # Для бінарної класифікації TreeExplainer повертає список
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            mean_abs = np.abs(shap_values).mean(axis=0)
            feature_names = list(X.columns)

            result = {
                name: round(float(val), 6)
                for name, val in zip(feature_names, mean_abs)
            }
            # Сортуємо за важливістю
            result = dict(
                sorted(result.items(), key=lambda x: -x[1])
            )
            log.info("SHAP global: %d ознак", len(result))
            return result

        except Exception as e:
            log.warning("Помилка SHAP global: %s", e)
            return {}

    def explain_local(
        self,
        X_row: pd.DataFrame,
    ) -> dict:
        """
        Локальне пояснення для одного рядка.
        Повертає словник {feature_name: shap_value}.
        """
        if self._explainer is None:
            return {}

        try:
            shap_values = self._explainer.shap_values(X_row)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            result = {
                name: round(float(val), 6)
                for name, val in zip(X_row.columns, shap_values[0])
            }
            return result

        except Exception as e:
            log.warning("Помилка SHAP local: %s", e)
            return {}

    @property
    def base_value(self) -> float:
        if self._explainer is None:
            return 0.5
        try:
            bv = self._explainer.expected_value
            return float(bv[1] if hasattr(bv, "__len__") else bv)
        except Exception:
            return 0.5