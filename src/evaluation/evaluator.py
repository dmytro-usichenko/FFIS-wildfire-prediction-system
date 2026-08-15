"""
src/evaluation/evaluator.py
─────────────────────────────
"""

from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    roc_curve,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class Evaluator:
    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        log.info("Оцінка: %s", model_name)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        metrics = {
            "model_name": model_name,
            "accuracy":   round(float(accuracy_score(y_test, y_pred)), 4),
            "precision":  round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":     round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1":         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc":    round(float(roc_auc_score(y_test, y_proba)), 4),
            "log_loss":   round(float(log_loss(y_test, y_proba)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_fpr":    fpr.tolist(),
            "roc_tpr":    tpr.tolist(),
        }

        log.info("AUC=%.4f  F1=%.4f  Acc=%.4f",
                 metrics["roc_auc"], metrics["f1"], metrics["accuracy"])
        return metrics