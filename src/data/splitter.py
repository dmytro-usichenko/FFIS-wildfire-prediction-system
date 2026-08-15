"""
src/data/splitter.py
─────────────────────
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
import config.settings as s

log = get_logger(__name__)


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_val:   pd.DataFrame
    X_test:  pd.DataFrame
    y_train: pd.Series
    y_val:   pd.Series
    y_test:  pd.Series

    def shapes(self) -> dict:
        return {
            "train": self.X_train.shape,
            "val":   self.X_val.shape,
            "test":  self.X_test.shape,
        }


class RandomSplitter:
    def __init__(
        self,
        test_size:    float = s.TEST_SIZE,
        val_size:     float = s.VAL_SIZE,
        random_state: int   = s.RANDOM_STATE,
    ) -> None:
        self.test_size    = test_size
        self.val_size     = val_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, feature_cols: list) -> SplitResult:
        X = df[feature_cols]
        y = df[s.TARGET]

        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y if s.STRATIFY else None,
            random_state=self.random_state,
        )
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=val_ratio,
            stratify=y_tv if s.STRATIFY else None,
            random_state=self.random_state,
        )

        result = SplitResult(X_train, X_val, X_test, y_train, y_val, y_test)
        log.info("Розбивка: %s", result.shapes())
        return result