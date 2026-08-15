"""
src/data/loader.py
──────────────────
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
import config.settings as s

log = get_logger(__name__)


class DataLoader:
    REQUIRED = [s.TARGET, s.DATE_COL, "NDVI", "SoilMoisture", "latitude", "longitude"]

    def __init__(
        self,
        data_path: str | Path = s.DATA_FILE,
        sample_size: Optional[int] = s.SAMPLE_SIZE,
        random_state: int = s.RANDOM_STATE,
    ) -> None:
        self.data_path    = Path(data_path)
        self.sample_size  = sample_size
        self.random_state = random_state
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        log.info("Завантажую: %s", self.data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Файл не знайдено: {self.data_path}\n"
                "Поклади датасет у папку data/"
            )

        raw = pd.read_parquet(self.data_path)
        log.info("Завантажено: %s | %.1f MB",
                 raw.shape, raw.memory_usage(deep=True).sum() / 1e6)

        self._validate(raw)
        df = self._postprocess(raw)

        if self.sample_size and len(df) > self.sample_size:
            df = self._sample(df)

        log.info("Фінальний розмір: %s", df.shape)
        self._df = df
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Відсутні колонки: {missing}")
        log.info("Валідація: OK (%d колонок)", len(df.columns))

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[s.DATE_COL] = pd.to_datetime(df[s.DATE_COL], errors="coerce")

        coord_lags = [c for c in df.columns
                      if c.startswith("lat_lag_") or c.startswith("lon_lag_")]
        if coord_lags:
            df.drop(columns=coord_lags, inplace=True)

        df[s.TARGET] = df[s.TARGET].astype("int8")
        df["_month"] = df[s.DATE_COL].dt.month
        df["_year"]  = df[s.DATE_COL].dt.year

        log.info("Баланс: %s", df[s.TARGET].value_counts().to_dict())
        return df

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Вибірка %d рядків", self.sample_size)
        per_class = self.sample_size // 2

        frames = []
        for label in [0, 1]:
            subset = df[df[s.TARGET] == label]
            n = min(per_class, len(subset))
            frames.append(subset.sample(n=n, random_state=self.random_state))

        result = pd.concat(frames).reset_index(drop=True)
        log.info("Після вибірки: %s | is_fire: %s",
                 result.shape, s.TARGET in result.columns)
        return result

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Спочатку виклич .load()")
        return self._df