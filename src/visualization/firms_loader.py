"""
src/visualization/firms_loader.py
───────────────────────────────────
Завантажує реальні дані про активні пожежі
з NASA FIRMS API (супутники MODIS / VIIRS).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

# Кеш-файл щоб не робити запит кожного разу
CACHE_FILE = Path("artifacts") / "firms_cache.csv"
CACHE_TTL_HOURS = 6


class FIRMSLoader:
    """
    Завантажує дані активних пожеж з NASA FIRMS API.

    Документація: https://firms.modaps.eosdis.nasa.gov/api/area/
    """

    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    # Обмеження для Марокко
    MOROCCO_BBOX = "-13.5,27.5,-1.0,35.8"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def load(
        self,
        days: int = 10,
        source: str = "VIIRS_SNPP_NRT",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Завантажує дані пожеж.

        Parameters
        ----------
        days     : кількість днів назад (1-10 для NRT)
        source   : VIIRS_SNPP_NRT / MODIS_NRT / VIIRS_NOAA20_NRT
        use_cache: використовувати кеш якщо є свіжі дані
        """
        # Перевіряємо кеш
        if use_cache and self._cache_valid():
            log.info("FIRMS: завантажую з кешу")
            return pd.read_csv(CACHE_FILE)

        # Якщо немає API ключа — повертаємо мок дані
        if not self.api_key:
            log.warning("FIRMS: немає API ключа, використовую мок дані")
            return self._mock_firms_data()

        try:
            url = f"{self.BASE_URL}/{self.api_key}/{source}/{self.MOROCCO_BBOX}/{days}"
            log.info("FIRMS: запит → %s", url)

            response = requests.get(url, timeout=15)
            response.raise_for_status()

            df = pd.read_csv(pd.io.common.StringIO(response.text))
            log.info("FIRMS: отримано %d точок пожеж", len(df))

            if len(df) > 0:
                df = self._clean(df)
                # Зберігаємо кеш
                CACHE_FILE.parent.mkdir(exist_ok=True)
                df.to_csv(CACHE_FILE, index=False)

            return df

        except Exception as e:
            log.warning("FIRMS: помилка API (%s), використовую мок дані", e)
            return self._mock_firms_data()

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Стандартизує колонки."""
        rename = {
            "latitude":   "latitude",
            "longitude":  "longitude",
            "bright_ti4": "brightness",
            "bright_t31": "brightness",
            "frp":        "frp",
            "confidence": "confidence",
            "acq_date":   "acq_date",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if "latitude" not in df.columns:
            return pd.DataFrame()

        # Нормалізуємо confidence
        if "confidence" in df.columns:
            conf_map = {"l": "low", "n": "nominal", "h": "high"}
            df["confidence"] = df["confidence"].astype(str).str.lower()
            df["confidence"] = df["confidence"].map(conf_map).fillna(df["confidence"])

        if "frp" not in df.columns:
            df["frp"] = 5.0

        if "brightness" not in df.columns:
            df["brightness"] = 320.0

        return df[["latitude", "longitude", "brightness",
                   "frp", "confidence", "acq_date"]].copy()

    def _cache_valid(self) -> bool:
        """Перевіряє чи кеш не застарів."""
        if not CACHE_FILE.exists():
            return False
        age_hours = (
            datetime.now().timestamp() - CACHE_FILE.stat().st_mtime
        ) / 3600
        return age_hours < CACHE_TTL_HOURS

    def _mock_firms_data(self, n: int = 45) -> pd.DataFrame:
        """
        Генерує реалістичні мок-дані FIRMS для Марокко.
        Використовується коли немає API ключа або немає інтернету.
        """
        rng = np.random.default_rng(99)

        # Кластеризуємо точки навколо відомих пожежонебезпечних регіонів Марокко
        centers = [
            (34.0, -5.0),   # Ріф
            (32.5, -6.5),   # Середній Атлас
            (31.0, -8.0),   # Марракеш
            (30.5, -9.5),   # Агадір
            (33.5, -4.5),   # Фес
        ]

        rows = []
        per_center = n // len(centers)
        for lat_c, lon_c in centers:
            for _ in range(per_center):
                rows.append({
                    "latitude":   round(lat_c + rng.normal(0, 0.4), 4),
                    "longitude":  round(lon_c + rng.normal(0, 0.4), 4),
                    "brightness": round(float(rng.uniform(310, 380)), 1),
                    "frp":        round(float(rng.exponential(12)), 1),
                    "confidence": rng.choice(["nominal", "high", "low"],
                                              p=[0.55, 0.35, 0.10]),
                    "acq_date":   (
                        datetime.now() - timedelta(days=int(rng.integers(0, 10)))
                    ).strftime("%Y-%m-%d"),
                })

        return pd.DataFrame(rows)