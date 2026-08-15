"""
config/settings.py
──────────────────
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Шляхи ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
DATA_FILE  = DATA_DIR / "Date_final_dataset_balanced_float32.parquet"

ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR    = ARTIFACT_DIR / "models"
SCALER_DIR   = ARTIFACT_DIR / "scalers"
REPORT_DIR   = ARTIFACT_DIR / "reports"
SHAP_DIR     = ARTIFACT_DIR / "shap_cache"
ASSETS_DIR   = ROOT / "assets"
ALGERIA_DATA_FILE   = DATA_DIR / "algeria_forest_fires_clean.csv"
ALGERIA_RF_MODEL    = MODEL_DIR / "algeria_rf_v1.joblib"
ALGERIA_LR_MODEL    = MODEL_DIR / "algeria_lr_v1.joblib"
ALGERIA_REPORT_FILE = REPORT_DIR / "algeria_model_comparison.json"


for _d in [DATA_DIR, MODEL_DIR, SCALER_DIR, REPORT_DIR, SHAP_DIR, ASSETS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Датасет ───────────────────────────────────────────────────────────────────
TARGET   = "is_fire"
DATE_COL = "acq_date"

SATELLITE_FEATURES: List[str] = ["NDVI", "SoilMoisture"]

GEO_FEATURES: List[str] = ["latitude", "longitude", "sea_distance"]

TEMPORAL_FEATURES: List[str] = [
    "day_of_week", "day_of_year", "is_weekend", "is_holiday",
]

METEO_VARS: List[str] = [
    "average_temperature", "maximum_temperature", "minimum_temperature",
    "precipitation", "wind_speed", "maximum_sustained_wind_speed",
    "wind_gust", "dew_point", "snow_depth", "fog", "thunder",
]

LAG_WINDOW: int = 7

ROLLING_SUFFIXES: List[str] = [
    "weekly_mean", "monthly_mean", "quarterly_mean",
    "yearly_mean", "last_1_year", "last_2_year", "last_3_year",
]

ENGINEERED_FEATURES: List[str] = [
    "temp_range_lag_1", "temp_wind_interaction",
    "drought_index", "ndvi_anomaly_flag", "fire_weather_score",
]

# ── Pipeline ──────────────────────────────────────────────────────────────────
SAMPLE_SIZE:  Optional[int] = 200_000
RANDOM_STATE: int           = 42
TEST_SIZE:    float         = 0.20
VAL_SIZE:     float         = 0.10
STRATIFY:     bool          = True

WINSOR_LOWER: float = 0.01
WINSOR_UPPER: float = 0.99

LOG_TRANSFORM_FEATURES: List[str] = [
    "precipitation_lag_1", "precipitation_lag_2",
    "snow_depth_lag_1", "fog_lag_1", "thunder_lag_1",
]

# ── Моделі ────────────────────────────────────────────────────────────────────
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "needs_scaling": True,
        "params": {
            "C": 1.0, "max_iter": 1000,
            "solver": "lbfgs", "random_state": RANDOM_STATE, "n_jobs": -1,
        },
    },
    "random_forest": {
        "display_name": "Random Forest",
        "needs_scaling": False,
        "params": {
            "n_estimators": 300, "max_depth": 12,
            "min_samples_leaf": 10, "max_features": "sqrt",
            "n_jobs": -1, "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        },
    },
    "xgboost": {
        "display_name": "XGBoost",
        "needs_scaling": False,
        "params": {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "gamma": 0.1,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE, "n_jobs": -1,
        },
    },
}

CV_FOLDS:       int = 5
N_ITER_SEARCH:  int = 20
SCORING_METRIC: str = "roc_auc"

# ── Карта ─────────────────────────────────────────────────────────────────────
MAP_CENTER_LAT: float = 31.5
MAP_CENTER_LON: float = -7.0
MAP_ZOOM_START: int   = 6
MAP_HEIGHT_PX:  int   = 560

FIRE_GRADIENT = {
    "0.0":  "#10b981",
    "0.35": "#84cc16",
    "0.55": "#f59e0b",
    "0.75": "#ff6b35",
    "1.0":  "#ff4d1a",
}

# ── SHAP ──────────────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE: int = 2_000
SHAP_CACHE_TTL:   int = 3600
TOP_N_FEATURES:   int = 15


@dataclass(frozen=True)
class _Config:
    root:         Path          = ROOT
    data_file:    Path          = DATA_FILE
    model_dir:    Path          = MODEL_DIR
    scaler_dir:   Path          = SCALER_DIR
    report_dir:   Path          = REPORT_DIR
    shap_dir:     Path          = SHAP_DIR
    assets_dir:   Path          = ASSETS_DIR
    target:       str           = TARGET
    date_col:     str           = DATE_COL
    sample_size:  Optional[int] = SAMPLE_SIZE
    random_state: int           = RANDOM_STATE
    test_size:    float         = TEST_SIZE
    val_size:     float         = VAL_SIZE
    lag_window:   int           = LAG_WINDOW
    map_lat:      float         = MAP_CENTER_LAT
    map_lon:      float         = MAP_CENTER_LON
    map_zoom:     int           = MAP_ZOOM_START
    shap_samples: int           = SHAP_SAMPLE_SIZE
    shap_ttl:     int           = SHAP_CACHE_TTL
    top_n:        int           = TOP_N_FEATURES


cfg = _Config()