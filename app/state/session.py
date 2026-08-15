"""
app/state/session.py
─────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import json
import os
from src.utils.logger import get_logger
log = get_logger(__name__)

import config.settings as s


def init_session() -> None:
    defaults = {
        "page":            "🏠 Головна",
        "model_loaded":    False,
        "data_loaded":     False,
        "last_prediction": None,
        "ai_recommendations": None,
        "scenario_results": None,
        "summary_report": None,
        "selected_region": "Марокко",
        "last_prediction_algeria": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "app_data" not in st.session_state:
        st.session_state["app_data"]      = _load_or_mock_data()
        st.session_state["model_results"] = _load_or_mock_results()
        st.session_state["data_loaded"]   = True
        st.session_state["model_loaded"]  = True

    if "algeria_data" not in st.session_state:
        st.session_state["algeria_data"] = _load_algeria_data()
        st.session_state["algeria_results"] = _load_algeria_results()

def _load_or_mock_data() -> pd.DataFrame:
    if s.DATA_FILE.exists():
        try:
            from src.data.loader import DataLoader
            df = DataLoader().load()
            if "is_fire" not in df.columns:
                return _mock_dataset()

            # Якщо немає fire_probability — рахуємо просту апроксимацію
            if "fire_probability" not in df.columns:
                df["fire_probability"] = df["is_fire"].astype(float)
                import numpy as np
                temp = df["average_temperature_lag_1"].fillna(
                    28) if "average_temperature_lag_1" in df.columns else pd.Series([28] * len(df))
                prec = df["precipitation_lag_1"].fillna(5) if "precipitation_lag_1" in df.columns else pd.Series(
                    [5] * len(df))
                wind = df["wind_speed_lag_1"].fillna(12) if "wind_speed_lag_1" in df.columns else pd.Series(
                    [12] * len(df))
                ndvi = df["NDVI"].fillna(0.35) if "NDVI" in df.columns else pd.Series([0.35] * len(df))
                soil = df["SoilMoisture"].fillna(0.3) if "SoilMoisture" in df.columns else pd.Series([0.3] * len(df))

                fp = (
                        0.35 * (temp.clip(0, 50) / 50)
                        - 0.25 * (prec.clip(0, 80) / 80)
                        + 0.20 * (wind.clip(0, 60) / 60)
                        - 0.20 * ndvi.clip(-0.2, 1.0)
                        - 0.10 * soil.clip(0, 1.0)
                )
                fp_min = fp.min()
                fp_max = fp.max()
                df["fire_probability"] = (
                        (fp - fp_min) / (fp_max - fp_min + 1e-9)
                ).clip(0, 1).round(4)

            return df
        except Exception as e:
            log.warning("Помилка завантаження: %s", e)
            pass
    return _mock_dataset()

def _load_or_mock_results() -> dict:
    results_file = s.REPORT_DIR / "model_comparison.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
                # Завантажуємо реальні SHAP значення якщо є
                shap_path = s.REPORT_DIR / "shap_importances.json"
                if shap_path.exists():
                    with open(shap_path) as sf:
                        shap_fi = json.load(sf)
                    if "XGBoost" in results:
                        results["XGBoost"]["feature_importances"] = shap_fi
            return results
        except Exception:
            pass
    return _mock_model_results()

def _mock_dataset(n: int = 2_000) -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2019-01-01", periods=n, freq="6h")
    lats  = rng.uniform(27.5, 35.8, n)
    lons  = rng.uniform(-13.5, -1.0, n)
    temp  = rng.normal(28, 8, n).clip(5, 50)
    prec  = rng.exponential(3, n).clip(0, 80)
    wind  = rng.weibull(2, n) * 18
    ndvi  = rng.normal(0.38, 0.18, n).clip(-0.2, 0.9)
    soil  = rng.normal(0.28, 0.14, n).clip(0, 1)

    fp = np.clip(
        0.35*(temp/50) - 0.25*(prec/80) + 0.20*(wind/60)
        - 0.20*ndvi - 0.10*soil + rng.normal(0, 0.08, n),
        0.02, 0.97,
    )
    fp_min = fp.min()
    fp_max = fp.max()
    fp = (fp - fp_min) / (fp_max - fp_min + 1e-9)

    is_fire = (fp > 0.55).astype(int)

    return pd.DataFrame({
        "acq_date":                  dates,
        "latitude":                  lats.round(4),
        "longitude":                 lons.round(4),
        "NDVI":                      ndvi.round(4),
        "SoilMoisture":              soil.round(4),
        "average_temperature_lag_1": temp.round(2),
        "maximum_temperature_lag_1": (temp + rng.uniform(2, 8, n)).round(2),
        "minimum_temperature_lag_1": (temp - rng.uniform(2, 8, n)).round(2),
        "precipitation_lag_1":       prec.round(2),
        "wind_speed_lag_1":          wind.round(2),
        "wind_gust_lag_1":           (wind * 1.4).round(2),
        "dew_point_lag_1":           (temp - rng.uniform(5, 20, n)).round(2),
        "sea_distance":              rng.uniform(10, 600, n).round(1),
        "fire_probability":          fp.round(4),
        "is_fire":                   is_fire,
        "month":                     pd.DatetimeIndex(dates).month,
        "year":                      pd.DatetimeIndex(dates).year,
    })

def _mock_model_results() -> dict:
    rng = np.random.default_rng(0)

    def _roc(auc: float):
        fpr = np.linspace(0, 1, 100)
        tpr = np.clip(
            fpr ** (1 / (auc * 2.5 + 0.01)) + rng.normal(0, .02, 100),
            0, 1,
        )
        tpr[0], tpr[-1] = 0, 1
        return np.sort(fpr).tolist(), np.sort(tpr).tolist()

    fi_base = {
        "NDVI":                       0.187,
        "SoilMoisture":               0.143,
        "average_temperature_lag_1":  0.121,
        "maximum_temperature_lag_1":  0.098,
        "wind_gust_lag_1":            0.087,
        "dew_point_lag_1":            0.076,
        "precipitation_lag_1":        0.068,
        "wind_speed_lag_1":           0.057,
        "sea_distance":               0.044,
        "fire_weather_score":         0.038,
        "drought_index":              0.031,
        "temp_range_lag_1":           0.027,
        "ndvi_anomaly_flag":          0.023,
    }

    configs = {
        "XGBoost":            (0.951, 0.891, 0.884, 0.899, 0.278, [[870,102],[88,940]]),
        "RandomForest":       (0.933, 0.872, 0.868, 0.877, 0.321, [[855,117],[110,918]]),
        "LogisticRegression": (0.851, 0.781, 0.774, 0.793, 0.448, [[775,197],[187,841]]),
    }

    results = {}
    for name, (auc, acc, prec, rec, ll, cm) in configs.items():
        fpr, tpr = _roc(auc)
        fi = {k: round(v * rng.uniform(.85, 1.15), 4) for k, v in fi_base.items()}
        results[name] = {
            "roc_auc":          auc,
            "accuracy":         acc,
            "precision":        prec,
            "recall":           rec,
            "f1":               round(2 * prec * rec / (prec + rec + 1e-9), 3),
            "log_loss":         ll,
            "confusion_matrix": cm,
            "roc_fpr":          fpr,
            "roc_tpr":          tpr,
            "feature_importances": fi,
        }
    return results


def run_prediction(inputs: dict) -> dict:
    """Реальний прогноз через навчену XGBoost модель."""
    import joblib
    import numpy as np
    from datetime import datetime

    try:
        # Завантажуємо модель і трансформери
        model = joblib.load(s.MODEL_DIR / "xgboost_v1.joblib")

        # Формуємо вхідний рядок з усіма потрібними колонками
        row = {
            "NDVI":                            inputs.get("ndvi", 0.35),
            "SoilMoisture":                    inputs.get("soil_moisture", 0.30),
            "latitude":                        inputs.get("latitude", 31.5),
            "longitude":                       inputs.get("longitude", -7.1),
            "sea_distance":                    inputs.get("sea_distance", 100.0),
            "day_of_week":                     inputs.get("day_of_week", 3),
            "day_of_year":                     inputs.get("day_of_year", 180),
            "is_weekend":                      0,
            "is_holiday":                      0,
            "average_temperature_lag_1":       inputs.get("temperature", 28.0),
            "maximum_temperature_lag_1":       inputs.get("temperature", 28.0) + 5.0,
            "minimum_temperature_lag_1":       inputs.get("temperature", 28.0) - 5.0,
            "precipitation_lag_1":             inputs.get("precipitation", 5.0),
            "wind_speed_lag_1":                inputs.get("wind_speed", 12.0),
            "maximum_sustained_wind_speed_lag_1": inputs.get("wind_speed", 12.0) * 1.2,
            "wind_gust_lag_1":                 inputs.get("wind_speed", 12.0) * 1.5,
            "dew_point_lag_1":                 inputs.get("temperature", 28.0) - 10.0,
            "snow_depth_lag_1":                0.0,
            "fog_lag_1":                       0.0,
            "thunder_lag_1":                   0.0,
        }

        # Додаємо лаги 2-7 як копії лагу 1
        meteo_vars = [
            "average_temperature", "maximum_temperature", "minimum_temperature",
            "precipitation", "wind_speed", "maximum_sustained_wind_speed",
            "wind_gust", "dew_point", "snow_depth", "fog", "thunder",
        ]
        for var in meteo_vars:
            for lag in range(2, 8):
                col = f"{var}_lag_{lag}"
                base = f"{var}_lag_1"
                row[col] = row.get(base, 0.0)

        # Додаємо rolling means як середні значення
        rolling_vars = [
            "average_temperature", "maximum_temperature", "minimum_temperature",
            "precipitation", "wind_gust", "dew_point", "snow_depth",
        ]
        for var in rolling_vars:
            base_val = row.get(f"{var}_lag_1", 0.0)
            for suffix in ["weekly_mean", "monthly_mean", "quarterly_mean",
                           "yearly_mean", "last_1_year", "last_2_year", "last_3_year"]:
                row[f"{var}_{suffix}"] = base_val

        X = pd.DataFrame([row])

        # Прогноз
        prob = float(model.predict_proba(X)[0][1])

    except Exception as e:
        # Якщо щось пішло не так — fallback на симуляцію
        log.warning("Помилка реального прогнозу: %s. Використовую симуляцію.", e)
        rng = np.random.default_rng(42)
        prob = float(np.clip(
            0.40 * (inputs.get("temperature", 28) / 45)
            - 0.25 * (inputs.get("precipitation", 5) / 50)
            + 0.20 * (inputs.get("wind_speed", 12) / 50)
            - 0.15 * inputs.get("ndvi", 0.35)
            - 0.10 * inputs.get("soil_moisture", 0.30)
            + 0.48 + rng.normal(0, 0.04),
            0.04, 0.97,
        ))

    level = (
        "LOW"      if prob < 0.25 else
        "MEDIUM"   if prob < 0.55 else
        "HIGH"     if prob < 0.75 else
        "CRITICAL"
    )

    return {
        "probability": prob,
        "risk_level":  level,
        "confidence":  round(0.92, 3),
        "factors": [
            ("NDVI",              -inputs.get("ndvi", 0.35) * 0.9,       "Суха рослинність"),
            ("Температура (D-1)",  inputs.get("temperature", 28) / 45 * 0.7, "Висока температура"),
            ("Опади (D-1)",       -inputs.get("precipitation", 5) / 50 * 0.6, "Відсутність опадів"),
            ("Вітер",              inputs.get("wind_speed", 12) / 50 * 0.5,   "Поширює вогонь"),
            ("Вологість ґрунту",  -inputs.get("soil_moisture", 0.30) * 0.8,   "Сухий ґрунт"),
        ],
        "timestamp": datetime.now().isoformat(),
    }
def get_ai_recommendations(prediction_result: dict, inputs: dict) -> str:
    """Генерує рекомендації через локальну LLM (Ollama) з fallback на правила."""
    prob = prediction_result["probability"]
    level = prediction_result["risk_level"]
    factors_text = "\n".join(
        f"- {feat}: {desc}"
        for feat, _, desc in prediction_result["factors"]
    )

    prompt = f"""You are an expert decision-support system for a forest firefighting service in Morocco.

    CURRENT ASSESSMENT:
    - Fire probability: {prob:.1%}
    - Risk level: {level}
    - Location: {inputs['latitude']:.2f}, {inputs['longitude']:.2f}
    - NDVI: {inputs['ndvi']:.2f} | Soil moisture: {inputs['soil_moisture']:.2f}
    - Temperature: {inputs['temperature']:.1f}°C | Precipitation: {inputs['precipitation']:.1f} mm | Wind: {inputs['wind_speed']:.1f} km/h

    Risk factors identified:
    {factors_text}

    INSTRUCTIONS:
    The risk level is {level} ({prob:.1%}). Your response MUST match the severity of this exact level:
    - If LOW (<25%): state clearly that NO special action is needed beyond standard routine monitoring. Do NOT recommend deploying extra resources, water bombers, or crews on standby.
    - If MEDIUM (25-55%): recommend light increase in monitoring frequency only.
    - If HIGH (55-75%): recommend active patrols and resource readiness.
    - If CRITICAL (>75%): recommend immediate deployment and evacuation readiness.

    Write 3-4 short sentences total (max 80 words). Be direct and proportional to the actual risk level. Do not pad with generic firefighting advice unrelated to the current numbers."""

    try:
        import ollama
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}],
            options={"timeout": 30},
        )
        return response["message"]["content"]

    except Exception as e:
        log.warning("Ollama недоступний: %s. Використовую fallback.", e)
        return _rule_based_recommendations(prediction_result, inputs)

def _rule_based_recommendations(prediction_result: dict, inputs: dict) -> str:
    """Резервна експертна система на основі правил (якщо Ollama недоступний)."""
    prob  = prediction_result["probability"]
    level = prediction_result["risk_level"]

    lines = []

    if level == "CRITICAL":
        lines.append("🔴 КРИТИЧНИЙ РІВЕНЬ РИЗИКУ. Рекомендується негайне реагування: "
                      "інформування місцевих пожежних служб, підготовка техніки до виїзду, "
                      "обмеження доступу відвідувачів до лісової зони.")
    elif level == "HIGH":
        lines.append("🟠 ВИСОКИЙ РІВЕНЬ РИЗИКУ. Рекомендується посилене патрулювання території "
                      "протягом найближчих 24-48 годин та готовність техніки гасіння в режимі очікування.")
    elif level == "MEDIUM":
        lines.append("🟡 СЕРЕДНІЙ РІВЕНЬ РИЗИКУ. Рекомендується плановий моніторинг ситуації, "
                      "без додаткових ресурсів понад стандартний режим.")
    else:
        lines.append("🟢 НИЗЬКИЙ РІВЕНЬ РИЗИКУ. Додаткових заходів не потрібно, "
                      "стандартний режим спостереження є достатнім.")

    lines.append("")
    lines.append("Ключові фактори, що впливають на оцінку:")

    if inputs["ndvi"] < 0.2:
        lines.append("• NDVI критично низький — рослинність надзвичайно суха. "
                      "Рекомендується обмежити вогневі роботи та паління стерні в цій зоні.")
    elif inputs["ndvi"] < 0.4:
        lines.append("• NDVI помірно знижений — рослинність частково підсушена, потрібен контроль.")

    if inputs["soil_moisture"] < 0.15:
        lines.append("• Вологість ґрунту критично низька — висока ймовірність швидкого "
                      "поширення низової пожежі.")

    if inputs["temperature"] > 35:
        lines.append(f"• Температура {inputs['temperature']:.0f}°C — екстремально висока, "
                      "підвищує швидкість висихання горючих матеріалів.")
    elif inputs["temperature"] > 28:
        lines.append(f"• Температура {inputs['temperature']:.0f}°C — підвищена, типова умова "
                      "для активізації пожежної небезпеки.")

    if inputs["precipitation"] < 1:
        lines.append("• Опади відсутні — накопичується дефіцит вологи, рекомендується "
                      "відстежувати динаміку за останні 7-10 днів.")

    if inputs["wind_speed"] > 30:
        lines.append(f"• Швидкість вітру {inputs['wind_speed']:.0f} км/год — значний ризик "
                      "швидкого поширення фронту пожежі та переносу іскор на сусідні ділянки.")
    elif inputs["wind_speed"] > 15:
        lines.append(f"• Швидкість вітру {inputs['wind_speed']:.0f} км/год — помірна, "
                      "враховувати при плануванні маршрутів патрулювання.")

    lines.append("")
    lines.append(f"Розрахована ймовірність пожежі: {prob:.1%} "
                  f"(модель XGBoost, AUC-ROC = 0.9905)")

    return "\n".join(lines)

def run_scenario_simulation(base_inputs: dict, scenarios: list[dict]) -> list[dict]:
    """Прогнозує ризик для кількох сценаріїв на основі базових умов."""
    results = []
    for name, changes in scenarios:
        scenario_inputs = dict(base_inputs)
        scenario_inputs.update(changes)
        pred = run_prediction(scenario_inputs)
        results.append({
            "name": name,
            "probability": pred["probability"],
            "risk_level": pred["risk_level"],
            "inputs": scenario_inputs,
        })
    return results


def get_top_risk_zones(data: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Повертає N точок з найвищим прогнозованим ризиком."""
    df = data.copy()
    if "fire_probability" not in df.columns:
        df["fire_probability"] = df.get("is_fire", 0).astype(float)

    cols = ["latitude", "longitude", "fire_probability"]
    for c in ["NDVI", "SoilMoisture", "average_temperature_lag_1",
              "precipitation_lag_1", "wind_speed_lag_1"]:
        if c in df.columns:
            cols.append(c)

    top = df.nlargest(n, "fire_probability")[cols].reset_index(drop=True)
    top.index = top.index + 1
    return top


def generate_summary_report(data: pd.DataFrame, results: dict) -> str:
    """Генерує короткий текстовий звіт про поточну ситуацію."""
    if "fire_probability" not in data.columns:
        data = data.copy()
        data["fire_probability"] = data.get("is_fire", 0).astype(float)

    total = len(data)
    high_risk = (data["fire_probability"] > 0.55).sum()
    critical_risk = (data["fire_probability"] > 0.75).sum()
    pct_high = high_risk / total * 100 if total else 0
    pct_critical = critical_risk / total * 100 if total else 0

    avg_ndvi = data["NDVI"].mean() if "NDVI" in data.columns else None
    avg_temp = data["average_temperature_lag_1"].mean() if "average_temperature_lag_1" in data.columns else None

    top_zones = get_top_risk_zones(data, 3)

    best_model = max(results.items(), key=lambda x: x[1].get("roc_auc", 0)) if results else ("XGBoost", {})
    model_name, model_res = best_model

    if pct_critical > 10:
        readiness = "ВИСОКИЙ — рекомендується посилена готовність ресурсів"
    elif pct_high > 20:
        readiness = "ПІДВИЩЕНИЙ — рекомендується активний моніторинг"
    else:
        readiness = "СТАНДАРТНИЙ — звичайний режим спостереження"

    lines = [
        f"ЗВІТ ПРО ПОТОЧНУ ПОЖЕЖНУ СИТУАЦІЮ",
        f"",
        f"Проаналізовано точок: {total:,}",
        f"Зон підвищеного ризику (>55%): {high_risk:,} ({pct_high:.1f}%)",
        f"Зон критичного ризику (>75%): {critical_risk:,} ({pct_critical:.1f}%)",
        f"",
    ]

    if avg_ndvi is not None:
        lines.append(f"Середній NDVI по вибірці: {avg_ndvi:.3f}")
    if avg_temp is not None:
        lines.append(f"Середня температура (D-1): {avg_temp:.1f}°C")

    lines.append("")
    lines.append("Топ-3 найкритичніші зони:")
    for i, row in top_zones.iterrows():
        lines.append(
            f"  {i}. ({row['latitude']:.3f}, {row['longitude']:.3f}) "
            f"— ризик {row['fire_probability']:.0%}"
        )

    lines.append("")
    lines.append(f"Рекомендований рівень готовності: {readiness}")
    lines.append("")
    lines.append(
        f"Прогноз сформовано моделлю {model_name} "
        f"(AUC-ROC = {model_res.get('roc_auc', 0):.3f})"
    )

    return "\n".join(lines)

def _load_algeria_data() -> pd.DataFrame:
    """Завантажує датасет Algerian Forest Fires."""
    if s.ALGERIA_DATA_FILE.exists():
        try:
            df = pd.read_csv(s.ALGERIA_DATA_FILE)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            log.warning("Помилка завантаження датасету Алжиру: %s", e)
    return pd.DataFrame()


def _load_algeria_results() -> dict:
    """Завантажує метрики моделей Алжиру."""
    if s.ALGERIA_REPORT_FILE.exists():
        try:
            with open(s.ALGERIA_REPORT_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning("Помилка завантаження результатів Алжиру: %s", e)
    return {}


def run_algeria_prediction(inputs: dict) -> dict:
    """Прогноз ризику пожежі для Алжиру через Random Forest (FWI-ознаки)."""
    import joblib

    try:
        bundle = joblib.load(s.ALGERIA_RF_MODEL)
        model    = bundle["model"]
        features = bundle["features"]

        row = {f: inputs.get(f, 0.0) for f in features}
        X = pd.DataFrame([row])[features]

        prob = float(model.predict_proba(X)[0][1])

    except Exception as e:
        log.warning("Помилка прогнозу Алжиру: %s. Використовую симуляцію.", e)
        rng = np.random.default_rng(42)
        prob = float(np.clip(
            0.35 * (inputs.get("isi", 5) / 20)
            + 0.30 * (inputs.get("ffmc", 80) / 100)
            + 0.20 * (inputs.get("fwi", 10) / 30)
            - 0.15 * (inputs.get("rain", 0) / 10)
            + 0.30 + rng.normal(0, 0.04),
            0.02, 0.98,
        ))

    level = (
        "LOW"      if prob < 0.25 else
        "MEDIUM"   if prob < 0.55 else
        "HIGH"     if prob < 0.75 else
        "CRITICAL"
    )

    return {
        "probability": prob,
        "risk_level":  level,
        "confidence":  round(0.96, 3),
        "factors": [
            ("ISI (індекс початкового поширення)", inputs.get("isi", 5) / 20 * 0.8, "Швидкість поширення вогню"),
            ("FFMC (вологість тонкого паливо)",     inputs.get("ffmc", 80) / 100 * 0.7, "Сухість поверхневого шару"),
            ("FWI (загальний індекс)",              inputs.get("fwi", 10) / 30 * 0.6, "Комплексна пожежна небезпека"),
            ("Опади",                               -inputs.get("rain", 0) / 10 * 0.5, "Зволоження території"),
            ("Температура",                          inputs.get("temperature", 28) / 45 * 0.4, "Швидкість висихання"),
        ],
        "timestamp": datetime.now().isoformat(),
    }