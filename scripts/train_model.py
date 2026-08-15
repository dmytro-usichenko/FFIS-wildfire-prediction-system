"""
scripts/train_model.py
───────────────────────
Використання:  python scripts/train_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd

from src.utils.logger import get_logger
from src.data.loader import DataLoader
from src.data.splitter import RandomSplitter
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.models.xgboost_model import XGBoostModel
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm_model import SVMModel
import config.settings as s

log = get_logger("train")


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {s.TARGET, s.DATE_COL, "_month", "_year"}
    result = []
    for c in df.columns:
        if c in exclude:
            continue
        if "_lag_" in c:
            try:
                lag_num = int(c.split("_lag_")[-1])
                if lag_num <= s.LAG_WINDOW:
                    result.append(c)
            except ValueError:
                result.append(c)
        else:
            result.append(c)
    log.info("Ознак для навчання: %d", len(result))
    return result


def main():
    log.info("=" * 55)
    log.info("  FFIS — НАВЧАННЯ МОДЕЛЕЙ")
    log.info("=" * 55)

    # 1. Завантаження даних
    log.info("Крок 1: Завантаження даних...")
    df = DataLoader().load()

    # 2. Вибір ознак
    log.info("Крок 2: Вибір ознак...")
    feature_cols = get_feature_cols(df)

    # 3. Розбивка на train/test
    log.info("Крок 3: Розбивка даних...")
    split = RandomSplitter().split(df, feature_cols)

    # 4. Навчання моделей
    models = [
        (XGBoostModel(), False),
        (RandomForestModel(), False),
        (LogisticRegressionModel(), True),
        (SVMModel(), True),
    ]

    evaluator   = Evaluator()
    all_results = {}

    for model, needs_scaling in models:
        log.info("Крок 4: Навчання %s...", model.name)
        trainer = Trainer(model, feature_cols, apply_scaling=needs_scaling)

        # SVM навчаємо на меншій вибірці — інакше дуже повільно
        if model.name == "SVM":
            X_tr = split.X_train.sample(20_000, random_state=42)
            y_tr = split.y_train.loc[X_tr.index]
            trainer.fit(X_tr, y_tr)
        else:
            trainer.fit(split.X_train, split.y_train)

        # Трансформуємо ДО збереження артефактів
        log.info("Крок 5: Оцінка %s...", model.name)
        X_test_t = trainer.transform(split.X_test)
        metrics = evaluator.evaluate(
            model, X_test_t, split.y_test, model.name
        )
        all_results[model.name] = metrics

        # Зберігаємо після оцінки
        trainer.save_artifacts(version="v1")
        metrics["train_time_s"] = trainer._train_time

        # Зберігаємо SHAP значення для XGBoost
        if model.name == "XGBoost":
            try:
                import shap, json
                log.info("Рахуємо SHAP значення...")
                xgb_model = model.sklearn_model
                explainer = shap.TreeExplainer(xgb_model)
                X_shap = pd.DataFrame(X_test_t).head(100)
                shap_values = explainer.shap_values(X_shap)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                mean_abs = abs(shap_values).mean(axis=0)
                feature_names = list(X_shap.columns) if hasattr(X_shap, 'columns') else [f"f{i}" for i in
                                                                                         range(len(mean_abs))]
                shap_fi = {name: round(float(val), 6)
                           for name, val in zip(feature_names, mean_abs)}
                shap_fi = dict(sorted(shap_fi.items(), key=lambda x: -x[1]))
                shap_path = s.REPORT_DIR / "shap_importances.json"
                with open(shap_path, "w") as f:
                    json.dump(shap_fi, f, indent=2)
                metrics["feature_importances"] = shap_fi
                log.info("SHAP збережено: %d ознак", len(shap_fi))
            except Exception as e:
                log.warning("SHAP помилка: %s", e)

    # 5. Зберігаємо результати
    out = s.REPORT_DIR / "model_comparison.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Результати збережено: %s", out)

    # 6. Підсумок
    log.info("=" * 55)
    log.info("  ПІДСУМОК")
    log.info("=" * 55)
    for name, m in all_results.items():
        log.info("%-25s  AUC=%.4f  F1=%.4f", name, m["roc_auc"], m["f1"])


if __name__ == "__main__":
    main()