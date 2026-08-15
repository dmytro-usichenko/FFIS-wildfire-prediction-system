"""
src/models/xgboost_model.py
"""

from xgboost import XGBClassifier
from src.models.base_model import BaseModel
import config.settings as s


class XGBoostModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(
            name="XGBoost",
            params=params or s.MODEL_CONFIGS["xgboost"]["params"],
        )

    def _build(self) -> XGBClassifier:
        return XGBClassifier(**self.params)