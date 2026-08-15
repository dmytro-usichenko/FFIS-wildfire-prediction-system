"""
src/models/logistic_regression.py
"""

from sklearn.linear_model import LogisticRegression
from src.models.base_model import BaseModel
import config.settings as s


class LogisticRegressionModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(
            name="LogisticRegression",
            params=params or s.MODEL_CONFIGS["logistic_regression"]["params"],
        )

    def _build(self) -> LogisticRegression:
        return LogisticRegression(**self.params)