"""
src/models/svm_model.py
"""

from sklearn.svm import SVC
from src.models.base_model import BaseModel
import config.settings as s


class SVMModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(
            name="SVM",
            params=params or {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "probability": True,
                "random_state": s.RANDOM_STATE,
                "cache_size": 1000,
            },
        )

    def _build(self) -> SVC:
        return SVC(**self.params)