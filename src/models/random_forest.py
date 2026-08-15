"""
src/models/random_forest.py
"""

from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel
import config.settings as s


class RandomForestModel(BaseModel):
    def __init__(self, params: dict = None):
        super().__init__(
            name="RandomForest",
            params=params or s.MODEL_CONFIGS["random_forest"]["params"],
        )

    def _build(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.params)