"""LightGBM expected-points model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from .features import TARGET

DEFAULT_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbose=-1,
    seed=42,
)


@dataclass
class PointsModel:
    features: list[str]
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    booster: lgb.Booster | None = None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None,
            num_boost_round: int = 3000) -> "PointsModel":
        dtrain = lgb.Dataset(train[self.features], train[TARGET])
        callbacks, valid_sets = [lgb.log_evaluation(0)], []
        if valid is not None:
            valid_sets = [lgb.Dataset(valid[self.features], valid[TARGET], reference=dtrain)]
            callbacks.append(lgb.early_stopping(100, verbose=False))
        self.booster = lgb.train(self.params, dtrain, num_boost_round=num_boost_round,
                                 valid_sets=valid_sets, callbacks=callbacks)
        return self

    @property
    def best_iteration(self) -> int:
        return self.booster.best_iteration or self.booster.current_iteration()

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.booster.predict(df[self.features], num_iteration=self.best_iteration),
                         index=df.index, name="pred")

    def feature_importance(self) -> pd.Series:
        imp = self.booster.feature_importance("gain")
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": self.features, "params": self.params,
                     "model": self.booster.model_to_string(num_iteration=self.best_iteration)}, path)

    @classmethod
    def load(cls, path: str | Path) -> "PointsModel":
        blob = joblib.load(path)
        return cls(blob["features"], blob["params"], lgb.Booster(model_str=blob["model"]))
