"""Next-ball outcome model v1 (build spec 6.3): XGBoost multinomial over the eight outcomes.

Trained on men's 20-over T20 deliveries (every competition, from 2012) before a cut-off date; scored on later
deliveries only. The benchmark it must beat is the base rate: each outcome's training frequency by innings and phase.
Any accuracy above 90% is treated as leakage until shown otherwise (build spec 6.7): the run fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from cricket.model.features import FEATURES
from cricket.model.metrics import multiclass_log_loss
from cricket.model.outcomes import K

PARAMS = {"objective": "multi:softprob", "num_class": K, "eta": 0.15, "max_depth": 6, "min_child_weight": 50,
          "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 5.0, "tree_method": "hist", "eval_metric": "mlogloss",
          "nthread": 4, "seed": 7}
LEAKAGE_ACCURACY = 0.90


def _dm(f: pd.DataFrame) -> xgb.DMatrix:
    return xgb.DMatrix(f[FEATURES].to_numpy(dtype=np.float32), label=f["cls"].to_numpy(), feature_names=FEATURES)


def choose_rounds(train: pd.DataFrame, valid: pd.DataFrame, max_rounds: int = 1500) -> tuple[int, float]:
    """Rounds by early stopping on a later validation block (never the test block)."""
    b = xgb.train(PARAMS, _dm(train), max_rounds, evals=[(_dm(valid), "valid")], early_stopping_rounds=40,
                  verbose_eval=False)
    return int(b.best_iteration) + 1, float(b.best_score)


def fit(train: pd.DataFrame, rounds: int) -> xgb.Booster:
    return xgb.train(PARAMS, _dm(train), rounds)


def predict(b: xgb.Booster, X: np.ndarray) -> np.ndarray:
    return b.inplace_predict(X.astype(np.float32, copy=False))


def base_rate(train: pd.DataFrame) -> pd.DataFrame:
    """Outcome frequencies by innings and phase: the benchmark."""
    oh = pd.get_dummies(train["cls"]).reindex(columns=range(K), fill_value=0)
    oh[["innings", "phase_code"]] = train[["innings", "phase_code"]].to_numpy()
    return oh.groupby(["innings", "phase_code"]).mean()


def evaluate(b: xgb.Booster, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    P = predict(b, test[FEATURES].to_numpy())
    br = base_rate(train)
    Pb = br.reindex(pd.MultiIndex.from_frame(test[["innings", "phase_code"]])).to_numpy(dtype=float)
    y = test["cls"].to_numpy()
    acc = float((P.argmax(1) == y).mean())
    if acc > LEAKAGE_ACCURACY:
        raise SystemExit(f"next-ball accuracy {acc:.3f} above {LEAKAGE_ACCURACY}: treated as leakage")
    return {"n": int(len(test)), "log_loss": multiclass_log_loss(y, P),
            "base_rate_log_loss": multiclass_log_loss(y, Pb),
            "accuracy": acc, "mean_p": P.mean(0).round(4).tolist(),
            "observed": np.bincount(y, minlength=K).astype(float).__truediv__(len(y)).round(4).tolist()}
