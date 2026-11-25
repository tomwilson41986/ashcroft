"""The back-to-lay strategy's model, frozen for the live trader: fitted on the warehouse, saved, loaded, applied.

racing² football, 25 Nov 2026 (the owner: back-to-lay live on all four cells, GBP 2 a bet, GBP 50 a day, below the
registered bar: ledger ``football-b2l-live-1125``). The model is the move model of ``football-b2l-live-model-1125``:
the features the live feed can see (``FEATURES``), a ridge and a LightGBM a role (home, draw, away), averaged, exactly
as ``football_bf_close_model.walk_forward`` fits them, here fitted once on every usable warehouse row and saved as one
JSON document (``football/b2l/model/latest.json`` in the capture bucket). The trader loads it and only predicts.

The four cells the trader acts on (``CELLS``): at the first book at or after 360 minutes before the off, a forecast
over 0.05 or over 0.03; at 15 minutes, over 0.05 or over 0.08. A forecast is the expected move of the runner's
normalised chance to the close, in log odds: positive, the chance is expected to rise and the price to shorten, so the
trade is to back now and lay at the close.

    python -m football.b2l --fit          # the weekly refit: fit, save, print the summary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_KEY = "football/b2l/model/latest.json"
ROLES = ("home", "draw", "away")
#: (checkpoint minutes, forecast threshold) by cell name; a runner trades once a market, at the first cell it clears
CELLS = {"360 over 0.05": (360, 0.05), "360 over 0.03": (360, 0.03), "15 over 0.05": (15, 0.05),
         "15 over 0.08": (15, 0.08)}
CHECKPOINTS = (360, 15)


def features() -> list[str]:
    """The live model's columns, in order (research/open_close/football_b2l_live_model.py)."""
    from research.open_close import football_b2l_live_model as lm
    from research.open_close.football_bf_features2 import IMP_COLS
    imp_delta = [f"idl_d_{c}" for c in IMP_COLS if c != "sup_gap"] + ["idl_own_dp"]
    return (lm.LEVEL + lm.IMPLIED + [f"pp_{c}" for c in lm.POLL_PATH] + [f"lv_{c}" for c in lm.MEMORY] + imp_delta)


# ---------------------------------------------------------------------------------------------------------------
# the ridge, fitted and kept (the same arithmetic as football_bf_close_model.ridge_fit_predict)
# ---------------------------------------------------------------------------------------------------------------

def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float, clip: float) -> dict:
    keep = np.mean(np.isfinite(X), axis=0) >= 0.2
    Xk = X[:, keep]
    miss = ~np.isfinite(Xk)
    mu, sd = np.nanmean(Xk, 0), np.nanstd(Xk, 0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    flag = miss.mean(0) >= 0.02
    Z = np.c_[np.clip(np.nan_to_num((Xk - mu) / sd), -clip, clip), miss[:, flag]]
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {"keep": keep.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "flag": flag.tolist(),
            "beta": beta.tolist(), "y_mean": float(y.mean()), "clip": clip}


def ridge_predict(r: dict, X: np.ndarray) -> np.ndarray:
    keep, flag = np.array(r["keep"], bool), np.array(r["flag"], bool)
    Xk = X[:, keep]
    miss = ~np.isfinite(Xk)
    z = np.clip(np.nan_to_num((Xk - np.array(r["mu"])) / np.array(r["sd"])), -r["clip"], r["clip"])
    return r["y_mean"] + np.c_[z, miss[:, flag]] @ np.array(r["beta"])


# ---------------------------------------------------------------------------------------------------------------
# fit, save, load, predict
# ---------------------------------------------------------------------------------------------------------------

def fit(d: pd.DataFrame, cols: list[str]) -> dict:
    """The model, a role at a time, on the usable rows of ``d`` (the frame of football_b2l_live_model.frame)."""
    from research.open_close import football_bf_close_model as bf
    out = {"features": cols, "roles": {}, "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "settings": {"ridge_alpha": bf.RIDGE_ALPHA, "clip": bf.CLIP}}
    for role in ROLES:
        tr = d[(d["role"] == role) & d["usable"]]
        y = tr["y_logit"].to_numpy(float)
        X = tr[cols].to_numpy(float)
        g = bf.gbm_fit(tr, cols, y)
        out["roles"][role] = {"rows": int(len(tr)), "ridge": ridge_fit(X, y, bf.RIDGE_ALPHA, bf.CLIP),
                              "gbm": g.booster_.model_to_string()}
        log.info("b2l model %s: %d rows", role, len(tr))
    kick = pd.to_datetime(d["kickoff"], utc=True)
    out["trained_on"] = {"rows": int(d["usable"].sum()), "first_kickoff": str(kick.min()), "last_kickoff": str(kick.max())}
    out["sha256"] = hashlib.sha256(json.dumps(out["roles"], sort_keys=True).encode()).hexdigest()[:16]
    return out


def predict(model: dict, rows: pd.DataFrame) -> np.ndarray:
    """The forecast for each row (its ``role`` and the model's features; a missing feature is blank)."""
    import lightgbm as lgb
    cols = model["features"]
    X = rows.reindex(columns=cols).to_numpy(float)
    out = np.full(len(rows), np.nan)
    for role, m in model["roles"].items():
        r = (rows["role"] == role).to_numpy()
        if not r.any():
            continue
        booster = lgb.Booster(model_str=m["gbm"])
        out[r] = 0.5 * (ridge_predict(m["ridge"], X[r]) + booster.predict(X[r]))
    return out


def cell_for(cp_min: int, forecast: float) -> str | None:
    """The cell a forecast at a checkpoint clears (the stricter first), or None."""
    hits = [(t, n) for n, (cp, t) in CELLS.items() if cp == cp_min and np.isfinite(forecast) and forecast > t]
    return max(hits)[1] if hits else None


def save(store, model: dict) -> None:
    store.put(MODEL_KEY, json.dumps(model).encode())


def load(store) -> dict | None:
    raw = store.get(MODEL_KEY)
    return json.loads(raw) if raw else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", action="store_true", help="fit on the warehouse and save to the capture bucket")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    store = Store(root=a.root) if a.root else Store()
    if a.fit:
        from research.open_close import football_b2l_live_model as lm
        d, _full, _live = lm.frame(store)
        model = fit(d, features())
        save(store, model)
        print(json.dumps({k: model[k] for k in ("fitted_utc", "trained_on", "sha256")}
                         | {"rows_by_role": {r: m["rows"] for r, m in model["roles"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
