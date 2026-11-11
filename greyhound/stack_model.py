"""The live greyhound price model from 10 Nov 2026 (the owner: "move the live greyhound pricing to the new model").

Batch 18's ``stack3_all3`` (reports/greyhound_model.md section 36; greyhound-fastloop.yml run 42853398485): on the test
of 1 Jun to 5 Nov 2026 its SP forecast's RMSE of log(BSP) is 0.3366 with the per-track stretch, against the log-BSP
recipe's 0.4017 as served. What it is:

- **Features:** every card-safe engine feature (greyhound.metrics, the fast loop's ``card_safe`` set), the race
  simulator and the weight block (greyhound/blocks ``sim``, ``wpred``), and the metrics database's tables
  (greyhound/pit: figs, ability, trouble, connections, market, breeding, life, form, draw, bend), less three
  prefixes batch 14 found wrong for the card (``DROP``).
- **Trees:** three LightGBM seeds, each a conditional logit over the race fitted to the Betfair SP book's shares
  (greyhound.fastloop ``race_soft_objective``), learning rate 0.08, early-stopped on the last tenth of the fit window.
- **Networks:** three DeepSets set models over the field (greyhound/nn.py), their scores averaged.
- **The stack:** the race softmax of ``a x trees + b x networks``, ``a`` and ``b`` fitted on the early-stopping rows.
- **The price:** the softmax's share of the SP book (``p_model``, the win chance), and the SP forecast ``pred_bsp`` =
  1 / (the SP book's usual total x the share stretched by track, greyhound.fastloop ``fit_stretch``).

This module mirrors greyhound.fastloop ``run_experiment`` for that recipe (tests/test_greyhound_stack_model.py checks
the two price the same rows the same), so the live price is the one the research scored.

    bundle = fit(df, features(df, base_features), before="2026-11-10")
    priced = price(bundle, card_rows)
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

#: the metrics database's tables the model reads (greyhound/pit), in their build order
PIT_TABLES = ("ability", "trouble", "connections", "market", "breeding", "life", "figs", "form", "draw", "bend")
MATRIX_BLOCKS = ("sim", "wpred") + tuple(f"pit_{t}" for t in PIT_TABLES)
#: batch 14 (run 42818559765): the weigh-in's own columns, a season count GBGB fills after the day, a trainer change
#: the card cannot see - the one list every card-safe feature set drops (greyhound/live.py)
from greyhound.live import CARD_UNSAFE_PREFIXES as DROP  # noqa: E402
RECIPE = {"name": "stack3_all3", "learning_rate": 0.08, "early_stop": 100, "rounds": 10000, "max_bin": 63,
          "seeds": 3, "nn_seeds": 3, "nn_seconds": 1800.0, "window_months": 29, "max_seconds": None,
          "calibration": "stretch_track"}


def features(df: pd.DataFrame, base: list[str], blocks=MATRIX_BLOCKS, drop=DROP) -> list[str]:
    """The model's columns: ``base`` (the card-safe engine features), each block's columns added to ``df`` in place
    (greyhound/blocks; ``pit_<table>`` joins the table already in ``df`` or from the store), the ``drop`` prefixes
    removed, in the fast loop's order."""
    from greyhound import blocks as gblocks
    feats = list(base)
    for b in blocks:
        feats += [f for f in gblocks.attach(df, b) if f not in feats]
    feats = [f for f in feats if not any(f.startswith(p) for p in drop)]
    return [f for f in feats if f in df.columns]


def _matrix(df: pd.DataFrame, feats: list[str], rows: np.ndarray | None = None) -> np.ndarray:
    rows = np.arange(len(df)) if rows is None else rows
    X = np.empty((len(rows), len(feats)), dtype=np.float32)
    for j, f in enumerate(feats):
        X[:, j] = df[f].to_numpy(np.float32, na_value=np.nan)[rows]
    return X


def _starts(race: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(race[1:] != race[:-1]) + 1]


def window_start(before: str, months: int) -> str:
    return f"{(pd.Timestamp(before) - pd.DateOffset(months=int(months))):%Y-%m-%d}"


def fit(df: pd.DataFrame, feats: list[str], before: str, recipe: dict | None = None,
        train_from: str | None = None, log=print) -> dict:
    """The bundle: the trees, the networks, the stack's weights, the stretch and the SP book, fitted on the modelling
    races wholly priced by the Betfair SP in [train_from, before) (``train_from``: ``recipe['window_months']`` before
    ``before`` unless given), rows grouped by race in time order, early-stopped on the window's last tenth."""
    import lightgbm as lgb
    from scipy.optimize import minimize

    from greyhound import nn
    from greyhound.fastloop import BASE_PARAMS, fit_stretch, race_soft_objective, soft_target, time_budget
    from greyhound.model import _race_softmax
    rc = dict(RECIPE, **(recipe or {}))
    t0 = time.monotonic()
    train_from = train_from or window_start(before, rc["window_months"])
    t = df.t.to_numpy().astype("datetime64[ns]")
    lo, edge = np.datetime64(pd.Timestamp(train_from), "ns"), np.datetime64(pd.Timestamp(before), "ns")
    fit_i = np.flatnonzero((t >= lo) & (t < edge))
    fit_i = fit_i[df.bsp_p_norm.notna().to_numpy()[fit_i]]
    if not len(fit_i):
        raise ValueError(f"no races wholly priced by the Betfair SP in [{train_from}, {before})")
    cut = np.quantile(t[fit_i].astype("int64"), 0.9).astype("datetime64[ns]")
    tr_i, es_i = fit_i[t[fit_i] < cut], fit_i[t[fit_i] >= cut]
    race = pd.factorize(df.race_id)[0]
    soft = soft_target(df, "share")
    won = df.won.to_numpy(float)
    params = dict(BASE_PARAMS, learning_rate=rc["learning_rate"], max_bin=rc["max_bin"])
    X_tr, X_es = _matrix(df, feats, tr_i), _matrix(df, feats, es_i)
    dtr = lgb.Dataset(X_tr, won[tr_i], feature_name=feats, free_raw_data=True, params={"max_bin": rc["max_bin"]})
    des = lgb.Dataset(X_es, won[es_i], reference=dtr, free_raw_data=True)
    fobj, _ = race_soft_objective(race[tr_i], soft[tr_i])
    _, feval = race_soft_objective(race[es_i], soft[es_i])
    trees = []
    for k in range(int(rc["seeds"])):
        es = lgb.early_stopping(rc["early_stop"], verbose=False)
        m = lgb.train(dict(params, seed=int(params.get("seed", 0)) + k, objective=fobj, metric="None"), dtr,
                      rc["rounds"], valid_sets=[des], feval=feval,
                      callbacks=[es, time_budget(rc.get("max_seconds"), es)])
        trees.append(m)
        log(f"tree seed {k}: {m.best_iteration} rounds, {time.monotonic() - t0:.0f} s")
    del dtr, des
    nets = []
    for k in range(int(rc["nn_seeds"])):
        nets.append(nn.fit(X_tr, race[tr_i], soft[tr_i], X_es, race[es_i], soft[es_i],
                           seconds=float(rc["nn_seconds"]) / int(rc["nn_seeds"]), seed=int(params.get("seed", 0)) + k))
        log(f"network seed {k}: {nets[-1]['epochs']} epochs, best {nets[-1]['best']}")
    bundle = {"feats": list(feats), "trees": trees, "nets": nets, "recipe": rc}
    sg, sn, st_es = _trees(bundle, X_es), _nets(bundle, X_es, race[es_i]), _starts(race[es_i])
    t_es = np.nan_to_num(soft[es_i])

    def xent(w):
        p = _race_softmax(w[0] * sg + w[1] * sn, st_es)
        return float(-(t_es * np.log(np.clip(p, 1e-12, None))).sum())
    w = minimize(xent, np.array([0.5, 0.5]), method="Nelder-Mead").x
    bundle["stack"] = {"a_trees": round(float(w[0]), 4), "b_network": round(float(w[1]), 4)}
    fr = pd.DataFrame({"r": df.race_id.to_numpy()[fit_i], "b": df.bsp.to_numpy(float)[fit_i]})
    bundle["book"] = float(np.median((1.0 / fr.b).groupby(fr.r).sum()))
    raw_es = bundle["stack"]["a_trees"] * sg + bundle["stack"]["b_network"] * sn
    lbsp = np.log(np.clip(df.bsp.to_numpy(float), 1.01, 1000.0))
    bundle["stretch"] = (fit_stretch(-raw_es, lbsp[es_i], won[es_i], race[es_i], df.track.to_numpy()[es_i],
                                     rc["calibration"]) if rc.get("calibration") else None)
    bundle["meta"] = {"recipe": rc["name"], "before": before, "train_from": train_from,
                      "fit_rows": int(len(fit_i)), "fit_races": int(len(np.unique(race[fit_i]))),
                      "rounds": [int(m.best_iteration) for m in trees],
                      "stopped_by": ["time" if rc.get("max_seconds") and m.current_iteration() < rc["rounds"]
                                     and m.current_iteration() - m.best_iteration < rc["early_stop"] else "early_stop"
                                     for m in trees],
                      "nets": [{k: nm[k] for k in ("epochs", "best", "seconds")} for nm in nets],
                      "stack": bundle["stack"], "book": round(bundle["book"], 5),
                      "stretch_b": (bundle["stretch"] or {}).get("b"), "features": len(feats),
                      "seconds": round(time.monotonic() - t0)}
    return bundle


def _trees(bundle: dict, X: np.ndarray) -> np.ndarray:
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in bundle["trees"]], axis=0)


def _nets(bundle: dict, X: np.ndarray, race: np.ndarray) -> np.ndarray:
    from greyhound import nn
    return np.mean([nn.predict(nm, X, race) for nm in bundle["nets"]], axis=0)


def raw_scores(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """The stack's score of each row (rows grouped by race): its log-odds within the race."""
    X = _matrix(df, bundle["feats"])
    race = pd.factorize(df.race_id)[0]
    s = bundle["stack"]
    return s["a_trees"] * _trees(bundle, X) + s["b_network"] * _nets(bundle, X, race)


def price(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """``df``'s rows priced (they must be grouped by race): ``p_model`` the race softmax (the SP book's share, the win
    chance), ``pred_bsp`` the stretched SP forecast as served, ``pred_bsp_raw`` the unstretched one."""
    from greyhound.model import _race_softmax
    if not len(df):
        return pd.DataFrame(index=df.index, columns=["p_raw", "p_model", "pred_bsp", "pred_bsp_raw"], dtype=float)
    race_codes = pd.factorize(df.race_id)[0]
    if (np.diff(race_codes) < 0).any():
        raise ValueError("the rows must be grouped by race")
    raw = raw_scores(bundle, df)
    st = _starts(race_codes)
    out = pd.DataFrame(index=df.index)
    out["p_raw"] = raw
    out["p_model"] = _race_softmax(raw.astype(float), st)
    book = float(bundle["book"])
    out["pred_bsp_raw"] = 1.0 / (book * out.p_model)
    fitb = bundle.get("stretch")
    if fitb:
        b = df.track.astype(str).map(fitb.get("by_track") or {}).fillna(fitb["b"]).to_numpy(float)
        lp = -raw
        c = lp - pd.Series(lp).groupby(race_codes).transform("mean").to_numpy()
        share = _race_softmax(-b * c, st)
        out["pred_bsp"] = 1.0 / (book * share)
    else:
        out["pred_bsp"] = out.pred_bsp_raw
    return out


# --------------------------------------------------------------------------------------------------------------------
# Saved and loaded as one file (torch's pickle: the trees as LightGBM's text, the networks as their weights)
# --------------------------------------------------------------------------------------------------------------------

def to_bytes(bundle: dict) -> bytes:
    import torch
    nets = [{"d": int(nm["net"].phi[0].in_features), "state": nm["net"].state_dict(),
             "knots": nm["tf"].knots, "z": nm["tf"].z, "flag": nm["tf"].flag,
             **{k: nm[k] for k in ("epochs", "best", "seconds")}} for nm in bundle["nets"]]
    obj = {"feats": bundle["feats"], "recipe": bundle["recipe"], "stack": bundle["stack"], "book": bundle["book"],
           "stretch": bundle["stretch"], "meta": bundle.get("meta", {}),
           "trees": [m.model_to_string(num_iteration=m.best_iteration) for m in bundle["trees"]], "nets": nets}
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def from_bytes(raw: bytes) -> dict:
    import lightgbm as lgb
    import torch

    from greyhound import nn
    obj = torch.load(io.BytesIO(raw), weights_only=False)        # our own file from our own bucket
    trees = [lgb.Booster(model_str=s) for s in obj["trees"]]
    nets = []
    for n in obj["nets"]:
        net = nn._net(n["d"])
        net.load_state_dict(n["state"])
        net.eval()
        tf = nn.Transform.__new__(nn.Transform)
        tf.knots, tf.z, tf.flag = n["knots"], n["z"], n["flag"]
        nets.append({"net": net, "tf": tf, **{k: n[k] for k in ("epochs", "best", "seconds")}})
    return {**{k: obj[k] for k in ("feats", "recipe", "stack", "book", "stretch", "meta")}, "trees": trees,
            "nets": nets}
