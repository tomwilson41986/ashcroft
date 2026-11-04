"""The production model bundle (paper trading's model): fitted on every delivery to date, versioned in the store.

    python -m cricket.model.production --root data/cricket/store --work data/cricket/eval

Writes ``cricket/models/<version>/`` and points ``cricket/models/latest.json`` at it:

- ``next_ball.ubj``: the next-ball model, every in-scope men's T20 delivery from 2012 to the latest Cricsheet day,
  the rounds the walk-forward evaluation chose;
- ``bat_ratings.parquet``, ``bowl_ratings.parquet``: each player's rating after their latest match;
- ``venues.parquet``, ``comps.parquet``: venue (three years) and competition (365 days) rates as of the day after
  the data ends;
- ``bundle.json``: the prior, the extras runs distribution, the feature list, the isotonic calibrators (chase, first
  innings) fitted on the walk-forward evaluation's out-of-sample simulator output, and where every number came from;
- ``model_card.md``: what it is, what it was scored at, what it leaves out.

The calibrators were fitted on the per-season models' output, not on this model's: stated in the card.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cricket.model import next_ball as nb
from cricket.model import calibration as calib
from cricket.model.calibration import METHODS, SELECT_ON
from cricket.model.data import load
from cricket.model.evaluate import choose_rounds
from cricket.model.features import FEATURES, build, t20_matches
from cricket.model.metrics import score
from cricket.model.outcomes import CLASSES, extras_runs_distribution
from cricket.ratings.player_ratings import C_PRIOR, DECAY, ratings_now
from cricket.ratings.venue_ratings import rates_now
from sources.common import Store

PREFIX = "cricket/models"


CHASE_BASE = ["rrr", "wkts_in_hand", "balls_left"]
FIRST_BASE = ["runs_before", "wickets_before", "legal_before"]


def starts_with_total(t: dict, f: pd.DataFrame) -> pd.DataFrame:
    from cricket.model.evaluate_first import over_starts
    totals = t["innings"][t["innings"].innings == 1].set_index("match_id")["runs"]
    s = over_starts(f)
    return s.assign(total=s.match_id.map(totals).to_numpy())


def fit_baselines(f: pd.DataFrame, starts: pd.DataFrame) -> dict:
    """The state-only logistics the stack layer combines with the simulator, fitted on every IPL season held: the
    chase on RRR, wickets in hand and balls left; the first innings on runs, wickets and legal balls at over starts."""
    from sklearn.linear_model import LogisticRegression
    from cricket.model.data import IPL
    ch = f[(f.event_name == IPL) & (f.innings == 2) & (f.balls_left > 0)]
    ch = ch.assign(wkts_in_hand=10 - ch.wickets_before)
    out = {}
    for name, rows, cols in (("chase", ch, CHASE_BASE), ("first_innings", starts, FIRST_BASE)):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(rows[cols], rows.bat_won)
        out[name] = {"features": cols, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                     "rows": int(len(rows))}
    return out


def baseline_p(base: dict, values: dict) -> float:
    z = base["intercept"] + sum(c * float(values[k]) for k, c in zip(base["features"], base["coef"]))
    return 1.0 / (1.0 + np.exp(-z))


def fit_bundle(store: Store, work: Path, log=print) -> dict:
    t = load(store)
    f, _, _, prior = build(t)
    rounds = choose_rounds(f, work, log)
    m = t20_matches(t["matches"])
    data_end = str(f.start_date.max())
    log(f"fit on {len(f):,} deliveries, {f.match_id.nunique():,} matches to {data_end}, {rounds} rounds")
    booster = nb.fit(f, rounds)
    ex_v, ex_p = extras_runs_distribution(f)
    # ratings from every in-scope match (also before 2012), the same frame the as-of ratings came from
    from cricket.model.data import ball_frame
    from cricket.model.outcomes import classify
    full = ball_frame(t, m)
    full["cls"] = classify(full)
    bat = ratings_now(full, t["players"], m, "bat", prior)
    bowl = ratings_now(full, t["players"], m, "bowl", prior)
    asof = (pd.Timestamp(data_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ctx = rates_now(full, m, asof, pop_before="2023-01-01")
    baselines = fit_baselines(f, starts_with_total(t, f))
    cal = {}
    sources = {}
    for name, fname in (("chase", "model_v1_preds.parquet"), ("first_innings", "model_v1_first_innings_preds.parquet")):
        preds = store.get_parquet(f"cricket/eval/{fname}")
        if preds is None:
            cal[name] = None
            sources[name] = f"no data found: {store.where('cricket/eval/' + fname)}"
            continue
        # the layer the evaluation's fixed rule chose, refitted on every evaluated season's out-of-sample output
        sel = preds[preds.year == SELECT_ON]
        method = min(METHODS, key=lambda m: score(sel.bat_won, sel[f"p_{m}"])["log_loss"]) if len(sel) else "isotonic"
        cal[name] = calib.fit(method, preds.p_sim.to_numpy(), preds.bat_won.to_numpy(), preds.p_base.to_numpy())
        sources[name] = (f"{method}, fitted on {store.where('cricket/eval/' + fname)}: {len(preds):,} rows, seasons "
                         f"{int(preds.year.min())}-{int(preds.year.max())}")
    version = f"v1-{data_end}"
    bundle = {
        "version": version, "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_end": data_end, "ratings_asof": asof, "train_rows": int(len(f)),
        "train_matches": int(f.match_id.nunique()),
        "rounds": rounds, "params": nb.PARAMS, "features": FEATURES, "classes": CLASSES,
        "prior": prior.tolist(), "extras_runs": {"values": ex_v.tolist(), "p": ex_p.tolist()},
        "rating_recipe": {"c": C_PRIOR, "r": DECAY}, "population": ctx["population"],
        "calibration": cal, "calibration_source": sources, "baselines": baselines,
        "scope": "men's 20-over six-ball T20 (match types T20, IT20), decided, no D/L, from 2012",
    }
    return {"bundle": bundle, "booster": booster, "bat": bat, "bowl": bowl, "venues": ctx["venues"],
            "comps": ctx["comps"]}


def model_card(b: dict, store: Store) -> str:
    L = [f"# Cricket model {b['version']} - model card", "",
         f"Built {b['built_utc']} by `python -m cricket.model.production`. Paper trading only: no live key.", "",
         f"- Scope: {b['scope']}.",
         f"- Training: {b['train_rows']:,} deliveries from {b['train_matches']:,} matches, to {b['data_end']} "
         "(Cricsheet).",
         f"- Next-ball model: XGBoost multinomial over {', '.join(b['classes'])}; {b['rounds']} rounds.",
         f"- Ratings: Dirichlet c = {b['rating_recipe']['c']}, decay r = {b['rating_recipe']['r']} a match, as of "
         f"{b['ratings_asof']}.",
         "", "## Calibration", ""]
    for k, v in b["calibration_source"].items():
        L.append(f"- {k}: {v}")
    L += ["", "The calibrators were fitted on the walk-forward per-season models' out-of-sample output, not on this "
          "model's own: a model trained on more data may need less (or other) correction.", "",
          "## Held-out scores (walk-forward evaluation)", ""]
    for name, fname in (("chase", "model_v1_preds.parquet"), ("first innings", "model_v1_first_innings_preds.parquet")):
        p = store.get_parquet(f"cricket/eval/{fname}")
        if p is None:
            L.append(f"- {name}: no data found")
            continue
        t = p[p.p_cal.notna()]
        s, sb = score(t.bat_won, t.p_cal), score(t.bat_won, t.p_base)
        L.append(f"- {name}: {len(t):,} rows, seasons {int(t.year.min())}-{int(t.year.max())}: calibrated log loss "
                 f"{s['log_loss']:.4f}, ECE {s['ece']:.4f}; baseline log loss {sb['log_loss']:.4f}, ECE "
                 f"{sb['ece']:.4f} (`{store.where('cricket/eval/' + fname)}`)")
    L += ["", "## Leaves out", "",
          "- Batting order beyond the crease is the listed XI order; future bowlers are the XI average.",
          "- No free hits, no runs on wicket balls, no wickets on wides or no-balls.",
          "- No bowler type, matchups, weather or toss effects.",
          "- Not checked against Betfair prices: the Benter combination and the strategy test wait on price history.",
          ""]
    return "\n".join(L)


def save(store: Store, fitted: dict) -> str:
    b = fitted["bundle"]
    base = f"{PREFIX}/{b['version']}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "next_ball.ubj"
        fitted["booster"].save_model(path)
        store.put(f"{base}/next_ball.ubj", path.read_bytes())
    for name in ("bat", "bowl", "venues", "comps"):
        store.put_parquet(f"{base}/{name if name in ('venues', 'comps') else name + '_ratings'}.parquet", fitted[name])
    store.put(f"{base}/bundle.json", json.dumps(b, indent=1).encode())
    store.put(f"{base}/model_card.md", model_card(b, store).encode())
    store.put(f"{PREFIX}/latest.json", json.dumps({"version": b["version"], "built_utc": b["built_utc"]}).encode())
    return base


def load_bundle(store: Store, version: str = "latest") -> dict:
    """The bundle the live engine reads: the booster, the ratings and context tables, bundle.json."""
    import xgboost as xgb
    if version == "latest":
        raw = store.get(f"{PREFIX}/latest.json")
        if raw is None:
            raise SystemExit(f"no data found: {store.where(PREFIX + '/latest.json')} (run cricket.model.production)")
        version = json.loads(raw)["version"]
    base = f"{PREFIX}/{version}"
    b = json.loads(store.get(f"{base}/bundle.json"))
    booster = xgb.Booster()
    booster.load_model(bytearray(store.get(f"{base}/next_ball.ubj")))
    out = {"bundle": b, "booster": booster}
    for name in ("bat_ratings", "bowl_ratings", "venues", "comps"):
        out[name] = pd.read_parquet(io.BytesIO(store.get(f"{base}/{name}.parquet")))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--work", default="data/cricket/eval")
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    fitted = fit_bundle(store, Path(a.work), log=lambda m: print(m, flush=True))
    base = save(store, fitted)
    print(json.dumps({"saved": store.where(base), "version": fitted["bundle"]["version"]}))
    print(model_card(fitted["bundle"], store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
