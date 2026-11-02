"""The card-safe full greyhound model, scored out of sample on the trade-out study's days (the owner, 2 Nov 2026:
"score the card-safe full model and rerun the study").

The trade-out study's first reading (reports/greyhound_model.md section 13) priced each dog with the forward test's
model of 5 Oct: the base 135 features, none of the horse model's families (pace and race shape, the normalised finish
position, the form windows, the market's view; ``greyhound/parity.py``) nor the feature library (Elo, the Kalman
rating, form lines, first-bend crossings; ``greyhound/lib.py``). The best research model (section 11: base, parity,
HorseRaceBase and the library, 342 features, fitted on the race's conditional logit) reads the dog's weight on the
day, which the morning card does not know. This is that model without those features (``greyhound.live.card_safe``):

1. Every GBGB run from 2018 to ``--to`` (``greyhound.track.runs_through``: the year tables and the raw days since),
   the metrics with the parity, HRB and library blocks and the Betfair SP history (the price-file tables).
2. One walk-forward fold: fitted on every modelling race from 2019 to the day before ``--from`` (6 Oct 2026, the first
   recorded greyhound books), early-stopped on its last tenth, the race's softmax objective; every race from ``--from``
   on predicted by it, so every price is out of sample.
3. Each day's predictions joined to the recorded Betfair greyhound markets (``greyhound.track.join_markets``) and
   written a day a file (the forward test's ``pred`` layout) to ``--pred-dir``, for
   ``research.greyhound.tradeout_study --pred-dir``.
4. The head to head on the same runners: log-loss against the result of the 5 Oct base model, this model and the
   Betfair SP (normalised within the race), on the races all three price whole.

Read-only on the store: everything goes to local files (the job's artifact).

    python -m research.greyhound.score_full --from 2026-10-06 --to 2026-11-01 --pred-dir out/pred_full \\
        --out out/greyhound_full_score.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound import tradeout as to

log = logging.getLogger(__name__)

VARIANT = "all+lib-card:softmax"
BLOCKS = ("parity", "hrb", "lib")
FIT_FROM = "2019-01-01"
PRICE_YEARS = "2018-2026"


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


def build(store, last: date):
    """The metrics on every GBGB run to ``last``, slimmed to the card-safe features."""
    from greyhound.live import card_safe
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices, slim
    from greyhound.track import runs_through
    t0 = time.monotonic()
    runs = runs_through(store, last)
    _log(f"runs: {len(runs):,} to {runs.race_date.max()} ({time.monotonic() - t0:.0f} s)")
    eng = GreyhoundMetricsEngine(blocks=BLOCKS)
    df = eng.calculate_all(runs, prices=load_prices(store, PRICE_YEARS))
    del runs
    gc.collect()
    feats = card_safe(eng.features)
    _log(f"features: {len(eng.features)} built, {len(feats)} card-safe; {eng.timings} ({time.monotonic() - t0:.0f} s)")
    return slim(df, eng.features), feats, eng


def score_days(pred: pd.DataFrame, first: date, last: date, store, pred_dir: Path) -> dict:
    """Each day's predictions joined to its Betfair markets and written; the head to head against the 5 Oct model and
    the SP."""
    from betfair_recorder import _read_day_csv
    from greyhound.model import logloss
    from greyhound.track import join_markets
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows, days, d = [], [], first
    while d <= last:
        ds = f"{d:%Y-%m-%d}"
        p = pred[pred.race_date.astype(str) == ds]
        markets = _read_day_csv(d, "markets_greyhound.csv")
        books = _read_day_csv(d, "books_greyhound.csv")
        if not len(p) or markets is None:
            days.append({"day": ds, "races": int(p.race_id.nunique()), "note": "no predictions or no markets"})
            d += timedelta(days=1)
            continue
        j = join_markets(p, markets)
        keep = ["race_id", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "field", "won",
                "sp_decimal", "p_model", "market_id", "selection_id"]
        j = j[[c for c in keep if c in j.columns]]
        j.to_parquet(pred_dir / f"{ds}.parquet", index=False)
        days.append({"day": ds, "races": int(p.race_id.nunique()), "races_on_betfair": int(j.race_id.nunique())})
        base = store.get_parquet(f"{to.TRACK_PRED}/{ds}.parquet")
        if base is not None and len(base) and books is not None:
            b = base.assign(market_id=base.market_id.astype(str),
                            selection_id=pd.to_numeric(base.selection_id, errors="coerce"))
            m = j.assign(market_id=j.market_id.astype(str),
                         selection_id=pd.to_numeric(j.selection_id, errors="coerce")).merge(
                b[["market_id", "selection_id", "p_model"]].rename(columns={"p_model": "p_base"}),
                on=["market_id", "selection_id"], how="inner")
            m = m.merge(to.closing_prices(books, markets, d), on=["market_id", "selection_id"], how="inner")
            whole = m.groupby("market_id").selection_id.transform("size") == m.field
            rows.append(m[whole & (m.bsp > 1)].assign(day=ds))
        d += timedelta(days=1)
    out = {"days": days}
    if rows:
        h = pd.concat(rows, ignore_index=True)
        q = (1.0 / h.bsp).groupby(h.market_id).transform(lambda s: s / s.sum())
        out["head_to_head"] = {
            "runners": int(len(h)), "races": int(h.market_id.nunique()), "days": int(h.day.nunique()),
            "logloss_base_5oct": round(logloss(h.p_base, h.won), 5),
            "logloss_full_card_safe": round(logloss(h.p_model, h.won), 5),
            "logloss_bsp": round(logloss(q, h.won), 5),
            "by_day": {k: {"base": round(logloss(g.p_base, g.won), 5), "full": round(logloss(g.p_model, g.won), 5),
                           "bsp": round(logloss(q[g.index], g.won), 5), "races": int(g.market_id.nunique())}
                       for k, g in h.groupby("day")}}
    return out


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from greyhound.model import score, walk_forward
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--from", dest="first", default=str(to.FIRST_BOOK_DAY), help="the first day predicted")
    ap.add_argument("--to", dest="last", default=None, help="the last day (default yesterday, UK)")
    ap.add_argument("--pred-dir", default="out/pred_full")
    ap.add_argument("--out", default="out/greyhound_full_score.json")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    store = Store(root=a.store)
    first = date.fromisoformat(a.first)
    last = date.fromisoformat(a.last) if a.last else uk_today() - timedelta(days=1)
    df, feats, eng = build(store, last)
    _, _, objective = VARIANT.partition(":")
    pred = walk_forward(df, feats, [f"{first:%Y-%m-%d}"], FIT_FROM, objective=objective)
    importance = pred.attrs.pop("importance")
    del df
    gc.collect()
    pred = pred[pred.race_date.astype(str) <= f"{last:%Y-%m-%d}"]
    _log(f"predicted {len(pred):,} runners in {pred.race_id.nunique():,} races, {pred.race_date.min()} to "
         f"{pred.race_date.max()} ({pred.rounds.iloc[0] if len(pred) else 0} rounds)")
    res = {"variant": VARIANT, "blocks": list(BLOCKS), "features": len(feats), "fit_from": FIT_FROM,
           "fit_before": f"{first}", "predicted": [f"{first}", f"{last}"], "timings": eng.timings,
           "score_vs_sp": score(pred), "top_features": {k: round(float(v), 4) for k, v in importance.head(30).items()}}
    res.update(score_days(pred, first, last, store, Path(a.pred_dir)))
    res["seconds"] = round(time.monotonic() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "days"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
