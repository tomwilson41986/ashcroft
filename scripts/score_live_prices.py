#!/usr/bin/env python3
"""The owner's staking on live Betfair prices: every runner's expected CLV at its best back price under the closing
model, and the backs (expected CLV at least the bar, at least MIN_VOL matched on the horse), staked to win GBP250.

    python scripts/score_live_prices.py --predictions now.csv --live live_odds.csv --out scored.csv
        [--bar 0.03] [--to-win 250] [--min-vol 100] [--draws 200000] [--model data/models/closing_model.json]

--live is betfair_sync.py --live's CSV (venue, market_start_time in UTC, runner_name, best back and lay with their
sizes, runner_matched and total_matched), or a transcribed one (race_time, track, horse, back, back_avail, lay,
lay_avail, race_matched: the owner's screenshots, where each runner's matched money is estimated as the race's total
shared by the prices). --predictions is predict_bfsp_today.py's CSV, after --non-runners.

The closing model (data/models/closing_model.json, model/race_book.py) forecasts the BSP book from the Betfair price
now and ours, with traded volume; expected CLV = E[price / BSP] - 1 over its draws. 'min_back' is the least price that
still clears the bar with the rest of the market as it is. Read-only: nothing here places a bet.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import race_book as rb  # noqa: E402

TIGHT = 1.25          # lay / back at most this: the mid is the market's price, else the back price


def load_model(path: str) -> rb.ClosingModel:
    j = json.loads(Path(path).read_text())
    return rb.ClosingModel(beta=np.asarray(j["beta"], float), vol_mean=j["vol_mean"], vol_sd=j["vol_sd"],
                           depth_mean=j["depth_mean"], depth_sd=j["depth_sd"],
                           band_edges=np.asarray(j["band_edges"], float), band_sigma=np.asarray(j["band_sigma"], float),
                           book=j["book"], n_train=j["n_train"])


def norm_name(name) -> str:
    """'Sod Hall Lane (IRE)' and 'Sod Hall Lane' alike: the name without its country, letters and digits only,
    lower case."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\s*\([A-Za-z]{2,3}\)\s*$", "", str(name)).lower())


def card_time(utc_iso: str) -> str:
    """A Betfair start time (UTC) as the card writes it: UK time, 12-hour, no leading zero ('1:38')."""
    t = pd.Timestamp(utc_iso)
    t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("Europe/London")
    return f"{t.hour % 12 or 12}:{t.minute:02d}"


def read_live(path: str) -> pd.DataFrame:
    """Either export as one frame: race_time, track, horse, back, back_avail, lay, lay_avail, runner_matched."""
    d = pd.read_csv(path, dtype={"race_time": str})
    if "market_start_time" in d.columns:                       # betfair_sync.py --live
        if "runner_status" in d.columns:
            d = d[d["runner_status"].fillna("ACTIVE").eq("ACTIVE")]
        if "market_status" in d.columns:
            d = d[d["market_status"].fillna("OPEN").eq("OPEN")]
        out = pd.DataFrame({"race_time": d["market_start_time"].map(card_time), "track": d["venue"],
                            "horse": d["runner_name"], "back": d["best_back_price"], "back_avail": d["best_back_size"],
                            "lay": d["best_lay_price"], "lay_avail": d["best_lay_size"],
                            "runner_matched": d.get("runner_matched"), "race_matched": d.get("total_matched")})
    else:                                                        # transcribed from the screens
        out = d.rename(columns={}).copy()
        out["runner_matched"] = np.nan
    return out.reset_index(drop=True)


def score(pred: pd.DataFrame, live: pd.DataFrame, model: rb.ClosingModel, bar: float = 0.03, to_win: float = 250.0,
          min_vol: float = 100.0, n_draws: int = 200000, seed: int = 930) -> pd.DataFrame:
    pred = pred.assign(key=pred["horse_name"].map(norm_name))
    live = live.assign(key=live["horse"].map(norm_name))
    d = live.merge(pred[["race_time", "track", "key", "horse_name", "predicted_bfsp"]], on=["race_time", "track", "key"],
                   how="left", validate="many_to_one")
    d["race"] = d["track"] + " " + d["race_time"]
    # a race is scored only when every runner Betfair has open is one we priced
    whole = d.groupby("race")["predicted_bfsp"].transform(lambda s: s.notna().all())
    d = d[whole & (d["back"] > 1)].copy()
    tight = d["lay"] / d["back"] <= TIGHT
    d["morningwap"] = np.where(tight, np.sqrt(d["back"] * d["lay"].where(tight, d["back"])), d["back"])
    inv = 1.0 / d["morningwap"]
    est = d["race_matched"] * inv / inv.groupby(d["race"]).transform("sum")
    d["morning_vol"] = d["runner_matched"].where(d["runner_matched"] > 0, est).fillna(0.0)
    d["vol_estimated"] = ~(d["runner_matched"] > 0)
    rng = np.random.default_rng(seed)
    parts = []
    for _, r in d.groupby("race", sort=True):
        r = r.copy()
        inputs = rb.closing_inputs(r)
        g = np.zeros(len(r), int)
        draws = rb.closing_draws(model.predict(inputs, g), model.sigma(inputs), book=model.book, n_draws=n_draws,
                                 rng=rng)
        r["exp_bsp"] = model.book / draws.mean(axis=0)
        r["ev"] = r["back"].to_numpy(float) * draws.mean(axis=0) - 1.0
        parts.append(r)
    if not parts:
        return d.assign(exp_bsp=np.nan, ev=np.nan, back_it=False, min_back=np.nan, stake=0.0)
    d = pd.concat(parts)
    d["back_it"] = (d["ev"] >= bar) & (d["morning_vol"] >= min_vol)
    d["min_back"] = np.where(d["back_it"], (1 + bar) * d["back"] / (1 + d["ev"]), np.nan)
    d["stake"] = np.where(d["back_it"], to_win / (d["back"] - 1.0), 0.0)
    return d.drop(columns=["key"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "data" / "models" / "closing_model.json"))
    ap.add_argument("--bar", type=float, default=0.03)
    ap.add_argument("--to-win", type=float, default=250.0)
    ap.add_argument("--min-vol", type=float, default=100.0)
    ap.add_argument("--draws", type=int, default=200000, help="closing-model draws a race (fewer is noisier at the bar)")
    a = ap.parse_args(argv)
    pred = pd.read_csv(a.predictions, dtype={"race_time": str})
    live = read_live(a.live)
    d = score(pred, live, load_model(a.model), a.bar, a.to_win, a.min_vol, n_draws=a.draws)
    d.to_csv(a.out, index=False)
    if len(d) and not (d["morning_vol"] > 0).any():
        print("no matched money in the snapshot (the delayed key returns none): the GBP"
              f"{a.min_vol:.0f}-matched rule cannot pass and the closing model's volume inputs are empty")
    races = d["race"].nunique()
    backs = d[d["back_it"]]
    print(f"{races} races scored, {len(d)} runners; {len(backs)} backs (expected CLV >= {a.bar:.0%}, "
          f">= GBP{a.min_vol:.0f} matched), staked to win GBP{a.to_win:.0f}: GBP{backs['stake'].sum():,.2f} in all")
    for r in backs.sort_values(["race_time", "track"]).itertuples():
        print(f"  {r.race_time:>5} {r.track:14s} {r.horse_name:28s} back {r.back:7.2f} (at or above {r.min_back:6.2f})  "
              f"ours {r.predicted_bfsp:7.2f}  expected CLV {r.ev:+.1%}  stake GBP{r.stake:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
