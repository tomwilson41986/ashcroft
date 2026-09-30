#!/usr/bin/env python3
"""The owner's staking on live Betfair prices: every runner's expected CLV at its best back price under the closing
model, and the backs (expected CLV at least the bar, at least MIN_VOL matched on the horse), staked to win GBP250.

    python scripts/score_live_prices.py --predictions now.csv --live live_odds.csv --out scored.csv
        [--bar 0.03] [--to-win 250] [--min-vol 100] [--draws 200000] [--model data/models/closing_model.json]

--live is betfair_sync.py --live's CSV (venue, market_start_time in UTC, runner_name, best back and lay with their
sizes, runner_matched and total_matched), or a transcribed one (race_time, track, horse, back, back_avail, lay,
lay_avail, race_matched: the owner's screenshots, where each runner's matched money is estimated as the race's total
shared by the prices). --predictions is predict_bfsp_today.py's CSV (after --non-runners), or the 06:00 record as
S3 keeps it (predictions/<day>.csv).

The closing model (data/models/closing_model.json, model/race_book.py) forecasts the BSP book from the Betfair price
now and ours, with traded volume; expected CLV = E[price / BSP] - 1 over its draws. 'min_back' is the least price that
still clears the bar with the rest of the market as it is. Read-only: nothing here places a bet.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import race_book as rb  # noqa: E402

NOVOL = ROOT / "data" / "models" / "closing_model_novol.json"


def load_model(path: str) -> rb.ClosingModel:
    return rb.load_closing_model(path)


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
          min_vol: float = 100.0, n_draws: int = 200000, seed: int = 930,
          novol: rb.ClosingModel | None = None, max_stake: float | None = None) -> pd.DataFrame:
    """A race with no matched money at all (the delayed key's feed) is read by ``novol``, the closing model fitted
    without volume, and is not held to ``min_vol``; without ``novol`` it is scored as the rest are."""
    pred = pred.assign(key=pred["horse_name"].map(norm_name))
    live = live.assign(key=live["horse"].map(norm_name))
    d = live.merge(pred[["race_time", "track", "key", "horse_name", "predicted_bfsp"]], on=["race_time", "track", "key"],
                   how="left", validate="many_to_one")
    d["race"] = d["track"] + " " + d["race_time"]
    # a race is scored only when every runner Betfair has open is one we priced
    whole = d.groupby("race")["predicted_bfsp"].transform(lambda s: s.notna().all())
    d = d[whole & (d["back"] > 1)].copy()
    d["morningwap"] = rb.market_now(d["back"], d["lay"])
    inv = 1.0 / d["morningwap"]
    est = d["race_matched"] * inv / inv.groupby(d["race"]).transform("sum")
    d["morning_vol"] = d["runner_matched"].where(d["runner_matched"] > 0, est).fillna(0.0)
    d["vol_estimated"] = ~(d["runner_matched"] > 0)
    rng = np.random.default_rng(seed)
    parts = []
    for _, r in d.groupby("race", sort=True):
        r = r.copy()
        blind = novol is not None and not (r["morning_vol"] > 0).any()
        r["ev"], r["exp_bsp"] = rb.race_expected_clv(r["back"], r["lay"], r["predicted_bfsp"], r["morning_vol"],
                                                     novol if blind else model, n_draws=n_draws, rng=rng)
        r["no_volume"] = blind
        parts.append(r)
    if not parts:
        return d.assign(exp_bsp=np.nan, ev=np.nan, no_volume=False, back_it=False, min_back=np.nan, stake=0.0)
    d = pd.concat(parts)
    d["back_it"] = (d["ev"] >= bar) & ((d["morning_vol"] >= min_vol) | d["no_volume"])
    d["min_back"] = np.where(d["back_it"], (1 + bar) * d["back"] / (1 + d["ev"]), np.nan)
    stake = to_win / (d["back"] - 1.0)
    d["stake"] = np.where(d["back_it"], stake if max_stake is None else np.minimum(stake, max_stake), 0.0)
    return d.drop(columns=["key"]).reset_index(drop=True)


def card_minutes(t) -> int:
    """A card time as minutes into the day, for ordering: '12:30' before '1:57' (1-9 are afternoon times)."""
    h, m = (int(x) for x in str(t).replace(".", ":").split(":")[:2])
    return (h + 12 if h < 10 else h) * 60 + m


def day_list(backs: pd.DataFrame, seen: pd.DataFrame, max_day: float) -> pd.DataFrame:
    """The day's backs against those already sent today (``seen``: race, horse_name, stake): each marked new or not,
    and the new ones kept, best expected CLV first, while the day's total stays within ``max_day``."""
    b = backs.copy()
    key = b["race"] + "|" + b["horse_name"]
    sent = set(seen["race"] + "|" + seen["horse_name"]) if len(seen) else set()
    b["new"] = ~key.isin(sent)
    room = max_day - float(seen["stake"].sum() if len(seen) else 0.0)
    fresh = b[b["new"]].sort_values("ev", ascending=False)
    b["over_limit"] = False
    b.loc[fresh.index[fresh["stake"].cumsum() > room + 1e-9], "over_limit"] = True
    return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(ROOT / "data" / "models" / "closing_model.json"))
    ap.add_argument("--model-novol", default=str(NOVOL),
                    help="the closing model for races with no matched money ('' to score them as the rest)")
    ap.add_argument("--bar", type=float, default=0.03)
    ap.add_argument("--to-win", type=float, default=250.0)
    ap.add_argument("--min-vol", type=float, default=100.0)
    ap.add_argument("--draws", type=int, default=200000, help="closing-model draws a race (fewer is noisier at the bar)")
    ap.add_argument("--max-stake", type=float, default=300.0, help="the owner's limit on one bet, GBP")
    ap.add_argument("--max-day", type=float, default=4000.0, help="the owner's limit on the day's stakes, GBP")
    ap.add_argument("--seen", default=None, help="a CSV of the backs already sent today (read, then brought up to date)")
    a = ap.parse_args(argv)
    pred = pd.read_csv(a.predictions, dtype={"race_time": str})
    # the 06:00 record as S3 keeps it names the course and horse as Betfair does
    pred = pred.rename(columns={k: v for k, v in {"venue": "track", "runner_name": "horse_name"}.items()
                                if k in pred.columns and v not in pred.columns})
    live = read_live(a.live)
    novol = load_model(a.model_novol) if a.model_novol else None
    d = score(pred, live, load_model(a.model), a.bar, a.to_win, a.min_vol, n_draws=a.draws, novol=novol,
              max_stake=a.max_stake)
    d.to_csv(a.out, index=False)
    if len(d) and d["no_volume"].any():
        print(f"{d.loc[d['no_volume'], 'race'].nunique()} races with no matched money (the delayed key's feed): "
              f"read by the closing model fitted without volume, not held to GBP{a.min_vol:.0f} matched")
    races = d["race"].nunique()
    seen_path = Path(a.seen) if a.seen else None
    seen = (pd.read_csv(seen_path, dtype={"race": str, "horse_name": str}) if seen_path and seen_path.exists()
            else pd.DataFrame(columns=["race", "horse_name", "stake"]))
    backs = day_list(d[d["back_it"]], seen, a.max_day)
    take = backs[backs["new"] & ~backs["over_limit"]]
    before = float(seen["stake"].sum()) if len(seen) else 0.0
    print(f"{races} races scored, {len(d)} runners; {len(backs)} backs (expected CLV >= {a.bar:.0%}, "
          f">= GBP{a.min_vol:.0f} matched), staked to win GBP{a.to_win:.0f}, at most GBP{a.max_stake:.0f} a bet")
    print(f"new: {len(take)} for GBP{take['stake'].sum():,.2f}; sent earlier today: {int((~backs['new']).sum())} "
          f"(GBP{before:,.2f}); the day's limit GBP{a.max_day:,.0f}"
          + (f"; {int(backs['over_limit'].sum())} over it, not sent" if backs["over_limit"].any() else ""))
    order = backs.assign(_t=backs["race_time"].map(card_minutes)).sort_values(["_t", "track", "ev"],
                                                                             ascending=[True, True, False])
    for r in order.itertuples():
        tag = "NEW " if r.new and not r.over_limit else ("over" if r.over_limit else "sent")
        print(f"  {tag} {r.race_time:>5} {r.track:14s} {r.horse_name:28s} back {r.back:7.2f} "
              f"(at or above {r.min_back:6.2f})  ours {r.predicted_bfsp:7.2f}  expected CLV {r.ev:+.1%}  "
              f"stake GBP{r.stake:,.2f}")
    if len(backs):
        print("trade out: once a back is matched, lay the same horse at the Betfair SP with liability "
              "stake x (price - 1) (GBP250 on a full to-win bet): the price's move is kept whatever the result")
    if seen_path is not None:
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([seen, take[["race", "horse_name", "stake"]]]).to_csv(seen_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
