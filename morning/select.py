"""The day's selections for the morning models from the 06:00 card (s3://$BUCKET/racecards/<day>_<hhmm>.csv) and
the HRB database's earlier days. The research and the paper trader read the day's runners from the results table;
live, the day has no results yet, so the card's rows are put into the history in the table's shape and the same
features are built (research/paper/morning_features.py). ``validate`` holds the two to each other on past days.

The card differs from the table in three ways (the 06:00 file of 4 Nov 2026: 22 columns):
  * no race_type and no race_code: the code comes from the race name (Chase, Hurdle, NH Flat / Bumper) and, for a
    Flat race, the going ("Standard ..." is the all-weather's going);
  * no jockeys_claim: the claim is in the jockey's name, "Young, Lauren(5)";
  * the race time may be 24-hour where the table's is 12-hour.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from research.paper import morning_features as MF

L = MF.L
_CLAIM = re.compile(r"\s*\((\d+)\)\s*$")


def split_claim(name) -> tuple[str, float]:
    """'Young, Lauren(5)' -> ('Young, Lauren', 5.0); no claim -> (name, 0.0)."""
    s = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name)
    m = _CLAIM.search(s)
    return (s[:m.start()].strip(), float(m.group(1))) if m else (s.strip(), 0.0)


def card_code(race_name: pd.Series, going: pd.Series) -> pd.Series:
    """chase / hurdle / nhflat from the race name, aw for a Flat race on Standard going, else flat."""
    n = race_name.fillna("").str.lower()
    g = going.fillna("").str.lower()
    return pd.Series(np.select([n.str.contains("chase"), n.str.contains("hurdle"),
                                n.str.contains(r"nh flat|n\.h\. flat|bumper|national hunt flat"),
                                g.str.contains("standard")],
                               ["chase", "hurdle", "nhflat", "aw"], default="flat"), index=race_name.index)


def time12(t) -> str:
    """'14:05' -> '2:05'; '2:05', '12:05' unchanged (the table's 12-hour clock)."""
    s = str(t).strip().replace(".", ":")
    h, _, m = s.partition(":")
    if not (h.isdigit() and m.isdigit()):
        return s
    h = int(h)
    return f"{h - 12 if h > 12 else h}:{int(m):02d}"


def card_rows(card: pd.DataFrame, day: str, like: pd.DataFrame) -> pd.DataFrame:
    """The card as rows of strat_lib.load(): every result and price column empty, the card's facts filled."""
    c = card.copy()
    out = pd.DataFrame(index=range(len(c)), columns=like.columns)
    for k in ("race_time", "track", "race_name", "race_class", "horse_name", "trainer", "headgear",
              "going_description", "stall"):
        if k in c and k in out:
            out[k] = c[k].to_numpy()
    out["race_date"] = day
    out["race_time"] = c["race_time"].map(time12).to_numpy()
    j = c["jockey_name"].map(split_claim)
    out["jockey_name"] = [a for a, _ in j]
    out["jockeys_claim"] = [b for _, b in j]
    for k in ("horse_age", "official_rating", "days_since_lr", "number_of_runners", "pounds"):
        if k in c and k in out:
            out[k] = pd.to_numeric(c[k], errors="coerce").to_numpy()
    out["code"] = card_code(c["race_name"], c.get("going_description", pd.Series("", index=c.index))).to_numpy()
    out["date"] = pd.Timestamp(day)
    out["rk"] = out["race_date"].astype(str) + "|" + out["track"].astype(str) + "|" + out["race_time"].astype(str)
    out["hcap"] = out["race_name"].fillna("").str.contains("handicap", case=False)
    out["year"] = out["date"].dt.year
    for k in MF.RESULT_COLS:
        if k in out:
            out[k] = np.nan
    out["won"] = np.nan
    out["placed"] = np.nan
    out["race_has_winner"] = False
    out["field"] = out.groupby("rk")["horse_name"].transform("size")
    for k in like.columns:                                  # the table's dtypes where the card leaves a column empty
        if like[k].dtype == bool and out[k].isna().all():
            out[k] = False
    return out


def history_with_card(hist: pd.DataFrame, card: pd.DataFrame, day: str) -> pd.DataFrame:
    """Earlier days of the history and the day's card, in the table's order."""
    h = hist[hist["race_date"] < day]
    rows = card_rows(card, day, hist)
    for k in hist.columns:
        if hist[k].dtype != object and k in rows:
            rows[k] = pd.to_numeric(rows[k], errors="coerce") if hist[k].dtype != bool else rows[k].astype(bool)
    out = pd.concat([h, rows], ignore_index=True)
    return out.sort_values(["date", "race_time", "track"], kind="stable").reset_index(drop=True)


def select(frame: pd.DataFrame, day: str, booster, threshold: float) -> pd.DataFrame:
    """One row per (model, runner): 'quick_return' (back within 1-4 days, flat/AW/hurdle/chase, 5+ runners) and
    'steamer' (the frozen CLEAN model's prediction at or above its threshold, 5+ runners)."""
    x = MF.build(frame)
    x = x[x["race_date"] == day].copy()
    if x.empty:
        return pd.DataFrame(columns=["model", "race_date", "race_time", "track", "horse_name", "pred"])
    x["pred"] = booster.predict(x[MF.CLEAN]) if booster is not None else np.nan
    q = x[MF.rule_qr(x)].assign(model="quick_return")
    s = x[(x["pred"] >= threshold) & (x["field"] >= 5)].assign(model="steamer") if booster is not None else x.iloc[:0]
    keep = ["model", "race_date", "race_time", "track", "horse_name", "code", "field", "dslr", "pred",
            "jockey_name", "trainer"]
    return pd.concat([q, s], ignore_index=True)[keep]


def read_card(get, day: str) -> pd.DataFrame | None:
    """The latest 06:00 card for the day from S3 (``get(key) -> bytes``, ``ls(prefix)`` on the getter if it has one)."""
    keys = sorted(get.ls(f"racecards/{day}_")) if hasattr(get, "ls") else []
    if not keys:
        return None
    return pd.read_csv(io.BytesIO(get(keys[-1])), dtype={"race_time": str})
