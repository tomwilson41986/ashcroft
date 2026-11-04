"""The morning edges' selections and the steamer model's features (paper only; docs/paper_trader_morning_edges.md).

Everything here reads the HRB database (race_results + betfair_prices) and, for a day being traded, only:
  * that day's RACECARD facts: course, time, horse, code, handicap, class, field, age, OR, jockey, trainer, allowance,
    headgear, days since the last run (the 06:00 card carries all of them); and
  * EARLIER DAYS for everything else: the last run's finish, market rank, BSP and steam, the trainer's and jockey's
    record.
Nothing from the day's result, its BSP or its morning trading enters a selection. ``card_view`` is the guard: it
blanks every result and price column of the day before the features are built.

Frozen 4 Nov 2026 (research: reports/strategy_research_2026-11-04.md):
  qr       back within 1-4 days, flat / AW / hurdle / chase, 5+ runners
  steam    the CLEAN steamer model's prediction at or above its frozen threshold (training set's top 5%)
  tf_pb    Flat/AW, last run a career-best Timeform performance rating by 5lb+ (research/paper/morning_edges.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "strat"))
import strat_lib as L  # noqa: E402

K = 15.0
MAIN_CODES = ("flat", "aw", "hurdle", "chase")
#: the CLEAN feature set, exactly as research/queries/done/strat_i_steamers_1104.py fitted it
CLEAN = ["code_i", "hcap", "cls", "field", "horse_age", "official_rating", "or_rank", "jockeys_claim", "dslr",
         "prev_fin", "prev_rank", "prev_bsp", "prev_steam", "runs_held", "trainer_change", "jockey_change", "hg_on",
         "hg_off", "month", "hour", "tr_n", "tr_sr", "tr_ae", "tr_steam", "jk_n", "jk_sr", "jk_ae", "jk_steam"]
#: the day's columns that are not on a 06:00 card: blanked for the day being traded
RESULT_COLS = ["placing_numerical", "place", "fin", "won", "placed", "bfsp", "bfsp_place", "bf_bsp", "bf_pbsp", "bsp",
               "pbsp", "morningwap", "morning_vol", "p_morningwap", "p_morning_vol", "odds", "comment",
               "total_dst_bt", "distbt", "comptime", "comptime_numeric", "bsp_rank", "race_has_winner"]


def card_view(df: pd.DataFrame, day: str) -> pd.DataFrame:
    """The frame as it stood at 06:00 on ``day``: earlier days whole, the day itself with its results and prices
    blanked (a test holds every selection to this)."""
    out = df[df["race_date"] <= day].copy()
    today = out["race_date"] == day
    for c in RESULT_COLS:
        if c in out:
            out.loc[today, c] = False if out[c].dtype == bool else np.nan
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature and rule input, each from earlier days or the card. ``df`` is strat_lib.load() output (or its
    card_view). Returns one row per runner with the CLEAN features, ``steam`` (the target, on days with prices) and
    the qr inputs."""
    df = df[df["code"].isin(MAIN_CODES)].copy()
    df["am"] = df["morningwap"].where((df["morning_vol"] >= L.MIN_AM_VOL) & (df["morningwap"] > 1))
    df["steam"] = np.log(df["am"] / df["bsp"]).where((df["bsp"] > 1) & df["am"].notna())
    h = df.groupby("horse_name", sort=False)
    for k, src in (("prev_date", "date"), ("prev_fin", "fin"), ("prev_rank", "bsp_rank"), ("prev_bsp", "bsp"),
                   ("prev_steam", "steam"), ("prev_trainer", "trainer"), ("prev_jockey", "jockey_name"),
                   ("prev_hg", "headgear")):
        df[k] = h[src].shift(1)
    df["runs_held"] = h.cumcount()
    df["dslr"] = df["days_since_lr"].where(df["days_since_lr"] > 0, (df["date"] - df["prev_date"]).dt.days)
    df["p"] = np.where(df["bsp"] > 1, 1 / df["bsp"], 0.0)
    df["has"] = (df["bsp"] > 1).astype(float)
    df["pw"] = df["won"].fillna(0) * df["has"]
    df["s_has"] = df["steam"].notna().astype(float)
    df["s_sum"] = df["steam"].fillna(0)
    for key, nm in (("trainer", "tr"), ("jockey_name", "jk")):
        pr = L.prior_cumsum(df, key, ["has", "pw", "p", "s_has", "s_sum"])
        df[f"{nm}_n"] = pr["has"]
        df[f"{nm}_sr"] = (pr["pw"] + 2) / (pr["has"] + 20)
        df[f"{nm}_ae"] = (pr["pw"] + K) / (pr["p"] + K)
        df[f"{nm}_steam"] = pr["s_sum"] / (pr["s_has"] + 30)
        df[f"{nm}_steam_n"] = pr["s_has"]
    df["field"] = df["number_of_runners"].where(df["number_of_runners"] > 0, df["field"])
    df["or_rank"] = df.groupby("rk")["official_rating"].rank(method="min", ascending=False)
    df["cls"] = pd.to_numeric(df["race_class"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
    hr = pd.to_numeric(df["race_time"].astype(str).str.extract(r"^(\d{1,2})")[0], errors="coerce")
    df["hour"] = np.where(hr < 11, hr + 12, hr)
    df["month"] = df["date"].dt.month
    df["code_i"] = df["code"].map({"flat": 0, "aw": 1, "hurdle": 2, "chase": 3})
    df["trainer_change"] = (df["prev_trainer"].notna() & (df["trainer"] != df["prev_trainer"])).astype(int)
    df["jockey_change"] = (df["prev_jockey"].notna() & (df["jockey_name"] != df["prev_jockey"])).astype(int)
    hg, phg = df["headgear"].fillna("").astype(str).str.strip(), df["prev_hg"].fillna("").astype(str).str.strip()
    df["hg_on"] = ((hg != "") & (phg == "")).astype(int)
    df["hg_off"] = ((hg == "") & (phg != "")).astype(int)
    df["hcap"] = df["hcap"].astype(int)
    return df


def rule_qr(x: pd.DataFrame) -> pd.Series:
    return x["dslr"].between(1, 4) & x["code"].isin(MAIN_CODES) & (x["field"] >= 5)


def train_steamer(df: pd.DataFrame, cutoff: str, start: str = "2018-01-01"):
    """The CLEAN steamer model on [start, cutoff]: LightGBM as the research fitted it. Returns (booster, threshold,
    rows used). The threshold is the 95th percentile of the fitted model's predictions on its own training rows."""
    import lightgbm as lgb
    x = build(df)
    x = L.priced(x)
    x = x[(x["race_date"] >= start) & (x["race_date"] <= cutoff) & x["steam"].notna()]
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=400, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    m.fit(x[CLEAN], x["steam"])
    thr = float(np.quantile(m.predict(x[CLEAN]), 0.95))
    return m.booster_, thr, len(x)


def day_selections(df: pd.DataFrame, day: str, booster=None, threshold: float | None = None) -> pd.DataFrame:
    """The day's qr and steam selections from the card view. One row per (strategy, runner)."""
    cv = card_view(df, day)
    x = build(cv)
    x = x[x["race_date"] == day].copy()
    if x.empty:
        return pd.DataFrame()
    out = []
    q = x[rule_qr(x)]
    out.append(q.assign(strategy="qr", pred=np.nan))
    if booster is not None and threshold is not None:
        x["pred"] = booster.predict(x[CLEAN])
        out.append(x[(x["pred"] >= threshold) & (x["field"] >= 5)].assign(strategy="steam"))
    keep = ["strategy", "race_date", "race_time", "track", "horse_name", "code", "field", "dslr", "pred", "trainer",
            "jockey_name"]
    return pd.concat(out, ignore_index=True)[keep]
