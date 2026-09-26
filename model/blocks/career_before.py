"""The horse's whole career: its runs before 2021, which the matrix does not hold, added to its runs since.

Every feature is built from 2021-01-01, in training and at 06:00 alike (the matrix's
start; the live path loads history from the model's own training start), so a horse's
runs before then do not exist for the model: a nine-year-old chaser whose Flat career
ended in 2020 reaches it with its jumps runs only. On the development window 8% of
runners have such runs (8.2 on average) and the 958 prices them +0.052 worse within age
and race code (research/queries/done/career_truncation.py); a history from 2018 does
not fit a runner's memory (iteration 82). The runs before 2021 are read from a table
(TABLE: research/queries/done/pre_window_careers.py, every run before 2021 measured by
model.form_windows.run_measures, as the engine measures the runs since) and added to
the horse's runs since 2021 on earlier days:

    cb_pw_runs            runs before 2021 (0 for a horse with none)
    cb_pw_share           their share of all its runs before today
    cb_years_since_first  years since its first run, before 2021 or since
    cb_car_win            won, over the whole career (the engine's fw_<m>_car
    cb_car_plc            placed           averages the runs since 2021 only)
    cb_car_nfp            normalised finishing position, 1 the winner .. 0 last
    cb_car_mkt            the market's view
    cb_car_ae             won less the market's chance
    cb_nfp_m3, cb_mkt_m3  the last three runs, the runs before 2021 filling the
                          window where fewer than three are held since
    cb_jumps_share        the share of its runs over hurdles, fences or in bumpers
    cb_or_peak            the highest official rating it has carried in today's
                          sphere: the Flat (turf and all-weather) or jumps
    cb_or_vs_peak         today's rating less that peak
    cb_flat_or_peak       the highest rating it has carried on the Flat (an
                          ex-Flat jumper's old mark)

A name is the same horse only if the record could be its own: a row whose horse was
born (its year less its age) after the year before the record's first run is a
namesake's, and reads as a horse with no runs before 2021. Rows before 2021 get
nothing (the table is their future), and only runs from 2021 count as runs since, so
a history loaded from earlier gives the same values. Days before today only: a 06:00
card and the same rows with their results in agree to the bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model.form_windows import run_measures
from model.freshness_features import _Days, _col
from model.race_shape import _codes, day_index, race_code

CUTOFF = "2021-01-01"
TABLE = Path(__file__).resolve().parents[2] / "data" / "careers" / f"pre_window_careers_{CUTOFF}.csv.gz"
#: Files the features are a function of besides the code (research_loop.block_code_hash).
DATA = (TABLE,)
MEASURES = ("win", "plc", "nfp", "mkt", "ae")
STITCHED = ("nfp", "mkt")
LAST = 3
JUMPS = ("hurdle", "chase", "nhflat")
#: A record may begin the year after the horse's birth year (a two-year-old's first
#: season, a year's grace for southern-hemisphere ages), not before.
AGE_MARGIN = 1

FEATURES = (["cb_pw_runs", "cb_pw_share", "cb_years_since_first"]
            + [f"cb_car_{m}" for m in MEASURES]
            + [f"cb_{m}_m{LAST}" for m in STITCHED]
            + ["cb_jumps_share", "cb_or_peak", "cb_or_vs_peak", "cb_flat_or_peak"])
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "horse_age", "number_of_runners",
         "placing_numerical", "bfsp", "official_rating", "race_type", "surface_type"]

_TABLE: pd.DataFrame | None = None
_BLANK = ("", "nan", "none")


def horse_key(names: pd.Series) -> pd.Series:
    """A horse as the engine keys it (form_windows, freshness): the name stripped, in lower case."""
    return names.fillna("").astype(str).str.strip().str.lower()


def _num(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(_col(df, name), errors="coerce").to_numpy(dtype=float)


def pre_window_table(history: pd.DataFrame, cutoff: str = CUTOFF) -> pd.DataFrame:
    """TABLE's rows from a history: each horse's runs before `cutoff`, one row per horse.

    The runs a block sees on the matrix (a usable price: blocks.matrix_rows), each
    measured by run_measures on its own race as the engine measures the runs since:
    runs and runs over jumps; the sum and count of each measure's known values; the
    last LAST runs' NFP and market view, last first (unknown where a run's is); the
    highest official rating carried on the Flat and over jumps; the first and last
    run. research/queries/done/pre_window_careers.py runs this on race_results."""
    from model.blocks import matrix_rows
    date = pd.to_datetime(history["race_date"], errors="coerce")
    d = history[((date < pd.Timestamp(cutoff)).to_numpy()) & matrix_rows(history)]
    key = horse_key(d["horse_name"])
    d, key = d[~key.isin(_BLANK)], key[~key.isin(_BLANK)]
    m = run_measures(d)
    jumps = np.isin(race_code(d).to_numpy(), JUMPS)
    orr = _num(d, "official_rating")
    orr = np.where(orr > 0, orr, np.nan)
    f = pd.DataFrame({"key": key.to_numpy(), "date": pd.to_datetime(d["race_date"]).to_numpy(),
                      "time": d["race_time"].astype(str).to_numpy(), "jumps": jumps.astype(float),
                      "or_flat": np.where(jumps, np.nan, orr), "or_jumps": np.where(jumps, orr, np.nan),
                      **{k: m[k] for k in MEASURES}})
    f = f.sort_values(["key", "date", "time"], kind="mergesort").reset_index(drop=True)
    g = f.groupby("key", sort=True)
    out = pd.DataFrame({"pw_runs": g.size(), "pw_jumps_runs": g["jumps"].sum()})
    for k in MEASURES:
        out[f"pw_{k}_sum"] = g[k].sum()
        out[f"pw_{k}_n"] = g[k].count()
    back = g.cumcount(ascending=False) + 1                 # 1 = the horse's last run before the cut-off
    for j in range(1, LAST + 1):
        last = f[back == j].set_index("key")
        for k in STITCHED:
            out[f"pw_{k}_l{j}"] = last[k]
    out["pw_or_max_flat"] = g["or_flat"].max()
    out["pw_or_max_jumps"] = g["or_jumps"].max()
    out["pw_first_run"] = g["date"].min().dt.strftime("%Y-%m-%d")
    out["pw_last_run"] = g["date"].max().dt.strftime("%Y-%m-%d")
    return out.rename_axis("horse_key").reset_index()


def use_table(t: pd.DataFrame | None) -> None:
    """Read `t` (pre_window_table's frame) in place of TABLE; None goes back to TABLE."""
    global _TABLE
    _TABLE = None if t is None else _prepare(t.copy())


def _prepare(t: pd.DataFrame) -> pd.DataFrame:
    t = t.set_index("horse_key")
    if not t.index.is_unique:
        raise ValueError("pre-window careers: a horse key appears twice")
    first = pd.to_datetime(t["pw_first_run"], errors="coerce")
    t["first_day"] = np.where(first.notna(), first.to_numpy().astype("datetime64[D]").astype(np.int64), np.nan)
    t["first_year"] = first.dt.year.astype(float)
    return t


def table() -> pd.DataFrame:
    """The runs before the cut-off, one row per horse, indexed by the engine's horse key."""
    global _TABLE
    if _TABLE is None:
        if not TABLE.exists():
            raise FileNotFoundError(f"{TABLE} is missing: research/queries/done/pre_window_careers.py writes it")
        _TABLE = _prepare(pd.read_csv(TABLE, dtype={"horse_key": str}, keep_default_na=False,
                                      na_values=[""]))
    return _TABLE


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    since = day >= np.datetime64(CUTOFF, "D").astype(np.int64)
    name = horse_key(df["horse_name"])
    named = ~name.isin(_BLANK).to_numpy()
    live = named & since                                   # the rows the block describes
    H = _Days(np.where(live, _codes(name), -1), day)       # runs since the cut-off only
    nb = len(H.u)
    prev = np.clip(np.arange(nb) - 1, 0, None)
    before = H.valid & (H.pos >= 1)

    def prior_sum(v_rows):
        """Per block: the horse's sum and count of known values over its earlier days."""
        v = H.per_block(v_rows)
        f = np.isfinite(v)
        cs = pd.Series(np.where(f, v, 0.0)).groupby(H.key).cumsum().to_numpy()
        cn = pd.Series(f.astype(float)).groupby(H.key).cumsum().to_numpy()
        return (H.to_rows(np.where(before, cs[prev], 0.0)), H.to_rows(np.where(before, cn[prev], 0.0)))

    def prior_max(v_rows):
        v = H.per_block(v_rows, how="max")
        cm = pd.Series(np.where(np.isfinite(v), v, -np.inf)).groupby(H.key).cummax().to_numpy()
        out = np.where(before, cm[prev], -np.inf)
        return H.to_rows(np.where(np.isfinite(out), out, np.nan))

    # the table's row for each row it may describe: the same name, a record that could be its own
    t = table()
    at = t.index.get_indexer(name.to_numpy())
    year = pd.to_datetime(df["race_date"], errors="coerce").dt.year.to_numpy(dtype=float)
    age = _num(df, "horse_age")
    rec_year = np.where(at >= 0, t["first_year"].to_numpy(dtype=float)[np.clip(at, 0, None)], np.nan)
    namesake = np.isfinite(age) & (rec_year < year - age + AGE_MARGIN)
    has = live & (at >= 0) & ~namesake

    def pw(col, fill=np.nan):
        return np.where(has, t[col].to_numpy(dtype=float)[np.clip(at, 0, None)], fill)

    m = run_measures(df)
    code = race_code(df).to_numpy()
    jumps = np.isin(code, JUMPS)
    orr = _num(df, "official_rating")
    orr = np.where(orr > 0, orr, np.nan)

    cols = {}
    pw_runs = pw("pw_runs", 0.0)
    iw_runs = H.to_rows(np.where(H.valid, H.pos, 0).astype(float))
    runs = pw_runs + iw_runs
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["cb_pw_runs"] = np.where(live, pw_runs, np.nan)
        cols["cb_pw_share"] = np.where(live & (runs > 0), pw_runs / runs, np.nan)
        first_since = H.to_rows(np.where(before, H.day[H.start], np.nan))
        first = np.fmin(pw("first_day"), first_since)
        cols["cb_years_since_first"] = np.where(live, (day - first) / 365.25, np.nan)
        for k in MEASURES:
            s, c = prior_sum(m[k])
            s = s + pw(f"pw_{k}_sum", 0.0)
            c = c + pw(f"pw_{k}_n", 0.0)
            cols[f"cb_car_{k}"] = np.where(live & (c > 0), s / c, np.nan)
        for k in STITCHED:
            v = H.per_block(m[k])
            s, c = np.zeros(nb), np.zeros(nb)
            for j in range(1, LAST + 1):                   # the runs since, as form_windows' windows read them
                x = np.where(H.valid & (H.pos >= j), v[np.clip(np.arange(nb) - j, 0, None)], np.nan)
                f = np.isfinite(x)
                s += np.where(f, x, 0.0)
                c += f
            s, c = H.to_rows(s), H.to_rows(c)
            for back in range(1, LAST + 1):                # then the runs before 2021, last first, into the room left
                x = np.where(iw_runs <= LAST - back, pw(f"pw_{k}_l{back}"), np.nan)
                f = np.isfinite(x)
                s = s + np.where(f, x, 0.0)
                c = c + f
            cols[f"cb_{k}_m{LAST}"] = np.where(live & (c > 0), s / c, np.nan)
        js, _ = prior_sum(jumps.astype(float))
        cols["cb_jumps_share"] = np.where(live & (runs > 0), (js + pw("pw_jumps_runs", 0.0)) / runs, np.nan)
        flat_peak = np.fmax(prior_max(np.where(jumps, np.nan, orr)), pw("pw_or_max_flat"))
        jumps_peak = np.fmax(prior_max(np.where(jumps, orr, np.nan)), pw("pw_or_max_jumps"))
        peak = np.where(jumps, jumps_peak, flat_peak)
        cols["cb_or_peak"] = np.where(live, peak, np.nan)
        cols["cb_or_vs_peak"] = np.where(live, orr - peak, np.nan)
        cols["cb_flat_or_peak"] = np.where(live, flat_peak, np.nan)
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
