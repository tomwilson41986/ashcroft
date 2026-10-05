"""What the horse sold for before it ran: Timeform's record of its foal, yearling, breeze-up and store sales.

The price error is largest where public form is thin (error-by-segment-958), and the
external-sales screen (reports/external_data_review.md, 5 Oct 2026) found that an
unexposed horse's auction price cuts the within-race error of the BSP forecast by about
3% in maidens and novice races, while adding nothing beyond the BSP itself: the market
prices it and the model could not see it. The prices come from Timeform's production
comments in the Blandford scraper's horse files (data/external/timeform_sales.csv.gz,
built by scripts/build_timeform_sales.py): only sales before a horse races, foal,
yearling and two-year-old (breeze-up) sales, and three- and four-year-old stores for
jumpers. A horse is found by name and foaling year (the race's year less its age), or
by name alone when one horse in the file has it and its foaling year is within one.

    sl_known            1 when the horse is in Timeform's file, else missing
    sl_sold             1 when it sold at a public sale before racing, 0 when known but not
                        (home-bred or bought privately), missing when unknown
    sl_ln_price         log of its last sale price before racing, in pounds
    sl_ln_yearling      log of its yearling price
    sl_ln_breeze        log of its two-year-old (breeze-up) price
    sl_ln_store         log of its store price (three- or four-year-old)
    sl_kind             the last sale's kind: 0 foal, 1 yearling, 2 breeze-up, 3 store
    sl_price_z          sl_ln_price against the field's, over the runners with one
    sl_price_rank       its rank among them by price, 1 the dearest, as a share of those priced
    sl_race_sold_share  the share of the field known to have sold

Nothing here reads a result: the sale came before the horse's first run, and the
field-relative readings use only the card. But the file itself is a snapshot (3 July
2025), and Timeform rewrites a horse's comment as its career grows: the sale clause
survives in 97-99% of horses with three runs or fewer by the snapshot and 3% of those
with sixteen or more (sales-block-coverage-1005). Before the snapshot, then, whether a
race's runner shows a price depends on how much it ran afterwards, which is the future.
So every reading is missing for a race on or before SNAPSHOT: the block reads the file
only as it stood before the race. Live gap: horses foaled in 2024 on (this year's
two-year-olds) have no record until a Timeform horse feed serves one.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from model.blocks.race_relative_wide import _z_gap
from model.race_shape import race_key

FEATURES = ["sl_known", "sl_sold", "sl_ln_price", "sl_ln_yearling", "sl_ln_breeze", "sl_ln_store", "sl_kind",
            "sl_price_z", "sl_price_rank", "sl_race_sold_share"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "horse_age"]
READS_RESULTS = False

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TABLE_PATH = os.path.normpath(os.path.join(_ROOT, "data", "external", "timeform_sales.csv.gz"))
# The synthetic history the block tests run on has its own horses, which Timeform's file does not
# hold; the tests read a fabricated table of them (mechanics only, never scored).
# The day the horse files were taken; a race on or before it reads nothing (see the docstring).
SNAPSHOT = "2025-07-03"
TEST_REACH = {"TABLE_PATH": os.path.normpath(os.path.join(_ROOT, "tests", "fixtures", "sales_synthetic.csv")),
              "SNAPSHOT": "2000-01-01"}


def norm_name(name) -> str:
    """'1. Casts Tasha (IRE)' -> 'caststasha': betfair_prices.normalise_horse, kept here so the
    block and the table builder cannot drift apart."""
    s = str(name or "").lower()
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


@lru_cache(maxsize=4)
def _table(path: str) -> pd.DataFrame:
    t = pd.read_csv(path, dtype={"name_norm": str})
    t["foal_year"] = pd.to_numeric(t["foal_year"], errors="coerce")
    return t


def _match(df: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Each row's record in the table (NaN where none): by name and foaling year, else by a name
    one horse in the table has, foaled within a year of the row's."""
    year = pd.to_datetime(df["race_date"], errors="coerce").dt.year
    age = pd.to_numeric(df["horse_age"], errors="coerce")
    q = pd.DataFrame({"name_norm": df["horse_name"].map(norm_name).to_numpy(),
                      "foal_year": (year - age).to_numpy(dtype=float)})
    exact = q.merge(t, on=["name_norm", "foal_year"], how="left")
    unique = t[~t["name_norm"].duplicated(keep=False)]
    loose = q[["name_norm", "foal_year"]].merge(unique, on="name_norm", how="left", suffixes=("", "_t"))
    near = ((loose["foal_year_t"] - loose["foal_year"]).abs() <= 1).to_numpy()
    vals = [c for c in t.columns if c not in ("name_norm", "foal_year")]
    loose = loose[vals].astype(float)
    loose.loc[~near, vals] = np.nan
    missing = exact["has_comment"].isna().to_numpy()
    out = exact[vals].copy()
    out.loc[missing, vals] = loose.loc[missing, vals].to_numpy()
    out.index = df.index
    return out


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rec = _match(df, _table(TABLE_PATH))
    after = (pd.to_datetime(df["race_date"], errors="coerce") > pd.Timestamp(SNAPSHOT)).to_numpy()
    rec.loc[~after, :] = np.nan                      # before the snapshot the file knows the future
    known = rec["has_comment"].notna().to_numpy()
    last = pd.to_numeric(rec["last_gbp"], errors="coerce").to_numpy(dtype=float)
    sold = np.isfinite(last) & (last > 0)

    def ln(col):
        v = pd.to_numeric(rec[col], errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.isfinite(v) & (v > 0), np.log(v), np.nan)

    cols = {"sl_known": np.where(known, 1.0, np.nan),
            "sl_sold": np.where(known, sold.astype(float), np.nan),
            "sl_ln_price": np.where(sold, np.log(np.where(sold, last, 1.0)), np.nan),
            "sl_ln_yearling": ln("yearling_gbp"), "sl_ln_breeze": ln("breeze_gbp"), "sl_ln_store": ln("store_gbp"),
            "sl_kind": np.where(sold, pd.to_numeric(rec["last_kind"], errors="coerce").to_numpy(dtype=float), np.nan)}
    rk = race_key(df).to_numpy()
    price = pd.Series(cols["sl_ln_price"], index=df.index)
    cols["sl_price_z"], _ = _z_gap(price, rk)
    rank = price.groupby(rk).rank(ascending=False, method="average")
    priced = price.notna().groupby(rk).transform("sum")
    cols["sl_price_rank"] = (rank / priced.where(priced > 0)).to_numpy(dtype=float)
    k = pd.Series(np.where(known, sold.astype(float), np.nan), index=df.index)
    n_known = k.notna().groupby(rk).transform("sum")
    cols["sl_race_sold_share"] = (k.fillna(0).groupby(rk).transform("sum") / n_known.where(n_known > 0)).to_numpy()
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    assert len(new) == n
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
