"""The point-in-time helpers every metrics table is built with.

The rule (the repo's, model/lagsafe.py): a feature of a run reads only what was known before that run's DAY. A dog's
own history steps back to its last earlier day; a group's (trainer, sire, track) aggregates its earlier days only. The
tests (tests/test_greyhound_pit.py) rewrite the last day's results and check that no table's last-day row moves.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_mean, race_lagged_expanding_sum  # noqa: F401  (re-exported)

LENGTH_S = 0.08


def norm_name(s) -> str:
    """A dog's name for matching across sources: lower case, letters and digits only ("Swift Blade's" -> swiftblades)."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def per_run_figures(runs: pd.DataFrame) -> pd.DataFrame:
    """The metrics engine's per-run figures (greyhound/metrics.py: gsr, esr, fsr, lbw, the parsed comment), so every
    table reads the same figures as the model."""
    from greyhound.metrics import GreyhoundMetricsEngine
    return GreyhoundMetricsEngine(blocks=())._per_run(runs.copy())


def day(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df.race_date, errors="coerce").dt.normalize()


def prev_day_values(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Each row's ``key`` group's values on its last EARLIER day (the last run of that day): a dog's previous run, never
    one later the same day. Aligned to ``df.index``; NaN when there is none."""
    d = df[[key, "t"] + [c for c in cols if c != "t"]].assign(_d=day(df))
    last = d.sort_values([key, "t"]).groupby([key, "_d"], sort=False).tail(1).sort_values([key, "_d"])
    prev = last.groupby(key)[cols].shift(1)
    prev[[key, "_d"]] = last[[key, "_d"]]
    m = d[[key, "_d"]].reset_index().merge(prev, on=[key, "_d"], how="left").set_index("index")
    return m[cols].reindex(df.index)


def prior_window(df: pd.DataFrame, keys: list[str], cols: list[str], days: int | None) -> pd.DataFrame:
    """Sums of ``cols`` (and the row count ``n``) over the group's runs on earlier days, within the last ``days`` days
    before the row's day (``None``: all earlier days). Aligned to ``df.index``; 0 where there is none."""
    d = df[keys + cols].copy()
    for k in keys:                                  # merge_asof cannot group on a missing key, nor on mixed types
        if not pd.api.types.is_numeric_dtype(d[k]):
            d[k] = d[k].astype(object).where(d[k].notna(), "?").astype(str)
    d["_d"] = day(df)
    d["n"] = 1.0
    agg = d.groupby(keys + ["_d"], dropna=False)[cols + ["n"]].sum().reset_index().sort_values(keys + ["_d"])
    cum = agg.groupby(keys, dropna=False)[cols + ["n"]].cumsum()
    agg[[f"_c_{c}" for c in cols + ["n"]]] = cum.to_numpy()

    def cum_before(when: pd.Series) -> pd.DataFrame:
        """The group's cumulative sums over days strictly before ``when`` (one value a row)."""
        q = d[keys].copy()
        q["_q"] = when.values
        q["_i"] = np.arange(len(q))
        q = q.dropna(subset=["_q"]).sort_values("_q")
        right = agg[keys + ["_d"] + [f"_c_{c}" for c in cols + ["n"]]].sort_values("_d")
        m = pd.merge_asof(q, right, left_on="_q", right_on="_d", by=keys, allow_exact_matches=False,
                          direction="backward")
        res = np.zeros((len(d), len(cols) + 1))
        res[m._i.to_numpy()] = m[[f"_c_{c}" for c in cols + ["n"]]].fillna(0.0).to_numpy()
        return pd.DataFrame(res, index=df.index, columns=cols + ["n"])

    upto = cum_before(d["_d"])
    if days is None:
        return upto
    return upto - cum_before(d["_d"] - pd.Timedelta(days=days))      # the days D - N .. D - 1


def shrunk_rate(hits: pd.Series, n: pd.Series, prior: float, k: float) -> pd.Series:
    """(hits + k x prior) / (n + k): a rate pulled to ``prior`` when there is little evidence."""
    return (hits + k * prior) / (n + k)


def ae(won: pd.Series, expected: pd.Series) -> pd.Series:
    """Actual over expected (wins over the market's implied chances); NaN with nothing expected."""
    return (won / expected).where(expected > 0)


def dog_lagged(df: pd.DataFrame, col: str, n: int | None = None, how: str = "mean", key: str = "dog_id") -> pd.Series:
    """Per ``key``: ``how`` (mean, sum, max) of ``col`` over its last ``n`` earlier DAYS (each day's runs averaged;
    ``n`` None: every earlier day). Aligned to ``df.index``; NaN with no earlier day."""
    d = df[[key, col]].assign(_d=day(df))
    per = d.groupby([key, "_d"])[col].mean().reset_index().sort_values([key, "_d"])
    g = per.groupby(key)[col]
    if n is None:
        stat = {"mean": g.expanding().mean(), "sum": g.expanding().sum(), "max": g.expanding().max()}[how]
    else:
        stat = {"mean": g.rolling(n, min_periods=1).mean(), "sum": g.rolling(n, min_periods=1).sum(),
                "max": g.rolling(n, min_periods=1).max()}[how]
    per["_v"] = stat.reset_index(level=0, drop=True).sort_index().reindex(per.index).to_numpy()
    per["_v"] = per.groupby(key)["_v"].shift(1)
    m = d[[key, "_d"]].reset_index().merge(per[[key, "_d", "_v"]], on=[key, "_d"], how="left").set_index("index")
    return m["_v"].reindex(df.index)


#: a run's trainer is the trainer at the race only when GBGB was read within this many days of it: GBGB serves each dog's
#: CURRENT trainer, owner and season on every past result (6 Nov 2026: none of 10,144 dogs raced in 2026 shows a second
#: trainer, in a fetch made that day), so history fetched later says who trains the dog now, not then
FRESH_DAYS = 4


def raw_fetch_lag(store) -> pd.Series:
    """Days between each GBGB day and the last time its raw file was written (S3 LastModified, or the file's mtime in a
    local store): small when the nightly fetch read it at the time, large for a day backfilled later."""
    import os
    from datetime import datetime, timezone
    days, written = [], []
    if store.root:
        base = store.root / "gbgb" / "raw"
        for p in base.rglob("*.json.gz"):
            days.append(p.name[:10])
            written.append(datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc))
    else:
        from sources.common import PREFIX
        tok = None
        while True:
            r = store.s3.list_objects_v2(Bucket=store.bucket, Prefix=f"{PREFIX}/gbgb/raw/",
                                         **({"ContinuationToken": tok} if tok else {}))
            for o in r.get("Contents", []):
                days.append(o["Key"].rsplit("/", 1)[1][:10])
                written.append(o["LastModified"])
            if not r.get("IsTruncated"):
                break
            tok = r["NextContinuationToken"]
    if not days:
        return pd.Series(dtype=float)
    d = pd.to_datetime(pd.Series(days), errors="coerce")
    w = pd.to_datetime(pd.Series(written), utc=True).dt.tz_localize(None).dt.normalize()
    return pd.Series(((w - d).dt.days).to_numpy(), index=d.dt.strftime("%Y-%m-%d")).groupby(level=0).min()


def trainer_at_the_time(runs: pd.DataFrame, lag: pd.Series | None) -> pd.Series:
    """The run's trainer where GBGB was read within FRESH_DAYS of the race; NaN where the file came later (or no
    fetch record is given)."""
    if lag is None or not len(lag):
        return pd.Series(np.nan, index=runs.index, dtype=object)
    ok = runs.race_date.map(lag).le(FRESH_DAYS)
    return runs.trainer.where(ok)
