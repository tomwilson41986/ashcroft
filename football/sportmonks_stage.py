"""The Sportmonks matches as lab columns: each canonical match's xG measures and statistics. ``smx_``.

racing² football research, 24 Nov 2026. ``sources/sportmonks.py`` builds one row a finished match; this pairs each
with the canonical ``match_id`` (``football.data.link_other``: competition, UK day, both team names) and keeps the
columns the block ``football/blocks/sm_xg.py`` reads, each renamed ``smx_<name>``. What happened in the match: the
block lags it to the collection day.
"""

from __future__ import annotations

import pandas as pd

KEEP = ["xg_xg", "xg_npxg", "xg_xgot", "xg_xpts", "xg_xgop", "xg_xgsp", "st_big_chances_created",
        "st_dangerous_attacks", "st_shots_insidebox", "st_shots_total", "st_key_passes", "st_successful_passes",
        "st_passes", "st_ball_possession"]
COLUMNS = [f"smx_{c}_{s}" for c in KEEP for s in ("h", "a")]


def build(gold: pd.DataFrame, sm: pd.DataFrame | None, aliases=None) -> pd.DataFrame:
    """``match_id`` + ``COLUMNS`` (blank where a column is missing from the source)."""
    from football.data import link_other
    if sm is None or sm.empty:
        return pd.DataFrame(columns=["match_id"] + COLUMNS)
    o = sm.copy()
    o["match_date"] = pd.to_datetime(o["kickoff_utc"], utc=True).dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    linked = link_other(gold, o, "sportmonks", aliases)
    if linked.empty:
        return pd.DataFrame(columns=["match_id"] + COLUMNS)
    out = linked[["match_id"]].copy()
    for c in KEEP:
        for s in ("h", "a"):
            src = f"{c}_{s}"
            out[f"smx_{src}"] = pd.to_numeric(linked[src], errors="coerce") if src in linked else float("nan")
    return out.dropna(subset=["match_id"]).drop_duplicates("match_id")[["match_id"] + COLUMNS]
