"""The Sportmonks matches as lab columns: each canonical match's xG measures and statistics. ``smx_``.

racing² football research, 24 Nov 2026. ``sources/sportmonks.py`` builds one row a finished match; this pairs each
with the canonical ``match_id`` (``football.data.link_other``: competition, UK day, both team names) and keeps the
columns the block ``football/blocks/sm_xg.py`` reads, each renamed ``smx_<name>``. What happened in the match: the
block lags it to the collection day.
"""

from __future__ import annotations

import pandas as pd

#: Sportmonks' own prediction (pr_) and the bookmakers' 1X2 consensus near the kick-off (od_q_: a closing benchmark,
#: never an early feature) ride along; the prediction's time is not known (sources/sportmonks.py)
EXTRA = ["pr_ft_h", "pr_ft_d", "pr_ft_a", "pr_o25", "pr_btts", "od_q_h", "od_q_d", "od_q_a", "od_books"]
KEEP = ["xg_xg", "xg_npxg", "xg_xgot", "xg_xpts", "xg_xgop", "xg_xgsp", "st_big_chances_created",
        "st_dangerous_attacks", "st_shots_insidebox", "st_shots_total", "st_key_passes", "st_successful_passes",
        "st_passes", "st_ball_possession"]
COLUMNS = [f"smx_{c}_{s}" for c in KEEP for s in ("h", "a")] + [f"smx_{c}" for c in EXTRA]


def build(gold: pd.DataFrame, sm: pd.DataFrame | None, aliases=None) -> pd.DataFrame:
    """``match_id`` + ``COLUMNS`` (blank where a column is missing from the source)."""
    from football.data import link_other
    from football.europe_form import COLUMNS as EU
    if sm is None or sm.empty:
        return pd.DataFrame(columns=["match_id"] + COLUMNS + EU)
    from football.reference import BY_ID
    o = sm[sm["competition_id"].isin(set(BY_ID))].copy()          # our competitions only: Europe is kept apart
    o["match_date"] = pd.to_datetime(o["kickoff_utc"], utc=True).dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    linked = link_other(gold, o, "sportmonks", aliases)
    if linked.empty:
        return pd.DataFrame(columns=["match_id"] + COLUMNS + EU)
    out = linked[["match_id"]].copy()
    for c in KEEP:
        for s in ("h", "a"):
            src = f"{c}_{s}"
            out[f"smx_{src}"] = pd.to_numeric(linked[src], errors="coerce") if src in linked else float("nan")
    for c in EXTRA:
        out[f"smx_{c}"] = pd.to_numeric(linked[c], errors="coerce") if c in linked else float("nan")
    out = out.dropna(subset=["match_id"]).drop_duplicates("match_id")[["match_id"] + COLUMNS]
    # each side's European matches as form (the owner, 24 Nov 2026), lagged to the collection day
    from football.europe_form import build as europe_build
    eu = europe_build(gold, sm, linked)
    return out.merge(eu, on="match_id", how="outer")
