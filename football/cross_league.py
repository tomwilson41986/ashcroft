"""Cross-league strength: one rating for every club across divisions (cups) and countries (UEFA).

racing² football research, 6 Nov 2026 (the owner: "work on cross-league strength, divisions in country for cups,
leagues across countries for Europe"). A goal-weighted Elo run once through every senior club match API-Football
holds (``api_football/fixtures.parquet``: domestic leagues, domestic cups, continental club competitions), in kick-off
order, keyed by API-Football's team id, which is the same club in every competition:

- each match's **pre-match** ratings are recorded before its result updates them (point in time by construction);
- expected score E = 1 / (1 + 10^(-(R_home + H - R_away) / 400)), H the home advantage (none at a neutral final);
- update K x G x (W - E), W = 1 / 0.5 / 0 and G = 1 + ln(1 + |goal difference|) (bigger wins move more);
- a club seen for the first time starts at its league's current mean rating (a promoted or a lower-division side
  enters at its level), or at ``START`` in a league with no rated clubs yet;
- cup ties carry strength between divisions and continental ties between countries: ``cross=False`` keys a club by
  (club, competition) instead, so nothing crosses (the baseline the cross-league rating must beat).

League strength (``league_strength``): each league's mean pre-match rating of its clubs, a season.

Excluded: friendlies, youth, reserve and women's competitions, unfinished matches.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

START, K, HOME = 1500.0, 20.0, 60.0
EXCLUDE_WORDS = ("Friendl", "Women", "Femen", "Feminine", "U17", "U18", "U19", "U20", "U21", "U23", "Youth",
                 "Reserve", "II", "B Team", "Juniors", "Olympic")


def senior_club_matches(fx: pd.DataFrame) -> pd.DataFrame:
    """Finished senior club matches with a score, in kick-off order, with each competition's kind."""
    from research.open_close.football_cross_league_data import kind
    f = fx[(fx["status"] == "FINISHED") & fx["ft_home"].notna() & fx["ft_away"].notna()
           & fx["apif_home_id"].notna() & fx["apif_away_id"].notna()].copy()
    name = f["league"].astype(str)
    f = f[~name.apply(lambda n: any(w in n for w in EXCLUDE_WORDS))]
    f["kind"] = [kind(lg, c) for lg, c in zip(f["league"], f["league_country"])]
    f = f[f["kind"].isin(["league", "cup", "uefa"])]
    f["kickoff"] = pd.to_datetime(f["kickoff_utc"], utc=True, errors="coerce")
    f = f.dropna(subset=["kickoff"]).sort_values(["kickoff", "apif_fixture_id"]).reset_index(drop=True)
    f["competition"] = f["league_country"].astype(str) + " | " + f["league"].astype(str)
    return f


def elo(f: pd.DataFrame, cross: bool = True, k: float = K, home: float = HOME) -> pd.DataFrame:
    """Each match's pre-match ratings (``r_home``, ``r_away``) from every earlier match only. ``cross=False``: a
    club's rating is kept apart in each competition (no strength carried between them)."""
    rating: dict = {}
    members: dict = {}                      # competition -> set of keys rated in it
    rh = np.empty(len(f))
    ra = np.empty(len(f))
    comp = f["competition"].to_numpy()
    hid, aid = f["apif_home_id"].to_numpy(), f["apif_away_id"].to_numpy()
    gh, ga = f["ft_home"].to_numpy(float), f["ft_away"].to_numpy(float)
    for i in range(len(f)):
        c = comp[i]
        kh = hid[i] if cross else (hid[i], c)
        ka = aid[i] if cross else (aid[i], c)
        mem = members.setdefault(c, set())
        for key in (kh, ka):
            if key not in rating:
                known = [rating[m] for m in mem if m in rating]
                rating[key] = float(np.mean(known)) if known else START
            mem.add(key)
        a, b = rating[kh], rating[ka]
        rh[i], ra[i] = a, b
        e = 1.0 / (1.0 + 10 ** (-(a + home - b) / 400.0))
        d = gh[i] - ga[i]
        w = 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
        g = 1.0 + math.log1p(abs(d))
        rating[kh] = a + k * g * (w - e)
        rating[ka] = b - k * g * (w - e)
    out = f[["apif_fixture_id", "kickoff", "competition", "kind", "apif_season", "apif_home_id", "apif_away_id",
             "ft_home", "ft_away"]].copy()
    out["r_home"], out["r_away"] = rh, ra
    return out


def league_strength(r: pd.DataFrame) -> pd.DataFrame:
    """Each domestic league's mean pre-match club rating a season (both sides of every league match)."""
    lg = r[r["kind"] == "league"]
    s = pd.concat([lg[["competition", "apif_season", "r_home"]].rename(columns={"r_home": "r"}),
                   lg[["competition", "apif_season", "r_away"]].rename(columns={"r_away": "r"})])
    return s.groupby(["competition", "apif_season"])["r"].agg(["mean", "count"]).reset_index()
