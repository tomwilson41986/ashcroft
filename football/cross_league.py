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


K_CROSS_CLUB, K_CROSS_LEAGUE = 10.0, 5.0      # fixed before any result was read (v2, 6 Nov 2026)


def elo_hier(f: pd.DataFrame, k: float = K, k_club_cross: float = K_CROSS_CLUB,
             k_league: float = K_CROSS_LEAGUE, home: float = HOME) -> pd.DataFrame:
    """v2: a club's rating = its domestic league's level + its place within the league (places centred on zero in
    each league). League matches move places only; matches between clubs of different leagues (cups across
    divisions, continental ties) move both leagues' levels (``k_league``) and, less, the two places
    (``k_club_cross``). A club's league is the domestic league it last played in; a promoted or relegated club keeps
    its total rating (its place is re-set against the new league). A club seen only in a cup or a continental
    competition so far is placed in that competition's pool. Pre-match totals recorded before each update."""
    level: dict = {}                     # league -> level
    place: dict = {}                     # club -> raw place
    league_of: dict = {}                 # club -> league
    sums: dict = {}                      # league -> [sum of raw places, count]

    def mean_place(lg):
        s = sums.get(lg)
        return s[0] / s[1] if s and s[1] else 0.0

    def total(c):
        lg = league_of[c]
        return level.get(lg, START) + place[c] - mean_place(lg)

    def join(c, lg):
        """Put club c in league lg, keeping its total rating if it had one."""
        tot = total(c) if c in league_of else None
        if c in league_of:
            s = sums[league_of[c]]
            s[0] -= place[c]
            s[1] -= 1
        if lg not in level:
            level[lg] = START if tot is None else tot
        league_of[c] = lg
        m = mean_place(lg)
        place[c] = m if tot is None else tot - level[lg] + m
        s = sums.setdefault(lg, [0.0, 0])
        s[0] += place[c]
        s[1] += 1

    def bump(c, d):
        place[c] += d
        sums[league_of[c]][0] += d

    rh, ra = np.empty(len(f)), np.empty(len(f))
    comp, kind_ = f["competition"].to_numpy(), f["kind"].to_numpy()
    hid, aid = f["apif_home_id"].to_numpy(), f["apif_away_id"].to_numpy()
    gh, ga = f["ft_home"].to_numpy(float), f["ft_away"].to_numpy(float)
    for i in range(len(f)):
        h, a, c = hid[i], aid[i], comp[i]
        for club in (h, a):
            if kind_[i] == "league":
                if league_of.get(club) != c:
                    join(club, c)
            elif club not in league_of:
                join(club, c)
        th, ta = total(h), total(a)
        rh[i], ra[i] = th, ta
        e = 1.0 / (1.0 + 10 ** (-(th + home - ta) / 400.0))
        d = gh[i] - ga[i]
        w = 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
        g = 1.0 + math.log1p(abs(d))
        lh, la = league_of[h], league_of[a]
        if lh == la:
            bump(h, k * g * (w - e))
            bump(a, -k * g * (w - e))
        else:
            level[lh] += k_league * g * (w - e)
            level[la] -= k_league * g * (w - e)
            bump(h, k_club_cross * g * (w - e))
            bump(a, -k_club_cross * g * (w - e))
    out = f[["apif_fixture_id", "kickoff", "competition", "kind", "apif_season", "apif_home_id", "apif_away_id",
             "ft_home", "ft_away"]].copy()
    out["r_home"], out["r_away"] = rh, ra
    return out
