"""The Tote's prices against Betfair's, pool by pool (the owner's ask of 6 Oct 2026: "access the tote prices and
compare to betfair"). Arithmetic only: the readers of the two sides feed it.

The pools (Britbet horse racing pool rules, October 2025): win 19.25% deducted; place 20% (5 or more runners; two
places for 5-7 runners, three for 8-15 and for non-handicaps of 16 or more, four for handicaps of 16 or more);
exacta 25%; trifecta 25%; swinger 30% (6 or more runners: two horses both in the first three, a third of the net
pool to each of 1st+2nd, 1st+3rd and 2nd+3rd). A horse taken out before coming under orders is refunded, so a Tote
bet carries no reduction factor. Bets placed direct at tote.co.uk carry two more terms, from articles of 2021-25
still to be confirmed against the current terms: the Tote Guarantee (a win bet pays at least the industry SP) and
Tote+ (10% more on win and place dividends over 1.20, 5% on the exotics).

Win and place against a Betfair lay. Backed on the Tote for S and laid on Betfair for S/(1 - c) at price L, the
lay's winnings cancel the Tote stake when the horse loses; when it wins the pair nets S x (D - b(L)), D the Tote's
return per unit staked and b(L) = 1 + (L - 1)/(1 - c) the break-even (10.18 at L = 10 and 2% commission). So the
pair pays only on winners, and only when the Tote pays more than b(L). Laid at Betfair's SP, both prices are fixed
only at the off: it is not an arbitrage but a bet that the Tote pays more than the BSP.

Exacta, trifecta, swinger. Betfair lists no such markets on GB/IE racing (the census of 6 Oct), so they are
compared with the fair dividend Betfair's win prices imply through model.ordering (Harville at gamma = delta = 1,
Benter's flattened exponents once fitted): a combination's probability p against the Tote's dividend D for it, the
expected return of a unit bet being p x D.
"""

from __future__ import annotations

import itertools

import numpy as np

from model.ordering import _flatten, exacta_matrix

DEDUCTION = {"win": 0.1925, "place": 0.20, "exacta": 0.25, "trifecta": 0.25, "swinger": 0.30}
TOTE_PLUS = {"win": 0.10, "place": 0.10, "exacta": 0.05, "trifecta": 0.05, "swinger": 0.05}
TOTE_PLUS_FLOOR = 1.20                          # Tote+ applies to dividends over this
SWINGER_SHARES = 3                              # the net pool is split between three winning pairs
COMMISSION = 0.02                               # the owner's Betfair rate


def place_terms(n_runners: int, handicap: bool = False) -> int:
    """Places the Tote's place pool pays: none under 5 runners, 2 for 5-7, 3 for 8-15, and for 16 or more 4 in
    a handicap, else 3."""
    if n_runners < 5:
        return 0
    if n_runners <= 7:
        return 2
    if n_runners <= 15:
        return 3
    return 4 if handicap else 3


def direct_return(dividend, pool: str = "win", sp=None, plus: bool = True, guarantee: bool = True):
    """What a winning unit bet placed direct returns, stake included: the dividend, with Tote+ over 1.20, and for
    a win bet at least the industry SP (decimal, stake included) when the guarantee is on."""
    d = np.asarray(dividend, float)
    if plus:
        d = np.where(d > TOTE_PLUS_FLOOR, d * (1.0 + TOTE_PLUS[pool]), d)
    if pool == "win" and guarantee and sp is not None:
        s = np.asarray(sp, float)
        d = np.where(np.isfinite(s) & (s > d), s, d)
    return d


def breakeven(lay_price, commission: float = COMMISSION):
    """The Tote return at which a back hedged by a Betfair lay at lay_price nets nothing when it wins."""
    return 1.0 + (np.asarray(lay_price, float) - 1.0) / (1.0 - commission)


def edge(tote_return, lay_price, commission: float = COMMISSION):
    """The Tote's return over the lay's break-even, as a share: above 0, the pair makes money if the horse wins."""
    return np.asarray(tote_return, float) / breakeven(lay_price, commission) - 1.0


def hedged_return(tote_return, lay_price, won, commission: float = COMMISSION):
    """Per unit on the Tote, laid for 1/(1 - c) at lay_price: nothing when the horse loses (the lay's winnings pay
    the Tote stake), the Tote's return less the break-even when it wins."""
    return np.where(np.asarray(won, bool), np.asarray(tote_return, float) - breakeven(lay_price, commission), 0.0)


def implied_share(dividend, pool: str = "win"):
    """The share of the pool's money on the winning horse or combination a declared dividend implies,
    (1 - deduction)/D; a third of that for the swinger, whose net pool is split three ways."""
    share = (1.0 - DEDUCTION[pool]) / np.asarray(dividend, float)
    return share / SWINGER_SHARES if pool == "swinger" else share


def market_probabilities(prices) -> np.ndarray:
    """Win probabilities from a race's prices (the reciprocals, normalised to one)."""
    q = 1.0 / np.asarray(prices, float)
    return q / q.sum()


def trifecta_tensor(p, gamma: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """P(i first, j second, k third) for every ordered triple (Benter's three-way form; Harville at 1, 1)."""
    q = np.asarray(p, float)
    q = q / q.sum()
    s, t = _flatten(q, gamma), _flatten(q, delta)
    n = len(q)
    out = np.zeros((n, n, n))
    for i, j in itertools.permutations(range(n), 2):
        rest = 1.0 - t[i] - t[j]
        if rest <= 1e-12:
            continue
        pij = q[i] * s[j] / (1.0 - s[i])
        for k in range(n):
            if k != i and k != j:
                out[i, j, k] = pij * t[k] / rest
    return out


def swinger_matrix(p, gamma: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """P(i and j both in the first three) for every pair, symmetric, zero on the diagonal. Each race has three
    winning pairs, so the pairs i < j sum to 3, and a runner's row sums to twice its chance of a top-three finish."""
    t = trifecta_tensor(p, gamma, delta)
    n = t.shape[0]
    m = np.zeros((n, n))
    for i, j, k in itertools.permutations(range(n), 3):
        v = t[i, j, k]
        m[i, j] += v
        m[i, k] += v
        m[j, k] += v
    return m + m.T


def fair_dividends(p, pool: str, gamma: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """The dividend at which a unit bet on each combination breaks even, from win probabilities p: 1/P for the
    exacta (winner x runner-up) and the trifecta (1st x 2nd x 3rd), 1/P(both in the first three) for the swinger
    pair. Impossible combinations get inf."""
    if pool == "exacta":
        prob = exacta_matrix(p, gamma)
    elif pool == "trifecta":
        prob = trifecta_tensor(p, gamma, delta)
    elif pool == "swinger":
        prob = swinger_matrix(p, gamma, delta)
    else:
        raise ValueError(f"no fair dividend for the {pool} pool")
    with np.errstate(divide="ignore"):
        return np.where(prob > 0, 1.0 / prob, np.inf)


# --------------------------------------------------------------------------------------------------------------------
# The record's answers, read (tote_recorder.py's day files) and joined to Betfair's catalogue
# --------------------------------------------------------------------------------------------------------------------

def _div(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def declared(records):
    """The declared dividends in race-card/race-results answers (the recorder's results file), as two frames.

    Runners: one row a placed runner: race_id, track, post_utc, runners, non_runners, cloth, name, finish, sp (the
    decimal SP), win and win_base (the win dividend as paid, with the Tote Guarantee, and the pool's own), place and
    place_base likewise. Pools: one row a pool: race_id, track, post_utc, pool, total (pounds), dividends and
    base_dividends in the order the answer gives them (finishing order; the swinger's three pairs). A race asked
    more than once (T+10, T+30, the day's end) is read from its latest answer; an abandoned race is left out."""
    import pandas as pd
    latest = {}
    for rec in sorted(records, key=lambda r: str(r.get("polled_utc") or "")):
        for race in ((rec.get("answer") or {}).get("results") or []) if isinstance(rec.get("answer"), dict) else []:
            if isinstance(race, dict) and race.get("raceId") is not None:
                latest[str(race["raceId"])] = race
    runners, pools = [], []
    for rid, race in latest.items():
        if race.get("abandoned"):
            continue
        base = {"race_id": rid, "track": race.get("trackName"), "post_utc": race.get("postTime"),
                "runners": race.get("runners"), "non_runners": len(race.get("nonRunners") or [])}
        for r in race.get("placeRunners") or []:
            win, plc = r.get("winDividend") or {}, r.get("placeDividend") or {}
            sp = r.get("decimalShowPrice")
            runners.append(dict(base, cloth=str(r.get("programNumber") or ""), name=r.get("name"),
                                finish=r.get("finishingPosition"),
                                sp=sp / 100.0 if isinstance(sp, (int, float)) and sp > 0 else None,
                                win=_div(win.get("dividend")), win_base=_div(win.get("baseDividend")),
                                place=_div(plc.get("dividend")), place_base=_div(plc.get("baseDividend"))))
        for p in race.get("pools") or []:
            pools.append(dict(race_id=rid, track=race.get("trackName"), post_utc=race.get("postTime"),
                              pool=str(p.get("name") or "").replace(" ", "").upper(), total=_div(p.get("total")),
                              dividends=[_div(x) for x in p.get("dividends") or []],
                              base_dividends=[_div(x) for x in p.get("baseDividends") or []]))
    return pd.DataFrame(runners), pd.DataFrame(pools)


def _course(name) -> str:
    import re
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", str(name or "").lower()))


def match_races(tote_races, catalogue, market_type: str = "WIN", tol_minutes: float = 2.0) -> dict:
    """Tote race id -> Betfair market id of market_type, by course and off time: the courses' names normalised, one
    inside the other ('CHELMSFORD' and 'Chelmsford City'), the offs within tol_minutes. tote_races has race_id,
    track and post_utc; catalogue is Betfair's (market_id, market_type, venue, market_start_utc)."""
    import pandas as pd
    if len(tote_races) == 0 or len(catalogue) == 0:
        return {}
    cat = catalogue[catalogue["market_type"].astype(str).eq(market_type)].drop_duplicates("market_id")
    cat = cat.assign(_c=cat["venue"].map(_course), _t=pd.to_datetime(cat["market_start_utc"], utc=True,
                                                                      errors="coerce"))
    out = {}
    races = tote_races.drop_duplicates("race_id")
    for r in races.itertuples():
        c, t = _course(r.track), pd.to_datetime(r.post_utc, utc=True, errors="coerce")
        if not c or pd.isna(t):
            continue
        hit = cat[cat["_c"].map(lambda v: bool(v) and (v in c or c in v))
                  & ((cat["_t"] - t).abs() <= pd.Timedelta(minutes=tol_minutes))]
        if len(hit) == 1:
            out[str(r.race_id)] = str(hit["market_id"].iloc[0])
    return out


def selection_ids(rows, catalogue, race_map: dict):
    """Each Tote runner's Betfair selection id in its race's market (race_map: Tote race id -> market id): by the
    cloth number, else by the horse's name normalised; None where neither finds it. rows has race_id, cloth, name."""
    import pandas as pd
    from betfair_prices import normalise_horse
    cat = catalogue.copy()
    cat["market_id"] = cat["market_id"].astype(str)
    cat["_cloth"] = cat["cloth_number"].map(lambda v: "" if pd.isna(v) else str(v).split(".")[0])
    by_cloth = {(m, c): s for m, c, s in zip(cat["market_id"], cat["_cloth"], cat["selection_id"]) if c}
    by_name = {(m, normalise_horse(n)): s for m, n, s in zip(cat["market_id"], cat["runner_name"], cat["selection_id"])}
    out = []
    for r in rows.itertuples():
        mid = race_map.get(str(r.race_id))
        if mid is None:
            out.append(None)
            continue
        s = by_cloth.get((mid, str(r.cloth)))
        if s is None:
            s = by_name.get((mid, normalise_horse(r.name)))
        out.append(None if s is None else int(s))
    return pd.Series(out, index=rows.index, dtype=object)
