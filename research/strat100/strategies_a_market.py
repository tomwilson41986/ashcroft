"""The a_market strategies of the 100-strategy harness, S069-S100 (racing², conventions K1-K10, 24 Nov 2026).

Betfair's price files and the results, 2018-01-01 to 2026-03-27, no model price (data_a_market.build_frames gives the
frames: win, pairs, runs). One strategy = one setting = one function returning a bets frame; the core settles it,
takes commission, and reads it. Research only: nothing here places a bet or reads the holdout.

WHAT A SELECTION MAY READ (the look-ahead rule; tests/test_strat100_a_market.py holds the line)
    morning entry (S077, S084, S086, S087, S090)   the card, earlier days, morningwap and morning_vol; nothing from
        ppwap, ppmax, ppmin, the BSP's value or the in-play columns. K6 is the one registered exception: an entry
        before the off is read only in races where no selection lacks a BSP (that reads whether a BSP exists, never
        its value; the core flags these 'no cuts').
    order at the SP decided before the off (the place and win SP strategies)   also ppwap and pp_vol, which are
        complete only at the off (each specification says so); the BSP only as the price the order fills at, or to
        test a limit. Never the in-play columns of the day.
    in-running strategies (S092-S100)   the BSP itself (the entry) and earlier days' in-play record; the day's own
        ipmin / ipmax only decide whether the resting order filled.

FILLS (every one an assumption the files cannot check)
    MARKET_ON_CLOSE          filled in full at the BSP; a runner with no BSP is no bet
    LIMIT_ON_CLOSE back      filled at the BSP only if BSP >= the limit; a lay only if BSP <= the limit. The limit is
                             the formula's price, not rounded to a tick, unless the specification rounds it (S079).
                             An order the SP does not reach is kept as a void row (no bet, counted in 'voids')
    morning entry            at morningwap, stake at most 10% of morning_vol, nothing under GBP2 (the core prints the
                             result at prices 2% and 5% worse)
    SP stake                 the specification's cap as a share of pp_vol, nothing under GBP2; an SP lay's liability
                             is never under GBP10 (K4)
    resting lay at L(f)      K8: filled only if the day's ipmin is above 1 and at or under core.tick_under(L); the
                             closing price is L itself (the order would rest on the tick at or under L, a slightly
                             better price, so the reading is conservative by less than a tick)
    resting back at H(m)     K8: filled only if ipmax >= core.tick_over(H); closed at H itself
    unfilled resting order   the SP bet stands to the result

A horse's earlier runs are the rows of EARLIER DAYS (race_results) or earlier offs (the price files, by horse_norm).
"""
from __future__ import annotations

import itertools
import re
import weakref

import numpy as np
import pandas as pd

from research.strat100 import core

DS = "a_market"
SEED = core.SEED
CODES = ("flat", "aw", "hurdle", "chase", "nhflat")
_CACHE: dict = {}
_TOKENS = itertools.count()


def _say(*a) -> None:
    print(*a, flush=True)


def _forget(name: str, token: int) -> None:
    hit = _CACHE.get(name)
    if hit is not None and hit[2] == token:
        del _CACHE[name]


def _cached(name: str, frame, fn):
    """``fn()`` once a shared frame. The frame is held weakly: when the runner lets the data set go, what was
    derived from it goes too (the next data set needs the memory), and a later frame is never mistaken for it."""
    hit = _CACHE.get(name)
    if hit is not None and hit[0]() is frame:
        return hit[1]
    val, token = fn(), next(_TOKENS)
    _CACHE[name] = (weakref.ref(frame), val, token)
    weakref.finalize(frame, _forget, name, token)
    return val


def _arr(v, n: int) -> np.ndarray:
    if isinstance(v, (pd.Series, pd.Index)):
        return v.to_numpy()
    a = np.asarray(v)
    return np.full(n, v) if a.ndim == 0 else a


def _bets(sel: pd.DataFrame, side: str, entry_price, *, stake=None, liability=None, exit="result", exit_price=None,
          won=None, market: str = "win", entry_type: str = "sp", void=False) -> pd.DataFrame:
    """The bets frame of the selected rows (race_date, rk, selection_id): one row a bet. ``void`` marks an order
    that did not fill (a limit the SP did not reach, a resting order never matched): no bet, counted as a void."""
    n = len(sel)
    if n == 0:
        return core.empty_bets()
    b = pd.DataFrame({
        "race_date": sel["race_date"].astype(str).to_numpy(),
        "race_key": sel["rk"].astype(str).to_numpy(),
        "selection": pd.to_numeric(sel["selection_id"]).astype(np.int64).astype(str).to_numpy(),
        "side": side,
        "stake": np.nan if stake is None else _arr(stake, n).astype(float),
        "entry_price": _arr(entry_price, n).astype(float),
        "exit": _arr(exit, n),
        "exit_price": np.nan if exit_price is None else _arr(exit_price, n).astype(float),
        "won": np.nan if won is None else _arr(won, n).astype(float),
        "void": _arr(void, n).astype(bool),
        "market": market,
        "entry_type": entry_type,
    })
    if liability is not None:
        b["liability"] = _arr(liability, n).astype(float)
    b["won"] = b["won"].clip(0.0, 1.0)
    for c in ("country", "code"):
        if c in sel.columns:
            b[c] = sel[c].to_numpy()
    return b


def _report(sid: str, bets: pd.DataFrame, by: tuple = (), extra: str = "") -> None:
    """A printed table of the bets (gross of commission): the specification's 'printed' cuts. Not the reading."""
    try:
        if not len(bets):
            _say(f"  {sid}: no bets{(' ' + extra) if extra else ''}")
            return
        s = core.settle(core.check_bets(bets))
        s = s[~s["void"]].copy()
        s["window"] = core.window_of(s["race_date"], DS)
        s["year"] = s["race_date"].str[:4]
        days = s.groupby("window")["race_date"].nunique()
        for keys in (("window",),) + ((tuple(by),) if by else ()):
            t = s.groupby(list(keys)).agg(bets=("pnl", "size"), risk=("risk", "sum"), gross=("pnl", "sum"))
            t["gross_ret"] = t["gross"] / t["risk"]
            if keys == ("window",):
                t["gbp_a_bet_day"] = t["gross"] / days.reindex(t.index)
            _say(f"  {sid} by {', '.join(keys)} (gross of commission){(' ' + extra) if extra else ''}")
            _say("    " + t.round(4).to_string().replace("\n", "\n    "))
    except Exception as exc:  # noqa: BLE001 -- a table is never worth the strategy
        _say(f"  {sid}: table not printed ({type(exc).__name__}: {exc})")


def _ae(won, price) -> float:
    won, price = np.asarray(won, float), np.asarray(price, float)
    ok = np.isfinite(won) & (price > 1.0)
    e = (1.0 / price[ok]).sum()
    return float(won[ok].sum() / e) if e > 0 else float("nan")


def _day(s) -> np.ndarray:
    """ISO days as integer day numbers."""
    return pd.to_datetime(pd.Series(np.asarray(s)), errors="coerce").to_numpy().astype("datetime64[D]").astype(np.int64)


def _name(s: pd.Series) -> pd.Series:
    """A person's name lower-cased, punctuation stripped, spaces collapsed ('' where there is none)."""
    t = s.astype(object).where(s.notna(), "").astype(str).str.lower()
    return t.str.replace(r"[^a-z0-9 ]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def _window(hkey, hday, values: dict, qkey, qday, back: int) -> dict:
    """The sum of each value over the history rows with the query's key on the ``back`` days BEFORE the query's day
    (days qday - back .. qday - 1; the query's own day and every later one never enter). An empty key is no key."""
    hk = pd.Series(np.asarray(hkey, dtype=object)).replace("", np.nan)
    qk = pd.Series(np.asarray(qkey, dtype=object)).replace("", np.nan)
    codes, _ = pd.factorize(pd.concat([hk, qk], ignore_index=True))
    hc, qc = codes[:len(hk)].astype(np.int64), codes[len(hk):].astype(np.int64)
    hd, qd = np.asarray(hday, np.int64), np.asarray(qday, np.int64)
    if len(hd) == 0 or len(qd) == 0:
        return {k: np.zeros(len(qd)) for k in values}
    base = min(hd.min(), qd.min()) - back - 1
    hd, qd = hd - base, qd - base
    m = int(max(hd.max(), qd.max())) + 2
    ok = hc >= 0
    comp = hc[ok] * m + hd[ok]
    order = np.argsort(comp, kind="stable")
    comp = comp[order]
    lo = np.searchsorted(comp, qc * m + (qd - back), "left")
    hi = np.searchsorted(comp, qc * m + qd, "left")
    out = {}
    for name, v in values.items():
        c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(np.asarray(v, float)[ok][order]))])
        out[name] = np.where(qc >= 0, c[hi] - c[lo], 0.0)
    return out


def _cap(stake, vol, share: float) -> np.ndarray:
    return np.asarray(core.capacity(stake, vol, share), float)


# ---------------------------------------------------------------------------------------------------------------
# the place market against the win market (S069-S076)
# ---------------------------------------------------------------------------------------------------------------

def _place_rows(frames) -> pd.DataFrame:
    """S069's universe: PAIR-A races with 5+ runners, every runner with a win BSP holding a place BSP, k passing
    K9, and K9's place chance p from the normalised 1 / win ppwap."""
    from research.strat100 import data_a_market as D
    x = frames["pairs"]
    ready = D.place_ready(x)
    if ready.any() and not x.loc[ready, "p_place"].notna().any():
        raise RuntimeError("K9's place chance was not built by the loader (see its log)")
    return x[ready & x["p_place"].notna()]


def _limit_back(x: pd.DataFrame, p, ev: float, stake: float) -> pd.DataFrame:
    """LIMIT_ON_CLOSE place back at L = 1 + ((1 + ev) / p - 1) / 0.98: filled at the place BSP only if BSP >= L."""
    limit = 1.0 + ((1.0 + ev) / np.asarray(p, float) - 1.0) / 0.98
    return _bets(x, "back", x["p_bsp"], stake=_cap(stake, x["p_pp_vol"], 0.02), won=x["p_win_lose"], market="place",
                 void=~(x["p_bsp"].to_numpy(float) >= limit))


def s069(frames):
    """Place pocket by limit-on-close back. Win ppwap <= 5.0 (and above 1.01), place pp_vol >= 1,000; the limit is
    the price at which the place bet is worth 5% after 2% commission by K9's chance; GBP20, at most 2% of the place
    pp_vol. Choices: an order whose limit the place SP does not reach is no bet (a void row); the limit is not
    rounded to a tick. Not blind: the pocket was read at the SPs on 2023-26Q1."""
    x = _place_rows(frames)
    x = x[(x["ppwap"] <= 5.0) & (x["p_pp_vol"] >= 1000)]
    b = _limit_back(x, x["p_place"], 0.05, 20.0)
    _report("S069", b, by=("country", "year"))
    return b


def s070(frames):
    """Place limit on close at mid prices: 5.0 < win ppwap <= 21.0, place pp_vol >= 500, the limit at a 10% value
    by K9's chance; GBP10, at most 2% of the place pp_vol."""
    x = _place_rows(frames)
    x = x[(x["ppwap"] > 5.0) & (x["ppwap"] <= 21.0) & (x["p_pp_vol"] >= 500)]
    return _limit_back(x, x["p_place"], 0.10, 10.0)


def harville_place(q, k: int) -> np.ndarray:
    """Harville's P(in the first k), exact for k <= 4 (the sums written out as arrays; core.harville_place is the
    same model, exact to three places and simulated for the fourth)."""
    q = np.asarray(q, float)
    q = q / q.sum()
    n = len(q)
    if k >= n:
        return np.ones(n)
    out = q.copy()
    if k == 1:
        return out
    eye = np.eye(n, dtype=bool)
    a = q[:, None] * q[None, :] / (1.0 - q[:, None])
    a[eye] = 0.0                                               # a[i, j]: i first, j second
    out += a.sum(axis=0)
    if k == 2:
        return out
    left2 = np.clip(1.0 - q[:, None] - q[None, :], 1e-12, None)
    b = (a / left2)[:, :, None] * q[None, None, :]
    b[eye[:, :, None] | eye[:, None, :] | eye[None, :, :]] = 0.0          # b[i, j, l]: l third
    out += b.sum(axis=(0, 1))
    if k == 3:
        return out
    if k > 4:
        raise ValueError("harville_place: k above 4 is not written out")
    left3 = np.clip(1.0 - q[:, None, None] - q[None, :, None] - q[None, None, :], 1e-12, None)
    c = (b / left3)[:, :, :, None] * q[None, None, None, :]
    i = np.arange(n)
    c[i, :, :, i] = 0.0                                        # the fourth is none of the first three
    c[:, i, :, i] = 0.0
    c[:, :, i, i] = 0.0
    return out + c.sum(axis=(0, 1, 2))


_HENERY_Z: dict = {}


def henery_theta(q, tol: float = 1e-4, max_iter: int = 200) -> np.ndarray:
    """Henery's means: each runner's time is N(theta_i, 1) and the lowest wins; theta solved so the win chances
    (the integral of the runner's density times the others' survival, on a fixed grid) match q to ``tol``. Each
    step is Newton's on the runner's own mean (d ln w_i / d theta_i is the mean of x - theta_i over the runner's
    winning times), damped, the means re-centred."""
    from scipy.special import log_ndtr
    q = np.clip(np.asarray(q, float), 1e-6, None)
    q = q / q.sum()
    theta = -np.log(q) * 0.5
    theta -= theta.mean()
    for _ in range(max_iter):
        x = np.linspace(theta.min() - 8.0, theta.max() + 8.0, 321)
        d = x[None, :] - theta[:, None]
        logsf = log_ndtr(-d)
        y = np.exp(-0.5 * d * d + logsf.sum(axis=0)[None, :] - logsf)
        w = np.clip(y.sum(axis=1), 1e-300, None)
        slope = (d * y).sum(axis=1) / w                         # below zero: a slower horse wins less
        w = w * (x[1] - x[0]) / np.sqrt(2.0 * np.pi)
        if np.abs(w - q).max() < tol:
            break
        step = (np.log(w) - np.log(q)) / np.minimum(slope, -0.2)
        theta -= 0.8 * np.clip(step, -2.0, 2.0)
        theta -= theta.mean()
    return theta


def henery_place(q, k: int, n_sims: int = 20000, seed: int = SEED) -> np.ndarray:
    """Henery's P(in the first k): theta by ``henery_theta`` (the win chances matched to 1e-4 by integration), the
    places counted on 20,000 simulated races, seed 7 (the same draws as core.henery_place, which solves theta on
    the simulated shares and is about forty times slower: a hundred thousand races would not finish)."""
    q = np.clip(np.asarray(q, float), 1e-6, None)
    q = q / q.sum()
    n = len(q)
    if k >= n:
        return np.ones(n)
    theta = henery_theta(q)
    z = _HENERY_Z.get((n, n_sims, seed))
    if z is None:
        z = _HENERY_Z[(n, n_sims, seed)] = np.random.default_rng(seed).standard_normal((n_sims, n))
    first = np.argpartition(z + theta, k - 1, axis=1)[:, :k]
    return np.bincount(first.ravel(), minlength=n) / n_sims


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


STACK_FIRST_YEAR, STACK_MIN_ROWS = 2021, 300


def _stack(frames) -> pd.DataFrame:
    """S071's stacked place chance: for each year Y from 2021 a logistic regression of 'placed' on logit(Harville),
    logit(Henery), logit(Harville) x 1{k=2}, logit(Harville) x 1{k=4} and ln(runners), fitted on the three calendar
    years before Y, its predictions rescaled in the race to sum to k. One row a runner of the predicted years."""
    def build():
        from sklearn.linear_model import LogisticRegression
        x = _place_rows(frames).copy()
        if x.empty:
            return x.assign(p_h=np.nan, p_n=np.nan, p_stack=np.nan)
        ph, pn = np.full(len(x), np.nan), np.full(len(x), np.nan)
        q_all, k_all = x["q_pp"].to_numpy(float), x["k"].to_numpy(float)
        for idx in x.groupby("event_id", sort=False).indices.values():
            q, k = q_all[idx], int(k_all[idx[0]])
            ph[idx], pn[idx] = harville_place(q, k), henery_place(q, k)
        _HENERY_Z.clear()                                      # the simulated draws are not needed again
        x["p_h"], x["p_n"] = np.clip(ph, 1e-6, 1 - 1e-6), np.clip(pn, 1e-6, 1 - 1e-6)
        x["placed"] = (x["p_win_lose"] == 1).astype(int)
        lh = _logit(x["p_h"])
        feats = np.column_stack([lh, _logit(x["p_n"]), lh * (x["k"] == 2), lh * (x["k"] == 4),
                                 np.log(x["n_win_bsp"].to_numpy(float))])
        year, y = x["year"].to_numpy(int), x["placed"].to_numpy(int)
        p = np.full(len(x), np.nan)
        for yr in range(STACK_FIRST_YEAR, int(year.max()) + 1):
            tr, te = (year >= yr - 3) & (year <= yr - 1), year == yr
            if tr.sum() < STACK_MIN_ROWS or not te.any() or len(np.unique(y[tr])) < 2:
                _say(f"  S071 stack {yr}: not fitted ({int(tr.sum())} rows in {yr - 3}-{yr - 1})")
                continue
            m = LogisticRegression(C=np.inf, max_iter=2000).fit(feats[tr], y[tr])
            p[te] = m.predict_proba(feats[te])[:, 1]
            _say(f"  S071 stack {yr}: fitted on {int(tr.sum()):,} runners of {yr - 3}-{yr - 1}; coefficients "
                 f"{np.round(m.coef_[0], 3).tolist()}, intercept {m.intercept_[0]:.3f}")
        x["p_stack"] = p
        tot = x.groupby("event_id", sort=False)["p_stack"].transform("sum")
        x["p_stack"] = (x["p_stack"] * x["k"] / tot).clip(1e-6, 0.999)
        out = x[x["p_stack"].notna()]
        if len(out):
            yv = out["placed"].to_numpy(float)

            def ll(pr):
                pr = np.clip(np.asarray(pr, float), 1e-6, 1 - 1e-6)
                return float(-(yv * np.log(pr) + (1 - yv) * np.log(1 - pr)).mean())
            _say(f"  S071 log loss on 'placed', {len(out):,} runners of the predicted years: Harville "
                 f"{ll(out['p_h']):.5f}, Henery {ll(out['p_n']):.5f}, K9 fixed exponents {ll(out['p_place']):.5f}, "
                 f"stack {ll(out['p_stack']):.5f}, 1 / place BSP {ll(1.0 / out['p_bsp']):.5f}")
        return out
    return _cached("stack", frames["pairs"], build)


def s071(frames):
    """Stacked place model (Harville and Henery), limit on close. Win ppwap <= 21.0, place pp_vol >= 500; the limit
    at a 5% value by the stacked chance; GBP10 (at most 2% of the place pp_vol, which the volume floor already
    grants). The stack for year Y is fitted on the three calendar years before Y, so the first bets are 2021 and
    SELECT is read 2021-2023. Choices: Harville is exact (the sums written out) and Henery's means are solved by
    integration, the places counted on 20,000 simulated races, seed 7."""
    x = _stack(frames)
    x = x[(x["ppwap"] <= 21.0) & (x["p_pp_vol"] >= 500)]
    return _limit_back(x, x["p_stack"], 0.05, 10.0)


def _place_sp_back(x: pd.DataFrame, stake: float) -> pd.DataFrame:
    """MARKET_ON_CLOSE place back at the place BSP (at most 2% of the place pp_vol)."""
    x = x[x["bsp"].notna() & x["p_bsp"].notna()]
    return _bets(x, "back", x["p_bsp"], stake=_cap(stake, x["p_pp_vol"], 0.02), won=x["p_win_lose"], market="place")


def _terms(frames) -> pd.DataFrame:
    """PAIR-A rows of races whose places paid pass K9 and whose runners all hold a win ppwap (ranks are whole)."""
    x = frames["pairs"]
    return x[x["k_ok"] & x["place_whole"] & x["rank_pp"].notna() & x["bsp"].notna() & x["p_bsp"].notna()]


def _bad_ew_base(frames) -> pd.DataFrame:
    x = _terms(frames)
    return x[x["n_win_bsp"].between(8, 12) & (x["k"] == 3) & x["ppwap"].between(3.5, 10.0) & (x["p_pp_vol"] >= 1000)]


def s072(frames):
    """Bad each-way race: 8-12 runners with a win BSP, 3 places, the shortest win ppwap <= 2.0; the runners ranked
    2nd and 3rd by win ppwap at 3.5-10.0 with place pp_vol >= 1,000, backed at the place SP, GBP20. Choice: ties
    share the lower rank (two runners level in second are both 'second')."""
    x = _bad_ew_base(frames)
    t = _terms(frames)
    fav = x["event_id"].map(t[t["rank_pp"] == 1].groupby("event_id")["ppwap"].min())
    return _place_sp_back(x[(fav <= 2.0) & x["rank_pp"].isin([2, 3])], 20.0)


def base_s072(frames):
    """S072's benchmark: every runner at win ppwap 3.5-10.0 with place pp_vol >= 1,000 in 8-12 runner races paying
    3 places, at the place SP, GBP20 (the race shape and the rank left out)."""
    return _place_sp_back(_bad_ew_base(frames), 20.0)


def _big_hcap(frames) -> pd.DataFrame:
    x = _terms(frames)
    hc = x["event_name"].str.contains(r"hcap|handicap", case=False, regex=True)
    x = x[hc & (x["n_win_bsp"] >= 16) & (x["k"] == 4) & x["ppwap"].between(6.0, 21.0) & (x["p_pp_vol"] >= 1000)]
    return x.assign(saturday=pd.to_datetime(x["race_date"]).dt.day_of_week.eq(5).to_numpy())


def s073(frames):
    """Extra-place proxy: Saturday handicaps (the win market's event_name holds Hcap or Handicap) with 16+ runners
    and 4 places; win ppwap 6.0-21.0, place pp_vol >= 1,000; the place SP, GBP20. Which races carried an offer is
    not held: the day and the race type stand in."""
    x = _big_hcap(frames)
    return _place_sp_back(x[x["saturday"]], 20.0)


def base_s073(frames):
    """S073's benchmark: the same band in 16+ runner handicaps paying 4 places on the other days of the week."""
    x = _big_hcap(frames)
    return _place_sp_back(x[~x["saturday"]], 20.0)


def _outsiders(frames) -> pd.DataFrame:
    x = _terms(frames)
    x = x[(x["rank_pp"] >= 3) & (x["ppwap"] <= 21.0) & (x["p_pp_vol"] >= 500) & (x["n_win_bsp"] >= x["k"] + 2)]
    n, k = x["n_win_bsp"], x["k"]
    kept = ((n <= 7) & (k == 3)) | ((n <= 15) & (k == 4))
    standard = ((k == 2) & n.between(5, 7)) | ((k == 3) & (n >= 8)) | ((k == 4) & (n >= 16))
    return x.assign(kept=kept.to_numpy(), standard=standard.to_numpy())


def s074(frames):
    """Places kept in a reduced field: at least k + 2 runners and (7 or fewer runners with 3 places) or (15 or fewer
    with 4); the runners ranked 3rd or lower by win ppwap at 21.0 or under with place pp_vol >= 500; the place SP,
    GBP10. That the places stood after a withdrawal is inferred from the count of runners, not recorded."""
    x = _outsiders(frames)
    return _place_sp_back(x[x["kept"]], 10.0)


def base_s074(frames):
    """S074's benchmark: the same ranks, prices and volume in races on standard terms (2 places with 5-7 runners,
    3 with 8 or more, 4 with 16 or more)."""
    x = _outsiders(frames)
    return _place_sp_back(x[x["standard"] & ~x["kept"]], 10.0)


def _steam_band(frames) -> pd.DataFrame:
    x = frames["pairs"]
    x = x[x["bsp"].notna() & x["p_bsp"].notna() & x["ppwap"].between(2.0, 15.0) & (x["p_pp_vol"] >= 1000)]
    with np.errstate(invalid="ignore", divide="ignore"):
        steam = np.log(x["morningwap"].to_numpy(float) / x["ppwap"].to_numpy(float))
    ok = (x["morning_vol"].to_numpy(float) >= 200) & (x["morningwap"].to_numpy(float) > 1.01) & (steam >= 0.15)
    return x.assign(steamer=ok)


def s075(frames):
    """Win-market steamer backed at the place SP: the win row's morning_vol >= 200, morningwap above 1.01,
    ln(morningwap / ppwap) >= 0.15, win ppwap 2.0-15.0, place pp_vol >= 1,000; GBP20 at the place SP."""
    x = _steam_band(frames)
    return _place_sp_back(x[x["steamer"]], 20.0)


def base_s075(frames):
    """S075's benchmark: every runner at win ppwap 2.0-15.0 with place pp_vol >= 1,000, at the place SP, GBP20."""
    return _place_sp_back(_steam_band(frames), 20.0)


def s076(frames):
    """Long-shot place lay by limit on close: S069's races and chance, win ppwap >= 50.0; a LIMIT_ON_CLOSE lay with
    maximum price M = min(10.0, 1 + ((1 - p) x 0.98 - 0.05) / p), filled at the place BSP only if BSP <= M;
    liability GBP20 (the layer's size is 20 / (BSP - 1)). Choices: no order where M is not above 1.01; an unfilled
    order is no bet (a void row). Not blind (the cell was read at the SPs on 2018-26Q1)."""
    x = _place_rows(frames)
    x = x[x["ppwap"] >= 50.0]
    p = x["p_place"].to_numpy(float)
    m = np.minimum(10.0, 1.0 + ((1.0 - p) * 0.98 - 0.05) / p)
    x, m = x[m > 1.01], m[m > 1.01]
    b = _bets(x, "lay", x["p_bsp"], liability=20.0, won=x["p_win_lose"], market="place",
              void=~(x["p_bsp"].to_numpy(float) <= m))
    _report("S076", b)
    return b


# ---------------------------------------------------------------------------------------------------------------
# the steamer model's lay side (S077, S078)
# ---------------------------------------------------------------------------------------------------------------

STEAM_CUTOFF, STEAM_FROM = "2021-12-31", "2022-01-01"


def _steamer(frames) -> pd.DataFrame:
    """The CLEAN steamer model (research/paper/morning_features.train_steamer, fitted on 2018-01-01..2021-12-31)
    scored from 2022-01-01: one row a runner (race_results id) with ``pred``, the card's ``field`` and ``low``, the
    10th percentile of the model's scores on its own training rows. The frame is put in strat_lib.load's shape (its
    race key, its BSP: race_results' own, else the price file's)."""
    def build():
        from research.paper import morning_features as MF
        r = frames["runs"]
        cols = ["id", "race_date", "date", "race_time", "track", "race_name", "race_class", "number_of_runners",
                "horse_name", "trainer", "jockey_name", "jockeys_claim", "horse_age", "official_rating",
                "days_since_lr", "headgear", "code", "hcap", "year", "morningwap", "morning_vol"]
        d = r[cols].copy()
        d["bsp"] = r["bfsp"].where(r["bfsp"] > 1, r["bsp"])
        d["fin"] = r["placing_numerical"]
        d["won"] = (d["fin"] == 1).astype(float)
        d["rk"] = d["race_date"] + "|" + d["track"] + "|" + d["race_time"]
        d = d.sort_values(["date", "race_time", "track"], kind="stable").reset_index(drop=True)
        d["bsp_rank"] = d.groupby("rk")["bsp"].rank(method="min")
        d["field"] = d.groupby("rk")["horse_name"].transform("size")
        d["race_has_winner"] = d.groupby("rk")["won"].transform("sum") >= 1
        x = MF.build(d)
        tr = MF.L.priced(x)
        tr = tr[(tr["race_date"] >= "2018-01-01") & (tr["race_date"] <= STEAM_CUTOFF) & tr["steam"].notna()]
        if len(tr) < 1000:
            _say(f"  steamer model: no data found ({len(tr)} training rows with a morning price to {STEAM_CUTOFF})")
            return pd.DataFrame(columns=["id", "pred", "field", "low"])
        booster, _, n = MF.train_steamer(d, STEAM_CUTOFF)
        low = float(np.quantile(booster.predict(tr[MF.CLEAN]), 0.10))
        te = x[x["race_date"] >= STEAM_FROM]
        out = pd.DataFrame({"id": te["id"].to_numpy(), "pred": booster.predict(te[MF.CLEAN]) if len(te) else [],
                            "field": te["field"].to_numpy(float), "low": low})
        _say(f"  steamer model: fitted on {n:,} runners to {STEAM_CUTOFF}; the 10th percentile of its training "
             f"scores {low:+.4f}; {len(out):,} runners scored from {STEAM_FROM}, {int((out['pred'] <= low).sum()):,} "
             "at or under it")
        return out
    return _cached("steamer", frames["runs"], build)


def _drifters(frames) -> pd.DataFrame:
    s = _steamer(frames)
    s = s[(s["pred"] <= s["low"]) & (s["field"] >= 5)]
    r = frames["runs"]
    return r[r["id"].isin(s["id"]) & r["bp_id"].notna()]


def s077(frames):
    """Predicted drifters laid in the morning: the steamer model's bottom tenth (by its training threshold), field
    5+, morningwap at 8.0 or under with morning_vol >= 200, in races where no selection lacks a BSP (K6); laid at
    morningwap for GBP100 at risk, the layer's size min(100 / (L - 1), 10% of morning_vol); closed by a back at the
    BSP. Scored from 2022 (the model is fitted to the end of 2021)."""
    x = _drifters(frames)
    x = x[(x["morningwap"] <= 8.0) & (x["morningwap"] > 1.01) & (x["morning_vol"] >= 200) & x["race_whole"].eq(True)]
    stake = _cap(100.0 / (x["morningwap"].to_numpy(float) - 1.0), x["morning_vol"], 0.10)
    return _bets(x, "lay", x["morningwap"], stake=stake, exit="bsp", exit_price=x["bsp"], entry_type="wap")


def s078(frames):
    """Predicted drifters laid at the SP with a limit of 8.0, held: the same model and threshold, field 5+, no
    morning-price filter; a LIMIT_ON_CLOSE lay at 8.0 (filled at the BSP only if BSP <= 8.0), liability GBP100, to
    the result."""
    x = _drifters(frames)
    x = x[x["bsp"].notna()]
    _say(f"  S078: the bottom tenth's win A/E against the BSP, {len(x):,} runners from {STEAM_FROM}: "
         f"{_ae(x['win_lose'], x['bsp']):.3f}")
    return _bets(x, "lay", x["bsp"], liability=100.0, won=x["win_lose"], void=~(x["bsp"] <= 8.0).to_numpy())


# ---------------------------------------------------------------------------------------------------------------
# the SP against the day's move, and segments at the SP (S079-S091)
# ---------------------------------------------------------------------------------------------------------------

def s079(frames):
    """Fade the steamer at the SP: win rows with morningwap at 8.0 or under and morning_vol >= 200, in races where no
    selection lacks a BSP and the count of selections with a BSP equals race_results.number_of_runners (no
    removal); a LIMIT_ON_CLOSE lay with its limit 0.80 x morningwap rounded down to a tick, filled at the BSP only
    if BSP <= the limit; liability GBP100, to the result. Choices: a race none of whose rows joins race_results has
    no runner count and is not read; an order the SP does not reach is a void row."""
    w = frames["win"]
    x = w[(w["morningwap"] <= 8.0) & (w["morning_vol"] >= 200) & w["race_whole"] & w["n_bsp"].eq(w["rr_runners"])]
    limit = np.asarray(core.tick_floor(0.80 * x["morningwap"].to_numpy(float)), float)
    with np.errstate(invalid="ignore"):
        fill = x["bsp"].to_numpy(float) <= limit
    f = x[fill]
    b = _bets(x, "lay", x["bsp"], liability=100.0, won=x["win_lose"], void=~fill)
    if len(f):
        _say(f"  S079: {len(f):,} fills of {len(x):,} orders; per GBP1 of backer's stake (gross) "
             f"{float(np.where(f['win_lose'] == 1, -(f['bsp'] - 1), 1.0).mean()):+.4f}")
    return b


def s080(frames):
    """Back the pre-off drifter at the BSP, held: win rows with a BSP, morning_vol >= 200, pp_vol >= 2,000, ppwap
    3-20 and ln(ppwap / morningwap) >= 0.18; GBP10 (at most 1% of pp_vol). The BSP is never a filter."""
    w = frames["win"]
    with np.errstate(invalid="ignore", divide="ignore"):
        drift = np.log(w["ppwap"].to_numpy(float) / w["morningwap"].to_numpy(float))
    x = w[w["bsp"].notna() & (w["morning_vol"] >= 200) & (w["pp_vol"] >= 2000) & w["ppwap"].between(3.0, 20.0)
          & (drift >= 0.18)]
    return _bets(x, "back", x["bsp"], stake=_cap(10.0, x["pp_vol"], 0.01), won=x["win_lose"])


PREV_COLS = ["race_date", "date", "code", "headgear", "trainer", "placing_numerical", "bsp", "ipmin", "win_lose"]


def _prev(frames) -> pd.DataFrame:
    """Each run's previous run: the horse's latest row of an EARLIER race_date in race_results (a second row of the
    same day is never 'previous'). Columns prev_<name>, row for row with frames['runs']."""
    def build():
        r = frames["runs"]
        key = r["horse_name"].str.lower().str.strip()
        d = r[PREV_COLS].assign(horse=key.to_numpy())
        last = d.drop_duplicates(["horse", "race_date"], keep="last")          # one row a horse a day
        sh = last.groupby("horse", sort=False)[PREV_COLS].shift(1)
        sh.columns = [f"prev_{c}" for c in PREV_COLS]
        sh = pd.concat([last[["horse", "race_date"]], sh], axis=1)
        out = d[["horse", "race_date"]].merge(sh, on=["horse", "race_date"], how="left")
        out.index = r.index
        return out.drop(columns=["horse", "race_date"])
    return _cached("prev", frames["runs"], build)


def _sp_runs(frames) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs with a win BSP in the price files from 2018, their previous runs)."""
    r, p = frames["runs"], _prev(frames)
    ok = r["bsp"].notna() & r["bp_id"].notna()
    return r[ok], p[ok]


def s081(frames):
    """Lay chasers sent back over hurdles, at the BSP: today a hurdle, the horse's previous run a chase; win pp_vol
    >= 1,000; a MARKET_ON_CLOSE lay with liability min(GBP50, 2% of pp_vol), never under GBP10; to the result. No
    price filter. Not blind (the tables covered 2022-26)."""
    r, p = _sp_runs(frames)
    hurdles = r[r["code"] == "hurdle"]
    _say(f"  S081 control: every hurdle runner at the BSP reads A/E {_ae(hurdles['win_lose'], hurdles['bsp']):.3f} "
         f"({len(hurdles):,} runners)")
    x = r[(r["code"] == "hurdle") & (p["prev_code"] == "chase") & (r["pp_vol"] >= 1000)]
    liab = np.minimum(50.0, 0.02 * x["pp_vol"].to_numpy(float))
    x, liab = x[liab >= 10.0], liab[liab >= 10.0]
    _say(f"  S081: the segment reads A/E {_ae(x['win_lose'], x['bsp']):.3f} ({len(x):,} runners)")
    return _bets(x, "lay", x["bsp"], liability=liab, won=x["win_lose"])


def s082(frames):
    """Back headgear removed, at the BSP: headgear worn at the horse's previous run and none today; all codes;
    GBP20 (at most 1% of pp_vol). Not blind."""
    r, p = _sp_runs(frames)
    x = r[(r["headgear"].str.strip() == "") & p["prev_date"].notna() & (p["prev_headgear"].fillna("").str.strip() != "")]
    return _bets(x, "back", x["bsp"], stake=_cap(20.0, x["pp_vol"], 0.01), won=x["win_lose"])


def _surnames(raw: pd.Series) -> list:
    """The surname tokens of a trainer: the words before a comma ('Henderson, N J'), else the last word."""
    out = []
    for t in raw.astype(object).where(raw.notna(), "").astype(str).str.lower():
        head = t.split(",")[0] if "," in t else (re.sub(r"[^a-z0-9 ]+", " ", t).split() or [""])[-1]
        out.append(frozenset(tok for tok in re.sub(r"[^a-z0-9 ]+", " ", head).split() if len(tok) > 1))
    return out


def s083(frames):
    """Back the first run for a new trainer, at the BSP: today's trainer differs from the trainer at the horse's
    previous run (lower-cased, punctuation stripped), left out where the two names share a surname token (the words
    before a comma, else the last word: a licence passed within a family or a name written two ways is not a
    switch); all codes; GBP20 (at most 1% of pp_vol). Not blind."""
    r, p = _sp_runs(frames)
    a, b = _name(r["trainer"]), _name(p["prev_trainer"])
    diff = (a != "") & (b != "") & (a != b)
    x, px = r[diff], p[diff]
    shared = np.array([bool(s & t) for s, t in zip(_surnames(x["trainer"]), _surnames(px["prev_trainer"]))], bool)
    _say(f"  S083: {len(x):,} runs with a different trainer's name; {int(shared.sum()):,} left out by the surname rule")
    x = x[~shared]
    return _bets(x, "back", x["bsp"], stake=_cap(20.0, x["pp_vol"], 0.01), won=x["win_lose"])


def _rr_race(r: pd.DataFrame) -> pd.Series:
    """race_results' own race: date|track|time."""
    return r["race_date"] + "|" + r["track"] + "|" + r["race_time"]


def s084(frames):
    """Bottom weight in chase handicaps, a morning back held: a chase whose race_name holds 'handicap'; the runner
    or runners with the lowest official_rating of the race's rated runners; morning_vol >= 50; GBP20 at morningwap,
    at most 10% of morning_vol; to the result. K6: only races where no selection lacks a BSP. Not blind."""
    r = frames["runs"]
    c = r[(r["code"] == "chase") & r["hcap"] & (r["official_rating"] > 0)]
    low = c["official_rating"].groupby(_rr_race(c)).transform("min")
    x = c[c["official_rating"].eq(low) & (c["morning_vol"] >= 50) & (c["morningwap"] > 1.01)
          & c["race_whole"].eq(True)]
    return _bets(x, "back", x["morningwap"], stake=_cap(20.0, x["morning_vol"], 0.10), won=x["win_lose"],
                 entry_type="wap")


def s085(frames):
    """Stable second string, at the BSP: a trainer with 2+ runners in a race, every one with morningwap above 1.01
    and morning_vol above 0; the trainer's lowest morningwap <= 6.0; its next-lowest runner is backed if its
    morningwap <= 26.0 (one a trainer a race, ties by seed 7); GBP20 (at most 1% of pp_vol). Stablemates are the
    runners in the result; a second string with no BSP is no bet."""
    r = frames["runs"]
    r = r[r["race_date"] >= "2018-01-01"]             # every runner of the trainer counts, priced or not
    tr = _name(r["trainer"])
    d = r.assign(g=(_rr_race(r) + "|" + tr).to_numpy(), tie=np.random.default_rng(SEED).random(len(r)))[tr != ""]
    priced = (d["morningwap"] > 1.01) & (d["morning_vol"] > 0)
    n = d.groupby("g")["id"].transform("size")
    d = d[(n >= 2) & priced.groupby(d["g"]).transform("all")]
    d = d.sort_values(["g", "morningwap", "tie"], kind="stable")
    pos = d.groupby("g", sort=False).cumcount()
    first = d["g"].map(d[pos == 0].set_index("g")["morningwap"])
    x = d[(pos == 1) & (first <= 6.0) & (d["morningwap"] <= 26.0) & d["bsp"].notna()]
    return _bets(x, "back", x["bsp"], stake=_cap(20.0, x["pp_vol"], 0.01), won=x["win_lose"])


def _morning_rows(frames) -> pd.DataFrame:
    """Runs with a morning price that reached the off: morningwap above 1.01, morning_vol above 0, a BSP; with
    r = min(morningwap / bsp - 1, 1.0) (the capped morning-to-SP move)."""
    def build():
        r = frames["runs"]
        x = r[(r["morningwap"] > 1.01) & (r["morning_vol"] > 0) & r["bsp"].notna()]
        return x.assign(r=np.minimum(x["morningwap"].to_numpy(float) / x["bsp"].to_numpy(float) - 1.0, 1.0),
                        day=_day(x["race_date"]))
    return _cached("morning_rows", frames["runs"], build)


def _clv_back(x: pd.DataFrame) -> pd.DataFrame:
    """A morning back to win GBP50 (at most GBP100, at most 10% of morning_vol), closed at the BSP."""
    stake = _cap(core.to_win_stake(50.0, x["morningwap"].to_numpy(float), 0.0, 100.0), x["morning_vol"], 0.10)
    return _bets(x, "back", x["morningwap"], stake=stake, exit="bsp", exit_price=x["bsp"], entry_type="wap")


SEG_FIT_TO, SEG_K, SEG_FLOOR, SEG_MIN_N = "2021-12-31", 300.0, 0.06, 300


def _segments(x: pd.DataFrame) -> pd.DataFrame:
    d = x["days_since_lr"]
    dband = np.select([d.isna() | (d <= 0), d <= 4, d <= 7, d <= 14, d <= 30, d <= 90],
                      ["none", "1-4", "5-7", "8-14", "15-30", "31-90"], default="91+")
    c = x["career_runs"]
    cband = np.select([c.isna() | (c <= 0), c <= 3, c <= 10], ["0", "1-3", "4-10"], default="11+")
    cls = pd.to_numeric(x["race_class"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
    clsb = np.select([cls.between(1, 3), cls == 4], ["1-3", "4"], default="5+/none")
    parent = x["code"].astype(str) + "|" + np.where(x["hcap"], "hcap", "non") + "|" + dband
    return pd.DataFrame({"parent": parent.to_numpy(), "cell": (parent + "|" + cband + "|" + clsb).to_numpy(),
                         "dband": dband}, index=x.index)


def s086(frames):
    """Segment steam table, backed in the morning and laid at the BSP. Cells: code (5) x handicap x days since the
    last run (none, 1-4, 5-7, 8-14, 15-30, 31-90, 91+) x career runs (0, 1-3, 4-10, 11+) x class (1-3, 4, 5+/none).
    On 2018-2021 each cell's mean r = min(morningwap / bsp - 1, 1) is shrunk to its parent (code x handicap x days
    band, k 300) and the parent to the global mean (k 300); a cell is selected at a shrunk mean of +0.06 or more
    with 300+ runners. From 2022 every runner of a selected cell with a morning price, in races where no selection
    lacks a BSP, to win GBP50 (at most GBP100, 10% of morning_vol). Choices: a runner with no career_runs or no
    days_since_lr goes to the '0' / 'none' band; the table is fitted on the same whole races it bets in."""
    m = _morning_rows(frames)
    m = m[m["race_whole"].eq(True)]
    seg = _segments(m)
    fit = (m["race_date"] <= SEG_FIT_TO).to_numpy()
    if not fit.any():
        _say("  S086: no data found (no morning price to the end of 2021)")
        return core.empty_bets()
    f = pd.DataFrame({"parent": seg["parent"][fit], "cell": seg["cell"][fit], "r": m["r"][fit]})
    glob = float(f["r"].mean())
    par = f.groupby("parent")["r"].agg(["sum", "size"])
    par_s = (par["sum"] + SEG_K * glob) / (par["size"] + SEG_K)
    cell = f.groupby("cell").agg(sum=("r", "sum"), n=("r", "size"), parent=("parent", "first"))
    cell["shrunk"] = (cell["sum"] + SEG_K * cell["parent"].map(par_s)) / (cell["n"] + SEG_K)
    chosen = cell[(cell["shrunk"] >= SEG_FLOOR) & (cell["n"] >= SEG_MIN_N)]
    _say(f"  S086: fitted on {int(fit.sum()):,} runners to {SEG_FIT_TO}, global mean r {glob:+.4f}; "
         f"{len(chosen)} of {len(cell)} cells selected")
    if len(chosen):
        _say("    " + chosen[["n", "shrunk"]].round(4).to_string().replace("\n", "\n    "))
    x = m[~fit & seg["cell"].isin(chosen.index).to_numpy()]
    b = _clv_back(x)
    _report("S086", b)
    quick = seg.loc[x.index, "dband"].eq("1-4").to_numpy()
    _report("S086", _clv_back(x[~quick]), extra="with the 1-4 day cells left out")
    return b


YARD_DAYS, YARD_K, YARD_FLOOR, YARD_MIN_N, YARD_MIN_CORR = 730, 100.0, 0.04, 50, 0.2


def _yard(frames) -> pd.DataFrame:
    """Each morning-priced run with its trainer's steam record over the 730 days before its day: n, the sum of r,
    g (the mean r of all runners in the same window) and score = (sum r + 100 g) / (n + 100)."""
    def build():
        m = _morning_rows(frames)
        tr = _name(m["trainer"])
        one = np.ones(len(m))
        own = _window(tr, m["day"], {"n": one, "s": m["r"]}, tr, m["day"], YARD_DAYS)
        allr = _window(np.zeros(len(m), int), m["day"], {"n": one, "s": m["r"]}, np.zeros(len(m), int), m["day"],
                       YARD_DAYS)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(allr["n"] > 0, allr["s"] / allr["n"], np.nan)
        return m.assign(tr=tr.to_numpy(), yard_n=own["n"], yard_g=g,
                        yard_score=(own["s"] + YARD_K * g) / (own["n"] + YARD_K))
    return _cached("yard", frames["runs"], build)


def _yard_persistence(y: pd.DataFrame) -> tuple[float, int]:
    """The correlation of a trainer's score at 1 January with its mean r over that calendar year, read on SELECT
    only (the years 2020-2023; trainers with 50+ runs in the window and 30+ in the year). (correlation, points)."""
    rows = []
    hist_day = y["day"].to_numpy()
    for yr in range(2020, 2024):
        nxt = y[y["race_date"].str[:4] == str(yr)]
        if nxt.empty:
            continue
        ahead = nxt.groupby("tr")["r"].agg(["mean", "size"])
        ahead = ahead[(ahead["size"] >= 30) & (ahead.index != "")]
        if ahead.empty:
            continue
        d0 = _day([f"{yr}-01-01"])[0]
        q = np.full(len(ahead), d0)
        one = np.ones(len(y))
        own = _window(y["tr"], hist_day, {"n": one, "s": y["r"]}, ahead.index.to_numpy(), q, YARD_DAYS)
        allr = _window(np.zeros(len(y), int), hist_day, {"n": one, "s": y["r"]}, np.zeros(1, int), q[:1], YARD_DAYS)
        if allr["n"][0] <= 0:
            continue
        g = allr["s"][0] / allr["n"][0]
        ok = own["n"] >= YARD_MIN_N
        rows.append(pd.DataFrame({"score": ((own["s"] + YARD_K * g) / (own["n"] + YARD_K))[ok],
                                  "ahead": ahead["mean"].to_numpy()[ok]}))
    if not rows:
        return float("nan"), 0
    t = pd.concat(rows, ignore_index=True)
    return (float(t["score"].corr(t["ahead"])) if len(t) >= 30 else float("nan")), len(t)


def s087(frames):
    """Gamble yards: back a runner in the morning when its trainer's shrunk steam record over the 730 days before
    the day is +4% or more on 50+ runs (score = (sum r + 100 g) / (n + 100), g the mean r of every runner in the
    window); races where no selection lacks a BSP; to win GBP50 (at most GBP100, 10% of morning_vol); closed at the
    BSP. The registered gate: the correlation of a trainer's score with its next-year mean r must be 0.2 or more,
    else the strategy is void (no bets). Choice: that correlation is read on SELECT only (1 January 2020-2023
    against the calendar year that follows), so CONFIRM plays no part in the decision; under 30 points is void."""
    y = _yard(frames)
    corr, n = _yard_persistence(y)
    _say(f"  S087: the correlation of a trainer's score with its next-year mean r, SELECT years 2020-2023: "
         f"{corr:+.3f} on {n} trainer-years (void under {YARD_MIN_CORR})")
    if not (corr >= YARD_MIN_CORR):
        _say("  S087: VOID by its registered gate; no bets")
        return core.empty_bets()
    x = y[(y["yard_score"] >= YARD_FLOOR) & (y["yard_n"] >= YARD_MIN_N) & y["race_whole"].eq(True)]
    return _clv_back(x)


def s088(frames):
    """In-running near miss last time, at the BSP: the horse's previous run was within 90 days, in the same code,
    not won, and its win price row had a BSP of 6.0 or more and traded at 2.0 or under in running (ipmin above 1);
    GBP20 at today's BSP (at most 1% of pp_vol)."""
    r, p = _sp_runs(frames)
    gap = (r["date"] - p["prev_date"]).dt.days
    x = r[gap.between(1, 90) & (p["prev_code"] == r["code"]) & (p["prev_placing_numerical"] != 1)
          & (p["prev_win_lose"] == 0) & (p["prev_bsp"] >= 6.0) & (p["prev_ipmin"] > 1.0) & (p["prev_ipmin"] <= 2.0)]
    m = x[(x["morningwap"] > 1.01) & (x["morning_vol"] >= 50)]
    _say(f"  S088: {len(x):,} runners; A/E at the BSP {_ae(x['win_lose'], x['bsp']):.3f}, against the morning "
         f"price {_ae(m['win_lose'], m['morningwap']):.3f} ({len(m):,} with GBP50+ matched in the morning)")
    return _bets(x, "back", x["bsp"], stake=_cap(20.0, x["pp_vol"], 0.01), won=x["win_lose"])


def s089(frames):
    """Last race of the meeting, the morning favourite at the BSP: the latest off of a (country, menu_hint,
    race_date) in the win rows; 5+ selections, every one with morningwap above 1.01; the single lowest morningwap,
    required <= 5.0; GBP20 at the BSP (at most 1% of pp_vol). Choices: two markets sharing the latest off, or two
    runners sharing the lowest morning price, are no bet; a favourite with no BSP is no bet."""
    w = frames["win"]
    meet = w["country"] + "|" + w["menu_hint"] + "|" + w["race_date"]
    last = w["event_ts"].eq(w["event_ts"].groupby(meet).transform("max"))
    x = w[last].assign(meet=meet[last].to_numpy())
    x = x[x.groupby("meet")["event_id"].transform("nunique").eq(1)]
    priced = x["morningwap"] > 1.01
    x = x[(x["n_sel"] >= 5) & priced.groupby(x["event_id"]).transform("all")]
    low = x.groupby("event_id")["morningwap"].transform("min")
    fav = x["morningwap"].eq(low)
    x = x[fav & fav.groupby(x["event_id"]).transform("sum").eq(1) & (x["morningwap"] <= 5.0) & x["bsp"].notna()]
    return _bets(x, "back", x["bsp"], stake=_cap(20.0, x["pp_vol"], 0.01), won=x["win_lose"])


def s090(frames):
    """One ride at the meeting, backed in the morning and laid at the BSP: the jockey has exactly one ride at that
    track that day (race_results rows), no claim, and 200+ rides with a win rate of 12% or more in the 365 days
    before; morning_vol above 0; races where no selection lacks a BSP; to win GBP50 (at most GBP100, 10% of
    morning_vol). Choices: a missing jockeys_claim is read as no claim; rides are counted from the results (a ride
    lost to a non-runner is not seen)."""
    r = frames["runs"]
    jk = _name(r["jockey_name"])
    day = _day(r["race_date"])
    rides = (jk + "|" + r["track"].str.lower() + "|" + r["race_date"]).to_frame("g").groupby("g")["g"].transform("size")
    past = _window(jk, day, {"n": np.ones(len(r)), "w": (r["placing_numerical"] == 1).astype(float)}, jk, day, 365)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = past["w"] / past["n"]
    ok = ((jk != "") & rides.eq(1) & r["jockeys_claim"].fillna(0).eq(0) & (past["n"] >= 200) & (rate >= 0.12)
          & (r["morningwap"] > 1.01) & (r["morning_vol"] > 0) & r["bsp"].notna() & r["race_whole"].eq(True))
    x = r[ok]
    b = _clv_back(x)
    stake = _cap(core.to_win_stake(50.0, x["bsp"].to_numpy(float), 0.0, 100.0), x["pp_vol"], 0.01)
    _report("S090", _bets(x, "back", x["bsp"], stake=stake, won=x["win_lose"]), extra="the held version at the BSP")
    return b


def s091(frames):
    """Win-market tail lay at the BSP: win rows with morningwap >= 80.0, morning_vol above 0 and pp_vol >= 200; a
    MARKET_ON_CLOSE lay, liability GBP10, to the result."""
    w = frames["win"]
    x = w[(w["morningwap"] >= 80.0) & (w["morning_vol"] > 0) & (w["pp_vol"] >= 200) & w["bsp"].notna()]
    b = _bets(x, "lay", x["bsp"], liability=10.0, won=x["win_lose"])
    _report("S091", b)
    return b


# ---------------------------------------------------------------------------------------------------------------
# in running (S092-S100): K8
# ---------------------------------------------------------------------------------------------------------------

def lay_target(bsp, f: float) -> np.ndarray:
    """L(f) = 1 + (bsp - 1) x f."""
    return 1.0 + (np.asarray(bsp, float) - 1.0) * f


def back_target(bsp, m: float) -> np.ndarray:
    """H(m) = 1 + (bsp - 1) x m."""
    return 1.0 + (np.asarray(bsp, float) - 1.0) * m


def lay_filled(ipmin, target) -> np.ndarray:
    """K8: a resting lay at ``target`` fills only if the in-play low is above 1 and at or under one tick below it."""
    ipmin = np.asarray(ipmin, float)
    with np.errstate(invalid="ignore"):
        return (ipmin > 1.0) & (ipmin <= np.asarray(core.tick_under(np.asarray(target, float)), float))


def back_filled(ipmax, target) -> np.ndarray:
    """K8: a resting back at ``target`` fills only if the in-play high is at or over one tick above it."""
    with np.errstate(invalid="ignore"):
        return np.asarray(ipmax, float) >= np.asarray(core.tick_over(np.asarray(target, float)), float)


def inrun_flags(x: pd.DataFrame) -> pd.DataFrame:
    """dob (traded at half the SP odds less a tick), d15 (traded at 1.5x the SP odds plus a tick), short2 (beaten
    after trading at 2.0 or under), won, lr = ln(ipmin / bsp): one run's in-running record (K8)."""
    bsp, lo = x["bsp"].to_numpy(float), x["ipmin"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({
            "dob": lay_filled(lo, lay_target(bsp, 0.5)).astype(float),
            "d15": back_filled(x["ipmax"], back_target(bsp, 1.5)).astype(float),
            "short2": ((x["win_lose"].to_numpy(float) == 0) & (lo > 1.0) & (lo <= 2.0)).astype(float),
            "won": (x["win_lose"].to_numpy(float) == 1).astype(float),
            "lr": np.where(lo > 1.0, np.log(lo / bsp), np.nan)}, index=x.index)


LAGS = 10


def _history(frames) -> pd.DataFrame:
    """Each recorded run's (a win row with a BSP) in-running record over the horse's EARLIER recorded runs, by
    horse_norm in off order, indexed by bp_id:
        n8, dob8, short2_n8, wins_n8        the last 8 within 730 days: count, dob share, short2 count, wins
        n10, d15_10, frame_n10, frame_d15   the last 10 within 730 days: count, d15 share, runs finishing 1st-3rd
                                            (race_results placing; an unjoined run is not counted), those with d15
        h_n, h_dob3, h_dob8, h_dob_car, h_lr3, h_lr8, h_short2_8, h_d15_8   no day limit (S092's features)"""
    def build():
        w = frames["win"]
        x = w.loc[w["bsp"].notna() & w["horse_norm"].ne(""),
                  ["bp_id", "horse_norm", "race_date", "event_ts", "bsp", "ipmin", "ipmax", "win_lose", "placing"]]
        x = x.sort_values(["horse_norm", "event_ts", "bp_id"], kind="stable")
        f = inrun_flags(x)
        f["frame"] = (x["placing"] <= 3).astype(float)
        f["frame_d15"] = f["frame"] * f["d15"]
        f["day"] = _day(x["race_date"]).astype(float)
        g = f.groupby(x["horse_norm"].to_numpy(), sort=False)
        day = f["day"].to_numpy()
        n = len(x)
        acc = {k: np.zeros(n) for k in ("n8", "dob8s", "short2_n8", "wins_n8", "n10", "d15_10s", "frame_n10", "frame_d15",
                                "c3", "c8", "dob3s", "dob8a", "lr3s", "lr3n", "lr8s", "lr8n", "sh8", "d158")}
        for k in range(1, LAGS + 1):
            s = g.shift(k)
            have = s["day"].notna().to_numpy()
            near = have & ((day - np.nan_to_num(s["day"].to_numpy(), nan=-1e9)) <= 730)
            v = {c: np.nan_to_num(s[c].to_numpy(float)) for c in ("dob", "d15", "short2", "won", "frame", "frame_d15")}
            lr = s["lr"].to_numpy(float)
            acc["n10"] += near
            acc["d15_10s"] += near * v["d15"]
            acc["frame_n10"] += near * v["frame"]
            acc["frame_d15"] += near * v["frame_d15"]
            if k <= 8:
                acc["n8"] += near
                acc["dob8s"] += near * v["dob"]
                acc["short2_n8"] += near * v["short2"]
                acc["wins_n8"] += near * v["won"]
                acc["c8"] += have
                acc["dob8a"] += have * v["dob"]
                acc["sh8"] += have * v["short2"]
                acc["d158"] += have * v["d15"]
                acc["lr8s"] += np.nan_to_num(lr)
                acc["lr8n"] += np.isfinite(lr)
            if k <= 3:
                acc["c3"] += have
                acc["dob3s"] += have * v["dob"]
                acc["lr3s"] += np.nan_to_num(lr)
                acc["lr3n"] += np.isfinite(lr)
        car_n = g.cumcount().to_numpy(float)
        car_dob = g["dob"].cumsum().to_numpy(float) - f["dob"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = pd.DataFrame({
                "n8": acc["n8"], "dob8": acc["dob8s"] / acc["n8"], "short2_n8": acc["short2_n8"],
                "wins_n8": acc["wins_n8"], "n10": acc["n10"], "d15_10": acc["d15_10s"] / acc["n10"],
                "frame_n10": acc["frame_n10"], "frame_d15": acc["frame_d15"],
                "h_n": car_n, "h_dob3": acc["dob3s"] / acc["c3"], "h_dob8": acc["dob8a"] / acc["c8"],
                "h_dob_car": car_dob / car_n, "h_lr3": acc["lr3s"] / acc["lr3n"], "h_lr8": acc["lr8s"] / acc["lr8n"],
                "h_short2_8": acc["sh8"] / acc["c8"], "h_d15_8": acc["d158"] / acc["c8"]},
                index=pd.Index(x["bp_id"].to_numpy(), name="bp_id"))
        return out
    return _cached("history", frames["win"], build)


def _win_with_history(frames) -> pd.DataFrame:
    w = frames["win"]
    w = w[w["bsp"].notna()]
    return w.join(_history(frames), on="bp_id")


def _back_to_lay(x: pd.DataFrame, f: float, stake: float = 20.0) -> pd.DataFrame:
    """Back at the BSP (at most 0.5% of pp_vol, nothing under GBP2) with a lay of stake x bsp / L resting at L(f):
    filled (K8), the trade is closed at L on every outcome; unfilled, the back stands to the result."""
    target = lay_target(x["bsp"], f)
    hit = lay_filled(x["ipmin"], target)
    return _bets(x, "back", x["bsp"], stake=_cap(stake, x["pp_vol"], 0.005), exit=np.where(hit, "inplay", "result"),
                 exit_price=np.where(hit, target, np.nan), won=x["win_lose"])


def _lay_to_back(x: pd.DataFrame, m: float, liability: float = 40.0) -> pd.DataFrame:
    """Lay at the BSP (size min(liability / (bsp - 1), 0.5% of pp_vol), nothing under GBP2) with a back of
    size x bsp / H resting at H(m): filled (K8), closed at H on every outcome; unfilled, the lay stands."""
    bsp = x["bsp"].to_numpy(float)
    target = back_target(bsp, m)
    hit = back_filled(x["ipmax"], target)
    return _bets(x, "lay", bsp, stake=_cap(liability / (bsp - 1.0), x["pp_vol"], 0.005),
                 exit=np.where(hit, "inplay", "result"), exit_price=np.where(hit, target, np.nan), won=x["win_lose"])


MISTAKE_RX = (r"\b(?:mistakes?|blunder(?:ed)?|not fluent|less than fluent|pecked|clouted"
              r"|jumped (?:left|right|slowly|awkwardly|poorly|badly|deliberately|sketchily))\b|\bhit\s+\d")


def _comment_shares(frames) -> pd.DataFrame:
    """From EARLIER days' comments, row for row with frames['runs']:
        slow8        share of the horse's last 8 runs with a comment that were slow starts (comment_features)
        chase_n, chase_err   the horse's last 8 chase runs with a comment before today: how many, and the share
                             whose comment matches jumps_form.FALL_RX or the mistake list (S099)"""
    def build():
        from model.blocks.jumps_form import FALL_RX
        from model.comment_features import COMMENT_CLASSES
        r = frames["runs"]
        c = r["comment"].str.lower()
        has = c.str.strip() != ""
        horse = r["horse_name"].str.lower().str.strip()
        d = pd.DataFrame({"horse": horse.to_numpy(), "race_date": r["race_date"].to_numpy()}, index=r.index)
        d["slow"] = c.str.contains(COMMENT_CLASSES["slow_start"], regex=True).astype(float).where(has)
        chase = (r["code"] == "chase") & has
        err = c.str.contains(FALL_RX, regex=True) | c.str.contains(MISTAKE_RX, regex=True)
        d["err"] = err.astype(float).where(chase)
        _say(f"  comments: the mistake-or-fall list matches {float(err[chase].mean()) if chase.any() else float('nan'):.1%} "
             f"of {int(chase.sum()):,} chase comments; the slow-start list {float(d['slow'].mean()):.1%} of "
             f"{int(has.sum()):,} comments")
        d["day"] = _day(d["race_date"])
        d["row"] = np.arange(len(d))
        d = d.sort_values(["day", "row"], kind="stable")
        out = pd.DataFrame(index=r.index)
        for col, name in (("slow", "slow8"), ("err", "chase_err")):
            sub = d[d[col].notna()]
            roll = sub.groupby("horse", sort=False)[col].rolling(8, min_periods=1)
            state = pd.DataFrame({"m": roll.mean().droplevel(0), "n": roll.count().droplevel(0)}).reindex(sub.index)
            state = pd.concat([sub[["horse", "day"]], state], axis=1).drop_duplicates(["horse", "day"], keep="last")
            # the state after the horse's latest such run on a STRICTLY EARLIER day
            got = pd.merge_asof(d[["horse", "day", "row"]], state, on="day", by="horse", allow_exact_matches=False)
            got = got.sort_values("row", kind="stable")
            out[name] = got["m"].to_numpy(float)
            if col == "err":
                out["chase_n"] = got["n"].fillna(0.0).to_numpy(float)
        return out
    return _cached("comment_shares", frames["runs"], build)


TOUCH_FROM, TOUCH_FIRST_BET, MIN_TRAIN, MIN_CAL = 2019, 2021, 200, 50
LGB_PARAMS = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 500,
              "colsample_bytree": 0.8, "random_state": SEED, "verbose": -1}
TOUCH_FEATS = ["f_logbsp", "f_bsp_rank", "number_of_runners", "code_i", "dist_furlongs", "going_i", "track_i", "cls",
               "p_lead", "p_prom", "p_rear", "lead_rank", "shape_bin", "style_n_eff", "stall_frac", "h_dob3",
               "h_dob8", "h_dob_car", "h_n", "h_lr3", "h_lr8", "h_short2_8", "h_d15_8", "jk_dob", "tr_dob",
               "f_log_bsp_am", "days_since_lr", "career_runs"]
NOTOUCH_FEATS = TOUCH_FEATS + ["slow8", "chase_err", "jk_d15w"]
TOUCH_CATS = ["code_i", "going_i", "track_i"]


def _touch_frame(frames) -> pd.DataFrame:
    """The learned models' rows: runs with a win BSP from 2019, their features (every one from the card, earlier
    days, or today's BSP and morning price, which the SP order is sized on) and today's outcomes dob, d15, won."""
    def build():
        from model.connections import going_group
        r, w = frames["runs"], frames["win"]
        if len(r) and not r["p_lead"].notna().any():
            raise RuntimeError("the run-style columns were not built by the loader (see its log)")
        ok = r["bsp"].notna() & r["bp_id"].notna()
        x = r[ok].copy()
        x["bp_id"] = x["bp_id"].astype(np.int64)
        x = x.join(_history(frames), on="bp_id").join(_comment_shares(frames)[ok.to_numpy()])
        fl = inrun_flags(x)
        x["dob"], x["d15"], x["won"] = fl["dob"], fl["d15"], fl["won"]
        day = _day(x["race_date"])
        one = np.ones(len(x))
        for key, tag in ((_name(x["jockey_name"]), "jk"), (_name(x["trainer"]), "tr")):
            s = _window(key, day, {"n": one, "dob": x["dob"], "w": x["won"], "wd": x["won"] * x["d15"]}, key, day, 730)
            with np.errstate(invalid="ignore", divide="ignore"):
                x[f"{tag}_dob"] = np.where(s["n"] > 0, s["dob"] / s["n"], np.nan)
                if tag == "jk":
                    x["jk_d15w"] = np.where(s["w"] > 0, s["wd"] / s["w"], np.nan)
        rank = w.loc[w["bsp"].notna()].groupby("event_id")["bsp"].rank(method="min")
        x["f_bsp_rank"] = x["bp_id"].map(pd.Series(rank.to_numpy(), index=w.loc[rank.index, "bp_id"].to_numpy()))
        x["f_logbsp"] = np.log(x["bsp"])
        with np.errstate(invalid="ignore", divide="ignore"):
            x["f_log_bsp_am"] = np.log(x["bsp"] / x["morningwap"])
        x["code_i"] = x["code"].map({c: i for i, c in enumerate(CODES)}).astype(float)
        x["going_i"] = going_group(x["going_description"]).map({"firm": 0, "good": 1, "soft": 2}).astype(float)
        x["track_i"] = pd.factorize(x["track"].str.lower().str.strip())[0].astype(float)
        x["cls"] = pd.to_numeric(x["race_class"].astype(str).str.extract(r"(\d)")[0], errors="coerce")
        flat = x["code"].isin(["flat", "aw"]) & (x["stall"] > 0) & (x["number_of_runners"] > 0)
        x["stall_frac"] = (x["stall"] / x["number_of_runners"]).where(flat)
        return x[x["year"] >= TOUCH_FROM]
    return _cached("touch_frame", frames["runs"], build)


def _walk_forward(x: pd.DataFrame, feats: list, y: np.ndarray, n_class: int, tag: str) -> np.ndarray:
    """For each bet year Y from 2021: LightGBM on the rows of 2019..Y-2, isotonic calibration (one a class,
    renormalised) on Y-1, the calibrated chances on Y. (rows, classes); NaN where no model could be fitted."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    out = np.full((len(x), n_class), np.nan)
    if x.empty:
        return out
    year = x["year"].to_numpy(int)
    data = x[feats].to_numpy(float)
    cats = [feats.index(c) for c in TOUCH_CATS if c in feats]
    for yr in range(TOUCH_FIRST_BET, int(year.max()) + 1):
        tr, ca, te = (year >= TOUCH_FROM) & (year <= yr - 2), year == yr - 1, year == yr
        if tr.sum() < MIN_TRAIN or ca.sum() < MIN_CAL or not te.any() or len(np.unique(y[tr])) < n_class:
            _say(f"  {tag} {yr}: not fitted ({int(tr.sum())} training rows, {int(ca.sum())} to calibrate on)")
            continue
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(data[tr], y[tr], categorical_feature=cats)
        raw_ca, raw_te = m.predict_proba(data[ca]), m.predict_proba(data[te])
        cal = np.empty((int(te.sum()), n_class))
        for c in range(n_class):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_ca[:, c], (y[ca] == c))
            cal[:, c] = iso.predict(raw_te[:, c])
        if n_class == 2:                                       # one isotonic fit: the chance of the event
            cal[:, 0] = 1.0 - cal[:, 1]
        tot = cal.sum(axis=1, keepdims=True)
        out[te] = np.where(tot > 0, cal / np.where(tot > 0, tot, 1.0), np.nan)
        _say(f"  {tag} {yr}: fitted on {int(tr.sum()):,} rows of {TOUCH_FROM}-{yr - 2}, calibrated on "
             f"{int(ca.sum()):,} of {yr - 1}, scored {int(te.sum()):,}")
    return out


def s092(frames):
    """Learned touch model, back at the SP and lay at half the SP odds: win rows with a BSP of 2.5-20; for each bet
    year Y from 2021 a LightGBM binary model of dob (K8) on 2019..Y-2, isotonic on Y-1, scored on Y; bet when
    p x (bsp / L - 1) x 0.98 - (1 - p) >= 0.05, L = L(0.5) (an unmatched back is counted lost in the test);
    back min(GBP20, 0.5% of pp_vol), a lay resting at L. Choices: the model is fitted inside the 2.5-20 band; the
    horse's short2 and d15 shares are over its last 8 recorded runs; the jockey's and trainer's dob rates are over
    all their priced runners of the 730 days before."""
    t = _touch_frame(frames)
    x = t[t["bsp"].between(2.5, 20.0)]
    p = _walk_forward(x, TOUCH_FEATS, x["dob"].to_numpy(int), 2, "S092")[:, 1] if len(x) else np.array([])
    bsp = x["bsp"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        ev = p * (bsp / lay_target(bsp, 0.5) - 1.0) * 0.98 - (1.0 - p)
        pick = ev >= 0.05
    _say(f"  S092: {int(np.isfinite(p).sum()):,} runners scored, {int(pick.sum()):,} at an expected value of 5%+")
    return _back_to_lay(x[pick], 0.5)


def s093(frames):
    """The horse's own half-price habit, back-to-lay: of its last 8 runs with a win price record within 730 days
    (at least 6), 62.5% or more traded at half the SP odds (dob); today's BSP 3-15; back min(GBP20, 0.5% of
    pp_vol), a lay resting at L(0.5)."""
    x = _win_with_history(frames)
    x = x[(x["n8"] >= 6) & (x["dob8"] >= 0.625) & x["bsp"].between(3.0, 15.0)]
    return _back_to_lay(x, 0.5)


def s094(frames):
    """Learned no-touch model, lay at the SP and back at 1.5x the SP odds: win rows with a BSP of 2-12; for each bet
    year Y from 2021 a LightGBM 3-class model (matched = d15; unmatched and won; unmatched and lost) on 2019..Y-2,
    an isotonic fit a class on Y-1, renormalised, scored on Y; bet when
    [P_m x (1 - bsp / H) x 0.98 - P_uw x (bsp - 1) + P_ul x 0.98] / (bsp - 1) >= 0.03, H = H(1.5); lay size
    min(GBP40 / (bsp - 1), 0.5% of pp_vol), a back resting at H. Features: S092's, the horse's share of slow starts
    over its last 8 commented runs, its share of mistakes or falls over its last 8 commented chases (S099's flag),
    and its jockey's share of winners that traded at H first, over the 730 days before."""
    t = _touch_frame(frames)
    x = t[t["bsp"].between(2.0, 12.0)]
    y = np.where(x["d15"].to_numpy() == 1, 0, np.where(x["won"].to_numpy() == 1, 1, 2)).astype(int)
    pr = _walk_forward(x, NOTOUCH_FEATS, y, 3, "S094") if len(x) else np.empty((0, 3))
    bsp = x["bsp"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        ev = (pr[:, 0] * (1.0 - bsp / back_target(bsp, 1.5)) * 0.98 - pr[:, 1] * (bsp - 1.0) + pr[:, 2] * 0.98) / (bsp - 1.0)
        pick = ev >= 0.03
    _say(f"  S094: {int(np.isfinite(pr[:, 0]).sum()):,} runners scored, {int(pick.sum()):,} at an expected value "
         "of 3%+ of liability")
    return _lay_to_back(x[pick], 1.5)


def s095(frames):
    """Grinders, lay-to-back: of the horse's last 10 recorded runs within 730 days (at least 6), 80% or more traded
    at 1.5x the SP odds (d15), and at least 3 finished 1st-3rd of which 80% or more have d15; today's BSP 2.5-10;
    lay size min(GBP40 / (bsp - 1), 0.5% of pp_vol), a back resting at H(1.5). Choice: a past run not joined to
    race_results has no placing and is not counted among the 1st-3rd."""
    x = _win_with_history(frames)
    with np.errstate(invalid="ignore", divide="ignore"):
        framed = x["frame_d15"] / x["frame_n10"]
    x = x[(x["n10"] >= 6) & (x["d15_10"] >= 0.80) & (x["frame_n10"] >= 3) & (framed >= 0.80)
          & x["bsp"].between(2.5, 10.0)]
    return _lay_to_back(x, 1.5)


def _styled(frames) -> pd.DataFrame:
    r = frames["runs"]
    if len(r) and not r["p_lead"].notna().any():
        raise RuntimeError("the run-style columns were not built by the loader (see its log)")
    return r[r["bsp"].notna() & r["bp_id"].notna() & r["code"].isin(["flat", "aw"])]


def s096(frames):
    """Lone projected leader on the Flat, back-to-lay: Flat or all-weather, 8 furlongs or under, 5-12 runners, one
    natural leader (shape_bin 1), the runner most likely to lead (lead_rank 1, ties by seed 7) with p_lead >= 0.45
    and style_n_eff >= 4, BSP 2.5-12; back min(GBP20, 0.5% of pp_vol), a lay resting at L(0.6)."""
    x = _styled(frames)
    x = x[(x["dist_furlongs"] <= 8) & x["number_of_runners"].between(5, 12) & (x["shape_bin"] == 1)
          & (x["lead_rank"] == 1) & (x["p_lead"] >= 0.45) & (x["style_n_eff"] >= 4)]
    x = x.assign(race=_rr_race(x).to_numpy(), tie=np.random.default_rng(SEED).random(len(x)))
    x = x.sort_values(["race", "tie"], kind="stable").drop_duplicates("race", keep="first")
    return _back_to_lay(x[x["bsp"].between(2.5, 12.0)], 0.6)


JOCKEY_MIN_WINNERS = 40


def _patient(frames) -> pd.DataFrame:
    """Runs with a BSP whose jockey, on the day, is in the top fifth by the share of its winners (40 or more with a
    price record in the 730 days before) that traded at 1.5x the SP odds before winning (d15), among every jockey
    with 40+ such winners as of that day."""
    def build():
        r = frames["runs"]
        x = r[r["bsp"].notna() & r["bp_id"].notna()]
        jk = _name(x["jockey_name"])
        fl = inrun_flags(x)
        day = _day(x["race_date"])
        wins = pd.DataFrame({"jk": jk.to_numpy(), "day": day, "w": fl["won"].to_numpy(),
                             "wd": (fl["won"] * fl["d15"]).to_numpy()})
        wins = wins[(wins["w"] == 1) & (wins["jk"] != "")]
        tot = wins.groupby("jk")["w"].sum()
        names = tot[tot >= JOCKEY_MIN_WINNERS].index                    # no other jockey can ever have 40 in a window
        _say(f"  S097: {jk[jk != ''].nunique():,} jockey names with a priced ride, {len(names):,} with "
             f"{JOCKEY_MIN_WINNERS}+ priced winners in all")
        if len(names) == 0 or x.empty:
            return x.iloc[:0]
        d0, d1 = int(day.min()), int(day.max())
        pos = {n: i for i, n in enumerate(names)}
        wins = wins[wins["jk"].isin(names)]
        ji = wins["jk"].map(pos).to_numpy(int)
        cw = np.zeros((len(names), d1 - d0 + 2))
        cd = np.zeros_like(cw)
        np.add.at(cw, (ji, wins["day"].to_numpy(int) - d0 + 1), 1.0)
        np.add.at(cd, (ji, wins["day"].to_numpy(int) - d0 + 1), wins["wd"].to_numpy(float))
        cw, cd = cw.cumsum(axis=1), cd.cumsum(axis=1)                   # column c: through day d0 + c - 1
        top = np.zeros(len(x), bool)
        jcode = jk.map(pos).to_numpy(float)
        for d in np.unique(day):
            hi, lo = d - d0, max(d - d0 - 730, 0)                       # days d - 730 .. d - 1
            w = cw[:, hi] - cw[:, lo]
            ok = w >= JOCKEY_MIN_WINNERS
            if not ok.any():
                continue
            share = np.where(ok, (cd[:, hi] - cd[:, lo]) / np.where(ok, w, 1.0), np.nan)
            cut = np.nanquantile(share, 0.80)
            rows = np.flatnonzero((day == d) & np.isfinite(jcode))
            j = jcode[rows].astype(int)
            top[rows] = ok[j] & (share[j] >= cut)
        return x[top]
    return _cached("patient", frames["runs"], build)


def s097(frames):
    """Patient riders, lay-to-back: the jockey has 40+ winners with a price record in the 730 days before and its
    share of them with d15 is in the top fifth (at or above the 80th percentile) of the jockeys with 40+ winners
    as of the day; the mount's BSP 2.5-10; lay size min(GBP40 / (bsp - 1), 0.5% of pp_vol), a back resting at
    H(1.5). Names are lower-cased with punctuation stripped; claims and spellings are not merged."""
    x = _patient(frames)
    return _lay_to_back(x[x["bsp"].between(2.5, 10.0)], 1.5)


def s098(frames):
    """Closers behind a contested pace, lay-to-back: Flat or all-weather, 7 furlongs or more, 10+ runners, a
    contested lead (shape_bin 2), p_rear >= 0.50, style_n_eff >= 4, BSP 3-12; lay size min(GBP40 / (bsp - 1), 0.5%
    of pp_vol), a back resting at H(1.5)."""
    x = _styled(frames)
    x = x[(x["dist_furlongs"] >= 7) & (x["number_of_runners"] >= 10) & (x["shape_bin"] == 2) & (x["p_rear"] >= 0.50)
          & (x["style_n_eff"] >= 4) & x["bsp"].between(3.0, 12.0)]
    return _lay_to_back(x, 1.5)


def s099(frames):
    """Sloppy jumpers over fences, lay-to-back: a chase; of the horse's last 8 chase runs with a comment (at least
    4, earlier days) half or more match jumps_form.FALL_RX or the mistake list; BSP 2-10; lay size
    min(GBP40 / (bsp - 1), 0.5% of pp_vol), a back resting at H(1.5). The word list is fixed here, unvalidated."""
    r = frames["runs"]
    cs = _comment_shares(frames)
    ok = (r["bsp"].notna() & r["bp_id"].notna() & (r["code"] == "chase") & (cs["chase_n"] >= 4)
          & (cs["chase_err"] >= 0.50) & r["bsp"].between(2.0, 10.0))
    return _lay_to_back(r[ok], 1.5)


def s100(frames):
    """Flatterers, a resting lay at 2.0: of the horse's last 8 recorded runs within 730 days (at least 6), 3 or more
    were beaten after trading at 2.0 or under (short2) and at most 1 was won; today's BSP 4-15. No SP bet: a lay of
    GBP20 at 2.0 kept in running, matched only if 1 < ipmin <= 1.99, held to the result. A market that jumps
    straight through 2.0 is counted filled at 2.0 (the adverse case); an unmatched order is no bet (a void row)."""
    x = _win_with_history(frames)
    x = x[(x["n8"] >= 6) & (x["short2_n8"] >= 3) & (x["wins_n8"] <= 1) & x["bsp"].between(4.0, 15.0)]
    hit = lay_filled(x["ipmin"], 2.0)
    f = x[hit]
    _say(f"  S100: {len(x):,} orders, {len(f):,} matched; the matched won "
         f"{float((f['win_lose'] == 1).mean()) if len(f) else float('nan'):.3f}")
    return _bets(x, "lay", 2.0, stake=20.0, won=x["win_lose"], entry_type="inplay", void=~hit)


# ---------------------------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------------------------

def _s(sid, name, family, side, market, build, **kw) -> core.Strategy:
    return core.Strategy(id=sid, name=name, family=family, dataset=DS, side=side, market=market, build=build,
                         entry=kw.pop("entry", "sp"), **kw)


STRATEGIES = [
    _s("S069", "Place pocket by limit-on-close back (win ppwap <= 5, 5% value)", "place v win: limit on close",
       "back", "place", s069, blind=False,
       tested_before="place-pocket-gate (fails at 5% commission); place-pocket-gate-2pc (passes at 2%: +1.72% "
                     "2021-22, +2.70% 2023-26Q1); place-rules-preplay-1002 (market on close after a pre-play "
                     "decision: -2.1 to +2.4% by year). The limit form (edge_investigation_1111 item 2): not run",
       note="ppwap includes trades up to the off; the fill and the order's own effect on the place SP are not in "
            "the files"),
    _s("S070", "Place limit on close at mid prices (win 5-21, 10% value)", "place v win: limit on close",
       "back", "place", s070,
       tested_before="edge-bsp-bias-1002 (14 rules chosen on 2010-17, 12 with intervals above zero on 2018-26Q1); "
                     "place-rules-preplay-1002 (5-8 EV>0.10 +10.5% then -0.6%). A limit version: none found",
       note="at long prices a limit fills on the late drifters; the ppwap look-ahead partly hides it"),
    _s("S071", "Stacked place model (Harville and Henery), limit on close", "place probability model",
       "back", "place", s071, select_from="2021-01-01",
       tested_before="place-market-vs-win (log loss: place market 0.49915, win-implied 0.50029, combination "
                     "0.49908); syndicate_papers item 7. A Henery or Stern fit: none found",
       note="three-year rolling fit, first bets 2021 (SELECT read 2021-2023); Harville exact, Henery's means by "
            "integration"),
    _s("S072", "Bad each-way race: second and third in the betting at the place SP", "place: race shape",
       "back", "place", s072, paired="BASE-S072",
       tested_before="this race shape: none found. edge-bsp-bias-1002 sp_segments (no back cell by band, group and "
                     "field held); discovery_1111 idea 56 'partly held', not tested",
       note="ppwap stands in for the price minutes out"),
    _s("BASE-S072", "Benchmark of S072: every runner at 3.5-10.0 in 8-12 runner, 3-place races at the place SP",
       "place: race shape", "back", "place", base_s072, benchmark=True),
    _s("S073", "Extra-place proxy: Saturday big-field handicaps at the place SP", "place: race shape",
       "back", "place", s073, paired="BASE-S073",
       tested_before="discovery_1111 idea 56: not tested. edge-bsp-bias-1002 (no weekday or handicap-by-day cell)",
       note="which races carried an extra-place offer is not held: a proxy"),
    _s("BASE-S073", "Benchmark of S073: the same band in 16+ runner, 4-place handicaps on other days",
       "place: race shape", "back", "place", base_s073, benchmark=True),
    _s("S074", "Places kept in a reduced field: outsiders at the place SP", "place: terms",
       "back", "place", s074, paired="BASE-S074",
       tested_before="none found. place-market-vs-win (calibrated overall); edge-bsp-bias-1002 (no cut by places "
                     "against the field)",
       note="that the places were kept after a withdrawal is inferred from the runner count; a probe"),
    _s("BASE-S074", "Benchmark of S074: the same ranks and prices in races on standard place terms",
       "place: terms", "back", "place", base_s074, benchmark=True),
    _s("S075", "Win-market steamer backed at the place SP", "place: win move as signal",
       "back", "place", s075, paired="BASE-S075",
       tested_before="no direct test. market-only-drift-1002; place-model-stage3-1003 (the win rule's horses' place "
                     "price shortens with the win price)",
       note="ppwap is complete only at the off"),
    _s("BASE-S075", "Benchmark of S075: every runner at win ppwap 2.0-15.0 at the place SP",
       "place: win move as signal", "back", "place", base_s075, benchmark=True),
    _s("S076", "Long-shot place lay by limit on close (win ppwap >= 50)", "place v win: limit on close",
       "lay", "place", s076, blind=False,
       tested_before="edge-bsp-bias-1002 (+9.4 to +13.5% a GBP1 of backer's stake, under 1% of liability); "
                     "place-rules-preplay-1002 (decided pre-play: +4.9 to +9.5% a GBP1 laid, +0.15 to +0.28% of "
                     "liability); place-t15-forward-1001 (only the SP is tradable). The limit version: none found",
       note="SP-lay capacity on an outsider's place market is not in the files"),
    _s("S077", "Predicted drifters: lay the steamer model's bottom tenth in the morning", "steamer model: lay side",
       "lay", "win", s077, entry="wap", select_from=core.FITTED_SELECT_FROM,
       tested_before="strat I (only the back side taken forward); morning-exit-bsp-1116. The bottom decile's drift "
                     "and a lay on it: not scored",
       note="model fitted 2018-2021, SELECT read 2022-2023; the WAP is not an offered lay price"),
    _s("S078", "Predicted drifters laid at the SP with a limit of 8.0, held", "steamer model: lay side",
       "lay", "win", s078, select_from=core.FITTED_SELECT_FROM,
       tested_before="strat I (the bottom decile at the BSP: no data found in the report); edge-bsp-bias-1002 (all "
                     "runners laid at the SP -0.2% of liability)",
       note="model fitted 2018-2021, SELECT read 2022-2023; a null is the likely outcome"),
    _s("S079", "Fade the steamer at the SP: limit-on-close lay at 80% of the morning price", "SP over-reaction",
       "lay", "win", s079,
       tested_before="steam-beyond-bsp ('if anything a slight over-reaction', Q1 2026); edge-bsp-bias-1002 (no lay "
                     "cell held, none defined by the move)",
       note="the entry is the BSP itself; no spread"),
    _s("S080", "Back the pre-off drifter at the BSP, held", "SP over-reaction", "back", "win", s080,
       tested_before="steam-beyond-bsp; morning-favourite-clv; strategy_research_2026-11-04 study G (no morning "
                     "band passes). Not run on the long window",
       note="ppwap is complete only at the off: live the drift would be read from the mid a minute out"),
    _s("S081", "Lay chasers sent back over hurdles, at the BSP", "over-bet segment", "lay", "win", s081,
       blind=False,
       tested_before="strat_screen_rules.csv: 2010-21 n 20,412 A/E 0.903; 2022-26 n 7,880 A/E 0.892 (p 0.004); "
                     "read as a back rule (failed), never as a lay",
       note="SP lay assumed filled"),
    _s("S082", "Back headgear removed, at the BSP", "segment back at the BSP", "back", "win", s082, blind=False,
       tested_before="strat_hypotheses_rules.csv H22: 2010-21 n 58,686 A/E 1.045 (p 0.002); 2022-26 n 24,454 A/E "
                     "1.053 (p 0.014), 3 of 5 years; failed the owner's gate on the train p"),
    _s("S083", "Back the first run for a new trainer, at the BSP", "segment back at the BSP", "back", "win", s083,
       blind=False,
       tested_before="strat_hypotheses_rules.csv H4: 2010-21 n 48,327 A/E 1.039; 2022-26 n 19,982 A/E 1.072 "
                     "(p 0.003), place A/E 0.979, 1 of 5 years both legs: failed the gate on the place leg"),
    _s("S084", "Bottom weight in chase handicaps, morning back held", "segment back in the morning",
       "back", "win", s084, entry="wap", blind=False,
       tested_before="strat_hypotheses_rules.csv H18b: 2010-21 A/E 1.044, morning 1.088; 2022-26 n 8,352 A/E "
                     "1.069, morning 1.116 (p < 0.001), 4 of 5 years; failed the gate on the train p",
       note="K6: only races where no selection lacks a BSP; no cuts"),
    _s("S085", "Stable second string, at the BSP", "stable intent", "back", "win", s085,
       tested_before="as a betting rule: none found. iter47-quick-stable (the stablemates block added nothing to "
                     "the price forecast)",
       note="stablemates are taken from the result (a declared non-runner is not seen)"),
    _s("S086", "Segment steam table: backed in the morning, laid at the BSP", "morning steam by segment",
       "back", "win", s086, entry="wap", select_from=core.FITTED_SELECT_FROM,
       tested_before="strat I; morning-exit-bsp-1116; open-close-horses-1101 (opening-book forecasts negative a "
                     "tick worse). A shrunk cell table by card facts: none found",
       note="table fitted once on 2018-2021, SELECT read 2022-2023; will largely rediscover the quick return"),
    _s("S087", "Gamble yards: the trainer's steam record with shrinkage", "trainer v market",
       "back", "win", s087, entry="wap",
       tested_before="strat I ('the gamble yard' alone failed the result gate; its CLV was not reported); strat H "
                     "(trainer A/E Spearman 0.09); phase4-segments-1119",
       note="void (no bets) if the trainer score's next-year correlation, read on SELECT, is under 0.2"),
    _s("S088", "In-running near miss last time, at the BSP", "hidden form", "back", "win", s088,
       tested_before="iter109-hrb-features, iter110-inrunning-served-blend (a forecast gain, not a bet); strat G H1 "
                     "hard luck (A/E 1.013 / 0.996 at the BSP, 0.957 in the morning)"),
    _s("S089", "Last race of the meeting: the morning favourite at the BSP", "calendar", "back", "win", s089,
       tested_before="strat G H24 evening A/E 1.000, Sunday 0.998 (all runners); bsp-calibration-1007 (favourites "
                     "calibrated overall). A last-race test: none found",
       note="the end-of-day effect is cited from memory (Ali 1977; Thaler and Ziemba 1988): unverified"),
    _s("S090", "One ride at the meeting: backed in the morning, laid at the BSP", "jockey intent",
       "back", "win", s090, entry="wap",
       tested_before="strat G H3 jockey upgrade (morning A/E 1.012 / 1.025, p 0.066); strat B. A single-ride "
                     "betting rule: none found",
       note="rides are counted from results"),
    _s("S091", "Win-market tail lay at the BSP (morningwap >= 80)", "favourite-longshot", "lay", "win", s091,
       tested_before="bsp-calibration-1007 (101+ returns -18.5% to backers); edge-bsp-bias-1002 (every runner laid "
                     "at the SP -0.2% of liability). A tail lay selected before the off: none found",
       note="pence a day expected: evaluated to close the question"),
    _s("S092", "Learned touch model: back at the SP, lay at half the SP odds", "in-running: back-to-lay",
       "both", "win", s092, select_from="2021-01-01",
       tested_before="edge-inplay-offsets-1002 (unconditional, every cell -14% to -39%; 37% matched against 55% "
                     "fair); hrb-features-1006, iter109/110 (a different target). A modelled touch: none found",
       note="walk-forward from 2021 (SELECT read 2021-2023); the BSP is known only at the off"),
    _s("S093", "The horse's own half-price habit, back-to-lay", "in-running: back-to-lay", "both", "win", s093,
       tested_before="edge-inplay-offsets-1002; hrb-features-1006 (ir_short adds nothing beside the BSP on the "
                     "win). The touch rate given the habit: none found"),
    _s("S094", "Learned no-touch model: lay at the SP, back at 1.5x the SP odds", "in-running: lay-to-back",
       "both", "win", s094, select_from="2021-01-01",
       tested_before="edge-inplay-offsets-1002 (lay-back -10% to -32% in every unconditional cell). A conditional "
                     "version: none found",
       note="walk-forward from 2021 (SELECT read 2021-2023); a beaten horse nearly always reaches H"),
    _s("S095", "Grinders: horses that run into the frame after trading long, lay-to-back",
       "in-running: lay-to-back", "both", "win", s095,
       tested_before="edge-inplay-offsets-1002; hrb-features-1006 (last run's in-running high adds nothing beside "
                     "the BSP on the win). As a lay-to-back selection: none found"),
    _s("S096", "Lone projected leader on the Flat, back-to-lay at 0.6 of the SP odds", "in-running: back-to-lay",
       "both", "win", s096,
       tested_before="Strategy J and leader-forecast (backing the projected leader at the BSP or morning price: A/E "
                     "0.95-1.04, no edge). The in-running exit: never tested"),
    _s("S097", "Patient riders, lay-to-back", "in-running: lay-to-back", "both", "win", s097,
       tested_before="none found (leader-forecast reads the jockey's share of leaders as an input; no in-running "
                     "test by rider)",
       note="claims and spellings of a jockey's name are not merged"),
    _s("S098", "Closers behind a contested pace, lay-to-back", "in-running: lay-to-back", "both", "win", s098,
       tested_before="edge-inplay-offsets-1002 (cut by field size, not run style); Strategy J (styles carry no edge "
                     "at the BSP). As a lay-to-back selection: none found"),
    _s("S099", "Sloppy jumpers over fences, lay-to-back", "in-running: lay-to-back", "both", "win", s099,
       tested_before="iter118 fx (-0.0006, not served); the jumps_form block (unresolved: discovery_1111 4.6); "
                     "chase-error-1120. An in-running trade on jumping record: none found",
       note="the mistake word list is fixed in the module and unvalidated"),
    _s("S100", "Flatterers: a resting lay at 2.0 on horses that keep trading odds-on and losing",
       "in-running: resting lay", "lay", "win", s100, entry="inplay",
       tested_before="hrb-features-1006 (A/E at the BSP by last in-running low 0.966-1.014: priced before the off). "
                     "The win rate after trading short, by the horse's record: none found",
       note="volume at 2.0 unknown; a jump straight through 2.0 is counted filled at 2.0"),
]
