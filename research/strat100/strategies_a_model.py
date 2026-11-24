"""The a_model strategies of the 100-strategy harness: S053-S068 (racing², conventions K1-K10, 24 Nov 2026).

Every strategy reads the frames of research/strat100/data_a.load_a_model() (the served blend gate_stn's out-of-sample
prices on the development window, on Betfair's price files) and returns a bets frame; research/strat100/core.py does
the settlement, the 2% commission, the worse-price prints, the bootstrap and the verdict. One strategy = one setting.
Research only: nothing here places a bet, and the locked holdout never reaches these frames (K1).

What a selection may read (the look-ahead rule, held by tests/test_strat100_a_model.py):
  a morning entry (S053-S067)   the model's price, the morning WAP and volume, the card, the closing model fitted on
                                earlier months, earlier days' results. Never ppwap, ppmax, ppmin, the BSP, the
                                result or the in-play columns of its own day
  the fill of an exit           the BSP (the close), ipmin through the target by a tick (K8), the result
  a pre-play entry (S068)       also the win ppwap and pp_vol (to the off: a look-ahead of minutes, as registered)
Two things read the close by registered convention and are flagged, not hidden: K6 keeps a race only if no selection
of its Betfair market lacks a BSP ('no cuts'), and K9 takes the places paid from the place rows' results.

Readings chosen where the specification left room (each also in its strategy's docstring):
  * S053's days are the days of the months the walk-forward closing model scores (from Feb 2025); January 2025 is
    only fitted on, so it is not a no-bet day. S061 starts a month later (its weights need a scored month).
  * 'ev' is race_book.expected_clv_walk_forward with the volume read as none (the refit arm of
    closing-refit-stn-1121, 22,950 bets there; the 18,992-bet figure quoted beside S053 is that query's served arm).
  * A half hedge (S062, S063) is the specification's: a lay at the BSP whose liability is half the back's winnings,
    size 0.5 s (p - 1) / (bsp - 1) (not the core's hedge_fraction 0.5, which halves the equal-profit lay).
  * S066's expected BSP is the same closing model's: exp_bsp = 1 / E[1 / BSP], so 1 - morningwap / exp_bsp = -ev.
  * S068's stake is GBP10 a fill, and at most 1% of the win market's pp_vol (K4's guide for an SP bet). Its
    benchmark S068-REST is stake-matched by day (a GBP-a-day pair between books of different sizes reads size).
  * No strategy here is blind (K10): the blend was chosen on a window that holds CONFIRM, and each rule's own
    earlier reading is named in its note. The best verdict is 'PASS (not blind)'.
"""
from __future__ import annotations

import weakref

import numpy as np
import pandas as pd

from research.strat100 import core
from research.strat100.data_a import PAIR_A, NoData

DS = "a_model"
BAR = 0.03                        # the live bar: expected CLV at least 3%
TO_WIN, FLOOR, CAP = 400.0, 15.0, 200.0     # the live staking: to win GBP400, GBP15 to GBP200 a horse
MAX_PRICE = 200.0
MIN_VOL = 100.0                   # GBP matched in the morning before the WAP is believed
SHARE = 0.10                      # K4: at most 10% of the morning volume
MAIN_CODES = ("flat", "aw", "hurdle", "chase")
WINTER = ("11", "12", "01", "02", "03")
DESCRIBE_DRAWS = 2000


def _say(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------------------------------------------

def _runners(frames) -> pd.DataFrame:
    r = frames.get("runners") if isinstance(frames, dict) else None
    if r is None or len(r) == 0:
        raise NoData("no data found: a_model holds no runner (the loader's frames are empty)")
    return r


def _meta(frames) -> dict:
    return frames.get("meta") or {}


def _need(r: pd.DataFrame, col: str, what: str) -> None:
    if col not in r.columns or not r[col].notna().any():
        raise NoData(f"no data found: {what} ({col} is empty in the loader's frames)")


def _scored_days(r: pd.DataFrame) -> list:
    """The race days of the months the walk-forward closing model scores (a month only fitted on is not read)."""
    return sorted(r.loc[r["ev"].notna(), "race_date"].astype(str).str[:10].unique())


def _all_days(frames) -> list:
    return sorted(str(d)[:10] for d in frames["days"])


def _picks(r: pd.DataFrame, ev_col: str = "ev", model_col: str = "predicted_bfsp", bar: float = BAR) -> pd.Series:
    """S053's selection: expected CLV at the morning WAP at least the bar, the WAP not under the model's price, at
    most 200, at least GBP100 matched in the morning, in a race K6 keeps. Morning facts only (and K6's flag)."""
    return ((r[ev_col] >= bar) & (r["morningwap"] >= r[model_col]) & (r["morningwap"] <= MAX_PRICE)
            & (r["morning_vol"] >= MIN_VOL) & r["k6_ok"].astype(bool))


def _base_stake(x: pd.DataFrame) -> np.ndarray:
    """BASE-A: to win GBP400, GBP15 to GBP200, at most 10% of the morning volume (K4)."""
    return np.asarray(core.capacity(core.to_win_stake(TO_WIN, x["morningwap"].to_numpy(float), FLOOR, CAP),
                                    x["morning_vol"].to_numpy(float), SHARE), float).reshape(-1)


def _base(frames) -> pd.DataFrame:
    """S053's picks, one row a runner, with ``stake`` (BASE-A). A copy: the shared frame is not touched."""
    r = _runners(frames)
    _need(r, "ev", "the walk-forward closing model scored no month")
    x = r[_picks(r)].copy()
    x["stake"] = _base_stake(x)
    return x[x["stake"] > 0]


def _bets(x: pd.DataFrame, side: str = "back", stake=None) -> pd.DataFrame:
    """A bets frame for entries at the morning WAP closed at the BSP (K3), one row a runner of ``x``."""
    s = x["stake"].to_numpy(float) if stake is None else np.asarray(stake, float)
    return pd.DataFrame({
        "race_date": x["race_date"].astype(str).str[:10].to_numpy(),
        "race_key": x["race"].astype(str).to_numpy(),
        "selection": x["horse_name"].astype(str).to_numpy(),
        "side": side,
        "stake": s,
        "entry_price": x["morningwap"].to_numpy(float),
        "exit": "bsp",
        "exit_price": x["bsp"].to_numpy(float),
        "won": pd.to_numeric(x["win_lose"], errors="coerce").clip(0, 1).to_numpy(float),
        "void": False,
        "market": "win",
        "entry_type": "wap",
    })


def _describe(label: str, bets: pd.DataFrame, days) -> None:
    """One printed line a window for a set of bets beside a strategy (descriptive: never a verdict). The same money
    arithmetic as the reading (core.daily_series), fewer bootstrap draws. A failure here never fails the strategy."""
    try:
        if bets is None or len(bets) == 0:
            _say(f"      {label}: no bets")
            return
        d = core.daily_series(core.check_bets(bets), days)
        win = core.window_of(d.index, DS)
        for w in ("select", "confirm"):
            x = d[win == w]
            est, lo, hi = core.day_bootstrap(x["pnl"], x["stake"], DESCRIBE_DRAWS, core.SEED)
            if not x["stake"].sum() > 0:
                _say(f"      {label}, {w}: no bets")
                continue
            _say(f"      {label}, {w}: {int(x['bets'].sum()):,} bets, stake GBP{x['stake'].sum():,.0f}, net "
                 f"GBP{x['pnl'].sum():+,.0f}, {100 * est:+.2f}% ({100 * lo:+.2f} to {100 * hi:+.2f}) [no cuts, at the WAP]")
    except Exception as exc:  # noqa: BLE001 -- a side table must not cost the strategy its reading
        _say(f"      {label}: not printed ({type(exc).__name__}: {exc})")


def _winter(label: str, bets: pd.DataFrame, days) -> None:
    """The winter months (Nov-Mar) apart, each winter by itself, and the other months."""
    if len(bets) == 0:
        return
    mm, yy = bets["race_date"].str[5:7], bets["race_date"].str[:4].astype(int)
    season = np.where(mm.isin(WINTER), np.where(mm.isin(("11", "12")), yy, yy - 1), -1)
    dd = pd.Series([str(d)[:10] for d in days], dtype=object)
    d_mm, d_yy = dd.str[5:7], dd.str[:4].astype(int)
    d_season = np.where(d_mm.isin(WINTER), np.where(d_mm.isin(("11", "12")), d_yy, d_yy - 1), -1)
    for s in sorted(set(d_season)):
        name = "the other months (Apr-Oct)" if s == -1 else f"winter {s}-{str(s + 1)[2:]} (Nov-Mar)"
        _describe(f"{label}, {name}", bets[season == s], list(dd[d_season == s]))


def _subset(frames, sid: str, keep, what: str, winter: bool = False):
    """A rule that only drops bets from S053: the kept bets are the strategy; the dropped bets' own reading is
    printed. ``keep(x)`` is a boolean mask over S053's picks, from what the morning knows."""
    x = _base(frames)
    days = _scored_days(_runners(frames))
    k = np.asarray(keep(x), bool)
    bets = _bets(x[k])
    _say(f"   {sid} {what}: keeps {int(k.sum()):,} of S053's {len(x):,} bets")
    _describe(f"{sid} kept", bets, days)
    _describe(f"{sid} dropped", _bets(x[~k]), days)
    if winter:
        _winter(sid, bets, days)
    return bets, days


# ---------------------------------------------------------------------------------------------------------------
# S053: the benchmark, and its restriction to the steamer's races (the pair of S061)
# ---------------------------------------------------------------------------------------------------------------

def s053(frames):
    """The rule at the morning WAP, 3% bar. Back every runner of a whole race with ev >= 0.03, morningwap >=
    predicted_bfsp, morningwap <= 200, morning_vol >= 100; stake to win GBP400 (15 to 200), at most 10% of the
    morning volume; closed at the BSP. Days: the scored months' race days (from Feb 2025). 'no cuts' (K6)."""
    x = _base(frames)
    days = _scored_days(_runners(frames))
    bets = _bets(x)
    _say(f"   S053: {len(bets):,} bets on {bets['race_date'].nunique()} of {len(days)} scored days "
         f"({days[0] if days else '-'} to {days[-1] if days else '-'})")
    _winter("S053", bets, days)
    return bets, days


def _steam_universe(r: pd.DataFrame) -> pd.Series:
    """The races the steamer scores: main codes (flat, all-weather, hurdle, chase), field 5+ (card facts)."""
    return r["steam_score"].notna() & (r["steam_field"] >= 5) & r["steam_code"].isin(MAIN_CODES)


def s053_main(frames):
    """S053's bets in the steamer's races only (main codes, field 5+): the benchmark S061 is paired against, so the
    pair is read 'on the same races'. Not a candidate."""
    _need(_runners(frames), "steam_score", "the steamer's score")
    x = _base(frames)
    return _bets(x[_steam_universe(x)]), _scored_days(_runners(frames))


# ---------------------------------------------------------------------------------------------------------------
# S054-S058: the rule by race type, rating uncertainty and code
# ---------------------------------------------------------------------------------------------------------------

def s054(frames):
    """Handicaps only: S053's bets in races whose race_name is a handicap by race_types_1120.kind_of (handicap,
    hcap, h'cap, nursery). A race with no race_name joined is not a handicap (dropped)."""
    return _subset(frames, "S054", lambda x: x["kind"].eq("handicap"), "handicaps only")


def s055(frames):
    """Rated horses only: S053's bets on runners with at least 3 earlier race_results rows (same horse_name, an
    earlier race_date) with a Betfair SP above 1. The race stays whole for the closing model: ev is S053's own."""
    _need(_runners(frames), "n_bsp", "the count of earlier Betfair SPs")
    return _subset(frames, "S055", lambda x: x["n_bsp"] >= 3, "at least 3 earlier Betfair SPs")


def s056(frames):
    """Jumps only: S053's bets where research.harness.code_of is 'jumps'. Both winters printed apart."""
    return _subset(frames, "S056", lambda x: x["code"].eq("jumps"), "jumps only", winter=True)


def s057(frames):
    """All-weather only: S053's bets where research.harness.code_of is 'aw'. Winter months printed apart."""
    return _subset(frames, "S057", lambda x: x["code"].eq("aw"), "all-weather only", winter=True)


def _class_chase(x: pd.DataFrame) -> pd.Series:
    """A UK chase of Class 1, 2 or 3: race_code contains 'Chase', the Betfair file's country is 'uk' and the first
    digit of race_class is 1-3. An Irish race carries no UK class and is never one."""
    chase = x["race_code"].astype(str).str.contains("Chase", case=False, na=False)
    return chase & x["country"].astype(str).str.lower().eq("uk") & x["cls"].isin((1, 2, 3))


def s058(frames):
    """Leave out Class 1-3 chases: S053's bets except in a UK race whose race_code contains Chase with race_class
    1, 2 or 3. Irish races stay in."""
    return _subset(frames, "S058", lambda x: ~_class_chase(x), "no Class 1-3 chases")


# ---------------------------------------------------------------------------------------------------------------
# S059-S061: blend agreement and the steamer's signal
# ---------------------------------------------------------------------------------------------------------------

def s059(frames):
    """The rule only where both blends clear the bar: S053's bets whose expected CLV under gate_st (the same
    walk-forward closing model fitted on gate_st's own prices) is also at least 3%. The guards are S053's (on
    gate_stn's price). A runner gate_st does not score is not backed. The two disagreement cells are printed."""
    r = _runners(frames)
    _need(r, "ev_st", "oos_gate_st.csv (the second blend)")
    out = _subset(frames, "S059", lambda x: x["ev_st"] >= BAR, "both blends clear 3%")
    days = out[1]
    x = _base(frames)
    _describe("S059 cell: gate_stn clears the bar, gate_st does not", _bets(x[~(x["ev_st"] >= BAR)]), days)
    only_st = r[_picks(r, "ev_st", "predicted_bfsp_st") & ~(r["ev"] >= BAR)].copy()
    only_st["stake"] = _base_stake(only_st)
    _describe("S059 cell: gate_st clears the bar, gate_stn does not (never a bet of S053's)",
              _bets(only_st[only_st["stake"] > 0]), days)
    return out


def _steam_cut(frames, key: str) -> float:
    v = _meta(frames).get(key)
    if v is None or not np.isfinite(v):
        raise NoData(f"no data found: the steamer's fit ({key} is not in the loader's meta)")
    return float(v)


def s060(frames):
    """The rule's picks in the steamer model's top 30%: S053's bets whose CLEAN steamer score (fitted once on
    2018-01-01..2024-11-30) is at or above the 70th percentile of that fit's own training scores, in main codes
    with a field of 5 or more. The predicted drifters (score at or under the fit's 20th percentile) are printed."""
    r = _runners(frames)
    _need(r, "steam_score", "the steamer's score")
    p70, p20 = _steam_cut(frames, "steam_p70"), _steam_cut(frames, "steam_p20")
    out = _subset(frames, "S060", lambda x: _steam_universe(x) & (x["steam_score"] >= p70),
                  f"steamer score >= its fit's 70th percentile ({p70:+.4f})")
    x = _base(frames)
    _describe(f"S060 cell: the bottom 20% of the score (<= {p20:+.4f}), the bets a veto would drop",
              _bets(x[_steam_universe(x) & (x["steam_score"] <= p20)]), out[1])
    return out


def _stack(r: pd.DataFrame, min_rows: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S061's stacked forecast. Universe: scored runners the steamer scores (main codes, field 5+). For each month
    after the first scored one, OLS of the realised morningwap / bsp - 1 on (1, ev_rule, ev_steam) over the EARLIER
    months' runners with at least GBP100 matched in the morning and a WAP of 200 or less; the month's runners get
    combined = a + w1 ev_rule + w2 ev_steam from those weights. Returns (the scored months' runners with
    ``combined``, the weights by month). A month's weights read earlier months' BSPs only."""
    u = r[r["ev"].notna() & _steam_universe(r)].copy()
    u["ev_steam"] = np.exp(u["steam_score"].astype(float)) - 1.0
    fit_ok = (u["morning_vol"] >= MIN_VOL) & (u["morningwap"] <= MAX_PRICE) & (u["bsp"] > 1)
    out, rows = [], []
    for m in sorted(u["month"].unique())[1:]:
        tr = u[(u["month"] < m) & fit_ok]
        if len(tr) < min_rows:
            continue
        X = np.column_stack([np.ones(len(tr)), tr["ev"].to_numpy(float), tr["ev_steam"].to_numpy(float)])
        y = tr["morningwap"].to_numpy(float) / tr["bsp"].to_numpy(float) - 1.0
        a, w1, w2 = np.linalg.lstsq(X, y, rcond=None)[0]
        te = u[u["month"] == m]
        out.append(te.assign(combined=a + w1 * te["ev"].to_numpy(float) + w2 * te["ev_steam"].to_numpy(float)))
        rows.append({"month": m, "fit_rows": len(tr), "a": a, "w_rule": w1, "w_steam": w2})
    scored = pd.concat(out) if out else u.iloc[:0].assign(combined=np.nan)
    return scored, pd.DataFrame(rows, columns=["month", "fit_rows", "a", "w_rule", "w_steam"])


def s061(frames):
    """Stacked expected CLV: back when a + w1 ev_rule + w2 ev_steam >= 0.03 (ev_steam = exp(steamer score) - 1;
    the three weights refitted each month by OLS on the months before, first scored month Mar 2025), with S053's
    guards, in main codes with a field of 5 or more. Stake BASE-A. Days: the months with weights. Paired against
    S053-MAIN (S053 on the same races)."""
    r = _runners(frames)
    _need(r, "steam_score", "the steamer's score")
    _need(r, "ev", "the walk-forward closing model scored no month")
    u, weights = _stack(r)
    _say("   S061 weights by month (OLS on the months before):")
    _say("      no month could be fitted" if weights.empty else
         "\n".join("      " + ln for ln in weights.round(4).to_string(index=False).splitlines()))
    x = u[_picks(u, "combined")].copy()
    x["stake"] = _base_stake(x)
    days = sorted(u["race_date"].astype(str).str[:10].unique())
    bets = _bets(x[x["stake"] > 0])
    try:          # the pair is GBP a day between two books: their sizes are printed beside it
        b0 = _base(frames)
        main = _bets(b0[_steam_universe(b0)])
        main = main[main["race_date"].isin(set(days))]
        for w in ("select", "confirm"):
            mine_w = bets.loc[core.window_of(bets["race_date"], DS) == w, "stake"]
            main_w = main.loc[core.window_of(main["race_date"], DS) == w, "stake"]
            _say(f"   S061 {w.upper()}: {len(mine_w):,} bets, stake GBP{mine_w.sum():,.0f}; S053-MAIN on the same days "
                 f"{len(main_w):,} bets, stake GBP{main_w.sum():,.0f} (the pair is GBP a day: read the two returns "
                 f"beside it)")
    except Exception as exc:  # noqa: BLE001 -- a side table
        _say(f"   S061 stakes beside S053-MAIN not printed ({type(exc).__name__}: {exc})")
    return bets, days


# ---------------------------------------------------------------------------------------------------------------
# S062-S065: the exit and the stake
# ---------------------------------------------------------------------------------------------------------------

def _half_hedged(x: pd.DataFrame, half) -> pd.DataFrame:
    """S053's bets where the rows marked ``half`` are hedged at the BSP for half the winnings: a lay of size
    0.5 s (p - 1) / (bsp - 1), so a winner keeps 0.5 s (p - 1) and a loser loses s less that lay's stake. The others
    are closed in full. (The lay's size is fixed at the WAP; the worse-price prints move only the back's price.)"""
    b = _bets(x)
    half = np.asarray(half, bool)
    s, p, bsp = b["stake"].to_numpy(float), b["entry_price"].to_numpy(float), b["exit_price"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        b["exit_stake"] = np.where(half, 0.5 * s * (p - 1.0) / (bsp - 1.0), np.nan)
    return b


def s062(frames):
    """Hold half where the expected CLV is 10% or more: S053's bets; a bet with ev >= 0.10 is laid at the BSP for
    half its winnings, every other bet closed in full at the BSP."""
    x = _base(frames)
    half = x["ev"] >= 0.10
    _say(f"   S062: {int(half.sum()):,} of {len(x):,} bets half-hedged (ev >= 10%)")
    return _half_hedged(x, half), _scored_days(_runners(frames))


def s063(frames):
    """Hold half on Irish races: S053's bets; a bet in a race of betfair_prices.country 'ire' is laid at the BSP for
    half its winnings, a UK bet closed in full. The model's conditional-logit weight beside the BSP on the Irish
    whole races held here is printed (research/harness.benter; descriptive)."""
    x = _base(frames)
    irish = x["country"].astype(str).str.lower().eq("ire")
    _say(f"   S063: {int(irish.sum()):,} of {len(x):,} bets half-hedged (Irish races)")
    try:
        from research import harness
        r = _runners(frames)
        bt = harness.benter(r[r["country"].astype(str).str.lower().eq("ire")])
        _say("      Benter test, Irish whole races only: " + ("; ".join(
            f"fold {w['fold']}: model {w['beta_model']:+.3f}, market {w['beta_market']:.3f}" for w in bt["weights"])
            or "no data found") + f" ({bt['races']:,} races scored)")
    except Exception as exc:  # noqa: BLE001 -- a side table
        _say(f"      Benter test on Irish races: not printed ({type(exc).__name__}: {exc})")
    return _half_hedged(x, irish), _scored_days(_runners(frames))


def _day_multiplier(daily_pnl: pd.Series, mult: float = 0.5) -> pd.Series:
    """Each race day's stake multiplier: ``mult`` when the PREVIOUS race day's net P&L was below zero, else 1 (the
    first day 1). ``daily_pnl`` is indexed by ISO day, every race day present (a no-bet day is 0, so not a loss)."""
    prev = daily_pnl.sort_index().shift(1)
    return pd.Series(np.where(prev < 0, mult, 1.0), index=prev.index)


def s064(frames):
    """Half stakes after a losing day: S053's bets; if the previous race day's net P&L of S053 (after commission,
    all of it settled at that day's BSPs, so known the next morning) was below zero, every stake today is halved.
    'Previous race day' is the previous scored race day of the data set; halving never changes a day's sign, so
    the full-stake series decides."""
    x = _base(frames)
    days = _scored_days(_runners(frames))
    full = core.daily_series(core.check_bets(_bets(x)), days)
    mult = _day_multiplier(full["pnl"])
    try:
        prev = full["pnl"].shift(1)
        ret = (full["pnl"] / full["stake"]).where(full["stake"] > 0)
        _say(f"   S064: {int((mult < 1).sum())} of {len(mult)} days at half stakes; mean day P&L after a losing day "
             f"GBP{full.loc[prev < 0, 'pnl'].mean():+,.2f} ({int((prev < 0).sum())} days), after any other day "
             f"GBP{full.loc[~(prev < 0), 'pnl'].mean():+,.2f}; lag-1 autocorrelation of the daily net CLV "
             f"{ret.corr(ret.shift(1)):+.3f}, of the daily P&L {full['pnl'].corr(prev):+.3f} [full stakes, no cuts]")
        _say("      the same table on the live fills (BASE-B) is the b_live builder's: no live frame is loaded "
             "into a_model (the holdout guard refuses every row dated after 27 Mar 2026)")
    except Exception as exc:  # noqa: BLE001 -- a side table
        _say(f"   S064 table not printed ({type(exc).__name__}: {exc})")
    m = x["race_date"].astype(str).str[:10].map(mult).fillna(1.0).to_numpy(float)
    return _bets(x, stake=x["stake"].to_numpy(float) * m), days


def s065(frames):
    """The rule's backs laid in running at half the SP odds: S053's bets; a lay of stake x morningwap / L rests at
    L = 1 + (bsp - 1) x 0.5 and is matched only if ipmin is at least one Betfair tick under L (K8:
    1 < ipmin <= core.tick_under(L); an ipmin at or under 1 is the files' mark for no in-running trade, never a
    fill); matched, the back is closed at L on every outcome; unmatched, it runs to the result. No lay at the SP. An upper bound: the files hold no volume at a price (K8)."""
    x = _base(frames)
    b = _bets(x)
    lay_at = 1.0 + (b["exit_price"].to_numpy(float) - 1.0) * 0.5
    ip = pd.to_numeric(x["ipmin"], errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore"):      # the files write 0 where nothing traded in running: not a low, no fill
        filled = (ip > 1.0) & (ip <= np.asarray(core.tick_under(lay_at), float).reshape(-1))
    b["exit"] = np.where(filled, "inplay", "result")
    b["exit_price"] = np.where(filled, lay_at, np.nan)
    _say(f"   S065: the in-running lay matched on {int(filled.sum()):,} of {len(b):,} bets")
    return b, _scored_days(_runners(frames))


# ---------------------------------------------------------------------------------------------------------------
# S066-S067: lays at the morning WAP
# ---------------------------------------------------------------------------------------------------------------

def s066(frames):
    """The lay book on the price files: lay a runner of a whole race when morningwap <= 10.0, morning_vol >= 200,
    1 - morningwap / E >= 0.03 and morningwap <= predicted_bfsp, E the expected BSP of the same monthly
    walk-forward closing model as S053 (E = 1 / E[1 / BSP], volume read as none). Size min(max(400 / (L - 1), 15),
    200, 10% of the morning volume); closed by a back at the BSP. One lay a runner."""
    r = _runners(frames)
    _need(r, "exp_bsp", "the walk-forward closing model scored no month")
    edge = 1.0 - r["morningwap"] / r["exp_bsp"]
    x = r[(edge >= BAR) & (r["morningwap"] <= 10.0) & (r["morningwap"] > 1.0) & (r["morning_vol"] >= 200.0)
          & (r["morningwap"] <= r["predicted_bfsp"]) & r["k6_ok"].astype(bool)].copy()
    x["stake"] = np.asarray(core.capacity(core.to_win_stake(TO_WIN, x["morningwap"].to_numpy(float), FLOOR, CAP),
                                          x["morning_vol"].to_numpy(float), SHARE), float).reshape(-1)
    return _bets(x[x["stake"] > 0], side="lay"), _scored_days(r)


def _false_favourites(r: pd.DataFrame) -> pd.DataFrame:
    """S067's selection: the single lowest morningwap of a whole race of 5 or more (a tie for lowest: no bet), at
    1.8 to 6.0 with at least GBP300 matched in the morning, the model's price at least 1.25 x the WAP, and some
    other runner strictly shorter on the model's prices. Morning facts only."""
    g = r.groupby("race", sort=False)
    rank_m = g["morningwap"].rank(method="min")
    lowest = rank_m.eq(1)
    alone = lowest.groupby(r["race"]).transform("sum").eq(1)
    rank_f = g["predicted_bfsp"].rank(method="min")
    keep = (lowest & alone & (r["n_field"] >= 5) & r["morningwap"].between(1.8, 6.0) & (r["morning_vol"] >= 300.0)
            & (r["predicted_bfsp"] >= 1.25 * r["morningwap"]) & (rank_f >= 2) & r["k6_ok"].astype(bool))
    x = r[keep].copy()
    x["stake"] = np.asarray(core.capacity(core.to_win_stake(200.0, x["morningwap"].to_numpy(float)),
                                          x["morning_vol"].to_numpy(float), SHARE), float).reshape(-1)
    return x[x["stake"] > 0]


def s067(frames):
    """Morning false favourite: lay the single lowest morningwap of a whole race (field 5+, a tie skipped) when
    1.8 <= morningwap <= 6.0, morning_vol >= 300, predicted_bfsp >= 1.25 x morningwap and it is not the model's
    shortest runner (joint-shortest counts as shortest: no bet). GBP200 at risk: size min(200 / (L - 1), 10% of the
    morning volume); closed by a back at the BSP. It needs no closing model, so every race day of the data set is
    read. The same lays held to the result are printed beside it."""
    r = _runners(frames)
    x = _false_favourites(r)
    days = _all_days(frames)
    bets = _bets(x, side="lay")
    _describe("S067 closed at the BSP", bets, days)
    _describe("S067 the same lays held to the result", bets.assign(exit="result", exit_price=np.nan), days)
    return bets, days


# ---------------------------------------------------------------------------------------------------------------
# S068: the rule's horses backed to place by limit on close (and the same limit rule on the rest)
# ---------------------------------------------------------------------------------------------------------------

PLACE_STAKE, PLACE_MAX_WIN_PPWAP, PLACE_SP_SHARE = 10.0, 21.0, 0.01
_ORDERS_CACHE: dict = {}


def _place_orders(frames) -> pd.DataFrame:
    """Every LIMIT_ON_CLOSE place order S068 or its benchmark could send: one row a runner of a scored whole race
    K6 keeps, whose every runner has a win ppwap above 1 and a place row (PAIR-A) in one place market with the
    places paid k in 2-4 and equal to the place book's round sum (K9), with a win ppwap of 21.0 or less.
        p      K9: core.place_prob(normalised 1 / win ppwap, k), one call a race
        limit  1 + (1 / p - 1) / 0.98 (expected value 0 after 2% commission)
        pick   S053 backs the runner
    The limit reads the win ppwap and k only; the place BSP decides the fill, never the order."""
    r = _runners(frames)
    pl = frames.get("place")
    if pl is None or len(pl) == 0:
        raise NoData("no data found: Betfair's place rows")
    hit = _ORDERS_CACHE.get("orders")
    if hit is not None and hit[0]() is r and hit[1]() is pl:           # the same two frames, by identity
        return hit[2]
    x = r[r["ev"].notna() & r["k6_ok"].astype(bool)].copy()
    x["pick"] = _picks(x).to_numpy(bool)
    x = _pair_place(x, pl)
    x["q"] = core.normalise(x["ppwap"].to_numpy(float), x["race"].to_numpy(), whole=True)
    paired = x["place_event_id"].notna() & x["k_ok"].eq(True)
    one = x.groupby("race")["place_event_id"].transform("nunique").eq(1)
    ok = (paired & one & x["q"].notna()).groupby(x["race"]).transform("all")
    x = x[ok].copy()
    p = np.full(len(x), np.nan)
    pos = pd.Series(np.arange(len(x)), index=x.index)
    for idx in x.groupby("race", sort=False).groups.values():
        g = x.loc[idx]
        k = int(g["k_places"].iloc[0])
        if k < len(g):
            p[pos.loc[idx].to_numpy()] = core.place_prob(g["q"].to_numpy(float), k)
    x["p_place"] = p
    x = x[(x["p_place"] > 0) & (x["p_place"] < 1) & (x["ppwap"] <= PLACE_MAX_WIN_PPWAP)].copy()
    x["limit"] = 1.0 + (1.0 / x["p_place"] - 1.0) / (1.0 - core.COMMISSION)
    _ORDERS_CACHE["orders"] = (weakref.ref(r), weakref.ref(pl), x)       # S068 and S068-REST share one pass
    return x


def _pair_place(x: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    """PAIR-A: each win runner's place row on (country, menu_hint, event_dt, selection_id)."""
    left = x.assign(selection_id=pd.to_numeric(x["selection_id"], errors="coerce"))
    right = pl.assign(selection_id=pd.to_numeric(pl["selection_id"], errors="coerce")).drop_duplicates(PAIR_A)
    cols = PAIR_A + ["place_event_id", "place_bsp", "place_pp_vol", "place_won", "k_places", "k_ok"]
    return left.merge(right[cols], on=PAIR_A, how="left", validate="m:1")


def _place_bets(o: pd.DataFrame) -> pd.DataFrame:
    """The orders that fill: the place BSP is at or above the limit. GBP10, at most 1% of the win market's pp_vol
    (K4), nothing under GBP2; held to the place result."""
    f = o[(o["place_bsp"] > 1) & (o["place_bsp"] >= o["limit"])]
    stake = np.asarray(core.capacity(np.full(len(f), PLACE_STAKE), f["pp_vol"].to_numpy(float), PLACE_SP_SHARE),
                       float).reshape(-1)
    b = pd.DataFrame({
        "race_date": f["race_date"].astype(str).str[:10].to_numpy(),
        "race_key": f["race"].astype(str).to_numpy(),
        "selection": f["horse_name"].astype(str).to_numpy(),
        "side": "back", "stake": stake, "entry_price": f["place_bsp"].to_numpy(float), "exit": "result",
        "exit_price": np.nan, "won": pd.to_numeric(f["place_won"], errors="coerce").clip(0, 1).to_numpy(float),
        "void": False, "market": "place", "entry_type": "sp"})
    return b[b["stake"] > 0].reset_index(drop=True)


def s068(frames):
    """The rule's horses backed to place by limit on close: S053's picks with a win ppwap of 21.0 or less, paired
    to their place rows; a LIMIT_ON_CLOSE place back at L = 1 + (1 / p - 1) / 0.98 (p by K9 from the normalised win
    ppwap), filled at the place BSP only if it is at least L; GBP10 a fill (at most 1% of the win pp_vol), held to
    the place result. Paired against S068-REST. The win ppwap includes trades up to the off (a look-ahead of
    minutes, as registered)."""
    o = _place_orders(frames)
    mine = o[o["pick"]]
    bets = _place_bets(mine)
    _say(f"   S068: {len(mine):,} limit orders on S053's picks, {len(bets):,} filled")
    return bets, _scored_days(_runners(frames))


def s068_rest(frames):
    """The same limit-on-close place rule on every runner S053 does not pick (the same races, the same ppwap cap and
    limit): the benchmark S068 is paired against. Not a candidate. STAKE-MATCHED BY DAY: the rest's fills are many
    times S068's, and a pair in GBP a day between books of different sizes is decided by size, not by edge; so each
    day's rest stakes are scaled by one factor to sum to S068's stake that day (a day S068 has no fill holds no
    benchmark bet). The pair is then S068's daily stake x (its return less the rest's). The rest at its own GBP10
    stakes is printed beside it."""
    o = _place_orders(frames)
    rest = o[~o["pick"]]
    bets = _place_bets(rest)
    days = _scored_days(_runners(frames))
    _say(f"   S068-REST: {len(rest):,} limit orders on the runners not picked, {len(bets):,} filled")
    _describe("S068-REST at its own stakes (GBP10 a fill, not stake-matched)", bets, days)
    return _stake_matched(bets, _place_bets(o[o["pick"]])), days


def _stake_matched(bench: pd.DataFrame, mine: pd.DataFrame) -> pd.DataFrame:
    """``bench`` with each day's stakes scaled by one factor so they sum to ``mine``'s stake that day; the bets of
    a day ``mine`` stakes nothing are dropped. Returns per GBP are unchanged; only the size is matched."""
    if len(bench) == 0:
        return bench
    target = mine.groupby("race_date")["stake"].sum() if len(mine) else pd.Series(dtype=float)
    day_sum = bench.groupby("race_date")["stake"].transform("sum").to_numpy(float)
    scale = bench["race_date"].map(target).fillna(0.0).to_numpy(float) / np.where(day_sum > 0, day_sum, 1.0)
    out = bench.assign(stake=bench["stake"].to_numpy(float) * scale)
    return out[out["stake"] > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------------------------

_SEEN = ("not blind: S053's bets on Oct 2025 - Mar 2026 were read on this artifact's rule before "
         "(closing-refit-stn-1121 to Dec 2025; race-types-1120 and phase4-rules-1119 on the blend before it)")
_BLEND = ("No a_model strategy is blind: gate_stn's blend was chosen on the development window to 31 Mar 2026 "
          "(iter130, iter131), which holds CONFIRM")
_WAP = "WAP entry (an average, not a price one order could take); no cuts (K6)"


def _s(id, name, family, build, side="back", market="win", entry="wap", paired="S053", blind=False, benchmark=False,
       tested_before="", note=""):
    return core.Strategy(id=id, name=name, family=family, dataset=DS, side=side, market=market, build=build,
                         entry=entry, paired=paired, blind=blind, benchmark=benchmark, tested_before=tested_before,
                         note=note)


STRATEGIES = [
    _s("S053", "The rule at the morning WAP, 3% bar (benchmark on the price files)", "benchmark", s053, paired=None,
       benchmark=True,
       tested_before="closing-refit-stn-1121 (same artifact and walk-forward closing model, GBP250 to win, no volume "
                     "cap, to Dec 2025: refit arm 22,950 bets net CLV +7.93%; its served-model arm 18,992 bets "
                     "+9.19%); phase4-rules-1119; phase6 gates 21 Nov (holdout, not read here). The 10% volume cap "
                     "and the GBP400 target are new.",
       note=f"BASE-A. {_WAP}. {_SEEN}. Live, the trader's picks earned -0.30% at the morning WAP "
            "(phase5-execution-1119): it ranks selections, it does not forecast pounds."),
    _s("S054", "Handicaps only", "rule by race type", s054,
       tested_before="race-types-1120 (descriptive: research CLV handicap +9.83% against novice/maiden/bumper "
                     "+7.42%, every cell positive; live +5.07% against +3.75%); model-qa-1113. As a rule: none found.",
       note=f"A subset of S053: read the pair (the dropped bets). {_WAP}. {_SEEN}."),
    _s("S055", "Rated horses only (at least 3 earlier Betfair SPs)", "rule by rating uncertainty", s055,
       tested_before="member-agreement-1120 (closed: the high-disagreement third had the higher CLV); "
                     "chase-error-1120. By count of earlier SPs: none found.",
       note=f"A subset of S053; namesakes are a small error in the count. {_WAP}. {_SEEN}."),
    _s("S056", "Jumps only", "rule by code", s056,
       tested_before="race-types-1120 (research hurdle +9.51%, chase +10.01%; live +3.84%, +3.88%); week-drop-1123; "
                     "iter117-nh-vs-flat. As a rule: none found.",
       note=f"A subset of S053; code by research.harness.code_of. {_WAP}. {_SEEN}."),
    _s("S057", "All-weather only", "rule by code", s057,
       tested_before="race-types-1120 (research +9.75%, live +5.90% to 19 Nov); week-drop-1123 (+0.16% on 15-21 Nov, "
                     "-6.18% on 22-23 Nov); model-qa-1113. As a rule: none found.",
       note=f"A subset of S053; code by research.harness.code_of. {_WAP}. {_SEEN}."),
    _s("S058", "Leave out Class 1-3 chases", "rule by race type", s058,
       tested_before="chase-error-1120 (error only: the morning WAP nearer the BSP than our price in Class 1-3 "
                     "chases); race-types-1120 (chases +10.01% CLV in research). As a rule: none found.",
       note=f"A subset of S053; UK races only carry a class. {_WAP}. {_SEEN}."),
    _s("S059", "The rule only where both blends clear the bar", "blend agreement", s059,
       tested_before="shadow-1122, shadow-1123 (mean |log ratio| 0.059-0.070, 4 days); iter130 (rule CLV +8.49% "
                     "against +8.38%); member-agreement-1120 closed the within-blend version. An agreement filter "
                     "across the two blends: none found.",
       note=f"A subset of S053; gate_st's file is iteration 131's if it holds one, else iteration 130's (the loader "
            f"prints which). {_WAP}. {_SEEN}."),
    _s("S060", "The rule's picks in the steamer model's top 30%", "rule with the steamer signal", s060,
       tested_before="strat I (reports/strategy_research_2026-11-04.md: steamer deciles monotone, A/E 0.82 to 1.10); "
                     "morning-exit-bsp-1116; paper_overlap_study_1104 (33 days, not significant); "
                     "edge_investigation_1111 item 5: not run.",
       note=f"A subset of S053; the steamer is fitted once on 2018-01-01..2024-11-30, before the window. {_WAP}. {_SEEN}."),
    _s("S061", "Stacked expected CLV: the rule's forecast and the steamer's, one bar", "rule with the steamer signal",
       s061, paired="S053-MAIN",
       tested_before="as S060 (strat I; morning-exit-bsp-1116; paper_overlap_study_1104). A combined forecast: none found.",
       note=f"Weights refitted monthly on earlier months; first scored month Mar 2025; paired against S053 on the "
            f"same races (S053-MAIN). {_WAP}. {_SEEN}."),
    _s("S062", "Hold half where the expected CLV is 10% or more", "exit: partial hedge", s062,
       tested_before="to-win-staking-0930 (Feb-Mar 2026: the >= 10% bets +20.71% held against +14.65% CLV, one "
                     "two-month window inside CONFIRM); against: Benter weight 0.00-0.01 (strategy-review-1017), held "
                     "+5.92% against CLV +6.83% (phase4-segments-1119), hedge-ratio-1021. By edge band on the long "
                     "window: none found.",
       note=f"S053's bets, another exit: the WAP entry is common to both arms. {_WAP}. {_SEEN}."),
    _s("S063", "Hold half on Irish races", "exit: partial hedge", s063,
       tested_before="The model's weight beside the BSP by country: none found. Overall 0.00-0.01 "
                     "(strategy-review-1017); bsp-calibration-1007 (pooled); fall-diagnosis-1118 (live first backs' "
                     "CLV larger in Ireland).",
       note=f"S053's bets, another exit. {_WAP}. {_SEEN}."),
    _s("S064", "Half stakes after a losing day", "loss stop across days", s064,
       tested_before="None found. The owner removed day stops on 7 Oct (no-stop-1007).",
       note=f"S053's bets at half stakes after a losing race day; research has few losing days and no cuts, so a "
            f"throttle only adds pounds if the conditional mean is below zero. {_WAP}. {_SEEN}."),
    _s("S065", "The rule's backs laid in running at half the SP odds", "exit: in-running", s065, side="both",
       tested_before="iter110; benter-blend-1020; strategy-review-1017; edge-inplay-offsets-1002 (the model adds "
                     "nothing beside the BSP at the result). An in-running exit: none found.",
       note=f"S053's bets, another exit; an upper bound (K8: no volume at a price, no time of the low). {_WAP}. {_SEEN}."),
    _s("S066", "The lay book on the price files: model drifters at 10 or under", "model lay book", s066, side="lay",
       paired=None, blind=False,
       tested_before="race-book-0930 (lays at 5%+ edge +0.7% of liability, no price cap, an earlier artifact); "
                     "open-close-horses-1101 (market-only lays not tradable a tick worse). The 10-or-under cut on "
                     "history: not run.",
       note="The WAP is not an offered lay price: the 2%- and 5%-worse columns are the test. No cuts (K6). The base "
            "lay book's blind forward figure (desk-lay-book-paper-1112) is not read or printed here. Not blind: "
            f"race-book-0930 read single lays from the same closing-model construction on Feb-Mar 2026. {_BLEND}."),
    _s("S067", "Morning false favourite: lay the leader the model does not rate", "false favourite", s067, side="lay",
       paired=None, blind=False,
       tested_before="morning-favourite-clv; morning-rank1-v4forecast (-2.95% backing the model's rank 1); "
                     "market-only-drift-1002. A lay of model-flagged favourites: none found.",
       note="The WAP is not an offered lay price: the worse-price columns are the test. No cuts (K6). Every race day "
            "of the data set is read (no closing model needed). Not blind: morning-favourite-clv and "
            f"market-only-drift-1002 read morning favourites on Jan-Mar 2026. {_BLEND}."),
    _s("S068", "The rule's horses backed to place by limit on close", "place: model-gated", s068, market="place",
       entry="sp", paired="S068-REST", blind=False,
       tested_before="place-model-stage3-1003 (the model adds nothing beyond the place SP; backs by the closing "
                     "model's book -1.1 to -1.9%); arms-full-1103 (place arms lose). A limit-on-close gate by the "
                     "win rule: none found.",
       note="SP entry (no worse-price print); p from the win ppwap, which includes trades up to the off (a "
            "look-ahead of minutes); places paid from the place rows' results (K9). The pair is against the rest "
            "stake-matched by day (S068's daily stake x its return less the rest's). Not blind: "
            "place-model-stage3-1003 read the rule's horses in the place market on Nov 2025 - Mar 2026. "
            f"{_BLEND}."),
    _s("S053-MAIN", "S053 in the steamer's races only (main codes, field 5+): S061's benchmark", "benchmark",
       s053_main, paired=None, benchmark=True,
       tested_before="as S053.", note=f"S053's bets restricted to the races S061 can bet in. {_WAP}. {_SEEN}."),
    _s("S068-REST", "The limit-on-close place rule on every runner the model does not pick: S068's benchmark",
       "benchmark", s068_rest, market="place", entry="sp", paired=None, blind=False, benchmark=True,
       tested_before="place-market-vs-win (the place price against the win price, fitted 2021-22); "
                     "place-model-stage3-1003.",
       note="The same limit and cap as S068 on the runners S053 does not back, in the same races; each day's stakes "
            "scaled by one factor to S068's stake that day (stake-matched: the return per GBP is the rest's own, "
            "printed also at its own GBP10 stakes). Not blind: place-model-stage3-1003 read the place market on Nov "
            f"2025 - Mar 2026. {_BLEND}."),
]
