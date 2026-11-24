"""The shared core of the 100-strategy harness (racing², conventions K1-K10, registered 24 Nov 2026).

One strategy = one setting = one ``Strategy`` record whose ``build(frames)`` returns a BETS FRAME. The core turns a
bets frame into money and into the reading; no strategy does its own settlement, commission, bootstrap or verdict.

    settle(bets)            each bet's gross P&L (K3): a closed trade locks its profit, a held bet settles on the result
    daily_series(bets, ..)  net P&L a race day after 2% of each market's positive net (K2)
    score(strategy, bets, days)   the reading: SELECT / CONFIRM / all, return per GBP, 90% interval by race day (K10)
    deflate(reading, ..)    the deflated Sharpe through research/overfit.dsr_check, then the verdict
    pair(reading, bench, ..)      a 'paired' strategy's daily net less its benchmark's on the same days (K7)
    verdict(reading)        the verdict exactly as K10 states it

THE BETS FRAME, one row a bet (a ladder order's levels are one row at the stake-weighted price):

    race_date    ISO day (YYYY-MM-DD)
    race_key     date|time|track, or the market_id
    selection    the runner (selection_id or name)
    side         'back' | 'lay'
    stake        the bet's size in GBP: the backer's stake, for a lay the layer's size s (not the liability)
    liability    a lay may give this instead of stake: s = liability / (entry_price - 1)
    entry_price  decimal price matched (before any non-runner cut and before any haircut)
    exit         'result' (held) | 'bsp' | 'mark' | 'inplay' (closed at exit_price)
    exit_price   the closing price; NaN for 'result'
    won          1 / 0 (a dead heat may be a fraction), NaN where the result is not needed
    void         True: no bet (non-runner, no BSP, no fill); P&L 0, stake 0, not counted
  optional
    market         the commission unit within the race ('win', 'place', '2tbp', ...); default Strategy.market
    hedge_fraction share of the equal-profit hedge taken at exit_price: the closing bet's size is
                   hedge_fraction x stake x entry_price / exit_price. Default 1 for a closed exit, 0 for 'result'.
                   0.5 is half the green
    exit_stake     the closing bet's size in GBP, given outright (it overrides hedge_fraction; the result is then
                   always needed): 'lay the stake back at L' is exit_stake = stake
    cut_factor     product of (1 - af/100) over the runners removed after the fill (K6 CUT); default 1
    entry_type     'book' | 'wap' (take the 2% and 5% worse prints, K4/K5) | 'sp' | 'inplay' (do not);
                   default Strategy.entry

Returns are per GBP staked for backs and per GBP of liability s x (entry_price - 1) for lays (K3).
Nothing here reads a database, S3 or Betfair, and nothing here places a bet.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import overfit

COMMISSION = 0.02                    # K2
SEED = 7                             # K10
DRAWS = 10_000                       # K10
LEVEL = 0.90                         # K10: the 5th-95th percentile
HOLDOUT_FROM = "2026-03-28"          # K1: never read in a_model or a_market
FITTED_SELECT_FROM = "2022-01-01"    # K1: a table fitted inside SELECT (2018-2021) is read on 2022-2023 only
WINDOWS = {                          # K1, both ends included
    "a_model": {"select": ("2024-12-28", "2025-09-30"), "confirm": ("2025-10-01", "2026-03-27")},
    "a_market": {"select": ("2018-01-01", "2023-12-31"), "confirm": ("2024-01-01", "2026-03-27")},
    "b_live": {"select": ("2026-10-01", "2026-10-31"), "confirm": ("2026-11-01", "2026-11-23")},
}
MIN_BETS = {"a_model": 300, "a_market": 300, "b_live": 100}      # K10 (d), on CONFIRM
DSR_WINDOW = {"a_model": "confirm", "a_market": "confirm", "b_live": "all"}      # K10 (c)
HAIRCUTS = {"a_model": (0.02, 0.05), "a_market": (0.02, 0.05), "b_live": (0.02,)}   # K4, K5
HAIRCUT_TYPES = ("book", "wap")
ENTRY_TYPES = ("book", "wap", "sp", "inplay")
EXITS = ("result", "bsp", "mark", "inplay")
PAIR_MODES = ("gbp", "return")
BASE_TRIALS, TRIALS_FLOOR = 100, 486                              # K10 (c): 100 + the ledger's rules, 486 on 24 Nov
REQUIRED = ["race_date", "race_key", "selection", "side", "entry_price", "exit"]
_ISO = r"^\d{4}-\d{2}-\d{2}$"


class HoldoutError(RuntimeError):
    """A row dated in the locked holdout reached the harness."""


@dataclass(frozen=True)
class Strategy:
    id: str                       # 'S001'; a benchmark may be 'BASE-B', 'REPLAY-B'
    name: str                     # one line: the rule and its one setting
    family: str
    dataset: str                  # 'a_model' | 'a_market' | 'b_live'
    side: str                     # 'back' | 'lay' | 'both'
    market: str                   # 'win' | 'place' | ... : the default commission unit of its bets
    build: Callable               # build(frames) -> bets frame, or (bets frame, the ISO race days its data holds)
    entry: str = "book"           # default entry_type of its bets
    paired: str | None = None     # the id of its benchmark (K7): read twice
    pair_mode: str = "gbp"        # 'gbp': daily net less the benchmark's (K7: a variant of the benchmark's own fills);
    #                               'return': less the benchmark's return on the strategy's own stake (a benchmark of
    #                               another size, a superset or another set of races: pounds would not compare)
    blind: bool = True            # False: its CONFIRM years were seen in an earlier table -> at best 'PASS (not blind)'
    select_from: str | None = None     # FITTED_SELECT_FROM for a table fitted inside SELECT
    benchmark: bool = False       # a benchmark is scored and printed, never a candidate
    tested_before: str = ""       # the ledger tag or report that already tested it, and its result
    note: str = ""

    def __post_init__(self):
        if self.dataset not in WINDOWS:
            raise ValueError(f"{self.id}: dataset {self.dataset!r} not one of {list(WINDOWS)}")
        if self.side not in ("back", "lay", "both"):
            raise ValueError(f"{self.id}: side {self.side!r}")
        if self.entry not in ENTRY_TYPES:
            raise ValueError(f"{self.id}: entry {self.entry!r} not one of {ENTRY_TYPES}")
        if self.pair_mode not in PAIR_MODES:
            raise ValueError(f"{self.id}: pair_mode {self.pair_mode!r} not one of {PAIR_MODES}")


# ---------------------------------------------------------------------------------------------------------------
# the bets frame
# ---------------------------------------------------------------------------------------------------------------

def empty_bets() -> pd.DataFrame:
    """A bets frame with no bet (a strategy that finds nothing returns this, not None)."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in (
        ("race_date", str), ("race_key", str), ("selection", str), ("side", str), ("stake", float),
        ("entry_price", float), ("exit", str), ("exit_price", float), ("won", float), ("void", bool))})


def drop_holdout(df: pd.DataFrame, col: str = "race_date") -> pd.DataFrame:
    """K1: the rows dated before the locked holdout. The first line of every a_model and a_market loader."""
    return df[df[col].astype(str).str[:10] < HOLDOUT_FROM]


def assert_no_holdout(df: pd.DataFrame, col: str = "race_date", what: str = "frame") -> None:
    if col in df.columns and len(df) and (df[col].astype(str).str[:10] >= HOLDOUT_FROM).any():
        raise HoldoutError(f"{what}: rows dated {HOLDOUT_FROM} or later (the locked holdout) are present")


def check_bets(bets: pd.DataFrame, strategy: Strategy | None = None) -> pd.DataFrame:
    """The bets frame checked and completed (a copy): defaults filled, lays given their size, the frame's own errors
    raised. A bet that cannot be settled (no result where one is needed, no closing price) is made void and counted
    in the column ``unsettled``."""
    if not isinstance(bets, pd.DataFrame):
        raise TypeError(f"a strategy returns a bets frame, not {type(bets).__name__}")
    missing = [c for c in REQUIRED if c not in bets.columns]
    if missing:
        raise ValueError(f"bets frame lacks {missing}")
    b = bets.reset_index(drop=True).copy()
    b["race_date"] = b["race_date"].astype(str).str[:10]
    if len(b) and not b["race_date"].str.match(_ISO).all():
        raise ValueError("race_date is not YYYY-MM-DD")
    for col, default in (("stake", np.nan), ("liability", np.nan), ("exit_price", np.nan), ("won", np.nan),
                         ("cut_factor", 1.0), ("exit_stake", np.nan)):
        b[col] = pd.to_numeric(b[col], errors="coerce") if col in b.columns else default
    b["cut_factor"] = b["cut_factor"].fillna(1.0)
    b["entry_price"] = pd.to_numeric(b["entry_price"], errors="coerce")
    b["void"] = b["void"].fillna(False).astype(bool) if "void" in b.columns else False
    b["market"] = (b["market"] if "market" in b.columns else pd.Series(np.nan, index=b.index, dtype=object)).fillna(
        strategy.market if strategy else "win").astype(str)
    b["entry_type"] = (b["entry_type"] if "entry_type" in b.columns else pd.Series(np.nan, index=b.index, dtype=object)
                       ).fillna(strategy.entry if strategy else "book").astype(str)
    b["side"], b["exit"] = b["side"].astype(str), b["exit"].astype(str)
    for col, allowed in (("side", ("back", "lay")), ("exit", EXITS), ("entry_type", ENTRY_TYPES)):
        bad = set(b[col].unique()) - set(allowed)
        if bad:
            raise ValueError(f"{col} holds {sorted(bad)}; allowed {allowed}")
    held = b["exit"].eq("result")
    b["hedge_fraction"] = pd.to_numeric(b["hedge_fraction"], errors="coerce") if "hedge_fraction" in b.columns else np.nan
    b["hedge_fraction"] = np.where(held, 0.0, b["hedge_fraction"].fillna(1.0))
    b["exit_stake"] = b["exit_stake"].where(~held)
    own = b["exit_stake"].notna()
    lay = b["side"].eq("lay")
    from_liab = lay & b["stake"].isna() & b["liability"].notna()
    b.loc[from_liab, "stake"] = b.loc[from_liab, "liability"] / (b.loc[from_liab, "entry_price"] - 1.0)
    live = ~b["void"]
    if (live & ~(b["entry_price"] > 1.0)).any():
        raise ValueError("a bet that is not void has no entry_price above 1")
    if (live & ~(b["stake"] >= 0)).any():
        raise ValueError("a bet that is not void has no stake (or, for a lay, no liability)")
    if (b["hedge_fraction"] < 0).any() or (b["exit_stake"] < 0).any() or (
            (b["cut_factor"] <= 0) | (b["cut_factor"] > 1)).any():
        raise ValueError("hedge_fraction or exit_stake below 0, or cut_factor outside (0, 1]")
    if (b["won"].notna() & ((b["won"] < 0) | (b["won"] > 1))).any():
        raise ValueError("won outside 0..1")
    needs_exit = live & ((b["hedge_fraction"] > 0) | own) & ~(b["exit_price"] > 1.0)
    needs_result = live & ((b["hedge_fraction"] != 1.0) | own) & b["won"].isna()
    b["unsettled"] = needs_exit | needs_result
    b["void"] = b["void"] | b["unsettled"] | (live & (b["stake"] == 0))
    return b


def settle(bets: pd.DataFrame, haircut: float = 0.0, sp_hedge: str | None = None) -> pd.DataFrame:
    """Each bet's gross P&L and the money it put at risk (K3), from a frame ``check_bets`` has passed.

    The entry price is first made ``haircut`` worse (backs x (1 - h), lays x (1 + h)) where the entry type takes
    haircuts, then cut by ``cut_factor`` (floor 1.01, K6). With p the price after both, s the size, x the exit price
    and f the hedge fraction, the closing bet is h = f x s x p / x (or ``exit_stake`` where given) and

        back:  wins  s (p - 1) - h (x - 1)      loses  -s + h           (f = 1: s (p / x - 1) either way)
        lay :  the same with the sign turned                            (f = 1: s (1 - p / x) either way)

    ``risk`` is the stake of a back and the liability s x (price - 1) of a lay at the price after the haircut and
    before the cut. A void bet has P&L 0 and risk 0.

    ``sp_hedge='liability'``: the BSP close as the order Betfair takes (the second reading; K3 stays the headline).
    A MARKET_ON_CLOSE lay is sized by a liability fixed before the SP is known, so a back fully closed at the BSP
    (exit 'bsp', hedge fraction 1, no exit_stake) is laid for the liability s x (q - 1), q the fill price before
    any cut (the live trader lays the back's winnings): a winner makes s (p - 1) - s (q - 1), a loser
    -s + s (q - 1) / (x - 1). A lay closed by an SP back of stake s: a winner makes s (x - p), a loser 0. The mean
    is K3's at a fair SP; the spread is wider. A bet with no result keeps K3's figure (column ``placeable`` False)."""
    b = bets.copy()
    sign = np.where(b["side"].eq("back"), 1.0, -1.0)
    hc = np.where(b["entry_type"].isin(HAIRCUT_TYPES), haircut, 0.0)
    quoted = b["entry_price"].to_numpy(float) * (1.0 - sign * hc)
    p = np.maximum(1.01, quoted * b["cut_factor"].to_numpy(float))
    s = b["stake"].to_numpy(float)
    f = b["hedge_fraction"].to_numpy(float)
    x = b["exit_price"].to_numpy(float)
    live = ~b["void"].to_numpy(bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        own = b["exit_stake"].to_numpy(float)
        h = np.where(np.isfinite(own), own, np.where(f > 0, f * s * p / x, 0.0))
        xm1 = np.where(h > 0, x - 1.0, 0.0)
        win = s * (p - 1.0) - h * xm1
        lose = -s + h
        w = np.where((f == 1.0) & ~np.isfinite(own), 0.0, b["won"].to_numpy(float))    # fully closed: any outcome
        pnl = sign * (w * win + (1.0 - w) * lose)
        if sp_hedge == "liability":
            won = b["won"].to_numpy(float)
            can = live & b["exit"].eq("bsp").to_numpy() & (f == 1.0) & ~np.isfinite(own) & np.isfinite(won)
            q1 = np.maximum(1.01, quoted) - 1.0
            back = won * (s * (p - 1.0) - s * q1) + (1.0 - won) * (-s + s * q1 / (x - 1.0))
            lay = won * s * (x - p)
            pnl = np.where(can, np.where(sign > 0, back, lay), pnl)
            b["placeable"] = can
        elif sp_hedge is not None:
            raise ValueError(f"sp_hedge {sp_hedge!r}: None or 'liability'")
    b["price"] = p
    b["pnl"] = np.where(live, pnl, 0.0)
    b["risk"] = np.where(live, np.where(sign > 0, s, s * (np.maximum(1.01, quoted) - 1.0)), 0.0)
    return b


def market_net(settled: pd.DataFrame, commission: float = COMMISSION) -> pd.DataFrame:
    """One row a market (race_date, race_key, market): gross, commission on its positive net (K2), net, risk, bets."""
    g = settled.assign(n=(~settled["void"]).astype(int)).groupby(["race_date", "race_key", "market"], sort=False).agg(
        gross=("pnl", "sum"), risk=("risk", "sum"), bets=("n", "sum")).reset_index()
    g["commission"] = commission * g["gross"].clip(lower=0.0)
    g["net"] = g["gross"] - g["commission"]
    return g


def daily_series(bets: pd.DataFrame, days=None, haircut: float = 0.0, commission: float = COMMISSION,
                 sp_hedge: str | None = None) -> pd.DataFrame:
    """Net P&L, stake (backs' stakes plus lays' liabilities) and bets a race day, indexed by ISO day. With ``days``
    the index is those days and the bets' own (a day with no bet is 0, K10)."""
    m = market_net(settle(bets, haircut, sp_hedge), commission)
    d = m.groupby("race_date").agg(pnl=("net", "sum"), stake=("risk", "sum"), bets=("bets", "sum"),
                                   commission=("commission", "sum"))
    if days is not None:
        idx = sorted(set(d.index) | {str(x)[:10] for x in days})
        d = d.reindex(idx, fill_value=0.0)
    d.index.name = "race_date"
    return d.sort_index()


# ---------------------------------------------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------------------------------------------

def window_of(days, dataset: str, select_from: str | None = None) -> np.ndarray:
    """'select', 'confirm' or '' for each ISO day (K1); ``select_from`` moves the start of the SELECT reading."""
    d = np.asarray([str(x)[:10] for x in days], dtype=object)
    (s0, s1), (c0, c1) = WINDOWS[dataset]["select"], WINDOWS[dataset]["confirm"]
    s0 = max(s0, select_from) if select_from else s0
    out = np.full(len(d), "", dtype=object)
    out[(d >= s0) & (d <= s1)] = "select"
    out[(d >= c0) & (d <= c1)] = "confirm"
    return out


def day_bootstrap(num, den=None, draws: int = DRAWS, seed: int = SEED, level: float = LEVEL) -> tuple:
    """(estimate, low, high): the ratio of sums over race days (the mean of ``num`` when ``den`` is None) and its
    interval, race days drawn with replacement. One value a race day in each array."""
    num = np.asarray(num, float)
    n = len(num)
    den = np.ones(n) if den is None else np.asarray(den, float)
    if n == 0 or not den.sum() > 0:
        return (float("nan"),) * 3
    est = float(num.sum() / den.sum())
    if n < 2:
        return est, float("nan"), float("nan")
    rng, out, step = np.random.default_rng(seed), np.empty(draws), max(1, 2_000_000 // n)
    for i in range(0, draws, step):
        idx = rng.integers(0, n, size=(min(step, draws - i), n))
        sn, sd = num[idx].sum(axis=1), den[idx].sum(axis=1)
        out[i:i + len(idx)] = np.where(sd > 0, sn / np.where(sd > 0, sd, 1.0), np.nan)
    if not np.isfinite(out).any():
        return est, float("nan"), float("nan")
    a = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(out, [a, 1.0 - a])
    return est, float(lo), float(hi)


def daily_sharpe(x) -> float:
    """The per-day Sharpe of a daily series (not annualised: the deflated Sharpe wants it per observation)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd) if sd > 0 else float("nan")


REGISTRATION_TAG = "strat100-1124"


def n_trials_for(n_registered: int = 0, ledger_path=None, sport: str = "horse_racing") -> dict:
    """K10 (c): 100 + the ledger's rules for the sport, recounted now, the larger of that and 486 used (and never
    fewer than the strategies registered plus the ledger's rules)."""
    from desk import trials
    lines, _ = trials.parse((ledger_path or trials.LEDGER).read_text())
    ledger = trials.count_rules(lines, sport)
    # The registration line (tag REGISTRATION_TAG, n_rules = the strategies) is already in the ledger's count once it
    # is written: then the 100 are not added a second time.
    held = any(d.get("tag") == REGISTRATION_TAG and isinstance(d.get("n_rules"), int) for d in lines)
    extra = 0 if held else max(BASE_TRIALS, int(n_registered))
    return {"ledger_rules": ledger, "registered": int(n_registered), "registration_in_ledger": held,
            "n_trials": max(TRIALS_FLOOR, extra + ledger)}


def _r(x, nd: int = 6):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def score(strategy: Strategy, bets: pd.DataFrame, days, n_trials: int | None = None, trial_sharpes=None,
          draws: int = DRAWS, seed: int = SEED) -> dict:
    """The reading of one strategy (K10). ``days`` are the ISO race days its data set holds; days outside SELECT and
    CONFIRM are not read. Per window (select, confirm, all = both): race days, days with a bet, bets, stake, net
    P&L, return per GBP with its 90% interval by race day, the daily Sharpe; the CONFIRM and SELECT returns at the
    worse prices (K4, K5); with ``n_trials`` the deflated Sharpe and the verdict (else call ``deflate`` after).
    The daily series is under ``_daily`` (race_date, window, bets, stake, pnl)."""
    ds = strategy.dataset
    b = check_bets(bets, strategy)
    if ds != "b_live":
        assert_no_holdout(b, what=f"{strategy.id} bets")
    d = daily_series(b, days)
    # the second reading: every full close at the BSP settled as the order Betfair takes (settle, sp_hedge)
    closed = ~b["void"] & b["exit"].eq("bsp") & (b["hedge_fraction"] == 1.0) & b["exit_stake"].isna()
    d["pnl_placeable"] = (daily_series(b, days, sp_hedge="liability")["pnl"].reindex(d.index).fillna(0.0)
                          if closed.any() else d["pnl"])
    d["window"] = window_of(d.index, ds, strategy.select_from)
    outside = int(d.loc[d["window"] == "", "bets"].sum())
    d = d[d["window"] != ""]
    r = {"id": strategy.id, "name": strategy.name, "family": strategy.family, "dataset": ds, "side": strategy.side,
         "market": strategy.market, "paired": strategy.paired, "pair_mode": strategy.pair_mode, "blind": strategy.blind,
         "benchmark": strategy.benchmark, "tested_before": strategy.tested_before, "note": strategy.note,
         "rows": len(b), "voids": int(b["void"].sum()), "unsettled": int(b["unsettled"].sum()),
         "bets_outside_windows": outside, "race_days": len(d), "error": None,
         "placeable_bets": int(closed.sum()), "placeable_no_result": int((closed & b["won"].isna()).sum())}
    for w in ("select", "confirm", "all"):
        x = d if w == "all" else d[d["window"] == w]
        est, lo, hi = day_bootstrap(x["pnl"], x["stake"], draws, seed)
        r.update({f"{w}_days": len(x), f"{w}_bet_days": int((x["bets"] > 0).sum()), f"{w}_bets": int(x["bets"].sum()),
                  f"{w}_stake": _r(x["stake"].sum(), 2), f"{w}_pnl": _r(x["pnl"].sum(), 2),
                  f"{w}_commission": _r(x["commission"].sum(), 2), f"{w}_ret": _r(est), f"{w}_lo": _r(lo),
                  f"{w}_hi": _r(hi), f"{w}_sharpe": _r(daily_sharpe(x["pnl"]))})
        est, lo, hi = (est, lo, hi) if not r["placeable_bets"] else day_bootstrap(x["pnl_placeable"], x["stake"], draws, seed)
        r.update({f"{w}_ret_placeable": _r(est), f"{w}_lo_placeable": _r(lo), f"{w}_hi_placeable": _r(hi),
                  f"{w}_sharpe_placeable": _r(daily_sharpe(x["pnl_placeable"]))})
    priced = bool((~b["void"] & b["entry_type"].isin(HAIRCUT_TYPES)).any())
    in_running = bool((~b["void"] & (b["entry_type"].eq("inplay") | b["exit"].eq("inplay"))).any())
    r["flags"] = "; ".join(f for f, on in (("no cuts", ds != "b_live" and priced),            # K6: price files
                                           ("upper bound", ds != "b_live" and in_running)) if on)   # K8
    for h in (0.02, 0.05):
        tag = f"{round(h * 100)}pct"
        r[f"confirm_ret_{tag}"] = r[f"select_ret_{tag}"] = None
        if h in HAIRCUTS[ds]:
            if not priced:                         # every entry at the SP or in running: no worse price to print
                r[f"confirm_ret_{tag}"], r[f"select_ret_{tag}"] = r["confirm_ret"], r["select_ret"]
                continue
            dh = daily_series(b, days, haircut=h)
            wh = window_of(dh.index, ds, strategy.select_from)
            for w in ("select", "confirm"):
                x = dh[wh == w]
                r[f"{w}_ret_{tag}"] = _r(x["pnl"].sum() / x["stake"].sum()) if x["stake"].sum() > 0 else None
    r["dsr_window"] = DSR_WINDOW[ds]
    r["dsr_sharpe"] = r[f"{r['dsr_window']}_sharpe"]
    r.update({"dsr": None, "sr_star": None, "n_trials": None, "dsr_passed": False, "dsr_reason": "not deflated",
              "pair_verdict": None, "dsr_placeable": None, "dsr_placeable_passed": False})
    r["_daily"] = d.reset_index()[["race_date", "window", "bets", "stake", "pnl", "pnl_placeable"]]
    r["fails"], r["verdict"] = _verdict(r)
    return deflate(r, n_trials, trial_sharpes) if n_trials else r


def _dsr_series(r: dict, col: str = "pnl") -> np.ndarray:
    d = r["_daily"]
    return (d if r["dsr_window"] == "all" else d[d["window"] == r["dsr_window"]])[col].to_numpy(float)


def deflate(r: dict, n_trials: int, trial_sharpes=None) -> dict:
    """K10 (c): the deflated Sharpe of the daily net P&L (CONFIRM for a_model and a_market, the whole series for
    b_live) by research/overfit.dsr_check; ``trial_sharpes`` are the ``dsr_sharpe`` of the same builder's
    strategies. Then the verdict."""
    res = overfit.dsr_check(_dsr_series(r), n_trials, trial_sharpes)
    r.update({"dsr": res.get("dsr"), "sr_star": res.get("sr_star"), "n_trials": res["n_trials"],
              "dsr_passed": bool(res["passed"]), "dsr_reason": res.get("reason", "")})
    # the same test on the placeable series; the trials' spread is K3's (the wider of the two: the harder test)
    alt = (overfit.dsr_check(_dsr_series(r, "pnl_placeable"), n_trials, trial_sharpes)
           if r.get("placeable_bets") and "pnl_placeable" in r["_daily"].columns else res)
    r.update({"dsr_placeable": alt.get("dsr"), "dsr_placeable_passed": bool(alt["passed"])})
    r["fails"], r["verdict"] = _verdict(r)
    return r


def _verdict(r: dict) -> tuple[str, str]:
    ds = r["dataset"]
    if r.get("error"):
        return "", "error"
    if ds == "b_live" and r["race_days"] < overfit.MIN_OBS:
        return "", "not read"
    if r["confirm_bets"] < MIN_BETS[ds]:
        return "", "not read: too few"
    fails = []
    if not (r["confirm_lo"] is not None and r["confirm_lo"] > 0):
        fails.append("a")
    if not (r["select_ret"] is not None and r["confirm_ret"] is not None
            and np.sign(r["select_ret"]) == np.sign(r["confirm_ret"]) and r["select_ret"] != 0):
        fails.append("b")
    if not r["dsr_passed"]:
        fails.append("c")
    if fails:
        return ",".join(fails), "FAIL"
    v = "PASS"
    if ds != "b_live" and not (r["confirm_ret_2pct"] is not None and r["confirm_ret_2pct"] > 0):
        v = "PASS at the WAP only"
    if r.get("placeable_bets") and not _placeable_ok(r):
        v = v.replace("PASS", "PASS on K3 only")
    return "", v if r["blind"] else f"{v} (not blind)"


def _placeable_ok(r: dict) -> bool:
    """(a)-(c) again on the series whose BSP closes are settled as the order Betfair takes (a lay at the SP for a
    fixed liability). A close whose row carries no result cannot be settled that way and keeps K3's figure; the
    reading counts them (``placeable_no_result``), so a strategy that gives no results is checked on K3 alone."""
    lo, sel, con = r.get("confirm_lo_placeable"), r.get("select_ret_placeable"), r.get("confirm_ret_placeable")
    return bool(lo is not None and lo > 0 and sel is not None
                and con is not None and sel != 0 and np.sign(sel) == np.sign(con) and r.get("dsr_placeable_passed"))


def verdict(r: dict) -> str:
    """K10, in its words: 'PASS', 'PASS (not blind)', 'PASS at the WAP only' (with '(not blind)' where both hold),
    'PASS on K3 only' (it passes with every BSP close locked as K3 writes it and not with the close settled as the
    liability order Betfair takes: columns ..._placeable),
    'FAIL' (``fails`` names which of a, b, c), 'not read: too few' (d), 'not read' (b_live under 30 race days),
    'error'. A PASS is a candidate for a registered forward paper test and nothing more."""
    return _verdict(r)[1]


def _pair_diff(r: dict, bench: dict) -> tuple:
    """(days, window, difference) on the days both hold. 'gbp' (K7): the strategy's daily net less its benchmark's.
    'return': the strategy's daily net less its own daily stake x the benchmark's return per GBP that day (on a day
    the benchmark has no stake, the benchmark's return over that window), so the sum is the strategy's stake x (its
    return - the benchmark's) and a benchmark's own size or loss cannot make the difference. NaN where the
    benchmark has no stake in a window the strategy bets in."""
    a, b = r["_daily"].set_index("race_date"), bench["_daily"].set_index("race_date")
    days = a.index.intersection(b.index)
    pa, pb = a.loc[days, "pnl"].to_numpy(float), b.loc[days, "pnl"].to_numpy(float)
    win = a.loc[days, "window"].to_numpy()
    if r.get("pair_mode", "gbp") != "return":
        return days, win, pa - pb
    sa, sb = a.loc[days, "stake"].to_numpy(float), b.loc[days, "stake"].to_numpy(float)
    rate = np.full(len(days), np.nan)
    for w in ("select", "confirm"):
        m = win == w
        if sb[m].sum() > 0:
            rate[m] = pb[m].sum() / sb[m].sum()
    rate = np.where(sb > 0, pb / np.where(sb > 0, sb, 1.0), rate)
    return days, win, np.where(sa > 0, pa - sa * rate, 0.0)


def pair(r: dict, bench: dict, n_trials: int, trial_sharpes=None, draws: int = DRAWS, seed: int = SEED) -> dict:
    """K7 and K10: the strategy's daily net less its benchmark's on the days both hold (``_pair_diff``: in GBP, or
    for ``pair_mode`` 'return' less the benchmark's return on the strategy's own stake); 'better than its benchmark'
    only if the difference's CONFIRM mean has its interval above zero (a), SELECT has the same sign (b) and its
    deflated Sharpe is at least 0.95 (c). ``trial_sharpes`` are the ``pair_sharpe`` of the same builder's pairs.
    A strategy that is not blind is at best 'better than its benchmark (not blind)': its CONFIRM days were seen."""
    days, win, diff = _pair_diff(r, bench)
    r["_pair"] = pd.DataFrame({"race_date": days, "window": win, "diff": diff})
    x = diff if r["dsr_window"] == "all" else diff[win == r["dsr_window"]]
    if not np.isfinite(diff).all():               # 'return': the benchmark has no stake in a window to compare with
        r.update({f"pair_{w}_{k}": None for w in ("select", "confirm") for k in ("mean", "lo", "hi")})
        r.update({"pair_select_days": int((win == "select").sum()), "pair_confirm_days": int((win == "confirm").sum()),
                  "pair_sharpe": None, "pair_dsr": None, "pair_dsr_passed": False,
                  "pair_verdict": "not read: the benchmark has no stake to compare with"})
        return r
    for w in ("select", "confirm"):
        est, lo, hi = day_bootstrap(diff[win == w], None, draws, seed)
        r.update({f"pair_{w}_days": int((win == w).sum()), f"pair_{w}_mean": _r(est, 4), f"pair_{w}_lo": _r(lo, 4),
                  f"pair_{w}_hi": _r(hi, 4)})
    res = overfit.dsr_check(x, n_trials, trial_sharpes)
    r.update({"pair_sharpe": _r(daily_sharpe(x)), "pair_dsr": res.get("dsr"), "pair_dsr_passed": bool(res["passed"])})
    if r.get("error") or bench.get("error") or (r["dataset"] == "b_live" and len(days) < overfit.MIN_OBS):
        r["pair_verdict"] = "not read"
    else:
        ok = ((r["pair_confirm_lo"] or 0) > 0 and (r["pair_select_mean"] or 0) > 0 and r["pair_dsr_passed"])
        v = "better than its benchmark" if ok else "not better than its benchmark"
        r["pair_verdict"] = v if (r.get("blind", True) or not ok) else f"{v} (not blind)"     # K10, as the verdict
    return r


def pair_sharpe(r: dict, bench: dict) -> float:
    """The daily Sharpe of the difference on the deflated-Sharpe window (the spread ``pair`` deflates by)."""
    _, win, diff = _pair_diff(r, bench)
    return daily_sharpe(diff if r["dsr_window"] == "all" else diff[win == r["dsr_window"]])


# ---------------------------------------------------------------------------------------------------------------
# helpers both data sets need
# ---------------------------------------------------------------------------------------------------------------

_BANDS = ((1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
          (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0))
TICKS = np.round(np.concatenate([lo + step * np.arange(round((hi - lo) / step)) for lo, hi, step in _BANDS]
                                + [np.array([1000.0])]), 2)          # Betfair's ladder: 350 prices, 1.01 to 1000


def _ticks(p, idx):
    p = np.asarray(p, float)
    ok = np.isfinite(p) & (idx >= 0) & (idx < len(TICKS))
    out = np.where(ok, TICKS[np.clip(idx, 0, len(TICKS) - 1)], np.nan)
    return float(out) if out.ndim == 0 else out


def _floor_idx(p):
    return np.searchsorted(TICKS, np.nan_to_num(np.asarray(p, float), nan=0.0) + 1e-9, side="right") - 1


def _ceil_idx(p):
    return np.searchsorted(TICKS, np.nan_to_num(np.asarray(p, float), nan=2000.0) - 1e-9, side="left")


def tick_floor(p):
    """The largest ladder price at or below p (NaN under 1.01)."""
    return _ticks(p, _floor_idx(p))


def tick_ceil(p):
    """The smallest ladder price at or above p (NaN over 1000)."""
    return _ticks(p, _ceil_idx(p))


def tick_move(p, n: int):
    """The ladder price n ticks from p: n > 0 longer, n < 0 shorter; a price between two ticks reaches the next
    tick with its first step; 0 is tick_floor. NaN past either end of the ladder."""
    return _ticks(p, (_floor_idx(p) if n >= 0 else _ceil_idx(p)) + n)


def tick_under(p):
    """One full tick under p, the conservative reading for a price between two ticks (the tick below the one under
    it). K8: a resting lay at L fills only if ipmin <= tick_under(L). NaN at the foot of the ladder: never filled."""
    return tick_move(tick_floor(p), -1)


def tick_over(p):
    """One full tick over p. K8: a resting back at H fills only if ipmax >= tick_over(H)."""
    return tick_move(tick_ceil(p), 1)


def normalise(price, race, whole: bool = True) -> np.ndarray:
    """Each runner's share of its race's book: (1 / price) / the race's sum. ``race`` is one key a row. With
    ``whole`` a race with any price missing or not above 1 gives NaN for all its runners (K5: whole race); without
    it the missing runners are left out of the sum."""
    p = pd.to_numeric(pd.Series(np.asarray(price)), errors="coerce")
    q = (1.0 / p).where(p > 1.0)
    key = pd.Series(np.asarray(race))
    tot = q.groupby(key).transform("sum")
    if whole:
        tot = tot.where(q.isna().groupby(key).transform("sum") == 0)
    return (q / tot).to_numpy(float)


ORDERING = ((7, 0.734, 0.557), (11, 0.804, 0.640), (15, 0.839, 0.697), (10 ** 9, 0.829, 0.735))   # K9, by field size


def ordering_exponents(n_runners: int) -> tuple[float, float]:
    return next((g, d) for top, g, d in ORDERING if n_runners <= top)


def place_prob(q, k: int, gamma: float | None = None, delta: float | None = None, n_sims: int = 20000,
               seed: int = SEED) -> np.ndarray:
    """K9: P(in the first k) for one race's win probabilities q, by model.ordering.place_probabilities with the
    registered exponents for the field size (pass gamma and delta to use others). Slow in a big field (a Python
    triple loop to three places): call it once a race and keep the answer."""
    from model.ordering import place_probabilities
    q = np.asarray(q, float)
    q = q / q.sum()
    if gamma is None or delta is None:
        gamma, delta = ordering_exponents(len(q))
    return place_probabilities(q, int(k), gamma, delta, n_sims=n_sims, rng=np.random.default_rng(seed))


def harville_place(q, k: int, n_sims: int = 20000, seed: int = SEED) -> np.ndarray:
    """Harville's P(in the first k): the exponents at 1."""
    return place_prob(q, k, 1.0, 1.0, n_sims, seed)


def henery_place(q, k: int, n_sims: int = 20000, seed: int = SEED, iters: int = 60) -> np.ndarray:
    """Henery's P(in the first k): each runner's time is normal with its own mean and unit variance; the means are
    solved so the simulated win shares match q (common random numbers, fixed seed), the places counted on the same
    draws. Against Harville it gives the favourite less of the places and the outsiders more."""
    q = np.clip(np.asarray(q, float), 1e-6, None)
    q = q / q.sum()
    n = len(q)
    if k >= n:
        return np.ones(n)
    z = np.random.default_rng(seed).standard_normal((n_sims, n))
    theta = -np.log(q)
    theta -= theta.mean()
    for _ in range(iters):
        w = (np.bincount((z + theta).argmin(axis=1), minlength=n) + 0.5) / (n_sims + 0.5 * n)
        theta += 0.4 * (np.log(w) - np.log(q))          # wins too often: slower
        theta -= theta.mean()
    first = np.argpartition(z + theta, k - 1, axis=1)[:, :k]
    return np.bincount(first.ravel(), minlength=n) / n_sims


def capacity(stake, available, share: float = 1.0, floor: float = 2.0):
    """The stake a market can take: at most ``share`` of ``available`` (K4: 0.10 of morning_vol or pp_vol; K8: 0.005
    of pp_vol; K5: 1.0 of the size shown), and nothing under ``floor`` (K5: no fill under GBP2). 0 where nothing fits."""
    s = np.minimum(np.asarray(stake, float), share * np.nan_to_num(np.asarray(available, float), nan=0.0))
    s = np.where(s >= floor, s, 0.0)
    return float(s) if s.ndim == 0 else s


def to_win_stake(to_win: float, price, floor: float = 0.0, cap: float = np.inf):
    """The stake that wins ``to_win`` at ``price``, held between ``floor`` and ``cap`` (the live rule: 400, 15, 200)."""
    p = np.asarray(price, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.clip(to_win / (p - 1.0), floor, cap)
    s = np.where(p > 1.0, s, 0.0)
    return float(s) if s.ndim == 0 else s
