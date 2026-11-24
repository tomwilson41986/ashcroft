"""The b_live strategies of the 100-strategy harness: S001-S052 on the live record (racing², conventions K1-K10,
registered 24 Nov 2026).

Every strategy reads the frames of research/strat100/data_b.load_b_live() (the recorder's books 1 Oct - 23 Nov 2026,
the served 06:00 prices, the live trader's own fills, the raw one-minute books) and returns (bets frame, the ISO race
days its data holds); research/strat100/core.py does the settlement, the 2% commission, the 2%-worse print, the
bootstrap and the verdict. One strategy = one setting. Research only: nothing here places a bet.

THE BOOKS LAG. The recorder reads Betfair's DELAYED key, so every book price here was shown some time after the live
market had it. A 'fill' is the size the delayed book showed at its best price (K5), never more; it is an assumption,
not a match. Every non-SP entry is printed again at a price 2% worse.

Three kinds of strategy:
  replays       S001-S007, S009, S010, S013, S014, S017-S020, S037: the live Session (research/arms) run a minute at
                a time on the raw books, one arm a strategy (REPLAY_ARMS); the loader runs them a day at a time and
                the strategy reads its own orders back. S001 is REPLAY-B (K7).
  real fills    BASE-B and S015, S016, S021-S023, S025-S027: the live trader's own matched backs (live_orders), only
                dropped, cut or re-settled; dropped money is not redeployed. No worse-price print: they were matched
                (their rows carry entry_type 'sp', the core's switch for 'no haircut', not a claim of an SP entry).
  marks         the rest: the book at a named mark (betfair_live_marks), or a raw poll where the rule names one (a
                day function in DAY_FUNCS, run by the loader on each day's raw books).

What a selection may read (held by tests/test_strat100_b_live.py): the book at its mark or poll and at earlier ones,
the model's 06:00 price, Betfair's forecast price, the card, earlier days' results and prices. Never the BSP, the
final status, a later mark or the day's result. What reads the close does so as an ORDER's mechanics and is said:
a limit on close fills against the BSP (S026, S048), an in-running lay against the in-play low (S027), and K9's
places paid fall back to the count of winners on the final mark where the raw book gives none.

Readings chosen where a specification left room (each also in its strategy's docstring):
  * K6 CUT: a back is cut by the factor of every runner removed after it (K6's words, research/arms/core.reduced);
    a lay only by factors of at least 2.5, Betfair's own floor (the literal rule would credit a layer with cuts
    Betfair does not make). S024 and its benchmark, whose premise is that floor, cut their backs by it too. BASE-B
    is scaled to Betfair's settled prices. A strategy that needs CUT reads only the days with raw books.
  * 'Whole race' at a mark: every runner ACTIVE at the mark has the prices the rule needs AND the frame holds as
    many rows as the book's active_runners (a runner the catalogue lost makes the race not whole).
  * LIVE-BAR at a mark is K5's, with the draws seeded 7 a race; the live rule's own fallback for a runner with no
    back price (its last trade) is not used: such a race is not whole.
  * A share move between two marks (S035, S043, S050) needs the same number of ACTIVE runners at both.
  * A walk-forward table's burn-in days are not scored (S008 from 11 Oct, S021 after 10 days of fills, S045 from
    20 Oct): the strategy returns its own days.
  * REPLAY-B is registered as S001 (benchmark); no second record of the same bets is registered.
"""
from __future__ import annotations

import math
import re
import weakref
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from research.strat100 import core
from research.strat100 import data_b as B
from research.strat100.data_b import KEY, PAIR_B, NoData, book

DS = "b_live"
ROOT = Path(__file__).resolve().parents[2]
BAR = 0.03                                  # the live bar: expected CLV at least 3%
TO_WIN, FLOOR, CAP = 400.0, 15.0, 200.0     # the live staking: to win GBP400, GBP15 to GBP200 a horse
MAX_PRICE = 200.0
DAY_LIMIT = 4000.0
DRAWS_CLV = 20000                           # K5: draws of the BSP book a race
SEED = 7                                    # K5
CLOSING_MODEL = "data/models/closing_model_novol.json"
NO_HAIRCUT = "sp"                           # core's entry_type that takes no worse-price print (real fills)
_CACHE: dict = {}


def _say(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------------------------------------------

def _meta(frames) -> dict:
    return (frames.get("meta") or {}) if isinstance(frames, dict) else {}


def _frame(frames, name: str) -> pd.DataFrame:
    f = frames.get(name) if isinstance(frames, dict) else None
    return f if isinstance(f, pd.DataFrame) else pd.DataFrame()


def _runners(frames) -> pd.DataFrame:
    r = _frame(frames, "runners")
    if r.empty:
        raise NoData("no data found: b_live holds no runner (the loader's frames are empty)")
    return r


def _days(frames, *needs: str, first: str | None = None, last: str | None = None) -> list:
    """The data set's race days that also have every source in ``needs`` ('books', 'replay', 'preds', 'fills',
    'steam': the loader's meta lists), from ``first`` to ``last``."""
    days = {str(d)[:10] for d in (frames.get("days") or [])}
    meta = _meta(frames)
    for n in needs:
        days &= {str(d)[:10] for d in (meta.get(f"days_{n}") or [])}
    return sorted(d for d in days if (first is None or d >= first) and (last is None or d <= last))


def _day_errors(frames, key: str) -> None:
    """A day function (or the replay) that failed on a day makes the strategy an error, never a silent gap."""
    err = (_meta(frames).get("day_errors") or {}).get(key) or {}
    if err:
        d, msg = next(iter(err.items()))
        raise RuntimeError(f"the pass over the raw books failed for {key} on {len(err)} days, e.g. {d}: "
                           f"{str(msg).split(' | ')[0]}")


def _none(sid: str, why: str, days=()) -> tuple:
    _say(f"   {sid}: no data found: {why}")
    return core.empty_bets(), list(days)


def _within(bets: pd.DataFrame, days) -> pd.DataFrame:
    if bets is None or len(bets) == 0:
        return core.empty_bets()
    return bets[bets["race_date"].astype(str).str[:10].isin(set(days))].reset_index(drop=True)


def _bets(x: pd.DataFrame, side, price, stake=None, *, liability=None, exit="bsp", exit_price=None, won=None,
          void=None, cut=None, market: str = "win", key: str = "market_id", bsp: str = "bsp",
          status: str = "final_status", entry_type=None, ts=None, **extra) -> pd.DataFrame:
    """A bets frame from selected runner rows. By default closed at the row's BSP; ``won`` is the final status
    (WINNER 1, LOSER 0); a runner REMOVED, or with no final status, or with no BSP where the exit needs one, is
    void (K6). Rows with no stake (no fill of GBP2) are not bets and are left out."""
    if x is None or len(x) == 0:
        return core.empty_bets()
    n = len(x)
    st = x[status].astype(str).to_numpy()
    settled = np.isin(st, ["WINNER", "LOSER"])
    ex = np.broadcast_to(np.asarray(exit, dtype=object), (n,))
    xp = x[bsp].to_numpy(float) if exit_price is None else np.broadcast_to(np.asarray(exit_price, float), (n,))
    v = ~settled | ((ex != "result") & ~(np.nan_to_num(xp, nan=0.0) > 1.0))
    if void is not None:
        v = v | np.asarray(void, bool)
    out = pd.DataFrame({
        "race_date": x["race_date"].astype(str).str[:10].to_numpy(), "race_key": x[key].astype(str).to_numpy(),
        "selection": x["selection_id"].astype(str).to_numpy(),
        "side": np.broadcast_to(np.asarray(side, dtype=object), (n,)),
        "stake": np.nan if stake is None else np.broadcast_to(np.asarray(stake, float), (n,)),
        "entry_price": np.broadcast_to(np.asarray(price, float), (n,)), "exit": ex,
        "exit_price": np.where(ex == "result", np.nan, xp),
        "won": np.where(settled, (st == "WINNER").astype(float), np.nan) if won is None
        else np.broadcast_to(np.asarray(won, float), (n,)),
        "void": v, "cut_factor": 1.0 if cut is None else np.asarray(cut, float), "market": market})
    if liability is not None:
        out["liability"] = np.broadcast_to(np.asarray(liability, float), (n,))
    if entry_type is not None:
        out["entry_type"] = entry_type
    if ts is not None:
        out["ts"] = pd.Series(np.asarray(ts)).astype(str).to_numpy()
    for k, val in extra.items():
        out[k] = np.asarray(val) if not np.isscalar(val) else val
    size = out["stake"] if liability is None else out["liability"]
    return out[np.nan_to_num(size.to_numpy(float), nan=0.0) > 0].reset_index(drop=True)


def _cut(frames, x: pd.DataFrame, side: str = "back", ts: str = "t", min_af: float | None = None) -> np.ndarray:
    """K6 CUT for rows matched at their column ``ts``: the removals after it, by every factor for a back and by
    factors of at least 2.5 for a lay (data_b.min_af_for), or by ``min_af`` where a strategy names its own."""
    return B.cut_factor(_frame(frames, "removals"), x["market_id"], x["selection_id"], x[ts],
                        B.min_af_for(side) if min_af is None else min_af)


def _mids(b: pd.DataFrame, fallback_last: bool = False) -> pd.DataFrame:
    """K5 on one mark's book: mid = sqrt(back1 x lay1), spread = lay1 / back1, share = (1/mid) / the race's sum
    over its ACTIVE runners, NaN through a race that is not whole (a runner with no two-sided quote, or fewer rows
    than the book's active runners). ``fallback_last``: a runner with no two-sided quote takes its last traded
    price (S048's own rule)."""
    b = b.copy()
    two = b["back1"].gt(1.0) & b["lay1"].gt(1.0)
    b["mid"] = np.sqrt(b["back1"] * b["lay1"]).where(two)
    if fallback_last:
        b["mid"] = b["mid"].fillna(b["last_traded"].where(b["last_traded"] > 1.0))
    b["spread"] = (b["lay1"] / b["back1"]).where(two)
    share = core.normalise(b["mid"], b["market_id"], whole=True) if len(b) else np.array([])
    b["whole"] = np.isfinite(share) & (b["n_rows"] == b["n_active"]).to_numpy()
    b["share"] = np.where(b["whole"], share, np.nan)
    return b


def _kept(key, runners):
    """A cached frame made from this very runners frame, else None."""
    ent = _CACHE.get(key)
    return ent[1] if ent is not None and ent[0]() is runners else None


def closing_model():
    if "model" not in _CACHE:
        from model import race_book as rb
        _CACHE["model"] = rb.load_closing_model(ROOT / CLOSING_MODEL)
    return _CACHE["model"]


def race_exp_bsp(back, lay, model_price, seed: int = SEED, market=None) -> tuple:
    """One whole race under the served no-volume closing model (K5): (expected CLV at each best back, the expected
    BSP 1 / E[1 / BSP]), DRAWS_CLV draws of the BSP book from ``seed``."""
    from model import race_book as rb
    back, lay = np.asarray(back, float), np.asarray(lay, float)
    return rb.race_expected_clv(back, lay, np.asarray(model_price, float), np.zeros(len(back)), closing_model(),
                                n_draws=DRAWS_CLV, rng=np.random.default_rng(seed), market=market)


def closing_view(b: pd.DataFrame, model_col: str = "predicted_bfsp") -> pd.DataFrame:
    """One mark's book with ``ev`` (the expected CLV at back1) and ``xb`` (the expected BSP), NaN through a race
    that is not whole: fewer than 2 runners, a runner ACTIVE at the mark with no back price or no model price, or
    fewer rows than the book's active runners."""
    b = b.copy()
    ev, xb = np.full(len(b), np.nan), np.full(len(b), np.nan)
    back, lay, mp = b["back1"].to_numpy(float), b["lay1"].to_numpy(float), b[model_col].to_numpy(float)
    full = (b["n_rows"] == b["n_active"]).to_numpy()
    for idx in b.groupby("market_id", sort=False).indices.values():
        if len(idx) < 2 or not full[idx].all() or not (back[idx] > 1.0).all() or not (mp[idx] > 1.0).all():
            continue
        ev[idx], xb[idx] = race_exp_bsp(back[idx], lay[idx], mp[idx])
    b["ev"], b["xb"] = ev, xb
    return b


def live_bar(frames, tag: str) -> pd.DataFrame:
    """A mark's WIN book with LIVE-BAR (K5): ``ev``, ``xb`` and ``bar`` = ev >= 0.03 at back1, back1 >=
    predicted_bfsp, back1 <= 200, whole race. Computed once a mark and kept while the runners frame is the same."""
    r = _runners(frames)
    k = ("bar", tag, DRAWS_CLV)
    if _kept(k, r) is None:
        b = closing_view(book(r, tag))
        b["bar"] = (b["ev"] >= BAR) & (b["back1"] >= b["predicted_bfsp"]) & (b["back1"] <= MAX_PRICE)
        _CACHE[k] = (weakref.ref(r), b)
    return _CACHE[k][1].copy()


def ladder(b: pd.DataFrame, target, ok2, ok3, side: str = "back") -> tuple:
    """K5's ladder: the target filled at level 1 up to its size, then level 2 and level 3 while ``ok2`` (and
    ``ok3``) hold, each at its own price and never more than the size shown. Returns (stake, the stake-weighted
    price); a fill under GBP2 is no fill (stake 0)."""
    p = [b[f"{side}{i}"].to_numpy(float) for i in (1, 2, 3)]
    s = [np.nan_to_num(b[f"{side}{i}_size"].to_numpy(float), nan=0.0) for i in (1, 2, 3)]
    ok2 = np.asarray(ok2, bool) & (np.nan_to_num(p[1], nan=0.0) > 1.0)
    ok3 = ok2 & np.asarray(ok3, bool) & (np.nan_to_num(p[2], nan=0.0) > 1.0)
    left = np.nan_to_num(np.asarray(target, float), nan=0.0)
    f1 = np.minimum(left, s[0])
    f2 = np.where(ok2, np.minimum(left - f1, s[1]), 0.0)
    f3 = np.where(ok3, np.minimum(left - f1 - f2, s[2]), 0.0)
    stake = f1 + f2 + f3
    with np.errstate(invalid="ignore", divide="ignore"):
        price = (f1 * p[0] + f2 * np.nan_to_num(p[1]) + f3 * np.nan_to_num(p[2])) / stake
    stake = np.where(stake >= 2.0, stake, 0.0)
    return stake, np.where(stake > 0, price, p[0])


def day_limit(x: pd.DataFrame, stake, order: list, limit: float = DAY_LIMIT, unit_risk=None,
              ascending=None) -> np.ndarray:
    """A day's money spent in ``order`` (columns of x): each bet takes what is left of ``limit`` (its stake for a
    back; ``unit_risk`` = price - 1 gives a lay's liability), the bet that crosses it is cut to the remainder and
    nothing under GBP2 is placed."""
    stake = np.asarray(stake, float).copy()
    risk = np.ones(len(x)) if unit_risk is None else np.asarray(unit_risk, float)
    o = x.assign(_i=np.arange(len(x))).sort_values(["race_date", *order], ascending=[True, *(ascending or [True] * len(order))],
                                                    kind="stable")
    for _, g in o.groupby("race_date", sort=False):
        left = limit
        for i in g["_i"].to_numpy():
            if not stake[i] > 0:
                continue
            take = min(stake[i], left / risk[i]) if risk[i] > 0 else 0.0
            stake[i] = take if take >= 2.0 else 0.0
            left -= stake[i] * risk[i]
    return stake


def _describe(sid: str, what: str, bets: pd.DataFrame, labels) -> None:
    """A descriptive line a group: bets, money at risk, P&L before commission and its return (never a verdict)."""
    try:
        if bets is None or len(bets) == 0:
            return
        s = core.settle(core.check_bets(bets.assign(_g=labels if np.ndim(labels) == 0 else np.asarray(labels))))
        g = s[~s["void"]].groupby("_g").agg(bets=("pnl", "size"), risk=("risk", "sum"), pnl=("pnl", "sum"))
        _say(f"   {sid} {what} (descriptive, before commission): " + "; ".join(
            f"{k}: {int(v.bets)} bets, GBP{v.risk:,.0f} at risk, GBP{v.pnl:+,.2f} ({v.pnl / v.risk:+.2%})"
            for k, v in g.iterrows() if v.risk > 0))
    except Exception as exc:  # noqa: BLE001 -- a print never fails a strategy
        _say(f"   {sid} {what}: not printed ({type(exc).__name__}: {exc})")


def _backed_by_trader(frames) -> set:
    f = _frame(frames, "fills")
    return set(zip(f["race_date"].astype(str), f["market_id"].astype(str), f["selection_id"].astype(int))) if len(f) else set()


def _trader_label(frames, bets: pd.DataFrame) -> list:
    backed = _backed_by_trader(frames)
    return ["live-trader horse" if (d, m, int(s)) in backed else "not backed live"
            for d, m, s in zip(bets["race_date"], bets["race_key"], bets["selection"])]


# ---------------------------------------------------------------------------------------------------------------
# the replays: the live Session with one arm's policy (research/arms), run by the loader a day at a time
# ---------------------------------------------------------------------------------------------------------------

#: id -> the arm: 'over' changes the pinned rule (data_b.RULE), 'policy' is read by the session below
REPLAY_ARMS = {
    "S001": {},
    "S002": {"over": {"clv_bar": 0.0}},
    "S003": {"over": {"clv_bar": 0.08}},
    "S004": {"policy": {"bar_hours": ((6.0, 0.03), (3.0, 0.05), (-1e9, 0.08))}},
    "S005": {"policy": {"bar_spread": (0.10, 0.03, 0.08)}},
    "S006": {"over": {"clv_bar": 0.0, "min_move": math.log(1.5)}},
    "S007": {"policy": {"forecast": True}},
    "S009": {"policy": {"first_hour": 60}},
    "S010": {"over": {"trade_until": "13:00"}, "policy": {"late_window": (600, 360.0)}},
    "S013": {"over": {"max_topups": 0}},
    "S014": {"policy": {"topup_window": 30.0}},
    "S017": {"over": {"ladder": False}},
    "S018": {"policy": {"floor_long": (21.0, 50.0)}},
    "S019": {"policy": {"tiers": ((6.0, 800.0, 300.0), (3.0, 400.0, 200.0), (-1e9, 200.0, 100.0))}},
    "S020": {"policy": {"level": 40.0}},
    "S037": {"kind": "lay", "over": {"trade_until": "08:00"},
             "policy": {"lays_only": True, "lay_bar": 0.03, "max_lay_price": 10.0, "lay_max_topups": 0,
                        "lay_day_liability": 4000.0, "max_spread": 1.10}},
}


def session_classes() -> tuple:
    """(the rule's session, the lay book's session): research/arms' ArmSession and LaySession with the arms'
    policies. Made on first use so that importing this module reads no trading code."""
    if "sessions" in _CACHE:
        return _CACHE["sessions"]
    from research.arms.core import UK
    from research.arms.sessions import ArmSession, LaySession

    class RuleArm(ArmSession):
        """Policy keys (all optional): bar_hours ((hours, bar), ...: the bar of the first row whose hours the
        market's time to the off reaches), bar_spread (cut, bar wide, bar tight), forecast ({(market, selection):
        F}: every price taken at or above F), late_window (UK minute, minutes to the off: after that minute only
        markets at least that far from the off), topup_window (minutes after the horse's first matched back),
        tiers ((hours, to win, cap), ...: the horse's target by the time to the off at its first back), level
        (a level stake a horse), and ArmSession's own floor_long. The plan is made at a 0% bar and min_move 0
        (data_b.replay_day); this session holds the arm's own bar and guard at each order and each ladder level."""

        def __init__(self, *a, policy=None, **k):
            super().__init__(*a, policy=policy, **k)
            self._first, self._tier, self._price_floor = {}, {}, None

        def execute(self, m, book_, d, now):
            pol, key = self.policy, (m.market_id, d.selection_id)
            pos = self.positions.get(key)
            held = bool(pos and pos.matched > 0)
            mins = (m.start - now).total_seconds() / 60.0
            if "bar_hours" in pol:
                self.cfg.clv_bar = next(bar for hours, bar in pol["bar_hours"] if mins / 60.0 >= hours)
            if "bar_spread" in pol:
                cut, wide, tight = pol["bar_spread"]
                q = book_.runners.get(d.selection_id) if book_ is not None else None
                spread = (q.lay[0][0] / q.back[0][0] - 1.0) if (q is not None and q.lay and q.back) else None
                self.cfg.clv_bar = wide if (spread is None or spread >= cut) else tight
            if not np.isfinite(d.move) or d.move < self.cfg.min_move - 1e-12:
                return
            if "late_window" in pol:
                after, need = pol["late_window"]
                uk = now.astimezone(UK)
                if uk.hour * 60 + uk.minute > after and mins < need:
                    return
            if "topup_window" in pol and held and key in self._first \
                    and now - self._first[key] > timedelta(minutes=float(pol["topup_window"])):
                return
            self._price_floor = None
            if "forecast" in pol:
                f = pol["forecast"].get(key)
                if not f or not d.price or d.price < f - 1e-12:
                    return
                self._price_floor = float(f)
            tier = None
            if "tiers" in pol:
                tier = self._tier.get(key) or next(t for t in pol["tiers"] if mins / 60.0 >= t[0])
                self.cfg.limits.max_stake = float(tier[2])
                d = replace(d, target=min(max(float(tier[1]) / (d.price - 1.0), self.cfg.stake_floor), float(tier[2])))
            if "level" in pol:
                d = replace(d, target=min(float(pol["level"]), self.cfg.limits.max_stake))
            before = pos.matched if pos else 0.0
            super().execute(m, book_, d, now)
            pos = self.positions.get(key)
            if pos is not None and pos.matched > before:
                self._first.setdefault(key, now)
                if tier is not None:
                    self._tier.setdefault(key, tier)

        def _ladder(self, book_, d):
            limit, offered = super()._ladder(book_, d)
            floor = self._price_floor
            if floor and limit < floor - 1e-12:
                q = book_.runners.get(d.selection_id)
                lim, size = d.price, 0.0
                for p, s in (q.back if q is not None else []):
                    if p < limit - 1e-12 or p < floor - 1e-12:
                        break
                    lim, size = p, size + s
                return (lim, size) if size > 0 else (d.price, d.back_size)
            return limit, offered

    class LayArm(LaySession):
        """LaySession with a spread test (policy max_spread: lay / best back at most that, for the lay taken and
        each level above it); a runner with no back price is not laid."""

        def lay_plan(self, m, book_):
            top = float(self.policy.get("max_spread", 1e9))
            out = []
            for c in super().lay_plan(m, book_):
                q = book_.runners[c["sid"]]
                if not q.back or c["price"] / q.back[0][0] > top + 1e-12:
                    continue
                c["levels"] = [(p, s) for p, s in c["levels"] if p / q.back[0][0] <= top + 1e-12]
                out.append(c)
            return out

    _CACHE["sessions"] = (RuleArm, LayArm)
    return _CACHE["sessions"]


def _replay(frames, sid: str) -> tuple:
    """A replayed arm's orders from the loader's ``replay`` frame, on the days the replay ran."""
    _day_errors(frames, "replay")
    days = _days(frames, "replay")
    rp = _frame(frames, "replay")
    if rp.empty or not days:
        return _none(sid, "no day was replayed (no raw books, no price file, or STRAT100_B_REPLAY=0)", days)
    b = rp[rp["id"].eq(sid)].drop(columns=["id"]).reset_index(drop=True)
    return _within(b, days), days


def _uk_hhmm(ts) -> pd.Series:
    t = pd.to_datetime(pd.Series(np.asarray(ts)), utc=True, errors="coerce").dt.tz_convert("Europe/London")
    return t.dt.strftime("%H:%M")


def s001(frames):
    """S001 = REPLAY-B (K7): the rule as it stands on 24 Nov 2026 (data_b.RULE: bar 3%, to win GBP400, GBP15-200,
    3 top-ups, 3 ladder levels, min_move 0, price at most 200, 05:15-10:00 UK, 15 minutes before the off, GBP4,000
    a day) replayed a minute at a time on the raw books by research/arms. One row a matched order (its ladder
    levels at the stake-weighted price), closed at the day's BSP (Betfair's price file) after CUT. Fills at the
    delayed book's sizes; the account holds GBP5,000 as research/arms' replays do."""
    b, days = _replay(frames, "S001")
    if len(b) and "ts" in b.columns:
        t = pd.to_datetime(b["ts"], utc=True, errors="coerce")
        n = b.groupby(["race_date", "race_key", "selection"]).cumcount()
        _describe("S001", "first orders and top-ups", b, np.where(n.to_numpy() == 0, "first order", "top-up"))
        _say(f"   S001: {b['race_date'].nunique()} days with a back of {len(days)} replayed; first back "
             f"{_uk_hhmm(t.groupby(b['race_date']).min()).min()} UK at the earliest")
    return b, days


def s002(frames):
    """S002: as S001 with the bar at 0% for first backs, top-ups and ladder levels (still never under our price)."""
    b, days = _replay(frames, "S002")
    if len(b) and "ts" in b.columns:
        last = _uk_hhmm(pd.to_datetime(b["ts"], utc=True, errors="coerce").groupby(b["race_date"]).max())
        staked = b.groupby("race_date")["stake"].sum()
        _say(f"   S002: the day's last back, UK: earliest {last.min()}, median {last.sort_values().iloc[len(last) // 2]}, "
             f"latest {last.max()}; days at the GBP4,000 limit (within GBP2): {int((staked >= DAY_LIMIT - 2).sum())} "
             f"of {len(staked)}")
    return b, days


def s003(frames):
    """S003: as S001 with the bar at 8% for first backs, top-ups and ladder levels."""
    return _replay(frames, "S003")


def s004(frames):
    """S004: as S001, the bar of each order and ladder level set by the hours from the read to the market's off:
    6 or more 3%, 3 to under 6 5%, under 3 8%."""
    return _replay(frames, "S004")


def s005(frames):
    """S005: as S001, the bar set by the runner's spread at the read (lay1 / back1 - 1): 10% or more, or no lay
    shown, 3%; under 10%, 8%. First backs, top-ups and ladder levels alike."""
    return _replay(frames, "S005")


def s006(frames):
    """S006: as S001 but the test is back1 >= 1.5 x predicted_bfsp (min_move ln 1.5) and expected CLV >= 0; top-ups
    and ladder levels need both at their own price."""
    b, days = _replay(frames, "S006")
    if len(b):
        _describe("S006", "by price band", b, pd.cut(b["entry_price"], [1, 5, 12, 25, 50, 1000]).astype(str))
    return b, days


def s007(frames):
    """S007: as S001 and every price taken (first back, top-up, each ladder level) at or above Betfair's forecast
    price F = 1 + forecast_num / forecast_den as betfair_live_markets stores it; a runner with no F is not backed.
    Unverified: that the stored forecast is the morning's (the loader's FCHECK pass counts the runners whose first
    recorded forecast differs from the stored one; it is printed here)."""
    fc = _frame(frames, "day_FCHECK")
    if len(fc):
        _say(f"   S007: forecast price, first recorded (markets.csv.gz) against stored: {int(fc['n'].sum()):,} "
             f"runners compared, {int(fc['n_diff'].sum()):,} differ ({fc['n_diff'].sum() / max(fc['n'].sum(), 1):.2%})")
    else:
        _say("   S007: no data found for the forecast check (no raw markets file read)")
    return _replay(frames, "S007")


def s009(frames):
    """S009: as S001, new backs and top-ups only in the 60 minutes from the day's first recorded book at or after
    05:15 UK (the first poll of any of the day's WIN markets in the trader's and the recorder's books)."""
    return _replay(frames, "S009")


def s010(frames):
    """S010: S001's session unchanged to 10:00 UK, then to 13:00 UK a market is tradable only if at least 360
    minutes from its off at the read; the same test, the horse's 3 top-ups and the GBP4,000 day limit shared."""
    b, days = _replay(frames, "S010")
    if len(b) and "ts" in b.columns:
        _describe("S010", "by window", b, np.where(_uk_hhmm(b["ts"]) > "10:00", "10:00-13:00, 6h+ out", "to 10:00"))
    f = _frame(frames, "fills")
    if len(f):
        cell = f[(f["race_date"] >= "2026-10-10") & (f["race_date"] <= "2026-11-04") & (f["minutes_to_off"] >= 360)
                 & _uk_hhmm(f["ts"]).between("10:00", "12:59").to_numpy()]
        if len(cell):
            _describe("S010", "the real fills 10 Oct - 4 Nov, 10:00-12:59 UK, 6h+ out", _fill_bets(cell), "real fills")
    return b, days


def s013(frames):
    """S013: as S001 for the first order on each horse (it takes back1-back3 by the ladder); no top-ups."""
    return _replay(frames, "S013")


def s014(frames):
    """S014: as S001; a top-up only at a read no more than 30 minutes after the horse's first matched back in the
    replay (at most 3)."""
    b, days = _replay(frames, "S014")
    f = _frame(frames, "fills")
    if len(f):
        t0 = f.groupby(["race_date"] + KEY)["ts"].transform("min")
        late = (f["ts"] - t0).dt.total_seconds() > 1800
        _describe("S014", "the same filter on BASE-B", _fill_bets(f),
                  np.where(f["first"], "first order", np.where(late, "top-up after 30 min (dropped)", "top-up within 30 min")))
    return b, days


def s017(frames):
    """S017: as S001 with no ladder: each order takes the best back only, up to its size."""
    return _replay(frames, "S017")


def s018(frames):
    """S018: as S001, and at a best back of 21.0 or more the horse's target is max(to-win-400 stake, GBP50), cap
    200 (research/arms' floor_long, the arm 'floor50')."""
    b, days = _replay(frames, "S018")
    if len(b):
        _describe("S018", "days seen by arms-full-1103 against 2-23 Nov", b,
                  np.where(b["race_date"] >= "2026-11-02", "2-23 Nov (unseen)", "to 1 Nov"))
    return b, days


def s019(frames):
    """S019: as S001 (bar 3% at every hour), the horse's target set by the hours to the off at its first matched
    back and kept for its top-ups: 6 or more to win GBP800 (cap 300), 3 to under 6 to win 400 (cap 200), under 3
    to win 200 (cap 100); floor 15, GBP4,000 a day."""
    return _replay(frames, "S019")


def s020(frames):
    """S020: as S001 with a level GBP40 a horse, top-ups to that stake, at most 3 (the live plan always stakes to
    win, so the level stake is set at the order, not by TradingConfig.staking)."""
    return _replay(frames, "S020")


def s037(frames):
    """S037: the lay book's core cut, replayed (research/arms LaySession): 05:15-08:00 UK, 15 minutes before the
    off, whole races; lay where 1.01 <= lay1 <= 10, lay1 / back1 <= 1.10 (a runner with no back is not laid),
    1 - lay1 / E >= 3% and lay1 <= our price; one lay a runner a day (no top-up), lay2 and lay3 only while each
    clears every test; GBP400 at risk a horse (stake min(max(400 / (L - 1), 15), 200)), GBP4,000 day liability;
    closed by a back at the BSP after CUT. The base lay book's own forward figure is not computed or printed."""
    b, days = _replay(frames, "S037")
    if len(b):
        _describe("S037", "by period", b, np.where(b["race_date"] <= "2026-11-12", "1 Oct - 12 Nov", "13-23 Nov"))
    return b, days


# ---------------------------------------------------------------------------------------------------------------
# BASE-B: the real fills, and the variants that drop, cut or re-settle them
# ---------------------------------------------------------------------------------------------------------------

def _fills(frames) -> pd.DataFrame:
    f = _frame(frames, "fills")
    if f.empty:
        raise NoData("no data found: live_orders holds no matched back in the window")
    return f


def _fill_bets(f: pd.DataFrame, stake=None, exit="bsp") -> pd.DataFrame:
    """The real fills as bets: the matched stake at the order's price, cut by the fill's cut_factor (K6), closed at
    the BSP of the horse's settle row; a horse with no settle row, no BSP or REMOVED is void."""
    if len(f) == 0:
        return core.empty_bets()
    return _bets(f.assign(final_status=f["result"].astype(object).where(f["result"].notna(), "").astype(str)), "back",
                 f["avg_price"], f["matched"] if stake is None else stake, exit=exit, cut=f["cut_factor"],
                 entry_type=NO_HAIRCUT, ts=f["ts"])


def base_b(frames):
    """BASE-B (K7): every matched back of the live trader, the rule as it was live each day, one row an order; a
    fill's P&L is matched x (p_cut / BSP - 1); 2% on each market's positive net. A day with no fill is 0."""
    f = _fills(frames)
    b = _fill_bets(f)
    _describe("BASE-B", "first orders and top-ups", b, np.where(f["first"], "first order", "top-up"))
    return b, _days(frames)


def s015(frames):
    """S015: BASE-B with every first order kept and a top-up kept only if its own ledger edge is at least 7%;
    nothing is redeployed. A top-up whose edge the ledger does not carry is dropped."""
    f = _fills(frames)
    keep = f["first"] | (f["edge"] >= 0.07)
    b = _fill_bets(f[keep])
    staked = f.groupby("race_date")["matched"].transform("sum")
    _describe("S015", "kept fills, by the day's real stake", b,
              np.where(staked[keep] < 3900, "days under GBP3,900", "days at the limit"))
    return b, _days(frames)


def s016(frames):
    """S016: BASE-B less every top-up matched at an average price under 4.0; first orders at any price stay."""
    f = _fills(frames)
    return _fill_bets(f[f["first"] | (f["avg_price"] >= 4.0)]), _days(frames)


EDGE_BINS = [0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.25]          # 3-4, 4-5, 5-6, 6-8, 8-10, 10-15, 15-25, 25%+
PRICE_BANDS = [3.0, 5.0, 8.0, 13.0, 21.0]                       # <3, 3-5, 5-8, 8-13, 13-21, 21+
HOUR_BANDS = [3.0, 6.0]                                         # <3, 3-6, 6+
KELLY_SCALE, KELLY_RATIO, KELLY_BURN, KELLY_MIN_MARKS = 80.0, 0.45, 10, 30


def s021(frames):
    """S021: BASE-B, each horse's stake capped at K = GBP80 x (mu / var) / 0.45, held in 0-200; its fills are kept
    in time order until K is reached, the fill that crosses it is cut.
    mu: the stake-weighted mean settled CLV (p_cut / BSP - 1) of the fills of ALL EARLIER DAYS whose horse's first
        order had its ledger edge in the same bin (3-4 ... 25%+; an edge under 3% reads as the first bin).
    var: the variance of log(back1 / BSP) over the runner-marks (every mark c08..o1, WIN markets) of ALL EARLIER
        DAYS in the same price band (of the horse's first fill) and hours-to-off band (at that fill); a cell with
        fewer than 30 runner-marks has no variance.
    mu <= 0, or no mu or var for the cell: no fill. The first 10 days of fills are burn-in and are not scored."""
    f = _fills(frames)
    r = _runners(frames)
    hk = ["race_date"] + KEY
    days_f = sorted(f["race_date"].unique())
    scored = [d for d in _days(frames) if len([x for x in days_f if x < d]) >= KELLY_BURN]
    if not scored:
        return _none("S021", "fewer than 10 earlier days of fills", scored)
    first = f.sort_values(["ts", "seq"], kind="stable").drop_duplicates(hk, keep="first")
    first = first.assign(ebin=np.digitize(first["edge"].fillna(-1.0), EDGE_BINS),
                         pband=np.digitize(first["avg_price"], PRICE_BANDS),
                         hband=np.digitize(first["minutes_to_off"].fillna(-1e9) / 60.0, HOUR_BANDS))
    f = f.merge(first[hk + ["ebin", "pband", "hband"]], on=hk, how="left")
    ok = f["bsp"].gt(1.0) & f["result"].isin(["WINNER", "LOSER"])
    clv = np.maximum(1.01, f["avg_price"] * f["cut_factor"]) / f["bsp"] - 1.0
    d = f[ok].assign(num=(f["matched"] * clv)[ok], den=f.loc[ok, "matched"]).groupby(["ebin", "race_date"])[["num", "den"]].sum()
    mu = {}
    for e, g in d.groupby(level=0):
        g = g.droplevel(0).reindex(days_f + [x for x in scored if x not in days_f]).sort_index().fillna(0.0)
        c = g.cumsum().shift(1).fillna(0.0)
        mu.update({(e, day): (c.loc[day, "num"] / c.loc[day, "den"]) for day in c.index if c.loc[day, "den"] > 0})
    cells = []
    for tag in [t for t in B.TAGS.values() if t != "last"]:
        b = book(r, tag)
        b = b[b["back1"].gt(1.0) & b["bsp"].gt(1.0)]
        cells.append(pd.DataFrame({"race_date": b["race_date"], "x": np.log(b["back1"] / b["bsp"]),
                                   "pband": np.digitize(b["back1"], PRICE_BANDS),
                                   "hband": np.digitize(b["mto"] / 60.0, HOUR_BANDS)}))
    cells = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame(columns=["race_date", "x", "pband", "hband"])
    v = cells.assign(x2=cells["x"] ** 2, n=1.0).groupby(["pband", "hband", "race_date"])[["n", "x", "x2"]].sum()
    var = {}
    all_days = sorted(set(cells["race_date"]) | set(scored))
    for (pb, hb), g in v.groupby(level=[0, 1]):
        c = g.droplevel([0, 1]).reindex(all_days).fillna(0.0).cumsum().shift(1).fillna(0.0)
        ok_n = c["n"] >= KELLY_MIN_MARKS
        vv = (c["x2"] - c["x"] ** 2 / c["n"].where(ok_n)) / (c["n"] - 1.0)
        var.update({(pb, hb, day): val for day, val in vv.items() if np.isfinite(val) and val > 0})
    f = f[f["race_date"].isin(set(scored))].sort_values(["ts", "seq"], kind="stable").reset_index(drop=True)
    m = np.array([mu.get((e, day), np.nan) for e, day in zip(f["ebin"], f["race_date"])], float)
    s2 = np.array([var.get((p, h, day), np.nan) for p, h, day in zip(f["pband"], f["hband"], f["race_date"])], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cap = np.clip(KELLY_SCALE * (m / s2) / KELLY_RATIO, 0.0, 200.0)
    cap = np.where(np.isfinite(cap) & (m > 0), cap, 0.0)
    before = f.groupby(hk)["matched"].cumsum() - f["matched"]
    stake = np.clip(cap - before.to_numpy(), 0.0, f["matched"].to_numpy())
    _say(f"   S021: {len(scored)} scored days; GBP{stake.sum():,.0f} kept of GBP{f['matched'].sum():,.0f}; horses with "
         f"no cap (mu <= 0 or an empty cell) {int((pd.Series(cap).groupby([f[c] for c in hk]).first() == 0).sum()):,}")
    return _fill_bets(f, stake=stake), scored


def s022(frames):
    """S022: BASE-B under a GBP400 cap on all backs in one market: its fills in time order are kept while the
    market's matched stake is under 400, the fill that crosses it is cut to the remainder, later fills dropped."""
    f = _fills(frames).sort_values(["ts", "seq"], kind="stable").reset_index(drop=True)
    before = f.groupby(["race_date", "market_id"])["matched"].cumsum() - f["matched"]
    stake = np.clip(400.0 - before.to_numpy(), 0.0, f["matched"].to_numpy())
    horses = f.groupby(["race_date", "market_id"])["selection_id"].transform("nunique")
    _describe("S022", "BASE-B by horses backed in the race", _fill_bets(f),
              np.where(horses >= 4, "4+", horses.astype(str) + " horse(s)"))
    return _fill_bets(f, stake=stake), _days(frames)


def s023(frames):
    """S023: BASE-B less every fill matched after t_m, the first raw-book poll at which a runner of its market
    shows REMOVED with a reduction factor of at least 2.5 (known to one poll). Only days with raw books are read."""
    f = _fills(frames)
    days = _days(frames, "books")
    if not days:
        return _none("S023", "no raw books (the removals are not known)", days)
    rem = _frame(frames, "removals")
    tm = rem[rem["af"] >= 2.5].groupby("market_id")["first_seen"].min() if len(rem) else pd.Series(dtype="datetime64[ns, UTC]")
    t_m = pd.to_datetime(f["market_id"].map(tm), utc=True)
    after = (f["ts"] > t_m).fillna(False).to_numpy()
    _describe("S023", "BASE-B by the fill's place against the market's first removal", _fill_bets(f),
              np.where(after, "after a removal (dropped)", "before any removal"))
    return _within(_fill_bets(f[~after]), days), days


def s025(frames):
    """S025: BASE-B held to the result (no SP lay): a winner pays s x (p_cut - 1), a loser costs s; 2%."""
    b = _fill_bets(_fills(frames), exit="result")
    return b, _days(frames)


def s026(frames):
    """S026: BASE-B closed by a LIMIT_ON_CLOSE lay at the fill's own cut price p_cut = max(1.01, price x cut): where
    the BSP is at or under p_cut the fill is hedged at the BSP as live (K3), where the BSP is over it the lay is
    not matched and the back runs to the result. The BSP decides the fill of the order, not the selection."""
    f = _fills(frames)
    p_cut = np.maximum(1.01, f["avg_price"] * f["cut_factor"])
    hedged = (f["bsp"] <= p_cut).to_numpy()
    return _fill_bets(f, exit=np.where(hedged, "bsp", "result")), _days(frames)


def s027(frames):
    """S027: every horse the live trader backed and settled to 22 Nov (one row a horse: the settle row's matched
    stake at Betfair's settled price), joined to betfair_prices' win row on (race_date, selection_id). No SP lay:
    a lay of matched x price / L rests in running at L = 1 + (BSP - 1) x 0.5 and is matched only if the in-play
    low is at or under one tick below L (K8); matched, the back is closed at L on every outcome; unmatched, it
    runs to the result. An upper bound: the files hold no volume at a price. A horse with no price-file row is
    void."""
    s = _frame(frames, "settles")
    if s.empty:
        raise NoData("no data found: live_orders holds no settle row")
    days = _days(frames, last="2026-11-22")
    s = s[s["race_date"].isin(set(days)) & s["matched"].gt(0) & s["settled_price"].gt(1.0)]
    p = _frame(frames, "prices")
    p = p[p["market_type"].eq("win")].drop_duplicates(["race_date", "selection_id"], keep=False) if len(p) else p
    if p.empty:
        return _none("S027", "betfair_prices holds no win row of the window", days)
    x = s.merge(p[["race_date", "selection_id", "bsp", "ipmin", "win_lose"]].rename(columns={"bsp": "bsp_file"}),
                on=["race_date", "selection_id"], how="left")
    lay_at = 1.0 + (x["bsp_file"] - 1.0) * 0.5
    with np.errstate(invalid="ignore"):
        hit = (x["ipmin"].to_numpy(float) > 1.0) & (x["ipmin"].to_numpy(float) <= np.asarray(
            core.tick_under(lay_at.to_numpy(float)), float).reshape(-1))
    known = x["bsp_file"].gt(1.0) & x["win_lose"].isin([0.0, 1.0]) & x["ipmin"].gt(1.0)
    _say(f"   S027: {len(x):,} settled horses to 22 Nov, {int(known.sum()):,} with a price-file row; the lay at half "
         f"the SP odds matched on {int((hit & known).sum()):,}")
    b = _bets(x.assign(final_status=np.where(x["win_lose"].eq(1.0), "WINNER", np.where(x["win_lose"].eq(0.0), "LOSER", ""))),
              "back", x["settled_price"], x["matched"], exit=np.where(hit, "inplay", "result"),
              exit_price=np.where(hit, lay_at, np.nan), void=~known.to_numpy(), entry_type=NO_HAIRCUT)
    return b, days


# ---------------------------------------------------------------------------------------------------------------
# the rule at a mark (S008, S011, S012, S024, S028, S029) and the morning controls (S030-S034)
# ---------------------------------------------------------------------------------------------------------------

def _rule_orders(b: pd.DataFrame, to_win: float = TO_WIN, guard: str = "predicted_bfsp") -> tuple:
    """The rule's one order at a mark for rows that clear the bar: to win ``to_win`` at back1 (GBP15-200), back1
    then back2 and back3 while each level's own expected CLV (price / xb - 1) is at least 3% and its price at
    least the guard; sizes shown. Returns (stake, price)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = [(b[f"back{i}"] / b["xb"] - 1.0 >= BAR) & (b[f"back{i}"] >= b[guard]) for i in (2, 3)]
    return ladder(b, core.to_win_stake(to_win, b["back1"], FLOOR, CAP), ok[0], ok[1])


def _s008(frames, corrected: bool):
    r = _runners(frames)
    days = _days(frames, "books", "preds", first="2026-10-11")
    if not days:
        return _none("S008", "no day with raw books and a model price after the 10-day burn-in", days)
    w = r[r["market_type"].eq("WIN") & r["bsp"].gt(1.0) & r["F"].gt(1.0) & r["predicted_bfsp"].gt(1.0)]
    x, y = np.log(w["F"] / w["predicted_bfsp"]), np.log(w["bsp"] / w["predicted_bfsp"])
    by = pd.DataFrame({"sxy": x * y, "sxx": x * x}).groupby(w["race_date"]).sum().reindex(
        sorted(set(w["race_date"]) | set(days))).fillna(0.0)
    cum = by.cumsum().shift(1)                                  # earlier days only
    slope = (cum["sxy"] / cum["sxx"].where(cum["sxx"] > 0)).to_dict()
    picks = []
    for tag in ("c08", "c09", "c10"):
        b = book(r, tag)
        b = b[b["race_date"].isin(set(days))].reset_index(drop=True)
        b["slope"] = b["race_date"].map(slope) if corrected else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.exp(np.log(b["predicted_bfsp"]) + b["slope"] * (np.log(b["F"]) - np.log(b["predicted_bfsp"])))
        raw = raw.where(b["F"].gt(1.0) & b["predicted_bfsp"].gt(1.0))
        g = b["market_id"]
        scale = (1.0 / raw).groupby(g).transform("sum") / (1.0 / b["predicted_bfsp"]).groupby(g).transform("sum")
        whole = raw.notna().groupby(g).transform("all")
        b["p2"] = (raw * scale).where(whole)                    # sum(1 / p2) = sum(1 / our price) in the race
        b = closing_view(b, "p2")
        hit = (b["ev"] >= BAR) & (b["back1"] >= b["p2"]) & (b["back1"] <= MAX_PRICE) & (b["mto"] >= 15)
        picks.append(b[hit].assign(mark=tag))
    p = pd.concat(picks, ignore_index=True).drop_duplicates(["race_date"] + KEY, keep="first")
    stake, price = _rule_orders(p, guard="p2")
    return _bets(p, "back", price, stake, cut=_cut(frames, p), ts=p["t"]), days


def s008(frames):
    """S008: the rule with our price corrected by Betfair's forecast price: p2 = exp(log p + b x (log F - log p)),
    rescaled so the race's sum of 1 / p2 equals that of 1 / p; b is the no-intercept OLS slope of log(BSP / p) on
    log(F / p) over every WIN runner of EARLIER days. 1-10 Oct is burn-in (not scored). At the first of the marks
    08:00, 09:00, 10:00 where LIVE-BAR holds with p2 in place of our price (whole race: every runner ACTIVE at
    the mark has a back, our price and F; at least 15 minutes to the off), one order: back1 then back2, back3
    while each clears the 3% bar and is at least p2; to win GBP400, GBP15-200. Closed at the BSP after CUT."""
    return _s008(frames, True)


def s008_base(frames):
    """S008's benchmark: the same code with b = 0 (our price unchanged), the same races (a forecast price on every
    runner is still required, so the two differ only by the correction)."""
    return _s008(frames, False)


def s011(frames):
    """S011: one look at the clock_08:00 mark (market OPEN, at least 15 minutes to the off): every runner where
    LIVE-BAR holds; one order a horse (back1 then back2, back3 while each level's own expected CLV is at least 3%
    and its price at least ours), to win GBP400, GBP15-200; GBP4,000 a day filled in descending expected CLV.
    Closed at the BSP after CUT for the removals after the mark."""
    days = _days(frames, "books", "preds")
    b = live_bar(frames, "c08")
    b = b[b["bar"] & (b["mto"] >= 15) & b["race_date"].isin(set(days))].reset_index(drop=True)
    stake, price = _rule_orders(b)
    stake = day_limit(b, stake, ["ev"], ascending=[False])
    return _bets(b, "back", price, stake, cut=_cut(frames, b), ts=b["t"]), days


def _clogit_print(sid: str, b: pd.DataFrame) -> None:
    """Descriptive: the conditional-logit weights of the winner on the mark's share and on our price's share,
    walk-forward by day (research/harness.clogit_fit). Reads results: never part of a selection."""
    try:
        from research.harness import clogit_fit
        x = _mids(b)
        x = x[x["whole"] & x["predicted_bfsp"].gt(1.0)].copy()
        x["q_model"] = core.normalise(x["predicted_bfsp"], x["market_id"], whole=True)
        x["won"] = x["final_status"].eq("WINNER").astype(int)
        one = x.groupby("market_id")["won"].transform("sum").eq(1) & x["q_model"].notna()
        x = x[one]
        days, w = sorted(x["race_date"].unique()), []
        for d in days[7:]:
            tr = x[x["race_date"] < d]
            w.append(clogit_fit(np.log(tr[["share", "q_model"]].to_numpy()), pd.factorize(tr["market_id"])[0],
                                tr["won"].to_numpy()))
        if w:
            w = np.array(w)
            _say(f"   {sid} (descriptive): conditional-logit weights walk-forward over {len(w)} days: the mark's mid "
                 f"{w[:, 0].mean():.3f} (last {w[-1, 0]:.3f}), our price {w[:, 1].mean():.3f} (last {w[-1, 1]:.3f})")
    except Exception as exc:  # noqa: BLE001
        _say(f"   {sid}: the conditional-logit weights were not printed ({type(exc).__name__}: {exc})")


def s012(frames):
    """S012: as S011 at each market's own off_120 mark, all day; the GBP4,000 is spent in the order of the marks'
    poll times (then by descending expected CLV)."""
    days = _days(frames, "books", "preds")
    b = live_bar(frames, "o120")
    _clogit_print("S012", b)
    b = b[b["bar"] & (b["mto"] >= 15) & b["race_date"].isin(set(days))].reset_index(drop=True)
    stake, price = _rule_orders(b)
    stake = day_limit(b, stake, ["t", "ev"], ascending=[True, False])
    return _bets(b, "back", price, stake, cut=_cut(frames, b), ts=b["t"]), days


def _s024(frames, other: bool):
    sid = "S024-OTHER" if other else "S024"
    _day_errors(frames, "S024")
    days = _days(frames, "books", "preds")
    t = _frame(frames, "day_S024")
    if t.empty or not days:
        return _none(sid, "no raw poll near 08:00 UK (the reduction factors of the runners still in are not known)", days)
    sub = t.assign(s=t["share"].where(t["af"] < 2.5, 0.0)).groupby("market_id")["s"].sum()
    free = set(sub.index[sub >= 0.05])
    small = set(zip(t.loc[t["af"] < 2.5, "market_id"], t.loc[t["af"] < 2.5, "selection_id"]))
    b = live_bar(frames, "c08")
    b = b[b["bar"] & b["race_date"].isin(set(days)) & b["market_id"].isin(set(sub.index))].reset_index(drop=True)
    in_free = b["market_id"].isin(free).to_numpy()
    is_small = np.array([(m, s) in small for m, s in zip(b["market_id"], b["selection_id"])], bool)
    if not other:
        _say(f"   S024: markets read at 08:00 {len(sub):,}; with 5% or more of the book on runners whose factor is "
             f"under 2.5: {len(free):,}; rule picks in them {int(in_free.sum()):,} (sub-2.5 runners left out "
             f"{int((in_free & is_small).sum()):,})")
    b = b[~in_free if other else (in_free & ~is_small)].reset_index(drop=True)
    stake, price = _rule_orders(b)
    return _bets(b, "back", price, stake, cut=_cut(frames, b, min_af=B.BETFAIR_MIN_AF), ts=b["t"]), days


def s024(frames):
    """S024: at the raw-book poll nearest 08:00 UK (within 5 minutes) a market qualifies if the summed share of its
    ACTIVE runners whose adjustment_factor is under 2.5 is at least 5% (share from the mid, from the back where no
    lay is shown; a runner with no back counts 0; a runner whose factor the book does not give is not counted).
    In those markets the runners where LIVE-BAR holds at clock_08:00 are backed, never one of the sub-2.5 runners:
    one order (back1-back3 by the bar and the guard), to win GBP400, GBP15-200, no day limit. Closed at the BSP
    after CUT read by Betfair's own rule, for this strategy and its benchmark only: a factor under 2.5% cuts
    nothing (the strategy's whole premise; every other back here is cut by every factor)."""
    return _s024(frames, False)


def s024_other(frames):
    """S024's benchmark: the same order on S011's 08:00 picks in the OTHER markets (read at 08:00, under 5% of the
    book on sub-2.5 runners), cut as S024 is (factors of at least 2.5)."""
    return _s024(frames, True)


def s028(frames):
    """S028: the runners where LIVE-BAR held at clock_08:00 that are two-sided at 08:00 and 10:00 with the same
    number of active runners at both and a mid at 10:00 at most 0.95 x the mid at 08:00, and at 10:00 back1 >=
    our price and <= 200. One order at the clock_10:00 mark: back1, then back2 and back3 at prices >= our price;
    sizes shown; to win GBP200, GBP15-200. The 3% bar is not required at 10:00. Closed at the BSP after CUT."""
    days = _days(frames, "books", "preds")
    r = _runners(frames)
    a = _mids(live_bar(frames, "c08"))
    c = _mids(book(r, "c10"))
    x = c.merge(a.loc[a["bar"], KEY + ["mid", "n_active", "t"]].rename(
        columns={"mid": "mid08", "n_active": "n08", "t": "t08"}), on=KEY, how="inner")
    x = x[x["race_date"].isin(set(days)) & (x["n_active"] == x["n08"]) & x["mid"].notna() & x["mid08"].notna()]
    drift = x[(x["mid"] / x["mid08"] >= 1.08)]
    x = x[(x["mid"] / x["mid08"] <= 0.95) & (x["back1"] >= x["predicted_bfsp"]) & (x["back1"] <= MAX_PRICE)]
    x = x.reset_index(drop=True)
    stake, price = ladder(x, core.to_win_stake(200.0, x["back1"], FLOOR, CAP), x["back2"] >= x["predicted_bfsp"],
                          x["back3"] >= x["predicted_bfsp"])
    b = _bets(x, "back", price, stake, cut=_cut(frames, x), ts=x["t"])
    if len(drift):
        d = drift.reset_index(drop=True)
        _describe("S028", "beside it: the 08:00 picks that drifted 8% or more, backed at 10:00's back1",
                  _bets(d, "back", d["back1"], core.capacity(core.to_win_stake(200.0, d["back1"], FLOOR, CAP), d["back1_size"])),
                  "drifters")
    _describe("S028", "by live-trader horse", b, _trader_label(frames, b))
    return b, days


def s029(frames):
    """S029: at clock_08:00, runners with back1 <= 30 and lay1 / back1 <= 1.5. Five votes, a missing input voting 0:
    (a) LIVE-BAR holds; (b) F <= 0.90 x back1; (c) the steamer's score at or above the 80th percentile of its
    training scores; (d) days since the last run 1-4; (e) the mid at 08:00 at most 0.97 x the mid at the runner's
    first raw poll at or after 05:15 UK with lay1 / back1 <= 1.5 (that poll before the mark). Back with 3 or more:
    back1, and back2, back3 at >= 0.97 x back1; sizes shown; to win GBP200, GBP15-200. Closed at the BSP after
    CUT."""
    _day_errors(frames, "S029")
    days = _days(frames, "books")
    meta = _meta(frames)
    b = _mids(live_bar(frames, "c08"))
    b = b[b["race_date"].isin(set(days)) & (b["back1"] <= 30.0) & (b["spread"] <= 1.5)].reset_index(drop=True)
    st, fm = _frame(frames, "steam"), _frame(frames, "day_S029")
    b = b.merge(st[KEY + ["steam_score"]].drop_duplicates(KEY), on=KEY, how="left") if len(st) else b.assign(steam_score=np.nan)
    b = b.merge(fm[KEY + ["first_mid", "first_t"]].drop_duplicates(KEY), on=KEY, how="left") if len(fm) \
        else b.assign(first_mid=np.nan, first_t=pd.NaT)
    p80 = meta.get("steam_p80")
    if p80 is None:
        _say("   S029: no data found for the steamer's vote (it votes 0)")
    early = pd.to_datetime(b["first_t"], utc=True) < b["t"]
    votes = (b["bar"].fillna(False).astype(int) + (b["F"] <= 0.90 * b["back1"]).astype(int)
             + ((b["steam_score"] >= p80) if p80 is not None else pd.Series(False, index=b.index)).astype(int)
             + b["dslr"].between(1, 4).astype(int)
             + (early & (b["mid"] <= 0.97 * b["first_mid"])).astype(int))
    all_b = _bets(b, "back", b["back1"], 10.0, votes=votes.to_numpy())
    if len(all_b):
        _describe("S029", "every candidate at back1 (GBP10) by its vote count", all_b, all_b["votes"].astype(str))
    x = b[votes >= 3].reset_index(drop=True)
    stake, price = ladder(x, core.to_win_stake(200.0, x["back1"], FLOOR, CAP), x["back2"] >= 0.97 * x["back1"],
                          x["back3"] >= 0.97 * x["back1"])
    return _bets(x, "back", price, stake, cut=_cut(frames, x), ts=x["t"]), days


def s030(frames):
    """S030: quick returners at the offered 08:00 price: WIN runners with days_since_last_run 1-4 in a Flat, Hurdle
    or Chase race; at clock_08:00 ACTIVE, 5 or more active runners, back1 and lay1 present, a last traded price,
    back1 <= 50. One bet: back1, to win GBP50, at most GBP100 and at most back1's size, none under GBP2. Closed
    at the BSP after CUT."""
    days = _days(frames, "books")
    b = book(_runners(frames), "c08")
    x = b[b["race_date"].isin(set(days)) & b["dslr"].between(1, 4)
          & b["race_type"].str.strip().str.lower().isin(["flat", "hurdle", "chase"])
          & (b["n_active"] >= 5) & b["back1"].gt(1.0) & b["lay1"].gt(1.0) & b["last_traded"].gt(1.0)
          & (b["back1"] <= 50.0)].reset_index(drop=True)
    stake = core.capacity(core.to_win_stake(50.0, x["back1"], 0.0, 100.0), x["back1_size"])
    out = _bets(x, "back", x["back1"], stake, cut=_cut(frames, x), ts=x["t"])
    _describe("S030", "by live-trader horse", out, _trader_label(frames, out))
    return out, days


def s031(frames):
    """S031: the CLEAN steamer (fitted once on 2018-01-01..2026-03-27, the loader) scored on each day's first
    racecard; runners at or above the fit's training 95th percentile in a field of 5 or more; at clock_08:00
    ACTIVE with back1 <= 200. One bet: back1, to win GBP50, at most GBP100 and back1's size, none under GBP2.
    Closed at the BSP after CUT. The live morning model (fitted to 31 Oct) is not used."""
    days = _days(frames, "books", "steam")
    st, p95 = _frame(frames, "steam"), _meta(frames).get("steam_p95")
    if st.empty or p95 is None or not days:
        return _none("S031", "the steamer's scores (no fit, no racecard, or STRAT100_B_STEAMER=0)", days)
    b = book(_runners(frames), "c08").merge(st[KEY + ["steam_score", "steam_field"]].drop_duplicates(KEY), on=KEY)
    x = b[b["race_date"].isin(set(days)) & (b["steam_score"] >= p95) & (b["steam_field"] >= 5) & b["back1"].gt(1.0)
          & (b["back1"] <= MAX_PRICE)].reset_index(drop=True)
    stake = core.capacity(core.to_win_stake(50.0, x["back1"], 0.0, 100.0), x["back1_size"])
    out = _bets(x, "back", x["back1"], stake, cut=_cut(frames, x), ts=x["t"])
    _describe("S031", "by live-trader horse", out, _trader_label(frames, out))
    return out, days


def s032_day(ctx) -> pd.DataFrame:
    """S032's day: at the first of the marks 08:00, 09:00, 10:00 where a runner has a last traded price, back1 >=
    1.12 x it, back1 <= 30, a lay shown, the market's matched >= GBP100 and 60 minutes or more to the off; then
    the first raw read of that runner 60 s to 6 minutes after the mark (ACTIVE, market OPEN, not in play) must
    still show back1 >= 1.12 x its own last traded price. Not confirmed there: no bet (a later mark is not
    tried). The bet: the confirming read's back1, to win GBP100, at most GBP50 and that back1's size."""
    cand = []
    for tag in ("c08", "c09", "c10"):
        b = book(ctx.runners, tag)
        cand.append(b[b["last_traded"].gt(1.0) & (b["back1"] >= 1.12 * b["last_traded"]) & (b["back1"] <= 30.0)
                      & b["lay1"].gt(1.0) & (b["mtm"] >= 100.0) & (b["mto"] >= 60.0)].assign(mark=tag))
    c = pd.concat(cand, ignore_index=True).drop_duplicates(KEY, keep="first")
    if c.empty:
        return core.empty_bets()
    raw = ctx.raw
    live = raw[raw["status"].eq("OPEN") & ~raw["inplay"] & raw["runner_status"].eq("ACTIVE")]
    j = c[KEY + ["t"]].rename(columns={"t": "t_mark"}).merge(
        live[KEY + ["t", "back1", "back1_size", "last_traded"]], on=KEY, how="inner")
    gap = (j["t"] - j["t_mark"]).dt.total_seconds()
    j = j[(gap >= 60.0) & (gap <= 360.0)].sort_values("t", kind="stable").drop_duplicates(KEY, keep="first")
    j = j[j["back1"].gt(1.0) & j["last_traded"].gt(1.0) & (j["back1"] >= 1.12 * j["last_traded"])]
    x = c.drop(columns=["t", "back1", "back1_size", "last_traded"]).merge(
        j[KEY + ["t", "back1", "back1_size", "last_traded"]], on=KEY, how="inner")
    stake = core.capacity(core.to_win_stake(100.0, x["back1"], 0.0, 50.0), x["back1_size"])
    return _bets(x, "back", x["back1"], stake, ts=x["t"],
                 cut=B.cut_factor(ctx.removals, x["market_id"], x["selection_id"], x["t"], B.min_af_for("back")))


def _from_day(frames, sid: str, why: str, first: str | None = None) -> tuple:
    _day_errors(frames, sid)
    days = _days(frames, "books", first=first)
    b = _frame(frames, f"day_{sid}")
    if b.empty or not days:
        return _none(sid, why if not days else "the rule found no bet on any day", days)
    return _within(b, days), days


def s032(frames):
    """S032: the rule with the last traded price in place of the model (s032_day): a best back 12% over the last
    trade that is still there a minute later. No model. Closed at the BSP after CUT."""
    b, days = _from_day(frames, "S032", "no raw books")
    _describe("S032", "by live-trader horse", b, _trader_label(frames, b) if len(b) else [])
    return b, days


def s033(frames):
    """S033: at clock_08:00, ACTIVE runners with back1 >= 1.30 x F, F <= 15 and lay1 / back1 <= 1.25: back1, to win
    GBP100 and at most back1's size (no cap is named), none under GBP2. No model. Closed at the BSP after CUT."""
    days = _days(frames, "books")
    b = _mids(book(_runners(frames), "c08"))
    x = b[b["race_date"].isin(set(days)) & (b["back1"] >= 1.30 * b["F"]) & (b["F"] <= 15.0)
          & (b["spread"] <= 1.25)].reset_index(drop=True)
    stake = core.capacity(core.to_win_stake(100.0, x["back1"]), x["back1_size"])
    return _bets(x, "back", x["back1"], stake, cut=_cut(frames, x), ts=x["t"]), days


def s034(frames):
    """S034: runners two-sided at clock_08:00 and clock_10:00 with the same number of active runners at both and a
    mid at 10:00 at most 0.92 x the mid at 08:00; at 10:00 lay1 / back1 - 1 <= 5%, mid <= 15, the market's matched
    >= GBP1,000 and 60 minutes or more to the off. One bet at 10:00's back1: to win GBP50, at most its size, none
    under GBP2. No model. Closed at the BSP after CUT."""
    days = _days(frames, "books")
    r = _runners(frames)
    a, c = _mids(book(r, "c08")), _mids(book(r, "c10"))
    x = c.merge(a[KEY + ["mid", "n_active"]].rename(columns={"mid": "mid08", "n_active": "n08"}), on=KEY, how="inner")
    x = x[x["race_date"].isin(set(days)) & (x["n_active"] == x["n08"]) & (x["mid"] <= 0.92 * x["mid08"])
          & (x["spread"] - 1.0 <= 0.05) & (x["mid"] <= 15.0) & (x["mtm"] >= 1000.0) & (x["mto"] >= 60.0)]
    x = x.reset_index(drop=True)
    stake = core.capacity(core.to_win_stake(50.0, x["back1"]), x["back1_size"])
    return _bets(x, "back", x["back1"], stake, cut=_cut(frames, x), ts=x["t"]), days


def _share_move(r: pd.DataFrame, tag_a: str, tag_b: str) -> pd.DataFrame:
    """ln(share at tag_b / share at tag_a) for the runners of WIN markets whole at both marks with the same number
    of active runners (a removal between the two moves every share by itself)."""
    a, b = _mids(book(r, tag_a)), _mids(book(r, tag_b))
    m = a.loc[a["whole"], KEY + ["share", "n_active", "mtm", "mid"]].rename(
        columns={"share": "share_a", "n_active": "n_a", "mtm": "mtm_a", "mid": "mid_a"}).merge(
        b.loc[b["whole"], KEY + ["share", "n_active", "mtm"]].rename(
            columns={"share": "share_b", "n_active": "n_b", "mtm": "mtm_b"}), on=KEY, how="inner")
    m = m[m["n_a"] == m["n_b"]]
    return m.assign(move=np.log(m["share_b"] / m["share_a"]))


def s035(frames):
    """S035: the 60-to-30-minute move followed at 15: a WIN market whole at off_60 and off_30 with the same active
    runners; move = ln(share at off_30 / share at off_60). At off_15 the runner is ACTIVE with lay1 / back1 - 1 <=
    3%: back if move >= +0.04 and back1 in 2-15 (to win GBP100, at most GBP50), lay if move <= -0.04 and lay1 in
    2-15 (liability GBP100); never more than the size shown, none under GBP2. Closed at the BSP after CUT."""
    days = _days(frames, "books")
    r = _runners(frames)
    mv = _share_move(r, "o60", "o30")
    x = _mids(book(r, "o15")).merge(mv[KEY + ["move"]], on=KEY, how="inner")
    x = x[x["race_date"].isin(set(days)) & (x["spread"] - 1.0 <= 0.03)]
    bk = x[(x["move"] >= 0.04) & x["back1"].between(2.0, 15.0)].reset_index(drop=True)
    ly = x[(x["move"] <= -0.04) & x["lay1"].between(2.0, 15.0)].reset_index(drop=True)
    backs = _bets(bk, "back", bk["back1"], core.capacity(core.to_win_stake(100.0, bk["back1"], 0.0, 50.0), bk["back1_size"]),
                  cut=_cut(frames, bk), ts=bk["t"])
    lays = _bets(ly, "lay", ly["lay1"], core.capacity(100.0 / (ly["lay1"] - 1.0), ly["lay1_size"]),
                 cut=_cut(frames, ly, "lay"), ts=ly["t"])
    out = pd.concat([backs, lays], ignore_index=True) if len(backs) or len(lays) else core.empty_bets()
    _describe("S035", "by side", out, out["side"].to_numpy() if len(out) else [])
    return out, days


def s036_day(ctx) -> pd.DataFrame:
    """S036's day: each removal with a factor of 5 or more whose removal time T is from 05:15 UK to 60 minutes
    before the off, and with no other runner of the market removed from 15 minutes before T to the entry. For
    each runner still in: P* = its mid at its last poll before T (within 15 minutes, lay1 / back1 <= 1.25) x
    (1 - af / 100). The entry poll is the second after the removal SHOWS in the book (the first poll at or after
    the one that first showed the runner REMOVED, then the first at least 60 s later; passed if that is more than
    10 minutes after T): on a delayed book a poll after T that still shows the runner in holds prices from before
    the removal. There: back if back1 >= 1.04 x
    P* and back1 <= 30 (to win GBP100, at most GBP50); lay if lay1 <= 0.96 x P* and lay1 <= 15 (liability
    GBP100); never more than the size shown. One entry a runner a removal."""
    rem, raw = ctx.removals, ctx.raw
    if rem.empty or raw.empty:
        return core.empty_bets()
    off = raw.drop_duplicates("market_id").set_index("market_id")["off"]
    rem = rem[rem["market_id"].isin(set(off.index))].copy()
    rem["off"] = rem["market_id"].map(off)
    big = rem[(rem["af"] >= 5.0) & (rem["rt"] >= B.uk_time(ctx.day, "05:15"))
              & (rem["rt"] <= rem["off"] - pd.Timedelta(minutes=60))]
    live = raw[raw["status"].eq("OPEN") & ~raw["inplay"] & raw["runner_status"].eq("ACTIVE")]
    out = []
    for e in big.itertuples(index=False):
        g = live[live["market_id"].eq(e.market_id) & live["selection_id"].ne(e.selection_id)]
        times = pd.DatetimeIndex(g.loc[g["t"] >= e.first_seen, "t"].unique()).sort_values()
        later = times[times >= times[0] + pd.Timedelta(seconds=60)] if len(times) else times
        if len(later) == 0:
            continue
        t_in = later[0]
        if t_in - e.rt > pd.Timedelta(minutes=10):              # the book showed the removal late: P* is stale by then
            continue
        others = rem[rem["market_id"].eq(e.market_id) & rem["selection_id"].ne(e.selection_id)]
        if ((others["rt"] >= e.rt - pd.Timedelta(minutes=15)) & (others["rt"] <= t_in)).any() \
                or ((others["first_seen"] >= e.rt - pd.Timedelta(minutes=15)) & (others["first_seen"] <= t_in)).any():
            continue
        pre = g[(g["t"] < e.rt) & (g["t"] >= e.rt - pd.Timedelta(minutes=15)) & g["back1"].gt(1.0) & g["lay1"].gt(1.0)]
        pre = pre.sort_values("t", kind="stable").drop_duplicates("selection_id", keep="last")
        pre = pre[pre["lay1"] / pre["back1"] <= 1.25]
        fair = (np.sqrt(pre["back1"] * pre["lay1"]) * (1.0 - e.af / 100.0)).to_numpy()
        now = g[g["t"].eq(t_in)].merge(pd.DataFrame({"selection_id": pre["selection_id"].to_numpy(), "fair": fair}),
                                       on="selection_id", how="inner")
        out.append(now.assign(removed=e.selection_id, af_removed=e.af))
    if not out:
        return core.empty_bets()
    x = pd.concat(out, ignore_index=True)
    x = x.merge(ctx.runners.loc[ctx.runners["market_type"].eq("WIN"), ["race_date"] + KEY + ["bsp", "final_status"]],
                on=KEY, how="inner")
    bk = x[x["back1"].gt(1.0) & (x["back1"] >= 1.04 * x["fair"]) & (x["back1"] <= 30.0)].reset_index(drop=True)
    ly = x[x["lay1"].gt(1.0) & (x["lay1"] <= 0.96 * x["fair"]) & (x["lay1"] <= 15.0)].reset_index(drop=True)

    def cut(z, side):
        return B.cut_factor(ctx.removals, z["market_id"], z["selection_id"], z["t"], B.min_af_for(side))
    backs = _bets(bk, "back", bk["back1"], core.capacity(core.to_win_stake(100.0, bk["back1"], 0.0, 50.0), bk["back1_size"]),
                  cut=cut(bk, "back"), ts=bk["t"], removed=bk["removed"].to_numpy() if len(bk) else None)
    lays = _bets(ly, "lay", ly["lay1"], core.capacity(100.0 / (ly["lay1"] - 1.0), ly["lay1_size"]), cut=cut(ly, "lay"),
                 ts=ly["t"], removed=ly["removed"].to_numpy() if len(ly) else None)
    return pd.concat([backs, lays], ignore_index=True) if len(backs) or len(lays) else core.empty_bets()


def s036(frames):
    """S036: the book after a withdrawal, traded where it stands 4% or more from the pro-rata price a minute after
    the removal shows (s036_day). Closed at the BSP; only removals after the fill cut the price (CUT)."""
    b, days = _from_day(frames, "S036", "no raw books")
    _describe("S036", "by side", b, b["side"].to_numpy() if len(b) else [])
    return b, days


# ---------------------------------------------------------------------------------------------------------------
# lays at a mark (S038-S040, S042-S045, S047) and at a raw poll (S041, S046)
# ---------------------------------------------------------------------------------------------------------------

def _lay_stake_400(x: pd.DataFrame) -> np.ndarray:
    """GBP400 at risk: min(max(400 / (L - 1), 15), 200, the size shown); none under GBP2."""
    return core.capacity(core.to_win_stake(400.0, x["lay1"], FLOOR, CAP), x["lay1_size"])


def _lay_stake_200(x: pd.DataFrame) -> np.ndarray:
    """GBP200 at risk: min(200 / (L - 1), the size shown, 200); none under GBP2."""
    return core.capacity(core.to_win_stake(200.0, x["lay1"], 0.0, 200.0), x["lay1_size"])


def _model_lays(frames, tags: tuple, days: list, extra=None) -> pd.DataFrame:
    """The closing model's lays at marks, in order: lay1 <= 10, lay1 / back1 <= 1.10, 1 - lay1 / E >= 3% (E the
    expected BSP of the whole race at the mark) and lay1 <= our price; a runner at the first mark it qualifies."""
    picks = []
    for i, tag in enumerate(tags):
        b = _mids(live_bar(frames, tag))
        ok = b["lay1"].gt(1.0) & (b["lay1"] <= 10.0) & (b["spread"] <= 1.10) & (1.0 - b["lay1"] / b["xb"] >= BAR) \
            & (b["lay1"] <= b["predicted_bfsp"]) & b["race_date"].isin(set(days))
        if extra is not None:
            ok = ok & extra(b)
        picks.append(b[ok].assign(mark_no=i, lay_ev=1.0 - b["lay1"] / b["xb"]))
    return pd.concat(picks, ignore_index=True).drop_duplicates(["race_date"] + KEY, keep="first").reset_index(drop=True)


def s038(frames):
    """S038: the lay book late in the morning: WIN markets at the marks 10:00, 11:00, 12:00 UK in order, 60 minutes
    or more to the off, whole races; lay where lay1 <= 10, lay1 / back1 <= 1.10, 1 - lay1 / E >= 3% and lay1 <=
    our price; one lay a runner at the first mark it qualifies, at lay1 for min(max(400 / (L - 1), 15), 200,
    lay1's size); GBP4,000 of liability a day in mark order, the larger expected value first. Closed by a back
    at the BSP after CUT."""
    days = _days(frames, "books", "preds")
    x = _model_lays(frames, ("c10", "c11", "c12"), days, extra=lambda b: b["mto"] >= 60.0)
    stake = day_limit(x, _lay_stake_400(x), ["mark_no", "lay_ev"], unit_risk=x["lay1"] - 1.0, ascending=[True, False])
    return _bets(x, "lay", x["lay1"], stake, cut=_cut(frames, x, "lay"), ts=x["t"]), days


def s039(frames):
    """S039: the lay book where no runner can come out uncut: WIN markets at clock_08:00 then clock_09:00 with at
    most 10 active runners and no ACTIVE runner at a back of 34.0 or more or with no back price (the frame must
    hold every active runner). The runner rule, stake and day liability are S038's."""
    days = _days(frames, "books", "preds")

    def safe(b):
        long_ = (~b["back1"].gt(1.0)) | (b["back1"] >= 34.0)
        return (b["n_active"] <= 10) & ~long_.groupby(b["market_id"]).transform("any") & (b["n_rows"] == b["n_active"])
    x = _model_lays(frames, ("c08", "c09"), days, extra=safe)
    stake = day_limit(x, _lay_stake_400(x), ["mark_no", "lay_ev"], unit_risk=x["lay1"] - 1.0, ascending=[True, False])
    return _bets(x, "lay", x["lay1"], stake, cut=_cut(frames, x, "lay"), ts=x["t"]), days


def s040(frames):
    """S040: at clock_09:00 (OPEN, 60 minutes or more to the off, whole race: every active runner has a back and
    our price), lay where lay1 <= 10, lay1 / back1 <= 1.10, lay1's size >= GBP10, our price >= 1.20 x lay1 and
    F >= 1.20 x lay1; at lay1 for min(200 / (L - 1), lay1's size, 200). Closed by a back at the BSP after CUT."""
    days = _days(frames, "books", "preds")
    b = _mids(live_bar(frames, "c09"))
    x = b[b["race_date"].isin(set(days)) & b["xb"].notna() & (b["mto"] >= 60.0) & b["lay1"].gt(1.0) & (b["lay1"] <= 10.0)
          & (b["spread"] <= 1.10) & (b["lay1_size"] >= 10.0) & (b["predicted_bfsp"] >= 1.20 * b["lay1"])
          & (b["F"] >= 1.20 * b["lay1"])].reset_index(drop=True)
    return _bets(x, "lay", x["lay1"], _lay_stake_200(x), cut=_cut(frames, x, "lay"), ts=x["t"]), days


def s041_day(ctx) -> pd.DataFrame:
    """S041's day: for each market with a real matched back, the first raw book at or within 2 minutes after the
    market's first matched back (OPEN, not in play). Candidates: ACTIVE runners the trader had not backed by that
    book, with lay1 <= 8 and lay1 / back1 <= 1.10. E is the closing model's expected BSP on that book as the lay
    session reads it (a runner with no back takes its last trade, else its lay; a race with a runner unpriced by
    the model or the market is passed). The candidate with the largest 1 - lay1 / E is laid if that is >= 0 and
    lay1 <= our price (else no lay in the market): lay1 for min(200 / (L - 1), lay1's size, 200)."""
    f, raw = ctx.fills, ctx.raw
    if f is None or len(f) == 0 or raw.empty:
        return core.empty_bets()
    from model import race_book as rb
    w = ctx.runners[ctx.runners["market_type"].eq("WIN")]
    live = raw[raw["status"].eq("OPEN") & ~raw["inplay"]]
    rows = []
    for mid, t0 in f.groupby("market_id")["ts"].min().items():
        g = live[live["market_id"].eq(mid) & (live["t"] >= t0) & (live["t"] <= t0 + pd.Timedelta(minutes=2))]
        if g.empty:
            continue
        t1 = g["t"].min()
        g = g[g["t"].eq(t1) & g["runner_status"].eq("ACTIVE")].merge(
            w[KEY + ["race_date", "predicted_bfsp", "bsp", "final_status"]], on=KEY, how="left")
        back, lay, mp = g["back1"].to_numpy(float), g["lay1"].to_numpy(float), g["predicted_bfsp"].to_numpy(float)
        fb = np.where(back > 1.0, back, np.where(g["last_traded"].to_numpy(float) > 1.0, g["last_traded"].to_numpy(float), lay))
        if len(g) < 2 or not (np.nan_to_num(mp) > 1.0).all() or not (np.nan_to_num(fb) > 1.0).all():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            market = np.where(back > 1.0, rb.market_now(back, lay), fb)
        _, xb = race_exp_bsp(back, lay, mp, seed=int(g["selection_id"].sum() % (2 ** 32)), market=market)
        had = set(f.loc[f["market_id"].eq(mid) & (f["ts"] <= t1), "selection_id"])
        g = g.assign(xb=xb, lay_ev=1.0 - lay / xb)
        c = g[~g["selection_id"].isin(had) & g["lay1"].gt(1.0) & (g["lay1"] <= 8.0) & g["back1"].gt(1.0)
              & (g["lay1"] / g["back1"] <= 1.10)]
        if c.empty:
            continue
        best = c.sort_values("lay_ev", ascending=False, kind="stable").iloc[[0]]
        if float(best["lay_ev"].iloc[0]) >= 0.0 and float(best["lay1"].iloc[0]) <= float(best["predicted_bfsp"].iloc[0]):
            rows.append(best)
    if not rows:
        return core.empty_bets()
    x = pd.concat(rows, ignore_index=True)
    return _bets(x, "lay", x["lay1"], _lay_stake_200(x), ts=x["t"],
                 cut=B.cut_factor(ctx.removals, x["market_id"], x["selection_id"], x["t"], B.min_af_for("lay")))


def s041(frames):
    """S041: where the live rule backed, lay the most over-bet rival (s041_day). The lay leg's own series is the
    strategy, with 2% on the lay's own positive net: the conservative reading of 'commission on the market's net
    of the real backs and the lay together' (netting a losing leg against a winning one can only lower the
    commission). Printed beside it: the real backs of those markets alone and with the lay."""
    b, days = _from_day(frames, "S041", "no raw books or no real fill")
    try:
        f = _frame(frames, "fills")
        if len(b) and len(f):
            fb = _fill_bets(f[f["market_id"].isin(set(b["race_key"]))])
            both = pd.concat([fb, b], ignore_index=True)
            for name, z in (("the real backs of the laid markets alone", fb), ("backs plus the lay", both)):
                d = core.daily_series(core.check_bets(z))["pnl"]
                _say(f"   S041 {name}: mean GBP{d.mean():+.2f} a day, sd {d.std():.2f}, worst day GBP{d.min():+.2f} "
                     f"({len(d)} days with a bet)")
    except Exception as exc:  # noqa: BLE001
        _say(f"   S041: the combined print failed ({type(exc).__name__}: {exc})")
    return b, days


def s042(frames):
    """S042: no model. WIN markets at clock_09:00 with 90 minutes or more to the off, 6 or more active runners and
    GBP500 or more matched. The favourite is the lowest mid among the two-sided ACTIVE runners, with the second
    lowest mid at least 1.10 x it; it is laid if its lay1 is in 2.0-6.0, lay1 / back1 <= 1.06 and lay1's size >=
    GBP20: lay1 for min(200 / (L - 1), lay1's size, 200). Closed by a back at the BSP after CUT."""
    days = _days(frames, "books")
    b = _mids(book(_runners(frames), "c09"))
    b = b[b["race_date"].isin(set(days)) & (b["mto"] >= 90.0) & (b["n_active"] >= 6) & (b["mtm"] >= 500.0) & b["mid"].notna()]
    rank = b.groupby("market_id")["mid"].rank(method="first")
    second = b["mid"].where(rank == 2).groupby(b["market_id"]).transform("max")
    x = b[(rank == 1) & (second >= 1.10 * b["mid"]) & b["lay1"].between(2.0, 6.0) & (b["spread"] <= 1.06)
          & (b["lay1_size"] >= 20.0)].reset_index(drop=True)
    out = _bets(x, "lay", x["lay1"], _lay_stake_200(x), cut=_cut(frames, x, "lay"), ts=x["t"])
    _describe("S042", "by live-trader horse", out, _trader_label(frames, out))
    return out, days


def s043(frames):
    """S043: a WIN market whole at clock_08:00 and clock_09:00 with the same active runners; a runner whose share
    moved ln(share 09:00 / share 08:00) <= -0.05 while the market's matched grew by 15% or more is laid at
    clock_10:00 if ACTIVE there with lay1 in 2.0-12.0, lay1 / back1 - 1 <= 6% and 120 minutes or more to the
    off: liability GBP100 (stake 100 / (lay1 - 1)), never more than lay1's size, none under GBP2. Closed by a
    back at the BSP after CUT."""
    days = _days(frames, "books")
    r = _runners(frames)
    mv = _share_move(r, "c08", "c09")
    mv = mv[(mv["move"] <= -0.05) & (mv["mtm_b"] >= 1.15 * mv["mtm_a"]) & (mv["mtm_a"] > 0)]
    x = _mids(book(r, "c10")).merge(mv[KEY + ["move"]], on=KEY, how="inner")
    x = x[x["race_date"].isin(set(days)) & x["lay1"].between(2.0, 12.0) & (x["spread"] - 1.0 <= 0.06)
          & (x["mto"] >= 120.0)].reset_index(drop=True)
    out = _bets(x, "lay", x["lay1"], core.capacity(100.0 / (x["lay1"] - 1.0), x["lay1_size"]),
                cut=_cut(frames, x, "lay"), ts=x["t"])
    _describe("S043", "by live-trader horse", out, _trader_label(frames, out))
    return out, days


def s044(frames):
    """S044: the control. A WIN market whole at clock_09:00 (every active runner two-sided); every ACTIVE runner
    that is not the lowest mid, with lay1 in 4-12, lay1 / back1 - 1 <= 5% and 180 minutes or more to the off, is
    laid: liability GBP100, never more than lay1's size, none under GBP2. Closed by a back at the BSP after CUT."""
    days = _days(frames, "books")
    b = _mids(book(_runners(frames), "c09"))
    b = b[b["whole"]]
    fav = b.groupby("market_id")["mid"].rank(method="first") == 1
    x = b[b["race_date"].isin(set(days)) & ~fav & b["lay1"].between(4.0, 12.0) & (b["spread"] - 1.0 <= 0.05)
          & (b["mto"] >= 180.0)].reset_index(drop=True)
    return _bets(x, "lay", x["lay1"], core.capacity(100.0 / (x["lay1"] - 1.0), x["lay1_size"]), cut=_cut(frames, x, "lay"),
                 ts=x["t"]), days


RELIABILITY_FROM = "2026-10-20"       # reliability_book_1112: d0 from the days before, the scored days from here


def reliability_book(b: pd.DataFrame, d0: float) -> pd.Series:
    """research/queries/done/reliability_book_1112.py's reliability-weighted share x for one mark's book: m = the
    middle of 1/back and 1/lay (no lay: 0.5 / back and the race's largest v); v = ((1/back - 1/lay) / 2)^2 /
    (1 + min(back size, lay size) / d0); x = m + (1 - sum m) v / sum v, floored at 0.001 and renormalised. Only
    for a race in which every runner ACTIVE at the mark has a back price (and the frame holds them all) and some
    runner has a lay; NaN elsewhere. Nothing of the close is read (the query's BSP filter was for its scoring)."""
    g = b["market_id"]
    bk, la = b["back1"], b["lay1"]
    has = la.gt(1.0)
    size = b[["back1_size", "lay1_size"]].min(axis=1).fillna(0.0)
    m = (1.0 / bk + 1.0 / la.where(has, bk)) / 2.0
    m = m.where(has, 0.5 / bk)
    v = (((1.0 / bk - 1.0 / la) / 2.0) ** 2 / (1.0 + size / d0)).where(has)
    v = v.fillna(v.groupby(g).transform("max"))
    vs = v.groupby(g).transform("sum")
    whole = bk.gt(1.0).groupby(g).transform("all") & (b["n_rows"] == b["n_active"]) & (b["n_rows"] >= 2) & (vs > 0)
    x = np.maximum(m + (1.0 - m.groupby(g).transform("sum")) * v / vs, 0.001)
    return (x / x.groupby(g).transform("sum")).where(whole)


def s045(frames):
    """S045: no model. WIN markets at clock_08:00 then clock_09:00, whole races; x = the reliability-weighted share
    exactly as reliability_book_1112 builds it (d0 = the median smaller size at the 08:00, 09:00 and 10:00 marks
    on the days before 20 Oct, fixed; the scored days are from 20 Oct), F2 = 1 / x. Lay where lay1 <= 0.94 x F2,
    lay1 <= 10 and lay1's size >= GBP10, one lay a runner at the first mark it qualifies: lay1 for min(200 /
    (L - 1), lay1's size, 200). Closed by a back at the BSP after CUT."""
    days = _days(frames, "books", first=RELIABILITY_FROM)
    r = _runners(frames)
    early = []
    for tag in ("c08", "c09", "c10"):
        b = book(r, tag)
        s = b.loc[b["race_date"] < RELIABILITY_FROM, ["back1_size", "lay1_size"]].min(axis=1)
        early.append(s[s > 0])
    early = pd.concat(early) if early else pd.Series(dtype=float)
    if early.empty or not days:
        return _none("S045", "no mark before 20 Oct to fix d0, or no scored day with raw books", days)
    d0 = float(early.median())
    picks = []
    for tag in ("c08", "c09"):
        b = book(r, tag)
        b = b[b["race_date"].isin(set(days))].reset_index(drop=True)
        f2 = 1.0 / reliability_book(b, d0)
        picks.append(b[b["lay1"].gt(1.0) & (b["lay1"] <= 0.94 * f2) & (b["lay1"] <= 10.0) & (b["lay1_size"] >= 10.0)])
    x = pd.concat(picks, ignore_index=True).drop_duplicates(["race_date"] + KEY, keep="first").reset_index(drop=True)
    _say(f"   S045: d0 GBP{d0:.2f} (the days before {RELIABILITY_FROM}); {len(days)} scored days")
    return _bets(x, "lay", x["lay1"], _lay_stake_200(x), cut=_cut(frames, x, "lay"), ts=x["t"]), days


def s046_day(ctx) -> pd.DataFrame:
    """S046's day: in each WIN market, raw books from 06:00 UK to 30 minutes before the off (OPEN, not in play),
    the first book that is whole (every ACTIVE runner two-sided) and holds an ACTIVE runner j with
    adjustment_factor >= 10 and a share (K5) at most half its factor / 100. One trigger a market. The lowest-mid
    runner among the others is laid if its lay1 / back1 - 1 <= 4% (if it is wider, no bet): lay1 for
    min(lay1's size, GBP50), liability at most GBP200."""
    raw = ctx.raw
    x = raw[raw["status"].eq("OPEN") & ~raw["inplay"] & raw["runner_status"].eq("ACTIVE")
            & (raw["t"] >= B.uk_time(ctx.day, "06:00")) & (raw["mto"] >= 30.0)].copy()
    if x.empty:
        return core.empty_bets()
    two = x["back1"].gt(1.0) & x["lay1"].gt(1.0)
    x["mid"] = np.sqrt(x["back1"] * x["lay1"]).where(two)
    g = [x["market_id"], x["t"]]
    x["whole"] = two.groupby(g).transform("all") & (x.groupby(g)["selection_id"].transform("size")
                                                    == x["number_of_active_runners"])
    x["share"] = ((1.0 / x["mid"]) / (1.0 / x["mid"]).groupby(g).transform("sum")).where(x["whole"])
    x["trig"] = x["whole"] & (x["adjustment_factor"] >= 10.0) & (x["share"] <= 0.5 * x["adjustment_factor"] / 100.0)
    hit = x[x["trig"]].groupby("market_id")["t"].min()
    if hit.empty:
        return core.empty_bets()
    b = x.merge(hit.rename("t_hit").reset_index(), on="market_id")
    b = b[b["t"].eq(b["t_hit"]) & ~b["trig"]]
    fav = b.sort_values("mid", kind="stable").drop_duplicates("market_id", keep="first")
    fav = fav[fav["lay1"] / fav["back1"] - 1.0 <= 0.04]
    fav = fav.merge(ctx.runners.loc[ctx.runners["market_type"].eq("WIN"), ["race_date"] + KEY + ["bsp", "final_status"]],
                    on=KEY, how="inner")
    stake = core.capacity(np.minimum(50.0, 200.0 / (fav["lay1"] - 1.0)), fav["lay1_size"])
    return _bets(fav, "lay", fav["lay1"], stake,
                 cut=B.cut_factor(ctx.removals, fav["market_id"], fav["selection_id"], fav["t"], B.min_af_for("lay")),
                 ts=fav["t"], triggers=len(hit))


def s046t_day(ctx) -> pd.DataFrame:
    """S046's first table, a day: among the runners ACTIVE in the day's WIN books, how many carry a reduction
    factor, and how many show more than one value of it during the day (a factor Betfair refreshes kills the
    idea)."""
    x = ctx.raw[ctx.raw["runner_status"].eq("ACTIVE") & ctx.raw["adjustment_factor"].notna()]
    if x.empty:
        return pd.DataFrame()
    n = x.groupby(KEY)["adjustment_factor"].agg(lambda s: s.round(2).nunique())
    rng = x.groupby(KEY)["adjustment_factor"].agg(lambda s: float(s.max() - s.min()))
    return pd.DataFrame([{"runners": int(len(n)), "changed": int((n > 1).sum()), "mean_range": float(rng.mean())}])


def s046(frames):
    """S046: the stale reduction factor (s046_day). Closed by a back at the BSP; if j (or any runner) is removed
    after the lay the lay's price is cut by its factor (CUT). The factor table is printed first."""
    t = _frame(frames, "day_S046T")
    if len(t):
        _say(f"   S046, first: {int(t['runners'].sum()):,} runner-days with a reduction factor in the raw WIN books; "
             f"{int(t['changed'].sum()):,} ({t['changed'].sum() / max(t['runners'].sum(), 1):.1%}) show more than one "
             f"value in the day (mean range {np.average(t['mean_range'], weights=t['runners']):.2f} points)")
    else:
        _say("   S046, first: no data found for the factor table (no raw books, or no factor on a runner still in)")
    b, days = _from_day(frames, "S046", "no raw books")
    _say(f"   S046: {len(b)} lays from the triggers" + (" (under 100: not read)" if len(b) < 100 else ""))
    return b, days


def s047(frames):
    """S047: the morning fade basket at the clock_10:00 lay, race days to 18 Nov (race_results ends 16 Nov). A WIN
    runner in any of: (a) a Hurdle whose last earlier run in race_results was a chase; (b) a first handicap
    (market_name has 'Hcap', no earlier handicap, 2 or more earlier runs); (c) first-time headgear on the Flat
    (wearing not empty, an earlier run and none of them in headgear); (d) a second career run (exactly one
    earlier run). The history is matched on the horse's normalised name and only rows dated before the race day
    are read. At 10:00: lay1 <= 12 and lay1 / back1 - 1 <= 6%; lay1 for min(lay1's size, GBP20). Closed by a
    back at the BSP after CUT."""
    days = _days(frames, "books", last="2026-11-18")
    h = _frame(frames, "hist")
    if h.empty or not days:
        return _none("S047", "race_results history of the cards' horses", days)
    b = _mids(book(_runners(frames), "c10")).merge(h[KEY + ["prev_n", "prev_type", "prev_hcap", "prev_headgear"]],
                                                  on=KEY, how="left")
    b["prev_n"] = b["prev_n"].fillna(0)
    kind = b["race_type"].str.strip().str.lower()
    seg_a = kind.eq("hurdle") & b["prev_type"].fillna("").str.lower().str.contains("chase")
    seg_b = b["market_name"].str.contains("Hcap", case=False, regex=False) & b["prev_hcap"].eq(0) & (b["prev_n"] >= 2)
    seg_c = kind.eq("flat") & b["wearing"].str.strip().ne("") & b["prev_headgear"].eq(0) & (b["prev_n"] >= 1)
    seg_d = b["prev_n"].eq(1)
    seg = np.where(seg_a, "a hurdle after a chase", np.where(seg_b, "b first handicap", np.where(
        seg_c, "c first headgear, Flat", np.where(seg_d, "d second run", ""))))
    keep = (seg != "") & b["race_date"].isin(set(days)).to_numpy() & b["lay1"].gt(1.0).to_numpy() \
        & (b["lay1"] <= 12.0).to_numpy() & (b["spread"] - 1.0 <= 0.06).to_numpy()
    x = b[keep].reset_index(drop=True)
    out = _bets(x, "lay", x["lay1"], core.capacity(20.0, x["lay1_size"]), cut=_cut(frames, x, "lay"), ts=x["t"],
                segment=seg[keep])
    _describe("S047", "by segment", out, out["segment"].to_numpy() if len(out) else [])
    return out, days


# ---------------------------------------------------------------------------------------------------------------
# the place and N TBP markets against the win market (S048-S052)
# ---------------------------------------------------------------------------------------------------------------

def _pair_place(win: pd.DataFrame, place: pd.DataFrame) -> pd.DataFrame:
    """PAIR-B (K5): each win row with its place row on (event_id, market_start_utc, selection_id); the place row's
    columns carry the suffix _pl. A key that does not name one row on each side is not paired."""
    w = win.drop_duplicates(PAIR_B, keep=False)
    p = place.drop_duplicates(PAIR_B, keep=False)
    p = p.rename(columns={c: f"{c}_pl" for c in p.columns if c not in PAIR_B})
    return w.merge(p, on=PAIR_B, how="inner", validate="1:1")


def _late_removal(b: pd.DataFrame) -> pd.Series:
    """K6 for place and TBP markets: True for every row of a market in which a runner ACTIVE at the mark is
    REMOVED at the final mark (a removal after the entry: the bet is void and counted)."""
    return b["final_status"].eq("REMOVED").groupby(b["market_id"]).transform("any")


def place_chance(frames, tag: str, fallback_last: bool = False) -> pd.DataFrame:
    """K9 at a mark: the WIN book with ``wmid`` (K5's mid; with ``fallback_last`` the last traded price where no
    two-sided quote), ``k`` (the places the paired place market pays) and ``p_place`` = core.place_prob(q, k) on
    q = the normalised 1 / wmid of the whole race; NaN where the race is not whole or k is not known. Kept while
    the runners frame is the same."""
    r = _runners(frames)
    key = ("k9", tag, fallback_last)
    if _kept(key, r) is None:
        w = _mids(book(r, tag), fallback_last).rename(columns={"mid": "wmid"})
        pl = r[r["market_type"].eq("PLACE") & r["places"].gt(0)].drop_duplicates(["event_id", "market_start_utc"])
        w = w.merge(pl[["event_id", "market_start_utc", "places"]].rename(columns={"places": "k"}),
                    on=["event_id", "market_start_utc"], how="left")
        p = np.full(len(w), np.nan)
        q = w["share"].to_numpy(float)
        for idx in w.groupby("market_id", sort=False).indices.values():
            k = w["k"].to_numpy(float)[idx[0]]
            if not np.isfinite(q[idx]).all() or not np.isfinite(k) or k < 1 or len(idx) < 2:
                continue
            p[idx] = 1.0 if k >= len(idx) else core.place_prob(q[idx], int(k))
        w["p_place"] = p
        w["late_nr_win"] = _late_removal(w)
        _CACHE[key] = (weakref.ref(r), w)
    return _CACHE[key][1].copy()


def _place_book(frames, tag: str) -> pd.DataFrame:
    p = book(_runners(frames), tag, "PLACE")
    p["late_nr"] = _late_removal(p)
    return p


def s048(frames):
    """S048: the place pocket by limit on close, decided at off_5. PAIR-B; the win mid of every ACTIVE runner is
    sqrt(back1 x lay1), else its last traded price; 5 or more active runners; p by K9 from those mids. For each
    runner with a win mid <= 5.0 a LIMIT_ON_CLOSE place back of GBP20 at L = 1 + (1.05 / p - 1) / 0.98, rounded
    up to a tick: matched at the place BSP only if that BSP is at or above L (the order's mechanics, not the
    selection), and held to the place result. A runner removed in the win or place market after off_5 voids the
    race's bets (counted)."""
    days = _days(frames)
    w = place_chance(frames, "o5", fallback_last=True)
    x = _pair_place(w, _place_book(frames, "o5"))
    x = x[x["race_date"].isin(set(days)) & (x["n_active"] >= 5) & (x["wmid"] <= 5.0) & x["p_place"].notna()
          & (x["p_place"] > 0)].reset_index(drop=True)
    limit = np.asarray(core.tick_ceil(np.maximum(1.01, 1.0 + (1.05 / x["p_place"] - 1.0) / 0.98).to_numpy(float)), float)
    psp = x["bsp_pl"].to_numpy(float)
    filled = np.nan_to_num(psp, nan=0.0) >= limit
    void = (x["late_nr_pl"] | x["late_nr_win"]).to_numpy()
    z = x.assign(market_id=x["market_id_pl"], bsp=x["bsp_pl"], final_status=x["final_status_pl"])
    all_b = _bets(z, "back", z["bsp"], 20.0, exit="result", void=void | ~(np.nan_to_num(psp) > 1.0), market="place",
                  entry_type="sp")
    b = _bets(z[filled], "back", z.loc[filled, "bsp"], 20.0, exit="result", void=void[filled], market="place",
              entry_type="sp")
    _say(f"   S048: {len(x):,} limit orders, {int(filled.sum()):,} matched at the place SP ({filled.mean() if len(x) else 0:.1%})")
    _describe("S048", "the same selections taken MARKET on close", all_b, "market on close")
    return b, days


def s049(frames):
    """S049: place favourites laid 15 minutes out. PAIR-B at off_15. Win side: back1 and lay1 present, lay1 / back1
    <= 1.10, mid <= 4.0, 5 or more active runners. Place side: OPEN, 1.10 <= lay1 <= 2.50, lay1 / back1 <= 1.08,
    lay1's size >= GBP20. One lay at the place lay1 for min(200 / (L - 1), lay1's size, 200), closed by a
    MARKET_ON_CLOSE back at the place SP: stake x (1 - L / PSP) on every outcome. A place market with a removal
    after the entry is void (counted)."""
    days = _days(frames)
    r = _runners(frames)
    pl_all = r[r["market_type"].eq("PLACE")]
    _say(f"   S049, first: place runners with a place BSP {pl_all['bsp'].gt(1.0).mean() if len(pl_all) else 0:.1%} of "
         f"{len(pl_all):,}")
    w = _mids(book(r, "o15"))
    p = _mids(_place_book(frames, "o15"))
    x = _pair_place(w, p)
    _say(f"   S049, first: {len(x):,} of {len(w):,} win runners at off_15 paired to a place row "
         f"({len(x) / max(len(w), 1):.1%})")
    x = x[x["race_date"].isin(set(days)) & (x["spread"] <= 1.10) & (x["mid"] <= 4.0) & (x["n_active"] >= 5)
          & x["lay1_pl"].between(1.10, 2.50) & (x["spread_pl"] <= 1.08) & (x["lay1_size_pl"] >= 20.0)].reset_index(drop=True)
    z = x.assign(market_id=x["market_id_pl"], bsp=x["bsp_pl"], final_status=x["final_status_pl"])
    stake = core.capacity(core.to_win_stake(200.0, x["lay1_pl"], 0.0, 200.0), x["lay1_size_pl"])
    return _bets(z, "lay", x["lay1_pl"], stake, void=x["late_nr_pl"].to_numpy(), market="place", ts=x["t_pl"]), days


def s050(frames):
    """S050: win steam the place price has not followed. PAIR-B. The WIN market whole at off_30 and off_10 with the
    same active runners and ln(share at off_10 / share at off_30) >= +0.08; the place back1 at off_10 at least
    0.99 x the place mid at off_30 (two-sided there); at off_5 the place runner ACTIVE with back1 in 1.3-6.0 and
    lay1 / back1 - 1 <= 4%. GBP20 at the place back1 at off_5 (at most its size, none under GBP2), laid at the
    place BSP: stake x (back1 / PSP - 1). A place market with a removal after the entry is void (counted)."""
    days = _days(frames)
    r = _runners(frames)
    mv = _share_move(r, "o30", "o10")
    mv = mv[mv["move"] >= 0.08]
    w = r.loc[r["market_type"].eq("WIN"), KEY + PAIR_B[:2] + ["race_date"]].merge(mv[KEY + ["move"]], on=KEY, how="inner")
    p30 = _mids(book(r, "o30", "PLACE"))[KEY + ["mid"]].rename(columns={"mid": "mid30"})
    p10 = book(r, "o10", "PLACE")[KEY + ["back1"]].rename(columns={"back1": "back10"})
    p5 = _mids(_place_book(frames, "o5")).merge(p30, on=KEY, how="inner").merge(p10, on=KEY, how="inner")
    x = _pair_place(w, p5)
    x = x[x["race_date"].isin(set(days)) & (x["back10_pl"] >= 0.99 * x["mid30_pl"]) & x["back1_pl"].between(1.3, 6.0)
          & (x["spread_pl"] - 1.0 <= 0.04)].reset_index(drop=True)
    z = x.assign(market_id=x["market_id_pl"], bsp=x["bsp_pl"], final_status=x["final_status_pl"])
    stake = core.capacity(20.0, x["back1_size_pl"])
    return _bets(z, "back", x["back1_pl"], stake, void=x["late_nr_pl"].to_numpy(), market="place", ts=x["t_pl"]), days


def s051(frames):
    """S051: a place offer 5% over the win-implied price at off_15, held. PAIR-B; p by K9 from the off_15 win mids
    of a whole win market (every active runner two-sided). Runners with a win mid <= 21.0 whose place back1 gives
    p x (1 + (back1 - 1) x 0.98) - 1 >= 0.05: the place back1 for min(GBP20, its size), held to the place result.
    A runner removed in the win or place market after off_15 voids the race's bets (counted)."""
    days = _days(frames)
    w = place_chance(frames, "o15")
    x = _pair_place(w, _place_book(frames, "o15"))
    ev = x["p_place"] * (1.0 + (x["back1_pl"] - 1.0) * 0.98) - 1.0
    x = x[x["race_date"].isin(set(days)) & (x["wmid"] <= 21.0) & (ev >= 0.05) & x["back1_pl"].gt(1.0)].reset_index(drop=True)
    z = x.assign(market_id=x["market_id_pl"], bsp=x["bsp_pl"], final_status=x["final_status_pl"])
    b = _bets(z, "back", x["back1_pl"], core.capacity(20.0, x["back1_size_pl"]), exit="result",
              void=(x["late_nr_pl"] | x["late_nr_win"]).to_numpy(), market="place", ts=x["t_pl"])
    ok = x["bsp_pl"].gt(1.0)
    if ok.any():
        _say(f"   S051 diagnostic: the place back1 against the place SP, mean back1 / PSP - 1 "
             f"{(x.loc[ok, 'back1_pl'] / x.loc[ok, 'bsp_pl'] - 1.0).mean():+.2%} over {int(ok.sum()):,} offers")
    return b, days


_TBP = re.compile(r"^\s*(\d)\s*TBP\b", re.IGNORECASE)


def s052_day(ctx) -> pd.DataFrame:
    """S052's day: the day's 'N TBP' markets (books_other_place, markets_other_place), each paired to its WIN
    market on (event_id, market_start_utc), runners on selection_id. At the TBP market's poll nearest 5 minutes
    before the off (within 2 minutes; OPEN, not in play): k = number_of_winners in that book (else N of the name);
    p by K9 from the off_5 win mids of a whole win market. Back where p x (1 + (back1 - 1) x 0.98) - 1 >= 0.05:
    back1 for min(GBP10, its size). Lay where lay1 <= 8 and (1 - p) x 0.98 - p x (lay1 - 1) >= 0.05: lay1 for
    min(lay1's size, 20 / (lay1 - 1)). Held to the result (the runner's status in the TBP market's last CLOSED
    book). A TBP or win market with a runner removed after the entry is void (counted). ``places_final`` is the
    count of winners the market settled (a market cut from N to fewer places is flagged by the strategy)."""
    mk = ctx.read("markets_other_place")
    bk = ctx.read("books_other_place")
    if mk is None or bk is None or len(mk) == 0 or len(bk) == 0:
        return core.empty_bets()
    mk = mk[mk["country"].astype(str).str.upper().isin(["GB", "IE"])].copy()
    mk["n"] = mk["market_name"].astype(str).str.extract(_TBP)[0].astype(float)
    mk = mk[mk["n"].notna()].drop_duplicates("market_id")
    if mk.empty:
        return core.empty_bets()
    mk["event_id"] = mk["event_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    mk["market_start_utc"] = mk["market_start_utc"].astype(str)
    mk["off"] = pd.to_datetime(mk["market_start_utc"], utc=True, errors="coerce")
    w = _mids(book(ctx.runners, "o5")).rename(columns={"market_id": "win_market"})
    w = w[w["whole"]]
    b = bk.merge(mk[["market_id", "n", "event_id", "market_start_utc", "off"]], on="market_id", how="inner")
    gap = ((b["off"] - b["t"]).dt.total_seconds() / 60.0 - 5.0).abs()
    pre = b[b["status"].eq("OPEN") & ~b["inplay"] & (gap <= 2.0)].assign(gap=gap)
    if pre.empty:
        return core.empty_bets()
    pre = pre[pre["gap"].eq(pre.groupby("market_id")["gap"].transform("min"))]
    pre = pre[pre["t"].eq(pre.groupby("market_id")["t"].transform("min"))]
    fin = b[b["status"].eq("CLOSED")].sort_values("t", kind="stable").drop_duplicates(KEY, keep="last")
    pre = pre.merge(fin[KEY + ["runner_status"]].rename(columns={"runner_status": "final_status"}), on=KEY, how="left")
    pre["final_status"] = pre["final_status"].fillna("")
    act = pre["runner_status"].eq("ACTIVE")
    pre["late_nr"] = (act & pre["final_status"].eq("REMOVED")).groupby(pre["market_id"]).transform("any")
    pre["places_final"] = pre["final_status"].eq("WINNER").groupby(pre["market_id"]).transform("sum")
    pre["k"] = pre["number_of_winners"].where(pre["number_of_winners"] > 0, pre["n"])
    x = pre[act].merge(w[["win_market", "event_id", "market_start_utc", "selection_id", "share", "final_status"]].rename(
        columns={"final_status": "win_final"}), on=["event_id", "market_start_utc", "selection_id"], how="inner")
    if x.empty:
        return core.empty_bets()
    n_win = w.groupby("win_market")["selection_id"].size()
    x = x[x.groupby("market_id")["selection_id"].transform("size") == x["win_market"].map(n_win)]   # the same field
    p = np.full(len(x), np.nan)
    x = x.reset_index(drop=True)
    for idx in x.groupby("market_id", sort=False).indices.values():
        q, k = x["share"].to_numpy(float)[idx], float(x["k"].to_numpy(float)[idx[0]])
        if np.isfinite(q).all() and np.isfinite(k) and k >= 1 and len(idx) >= 2:
            p[idx] = 1.0 if k >= len(idx) else core.place_prob(q, int(k))
    x["p"] = p
    win_nr = w.assign(nr=w["final_status"].eq("REMOVED")).groupby("win_market")["nr"].any()
    x["void"] = x["late_nr"] | x["win_market"].map(win_nr).fillna(False).astype(bool)
    x["race_date"], x["bsp"] = ctx.day, np.nan
    bk_ = x[x["back1"].gt(1.0) & (x["p"] * (1.0 + (x["back1"] - 1.0) * 0.98) - 1.0 >= 0.05)].reset_index(drop=True)
    ly_ = x[x["lay1"].gt(1.0) & (x["lay1"] <= 8.0) & ((1.0 - x["p"]) * 0.98 - x["p"] * (x["lay1"] - 1.0) >= 0.05)]
    ly_ = ly_.reset_index(drop=True)

    def out(z, side, price, stake):
        return _bets(z, side, price, stake, exit="result", void=z["void"].to_numpy(), ts=z["t"],
                     market=(z["n"].astype(int).astype(str) + "tbp").to_numpy() if len(z) else "tbp",
                     places=z["k"].to_numpy() if len(z) else None,
                     places_final=z["places_final"].to_numpy() if len(z) else None)
    backs = out(bk_, "back", bk_["back1"], core.capacity(10.0, bk_["back1_size"]))
    lays = out(ly_, "lay", ly_["lay1"], core.capacity(20.0 / (ly_["lay1"] - 1.0), ly_["lay1_size"]))
    return pd.concat([backs, lays], ignore_index=True) if len(backs) or len(lays) else core.empty_bets()


def s052(frames):
    """S052: N TBP offers 5% off the win-implied price, held (s052_day), from 7 Oct. Backs GBP10, lays liability
    GBP20, both capped by the size shown; 2% on each TBP market's own positive net (its own commission unit)."""
    _day_errors(frames, "S052")
    days = _days(frames, "books", first="2026-10-07")
    b = _frame(frames, "day_S052")
    if b.empty or not days:
        return _none("S052", "no N TBP book recorded (books_other_place) or no offer cleared 5%", days)
    b = _within(b, days)
    if len(b):
        _describe("S052", "by side and N", b, (b["side"] + " " + b["market"]).to_numpy())
        if {"places", "places_final"} <= set(b.columns):
            cut_ = b[(b["places_final"] > 0) & (b["places_final"] < b["places"])]
            _say(f"   S052 flag: {cut_['race_key'].nunique()} TBP markets settled fewer winners than the places their "
                 f"book showed at the entry ({len(cut_)} bets)")
    return b, days


# ---------------------------------------------------------------------------------------------------------------
# the small tables other strategies read from the raw books
# ---------------------------------------------------------------------------------------------------------------

def s024_day(ctx) -> pd.DataFrame:
    """S024's table, a day: at each WIN market's raw poll nearest 08:00 UK (within 5 minutes; OPEN, not in play,
    before the off) each ACTIVE runner's adjustment_factor (af) and its share of the book (from the mid, from the
    back where no lay is shown; a runner with no back counts 0)."""
    raw = ctx.raw
    x = raw[raw["status"].eq("OPEN") & ~raw["inplay"] & (raw["mto"] > 0)].copy()
    x["gap"] = (x["t"] - B.uk_time(ctx.day, "08:00")).abs().dt.total_seconds()
    x = x[x["gap"] <= 300.0]
    if x.empty:
        return pd.DataFrame()
    x = x[x["gap"].eq(x.groupby("market_id")["gap"].transform("min"))]
    x = x[x["t"].eq(x.groupby("market_id")["t"].transform("min")) & x["runner_status"].eq("ACTIVE")]
    price = np.sqrt(x["back1"] * x["lay1"]).where(x["back1"].gt(1.0) & x["lay1"].gt(1.0), x["back1"].where(x["back1"] > 1.0))
    inv = (1.0 / price).fillna(0.0)
    tot = inv.groupby(x["market_id"]).transform("sum")
    return pd.DataFrame({"market_id": x["market_id"].to_numpy(), "selection_id": x["selection_id"].to_numpy(),
                         "af": x["adjustment_factor"].to_numpy(float), "share": (inv / tot.where(tot > 0)).to_numpy(),
                         "t": x["t"].to_numpy()})


def s029_day(ctx) -> pd.DataFrame:
    """S029's table, a day: each WIN runner's first raw poll at or after 05:15 UK with a two-sided quote of lay1 /
    back1 <= 1.5 (ACTIVE, market OPEN, not in play): its time and mid."""
    raw = ctx.raw
    x = raw[raw["status"].eq("OPEN") & ~raw["inplay"] & raw["runner_status"].eq("ACTIVE")
            & (raw["t"] >= B.uk_time(ctx.day, "05:15")) & raw["back1"].gt(1.0) & raw["lay1"].gt(1.0)]
    x = x[x["lay1"] / x["back1"] <= 1.5].sort_values("t", kind="stable").drop_duplicates(KEY, keep="first")
    return pd.DataFrame({"market_id": x["market_id"].to_numpy(), "selection_id": x["selection_id"].to_numpy(),
                         "first_t": x["t"].to_numpy(), "first_mid": np.sqrt(x["back1"] * x["lay1"]).to_numpy()})


def fcheck_day(ctx) -> pd.DataFrame:
    """S007's check, a day: each WIN runner's forecast price as first recorded in the day's raw catalogue against
    the one betfair_live_markets stores (is the stored forecast the morning's?)."""
    parts = [ctx.read(n) for n in ("markets_trader", "markets")]
    parts = [p for p in parts if p is not None and len(p) and {"forecast_num", "forecast_den"} <= set(p.columns)]
    if not parts:
        return pd.DataFrame()
    m = pd.concat(parts, ignore_index=True)
    m["num"], m["den"] = pd.to_numeric(m["forecast_num"], errors="coerce"), pd.to_numeric(m["forecast_den"], errors="coerce")
    m = m[(m["num"] > 0) & (m["den"] > 0)].copy()
    m["selection_id"] = pd.to_numeric(m["selection_id"], errors="coerce")
    order = pd.to_datetime(m["recorded_utc"], utc=True, errors="coerce") if "recorded_utc" in m.columns else None
    if order is not None:
        m = m.assign(_o=order).sort_values("_o", kind="stable")
    first = m.drop_duplicates(KEY, keep="first").assign(F_first=lambda z: 1.0 + z["num"] / z["den"])
    r = ctx.runners[ctx.runners["market_type"].eq("WIN") & ctx.runners["F"].gt(1.0)]
    j = r[KEY + ["F"]].merge(first[KEY + ["F_first"]], on=KEY, how="inner")
    return pd.DataFrame([{"n": int(len(j)), "n_diff": int((np.abs(j["F"] - j["F_first"]) > 1e-9).sum())}])


#: what the loader runs on each day's raw books (data_b.day_pass): key -> function(ctx) -> frame (bets or a table),
#: kept as frames['day_<key>']
DAY_FUNCS = {"S024": s024_day, "S029": s029_day, "S032": s032_day, "S036": s036_day, "S041": s041_day,
             "S046": s046_day, "S046T": s046t_day, "S052": s052_day, "FCHECK": fcheck_day}


# ---------------------------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------------------------

def _s(sid, name, family, side, build, market="win", **kw) -> core.Strategy:
    return core.Strategy(id=sid, name=name, family=family, dataset=DS, side=side, market=market, build=build, **kw)


_REPLAY_NOTE = ("replayed on the DELAYED key's books at the sizes shown (the replay fills more kindly than live); "
                "the SP lay assumed matched in full")
_REAL_NOTE = "the real fills, only dropped, cut or re-settled; nothing redeployed; no fill assumption"
_MARK_NOTE = "a fill at the delayed book's best price and size is assumed; also printed 2% worse"

STRATEGIES = [
    _s("S001", "REPLAY-B: the live rule as it stands (benchmark)", "benchmark", "back", s001, benchmark=True,
       tested_before="live 45 days +GBP6,929 (TRADING.md); arms-full-1103 (older rule GBP6,101 over 21 days); "
                     "phase5-reductions-1119 (first backs +9.39% after cuts, top-ups +1.78%); week-drop-1123 "
                     "(CLV +1.47% on 17-23 Nov); live earned 0.80 of the replay on 10-19 Oct (edge_investigation_1111)",
       note=_REPLAY_NOTE),
    _s("BASE-B", "BASE-B: the live trader's real fills (benchmark)", "benchmark", "back", base_b, benchmark=True,
       entry=NO_HAIRCUT, blind=False,
       tested_before="the live record itself: TRADING.md, week-drop-1123, phase4-segments-1119",
       note=_REAL_NOTE + "; entry_type 'sp' is only the core's switch for 'no worse-price print'"),
    _s("S002", "Bar 0%", "live rule: bar", "back", s002, paired="S001",
       tested_before="phase4-rules-1119 (price files: 0% most pounds, +GBP102,787 against 91,876 at 3%); "
                     "phase4-segments-1119 (EV 0-3%: CLV +2.03%); on the recorded books: none found", note=_REPLAY_NOTE),
    _s("S003", "Bar 8%", "live rule: bar", "back", s003, paired="S001",
       tested_before="rule-candidates-replay-1020 (bar 6% GBP1,587 against 2,391, old window); phase4-rules-1119 "
                     "(8%: +12.72% on half the turnover); not replayed since the 10:00 stop", note=_REPLAY_NOTE),
    _s("S004", "Bar that rises as the off nears (3 / 5 / 8% at 6 and 3 hours)", "live rule: bar by time to the off",
       "back", s004, paired="S001",
       tested_before="a bar by hours to the off: none found; near: bar1/bar2 arms GBP5,766 / 5,402 against 6,101 "
                     "(arms-full-1103); closing-correction-1009 (not robust)", note=_REPLAY_NOTE),
    _s("S005", "Bar by the book's spread (3% wide, 8% tight)", "live rule: bar by book state", "back", s005,
       paired="S001",
       tested_before="a bar by spread with the served model: none found; the spread blend lost in replay "
                     "(blend-ladder-replay-1020; closing-refit-replay-1109 -GBP51.68 a day); market-read-replay-1112 lost",
       note=_REPLAY_NOTE),
    _s("S006", "Raw overlay: price at least 50% above our forecast, EV >= 0", "live rule: model-market gap", "back",
       s006, paired="S001",
       tested_before="descriptive only (below-model-1020, topups-1020: top-ups over 50% above our price +20.7%); "
                     "a raw-gap rule replayed: none found", note=_REPLAY_NOTE),
    _s("S007", "Tissue agreement: price at or above Betfair's forecast price as well", "forecast price", "back", s007,
       paired="S001",
       tested_before="move-model-1021 (error only); combo-replay-1021 (its LightGBM veto lost GBP1,645 against "
                     "1,826); edge_investigation_1111 lever 8 never followed up; as a selection rule: none found",
       note=_REPLAY_NOTE + "; unverified: that the stored forecast is the morning's (printed)"),
    _s("S008", "The rule with our price corrected by Betfair's forecast price (marks 08:00/09:00/10:00)",
       "forecast price", "back", s008, paired="S008-BASE",
       tested_before="move-model-1021; benter-blend-1020 and closing-refit-replay-1109 (refits priced better but "
                     "made less in replay); not tested as a price correction", note=_MARK_NOTE),
    _s("S008-BASE", "S008's benchmark: the same marks and code with our price unchanged", "benchmark", "back",
       s008_base, benchmark=True, tested_before="none found (a control made for S008)", note=_MARK_NOTE),
    _s("S009", "First hour only", "live rule: time of entry", "back", s009, paired="S001",
       tested_before="live-by-hour-1104 (06:00 UK hour GBP3,292 of 4,986); stop0900 arm GBP5,457 against 6,101 "
                     "(arms-full-1103); window-1000; a one-hour window: none found", note=_REPLAY_NOTE),
    _s("S010", "The rule, and 10:00-13:00 only on races 6 hours or more away", "live rule: time of entry", "back",
       s010, paired="S001",
       tested_before="live-by-hour-1104 (hour and time to the off read apart, the cross cell never); "
                     "time-cutoff-replay-1007/1008; day-limit-replay-1020", note=_REPLAY_NOTE),
    _s("S011", "Single look at 08:00 UK", "live rule: time of entry", "back", s011,
       tested_before="from0800 arm GBP73 a day against 285 (timing-1104); the rule from 08:00 GBP4,393 against "
                     "5,316 (strategy-replay-1017); a one-shot sweep: closing-model-by-hour-1007 on 4 days",
       note=_MARK_NOTE),
    _s("S012", "Single look two hours before each off", "live rule: time of entry", "back", s012,
       tested_before="timing-1104 (2-4h 1.2%, 1-2h -1.3% after cuts); late-entry-0930; small-book-late-1111 "
                     "(T-5 to T-1: weight 0.00-0.08, 0 of 54 pass)", note=_MARK_NOTE),
    _s("S013", "First backs only (no top-ups)", "live rule: top-ups", "back", s013, paired="S001",
       tested_before="topups0 GBP2,583 against 6,101 over 21 October days (arms-full-1103); one_back GBP1,640 "
                     "against 4,950 (stake-per-horse-1016); not read on the winter card", note=_REPLAY_NOTE),
    _s("S014", "Top-ups only within 30 minutes of the first back", "live rule: top-ups", "back", s014, paired="S001",
       tested_before="named in topups-1020 and edge_investigation_1111 lever 5, never run; topup-flow-guard-1115 "
                     "FAIL (deflated Sharpe 0.04-0.20)", note=_REPLAY_NOTE),
    _s("S015", "Top-up bar 7% on the real fills", "live rule: top-ups", "back", s015, paired="BASE-B",
       entry=NO_HAIRCUT,
       tested_before="with redeployment it lost three times (blend-ladder-replay-1020 GBP1,822 against 2,131; "
                     "topup-policy-replay-1004; stake-per-horse-1016); as a pure drop: none found", note=_REAL_NOTE),
    _s("S016", "No top-ups under 4.0 on the real fills", "live rule: top-ups by price", "back", s016, paired="BASE-B",
       entry=NO_HAIRCUT,
       tested_before="topup-policy-replay-1004 (485 and 435 against 656, 4 days); horse caps 300/200/150 "
                     "(stake-per-horse-1016); first-back-only under 4.0: none found", note=_REAL_NOTE),
    _s("S017", "No ladder (best back only)", "live rule: execution", "back", s017, paired="S001", blind=False,
       tested_before="blend-ladder-replay-1020 (ladder GBP2,388 against 2,131); depth-refill-1020; "
                     "phase5-execution-1119; the 22 Nov replay (one CONFIRM day seen: not blind)", note=_REPLAY_NOTE),
    _s("S018", "Stake floor GBP50 at prices 21 and over", "live rule: staking", "back", s018, paired="S001",
       blind=False,
       tested_before="arms-full-1103: floor50 GBP6,504 against 6,101 (21 days, not significant); "
                     "paper-agents-rescore-1104b GBP6,153.86 against 5,868.76 (CONFIRM days to 3 Nov seen: not "
                     "blind); retired 4 Nov before its forward test",
       note=_REPLAY_NOTE + "; the SP lay at 100+ can part-fill"),
    _s("S019", "Target scaled by hours to the off (to win 800 / 400 / 200)", "live rule: staking", "back", s019,
       paired="S001",
       tested_before="win1000 (timing-1104: +GBP26 a day); to_win_600 GBP5,381 against 4,950 "
                     "(stake-per-horse-1016); a target falling with time: none found", note=_REPLAY_NOTE),
    _s("S020", "Level GBP40 a horse", "live rule: staking", "back", s020, paired="S001",
       tested_before="stake-scheme-replay-1017 (1-16 Oct): level_40 GBP4,735 against 5,776; to-win-staking-0930; "
                     "not replayed since the three-level ladder", note=_REPLAY_NOTE),
    _s("S021", "Kelly cap on the calibrated edge over the move's variance (real fills)", "staking: fractional Kelly",
       "back", s021, paired="BASE-B", entry=NO_HAIRCUT,
       tested_before="stake-scheme-replay-1017 (stakes by the raw edge about level); race-book-0930 (Kelly books); "
                     "with a calibrated edge or a variance term: none found", note=_REAL_NOTE),
    _s("S022", "Race cap GBP400 on the real fills", "staking: per-race cap", "back", s022, paired="BASE-B",
       entry=NO_HAIRCUT,
       tested_before="realised CLV by horses backed in a race: none found; removal-tilt-replay-1109 (dropping "
                     "positive-CLV money cost GBP1,444)", note=_REAL_NOTE),
    _s("S023", "No new money in a market after a removal (real fills)", "non-runner", "back", s023, paired="BASE-B",
       entry=NO_HAIRCUT,
       tested_before="fills after a removal: none found; removal-by-market-1109 (the cut is fair on average); "
                     "replay-nr-bar-1021 (lost GBP231); removal-tilt-replay-1109 failed",
       note=_REAL_NOTE + "; the removal's time is known to one poll"),
    _s("S024", "The rule's 08:00 picks in markets with a free removal to come", "non-runner", "back", s024,
       paired="S024-OTHER",
       tested_before="removal-by-market-1109 (rule runners with a small removal to come +6.57% after the cut "
                     "against +3.40%); a selection on the sub-2.5 share: none found", note=_MARK_NOTE),
    _s("S024-OTHER", "S024's benchmark: the 08:00 picks in the other markets", "benchmark", "back", s024_other,
       benchmark=True, tested_before="none found (a control made for S024)", note=_MARK_NOTE),
    _s("S025", "The live backs held to the result", "exit", "back", s025, paired="BASE-B", entry=NO_HAIRCUT,
       tested_before="strategy-review-1017 (held GBP550 against 1,555, sd 1,013 against 259); hedge-ratio-1021; "
                     "phase4-segments-1119 (held +5.92% against CLV +6.83%); not re-read on the days since",
       note=_REAL_NOTE),
    _s("S026", "Conditional hedge: limit-on-close lay at the fill's own price", "exit", "back", s026, paired="BASE-B",
       entry=NO_HAIRCUT,
       tested_before="morning-exit-bsp-1116 (COND, morning models only); strategy-review-1017 (holding the "
                     "shorteners: noise); holding the drifters on the live fills: none found",
       note=_REAL_NOTE + "; a limit lay is taken as matched whenever the BSP is at or under the limit"),
    _s("S027", "The live backs laid in running at half the SP odds", "exit: in-running", "both", s027,
       paired="BASE-B", entry=NO_HAIRCUT,
       tested_before="strategy-review-1017; morning-exit-bsp-1116; edge-inplay-offsets-1002 (unconditional "
                     "back-to-lay -14% to -39%); an in-running exit for the live backs: none found",
       note="the real stakes at Betfair's settled prices; the in-running lay is an UPPER BOUND (K8: no volume at "
            "a price, no time of the low); betfair_prices to 22 Nov"),
    _s("S028", "Early money agrees: back the 08:00 pick that has shortened by 10:00", "early money", "back", s028,
       tested_before="live-by-hour-1104 (10:00 entries -0.4% after cuts, unconditional); steam-beyond-bsp; an entry "
                     "conditional on agreement: none found",
       note=_MARK_NOTE + "; the trader's own takes moved some of these prices"),
    _s("S029", "Three of five signals agree at 08:00", "signal vote", "back", s029,
       tested_before="each leg alone: TRADING.md, strat E, strat I, move-model-1021; combinations: "
                     "paper_overlap_study_1104 only; a vote: none found", note=_MARK_NOTE),
    _s("S030", "Quick returners at the offered 08:00 price, laid at the BSP", "morning models", "back", s030,
       tested_before="strat E (A/E 1.148 at the morning WAP, fails at the BSP); morning-exit-bsp-1116 (trade-out "
                     "+7.8% at the WAP, deflated Sharpe 0.214, fails); forward 5-14 Nov at 10:00: 6 backs",
       note=_MARK_NOTE),
    _s("S031", "Steamer model's top 5% at the offered 08:00 price, laid at the BSP", "morning models", "back", s031,
       tested_before="strat I (5/5 years); morning-exit-bsp-1116 (trade-out +11.3% at 95% of the WAP, deflated "
                     "Sharpe 1.0); edge_investigation_1111 item 5 asked for the first read: not run",
       note=_MARK_NOTE + "; the steamer is fitted to 27 Mar 2026, the live one is not used"),
    _s("S032", "The rule with the last traded price in place of the model", "model-free control", "back", s032,
       tested_before="liquidity-1020 (model-picked fills only); model_vs_market_1124 (every runner's 08:00 best "
                     "back -10.5% to the BSP, the model's picks +1.9%); a model-free anchor: none found",
       note=_MARK_NOTE + "; a last trade may be hours old"),
    _s("S033", "The 08:00 price far above Betfair's forecast price (no model)", "model-free control", "back", s033,
       tested_before="move-model-1021 (the forecast alone has error 0.647 against the model's 0.396); "
                     "model_vs_market_1124; expected to fail alone", note=_MARK_NOTE),
    _s("S034", "Morning momentum: follow the 08:00 to 10:00 shortener", "morning price path", "back", s034,
       tested_before="open-close-horses-1101 (opening book only); move-model-1021 (the path carries 2% of the "
                     "gain); price-diffusion-1112; a market-only momentum rule: none found", note=_MARK_NOTE),
    _s("S035", "Pre-off momentum: follow the 60-to-30-minute move, enter at 15", "pre-off momentum", "both", s035,
       tested_before="small-book-late-1111 (no selection by move; takers -1.6 to -2.0%); stream-late-signal-1117 "
                     "(flow and ladder pressure at T-5: nothing; price momentum not tested)", note=_MARK_NOTE),
    _s("S036", "The book after a withdrawal: trade the gap to the pro-rata price", "non-runner", "both", s036,
       tested_before="substitution-1112 (pro rata holds); reductions-1021, replay-nr-bar-1021, "
                     "removal-tilt-replay-1109 (all act before the removal, all fail); the speed of re-formation: "
                     "none found",
       note=_MARK_NOTE + "; unverified: Betfair's handling of unmatched offers at a removal"),
    _s("S037", "Lay book core: first lay only, 10 or under, tight book, 05:15-08:00", "model lay book", "lay", s037,
       blind=False,
       tested_before="TRADING.md lay arm +GBP827 over 21 days; desk-lay-book-1113 +GBP1,243.56 over 30 days, "
                     "deflated Sharpe 0.911 (fails; its days reach into CONFIRM: not blind); "
                     "desk-lay-book-paper-1112 running (its figure is not read here); this cut: not read",
       note="replayed; taker fills at the delayed lay1 on one-minute steps assumed"),
    _s("S038", "Late-morning lay book: 10:00-12:00 UK", "model lay book", "lay", s038,
       tested_before="closing-model-by-hour-1007 (backs, 4 days); paper lays 'all before 07:00'; a lay after "
                     "10:00: none found", note=_MARK_NOTE),
    _s("S039", "Removal-safe lay book: no runner that can come out uncut", "model lay book: non-runner", "lay", s039,
       tested_before="removal-by-market-1109; removal-tilt-replay-1109 (the back-side skip fails); the lay side by "
                     "removal type: not read", note=_MARK_NOTE),
    _s("S040", "Doubly confirmed short: model and tissue both 20% over the lay at 09:00", "lay: two priors", "lay",
       s040,
       tested_before="move-model-1021; edge_investigation_1111 lever 8; the lay arm for the model-only version; "
                     "the two-prior lay: none found", note=_MARK_NOTE),
    _s("S041", "Race-neutral pair: lay the most over-bet rival where the live rule backed", "whole-race book", "lay",
       s041,
       tested_before="TRADING.md arm 3 (race book with lays GBP3,804 against the rule's 6,101); race-book-0930; a "
                     "lay paired to the rule's real backs: none found",
       note=_MARK_NOTE + "; the lay leg alone, commission on its own positive net (conservative)"),
    _s("S042", "Market-only morning favourite lay at 09:00", "false favourite", "lay", s042,
       tested_before="morning-favourite-clv (Q1 2026, at the WAP); market-only-drift-1002 (no drift by band); "
                     "open-close-lay-cell-1101 (lays at 21+ all negative); the short favourite at an offered price: "
                     "not a cell", note=_MARK_NOTE),
    _s("S043", "Lay the runner drifting on volume between 08:00 and 09:00", "morning momentum", "lay", s043,
       tested_before="paper-agents-rescore-1104b (quotes_mid2 -GBP17,554); open-close-lay-cell-1101 (21+ only); "
                     "market-only-drift-1002; move-conditioned lays under 21: not tested",
       note=_MARK_NOTE + "; the matched money is the market's, not the runner's"),
    _s("S044", "Control: lay the formed middle of the book at 09:00", "baseline lay", "lay", s044,
       tested_before="open-close-lay-cell-1101 (21+: -0.62% on liability; under 21 not read); "
                     "market-only-drift-1002; paper-agents-rescore-1104b", note=_MARK_NOTE),
    _s("S045", "Coherent-book lay: a short offer against the reliability-weighted book", "microstructure lay", "lay",
       s045,
       tested_before="reliability-book-1112 (passes as a pricing read); market-read-replay-1112 (in the back "
                     "trader it loses GBP786, deflated Sharpe 0.029); the lay side: not read",
       note=_MARK_NOTE + "; scored from 20 Oct (d0 fixed on the days before)"),
    _s("S046", "Stale reduction factor: lay the favourite when a rival has drifted far below its factor",
       "non-runner", "lay", s046,
       tested_before="removal-by-market-1109 (the factor is fair on average: 11.20% against 10.97%); "
                     "desk-lay-book-1113; this trigger: none found",
       note=_MARK_NOTE + "; unverified: whether Betfair refreshes factors in the day (the table is printed first)"),
    _s("S047", "Morning fade basket at offered lay prices", "over-bet segments", "lay", s047,
       tested_before="strat_screen_rules.csv, strat_hypotheses_rules.csv (morning A/E 0.81-0.95 at the WAP); "
                     "open-close-lay-cell-1101; a card-conditioned lay at offered prices: none found",
       note=_MARK_NOTE + "; race days to 18 Nov (race_results ends 16 Nov)"),
    _s("S048", "Place pocket by limit on close, decided five minutes out", "place v win: limit on close", "back",
       s048, market="place", entry="sp",
       tested_before="place-pocket-gate-2pc (passes at the SPs: +2.70%, 13,829 bets); place-t15-forward-1001 (one "
                     "day, 3 bets); edge_investigation_1111 item 2 asked for it: not run",
       note="filled at the place SP where it is at or above the limit; our own effect on the place SP unknown"),
    _s("S049", "Place favourites: lay 15 minutes out, back at the place SP", "place closing-line", "lay", s049,
       market="place",
       tested_before="place-fields-check-1002; place-t15-forward-1001 (price path, one day); "
                     "edge_investigation_1111 item 6 asked for the pooled read: not run",
       note=_MARK_NOTE + "; a full fill of the SP back assumed"),
    _s("S050", "Win steam the place price has not followed", "place lead-lag", "back", s050, market="place",
       tested_before="place-t15-forward-1001; place-clv-forward-1002; arms-full-1103 place arms -GBP3,204; a "
                     "win-led trigger near the off: none found", note=_MARK_NOTE),
    _s("S051", "Place offer 5% over the win-implied price at T-15, held", "place v win: book value", "back", s051,
       market="place",
       tested_before="place-t15-forward-1001 (too few bets); the nightly scans count such offers but settled only "
                     "model-flagged ones; edge_investigation_1111 item 6: not run", note=_MARK_NOTE),
    _s("S052", "N TBP offers 5% off the win-implied price, held", "other-place: book value", "both", s052,
       market="other-place",
       tested_before="all-markets-scan-1006b (0-30 offers a market and mark; only model-flagged offers settled, "
                     "negative: other-markets-1109 to -1122); a pooled settle of market-flagged offers: none found",
       note=_MARK_NOTE + "; capacity is pounds; stale quotes last seconds on a delayed key"),
]
