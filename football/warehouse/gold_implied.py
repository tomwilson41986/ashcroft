"""gold/implied: what the market expects of the match at each checkpoint: chances, goals, and where markets disagree.

One row an event a checkpoint, from the checkpoint books of the event's three markets:

- the normalised 1X2 chances and the over 2.5 chance (complete books only), with each book's sum and stalest price;
- the Asian handicap's main line (the complete line nearest evens) and the home side's chance on it;
- the goal expectancies those prices imply. Two statistics of a Dixon-Coles score model do not depend on its
  low-score correction rho: the chance of over 2.5 goals fixes the total (home + away rate), and the home chance
  less the away chance then fixes the supremacy (home - away rate). So the rates are read off the over/under and
  the 1X2 exactly, with no grid and no assumption about rho (the earlier ``implied_rates`` searched a 0.02 grid
  of independent Poisson rates: the model review's finding A3);
- ``rho``: the low-score correction the market's draw price then implies (the draw chance against the
  independent-Poisson draw at those rates): the market's draw premium, match by match;
- ``sup_ah``: the supremacy the Asian handicap main line implies at the same total, and ``sup_gap`` = sup_ah less
  the 1X2's: the disagreement between the two markets (in goals), a lead-lag signal.

A fair Asian handicap price on a line with a push pays on win against loss only, and a quarter line is half a stake
on each neighbour, so the home side's normalised chance on line L is W / (W + Lo) with W and Lo the win and loss
chances summed over the line's halves; ``_cover`` computes it from the goal-difference (Skellam) distribution.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from football.warehouse.schema import TS, Column, TableSpec

VERSION = 1
SCOPE = "part"
INPUTS = ("gold/checkpoint", "gold/event")
COLUMNS = {
    "gold/checkpoint": ["market_id", "event_id", "market_type", "role", "cp_min", "line", "cp_time", "p_norm",
                        "book_complete", "book_sum", "book_max_age_s"],
    "gold/event": ["event_id", "mo_market_id", "ou_market_id", "ah_market_id"],
}

T_LO, T_HI = 0.05, 12.0

SPEC = TableSpec(
    "gold/implied",
    "One row an event a checkpoint: the market's chances (1X2, over 2.5, the handicap's main line), the goal "
    "expectancies they imply, the draw premium (rho) and the disagreement between the handicap and the 1X2.",
    (
        Column("event_id", pa.int64(), "Betfair's event id.", False),
        Column("cp_min", pa.int32(), "Minutes before the scheduled kick-off; 0 is the close.", False),
        Column("cp_time", TS, "The checkpoint's moment."),
        Column("p_home", pa.float64(), "The chance of a home win, normalised (complete 1X2 book only)."),
        Column("p_draw", pa.float64(), "The chance of a draw."),
        Column("p_away", pa.float64(), "The chance of an away win."),
        Column("mo_book_sum", pa.float64(), "Sum of 1 / price over the 1X2 book."),
        Column("mo_age_s", pa.float64(), "The 1X2 book's stalest price, seconds."),
        Column("p_over", pa.float64(), "The chance of over 2.5 goals, normalised (complete book only)."),
        Column("ou_book_sum", pa.float64(), "Sum of 1 / price over the over/under book."),
        Column("ou_age_s", pa.float64(), "The over/under book's stalest price, seconds."),
        Column("ah_line", pa.float64(), "The Asian handicap main line as the home side's handicap (-0.5: home "
               "gives half a goal): the complete line nearest evens."),
        Column("ah_p_home", pa.float64(), "The home side's normalised chance on that line."),
        Column("ah_age_s", pa.float64(), "That line's stalest price, seconds."),
        Column("ah_lines", pa.int16(), "Asian handicap lines with both sides priced."),
        Column("total", pa.float64(), "Expected goals in the match (home + away rate), from the over 2.5 chance."),
        Column("supremacy", pa.float64(), "Expected goal difference (home - away rate), from the 1X2 at that "
               "total."),
        Column("lam_home", pa.float64(), "The home side's goal expectancy."),
        Column("lam_away", pa.float64(), "The away side's goal expectancy."),
        Column("rho", pa.float64(), "The Dixon-Coles low-score correction the market's draw price implies at "
               "those rates (negative: more draws than independent Poisson)."),
        Column("sup_ah", pa.float64(), "The supremacy the handicap's main line implies at the same total."),
        Column("sup_gap", pa.float64(), "sup_ah less supremacy: the handicap market against the 1X2, in goals."),
    ),
    key=("event_id", "cp_min"), sort=("event_id", "cp_min"),
)
SPECS = (SPEC,)


# ---------------------------------------------------------------------------------------------------------------
# the score model's statistics (vectorised)
# ---------------------------------------------------------------------------------------------------------------

def p_over25(total):
    """The chance of three goals or more when the total is Poisson with this mean (rho leaves it unchanged)."""
    t = np.asarray(total, float)
    return 1.0 - np.exp(-t) * (1.0 + t + 0.5 * t * t)


def total_from_over(p_over) -> np.ndarray:
    """The total whose over 2.5 chance is ``p_over`` (bisection; blank outside the solvable range)."""
    p = np.asarray(p_over, float)
    ok = np.isfinite(p) & (p > p_over25(T_LO)) & (p < p_over25(T_HI))
    lo, hi = np.full(p.shape, T_LO), np.full(p.shape, T_HI)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        up = p_over25(mid) < p
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def _skellam(total, sup):
    from scipy.stats import skellam
    t, s = np.asarray(total, float), np.asarray(sup, float)
    return skellam, 0.5 * (t + s), 0.5 * (t - s)


def home_minus_away(total, sup) -> np.ndarray:
    """P(home win) - P(away win) at these rates: the same with or without the low-score correction."""
    sk, lh, la = _skellam(total, sup)
    return sk.sf(0, lh, la) - sk.cdf(-1, lh, la)


def draw_independent(total, sup) -> np.ndarray:
    sk, lh, la = _skellam(total, sup)
    return sk.pmf(0, lh, la)


def _solve_sup(f, target, total) -> np.ndarray:
    """The supremacy in (-total, total) at which the increasing function ``f(total, sup)`` equals ``target``."""
    t, y = np.asarray(total, float), np.asarray(target, float)
    ok = np.isfinite(t) & np.isfinite(y)
    tt = np.where(ok, t, 2.5)
    lo, hi = -tt * (1 - 1e-6), tt * (1 - 1e-6)
    with np.errstate(invalid="ignore"):
        ok &= (f(tt, lo) < y) & (f(tt, hi) > y)
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            up = f(tt, mid) < y
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def supremacy_from_1x2(p_home, p_away, total) -> np.ndarray:
    return _solve_sup(home_minus_away, np.asarray(p_home, float) - np.asarray(p_away, float), total)


def _win_loss(sk, lh, la, line):
    """(win, loss) chances of the home side with ``line`` goals start on a half or whole line."""
    k = np.floor(-line)                       # D + line > 0  <=>  D > -line
    whole = np.isclose(line, np.round(line))
    win = sk.sf(np.where(whole, -line, k), lh, la)
    loss = sk.cdf(np.where(whole, -line - 1, k), lh, la)
    return win, loss


def _cover(total, sup, line) -> np.ndarray:
    """The home side's fair normalised chance on Asian handicap ``line`` (its handicap): win over win plus loss,
    a quarter line being half a stake on each neighbouring line."""
    sk, lh, la = _skellam(total, sup)
    line = np.asarray(line, float)
    quarter = np.isclose(np.abs(line * 4) % 2, 1)            # x.25 or x.75
    l1, l2 = np.where(quarter, line - 0.25, line), np.where(quarter, line + 0.25, line)
    w1, x1 = _win_loss(sk, lh, la, l1)
    w2, x2 = _win_loss(sk, lh, la, l2)
    w, x = w1 + w2, x1 + x2
    return w / np.where(w + x > 0, w + x, np.nan)


def supremacy_from_handicap(line, p_home, total) -> np.ndarray:
    line = np.asarray(line, float)
    return _solve_sup(lambda t, s: _cover(t, s, np.where(np.isfinite(line), line, 0.0)), p_home, total)


# ---------------------------------------------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------------------------------------------

def build(inputs: dict) -> dict:
    cp = inputs["gold/checkpoint"].select(COLUMNS["gold/checkpoint"]).to_pandas()
    ev = inputs["gold/event"].select(COLUMNS["gold/event"]).to_pandas()
    cp = cp[cp["book_complete"]]
    if cp.empty:
        return {SPEC.name: SPEC.schema.empty_table()}
    key = ["event_id", "cp_min"]

    mo = cp[cp["market_id"].isin(ev["mo_market_id"].dropna())]
    wide = mo.pivot_table(index=key, columns="role", values="p_norm", aggfunc="first")
    for c in ("home", "draw", "away"):
        if c not in wide.columns:
            wide[c] = np.nan
    one = mo.groupby(key).agg(cp_time=("cp_time", "first"), mo_book_sum=("book_sum", "first"),
                              mo_age_s=("book_max_age_s", "first"))
    out = wide[["home", "draw", "away"]].rename(columns={"home": "p_home", "draw": "p_draw", "away": "p_away"}) \
        .join(one)

    ou = cp[cp["market_id"].isin(ev["ou_market_id"].dropna()) & (cp["role"] == "over")]
    ou = ou.groupby(key).agg(p_over=("p_norm", "first"), ou_book_sum=("book_sum", "first"),
                             ou_age_s=("book_max_age_s", "first"), ou_time=("cp_time", "first"))

    ah = cp[cp["market_id"].isin(ev["ah_market_id"].dropna()) & (cp["role"] == "home")].copy()
    ah["dist"] = (ah["p_norm"] - 0.5).abs()
    n_lines = ah.groupby(key).size().rename("ah_lines")
    ah = ah.sort_values(key + ["dist", "line"]).groupby(key).agg(
        ah_line=("line", "first"), ah_p_home=("p_norm", "first"), ah_age_s=("book_max_age_s", "first"),
        ah_time=("cp_time", "first")).join(n_lines)

    out = out.join(ou, how="outer").join(ah, how="outer")
    out["cp_time"] = out["cp_time"].fillna(out["ou_time"]).fillna(out["ah_time"])
    out = out.reset_index()

    total = total_from_over(out["p_over"].to_numpy(float))
    sup = supremacy_from_1x2(out["p_home"].to_numpy(float), out["p_away"].to_numpy(float), total)
    lam_h, lam_a = 0.5 * (total + sup), 0.5 * (total - sup)
    with np.errstate(invalid="ignore", divide="ignore"):
        d0 = np.where(np.isfinite(sup), draw_independent(np.where(np.isfinite(total), total, 2.5),
                                                         np.where(np.isfinite(sup), sup, 0.0)), np.nan)
        denom = 2.0 * lam_h * lam_a * np.exp(-total)
        rho = np.where(denom > 1e-6, (d0 - out["p_draw"].to_numpy(float)) / denom, np.nan)
        sup_ah = supremacy_from_handicap(out["ah_line"].to_numpy(float), out["ah_p_home"].to_numpy(float), total)

    table = pa.table({
        "event_id": pa.array(out["event_id"], pa.int64()),
        "cp_min": pa.array(out["cp_min"], pa.int32()),
        "cp_time": pa.Array.from_pandas(out["cp_time"]).cast(TS),
        "p_home": pa.array(out["p_home"], pa.float64(), from_pandas=True),
        "p_draw": pa.array(out["p_draw"], pa.float64(), from_pandas=True),
        "p_away": pa.array(out["p_away"], pa.float64(), from_pandas=True),
        "mo_book_sum": pa.array(out["mo_book_sum"], pa.float64(), from_pandas=True),
        "mo_age_s": pa.array(out["mo_age_s"], pa.float64(), from_pandas=True),
        "p_over": pa.array(out["p_over"], pa.float64(), from_pandas=True),
        "ou_book_sum": pa.array(out["ou_book_sum"], pa.float64(), from_pandas=True),
        "ou_age_s": pa.array(out["ou_age_s"], pa.float64(), from_pandas=True),
        "ah_line": pa.array(out["ah_line"], pa.float64(), from_pandas=True),
        "ah_p_home": pa.array(out["ah_p_home"], pa.float64(), from_pandas=True),
        "ah_age_s": pa.array(out["ah_age_s"], pa.float64(), from_pandas=True),
        "ah_lines": pa.array(out["ah_lines"].astype("Int64"), pa.int16(), from_pandas=True),
        "total": pa.array(total, pa.float64(), from_pandas=True),
        "supremacy": pa.array(sup, pa.float64(), from_pandas=True),
        "lam_home": pa.array(lam_h, pa.float64(), from_pandas=True),
        "lam_away": pa.array(lam_a, pa.float64(), from_pandas=True),
        "rho": pa.array(rho, pa.float64(), from_pandas=True),
        "sup_ah": pa.array(sup_ah, pa.float64(), from_pandas=True),
        "sup_gap": pa.array(sup_ah - sup, pa.float64(), from_pandas=True),
    })
    return {SPEC.name: table}
