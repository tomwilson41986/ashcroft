"""Where football makes money on Betfair, if anywhere: every strategy type, every market we hold Betfair prices for.

racing² football research, 3 Nov 2026 (the owner: "look at any strategies for making profit on a market: back to win,
back to lay, lay outright, lay to back, over/under, handicaps"). Read-only over the lab matrix (football/lab.py), real
prices and real results only. Nothing is fitted on prices here; the one model is the walk-forward Dixon-Coles price
(p_h/p_d/p_a/p_o25, lam_h/lam_a, each fitted on matches before its collection day).

Markets and prices held (football-data.co.uk via the gold tables):

- Betfair Exchange only (the owner, 3 Nov 2026: work on Betfair alone): football-data's Betfair early and closing
  prices, from 2024/25, 2% commission on each winning bet.
- Match odds (home, draw, away) and over/under 2.5 goals: Betfair's own early (BFE) and closing (BFEC) columns.
  The Asian handicap's main line where its source is Betfair (early from 2024/25, close from 2025/26).
- No in-play prices, no correct score, both-teams-to-score or corners prices: those strategies cannot be scored here.

Strategy families, each scored as return on stake (a lay's stake is the backer's stake it lays), with a 90% bootstrap
interval by match day (2,000 draws):

1. **Back to win** (hold to the result): every selection, by price band; and the model's value bets (expected return
   at the price above a threshold).
2. **Lay outright** (hold): the same, laid. The lay price is not in the data: a lay is costed at a book symmetric
   about the fair price (lay = the next Betfair tick at or above fair^2 / back), the lab's costing.
3. **Back to lay / lay to back** (trade the move, greened at the close): every selection by band, and the model's
   disagreement with the open (back what the model thinks too long, lay at the close; and the reverse). Each end
   costed at the quoted back price and the symmetric book's lay price.
4. **Over/under 2.5** and **Asian handicap**: the same families on those markets (quarter lines settled as two
   half-stakes).

Seasons from 2025/26 have been read by earlier pre-registered rules: they are shown apart, as development evidence.

    python -m research.open_close.football_market_scan --cache <lab cache> --out-json reports/football_runs/market_scan.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COMMISSION = 0.02
BANDS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1001.0]
BAND_LABELS = ["1.0-1.5", "1.5-2", "2-3", "3-5", "5-10", "10+"]
EDGES = (0.0, 0.05, 0.10)
N_BOOT = 2000


# --------------------------------------------------------------------------------------------------------------------
# prices
# --------------------------------------------------------------------------------------------------------------------

def tick_up(p: np.ndarray) -> np.ndarray:
    from trading.exchange import tick_up as tu
    return np.array([tu(x) if np.isfinite(x) and x > 1 else np.nan for x in p])


def tick_down(p: np.ndarray) -> np.ndarray:
    from trading.exchange import tick_down as td
    return np.array([td(x) if np.isfinite(x) and x > 1 else np.nan for x in p])


def fair(prices: list[np.ndarray]) -> list[np.ndarray]:
    """De-vigged fair prices (proportional) for one market's selections."""
    inv = np.vstack([1 / np.asarray(p, float) for p in prices])
    s = inv.sum(0)
    return [s / i for i in inv]


def sym_lay(back: np.ndarray, fair_p: np.ndarray, ticks: bool = True) -> np.ndarray:
    """The lay price of a book symmetric about the fair price: fair^2 / back, on Betfair the next tick at or above."""
    raw = fair_p ** 2 / back
    return tick_up(raw - 1e-9) if ticks else raw


# --------------------------------------------------------------------------------------------------------------------
# settlement
# --------------------------------------------------------------------------------------------------------------------

def ah_home_units(gd: np.ndarray, line: np.ndarray) -> np.ndarray:
    """The home side's Asian handicap result per unit, as (win share, push share, lose share) collapsed to a number
    in [-1, 1] of 'wins' where 0.5 is a half win: returns (w, p, l) fractions for two half-stakes at quarter lines."""
    out = np.zeros((len(gd), 3))
    for k, (g, ln) in enumerate(zip(gd, line)):
        if not (np.isfinite(g) and np.isfinite(ln)):
            out[k] = np.nan
            continue
        parts = [ln] if abs((ln * 4) % 2) != 1 else [ln - 0.25, ln + 0.25]
        for pl in parts:
            r = g + pl
            out[k, 0 if r > 0 else 1 if r == 0 else 2] += 1 / len(parts)
    return out


def back_return(price: np.ndarray, won: np.ndarray, com: float, push: np.ndarray | None = None) -> np.ndarray:
    """Per unit staked, held to the result; commission on a winning bet's profit. ``won``/``push`` may be fractions."""
    push = np.zeros_like(price) if push is None else push
    lose = 1 - won - push
    return won * (price - 1) * (1 - com) - lose


def lay_return(price: np.ndarray, won: np.ndarray, com: float, push: np.ndarray | None = None) -> np.ndarray:
    """Per unit of the backer's stake laid, held to the result: +1 (less commission) when the selection loses."""
    push = np.zeros_like(price) if push is None else push
    lose = 1 - won - push
    return lose * (1 - com) - won * (price - 1)


def green(entry: np.ndarray, exit_: np.ndarray, side: int) -> np.ndarray:
    """Greened at the exit, per unit staked at the entry (before commission): a back o/c - 1, a lay 1 - o/c."""
    return entry / exit_ - 1 if side > 0 else 1 - entry / exit_


# --------------------------------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------------------------------

def summary(ret: np.ndarray, day: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    ok = np.isfinite(ret)
    r, d = ret[ok], np.asarray(day)[ok]
    if len(r) == 0:
        return {"bets": 0}
    out = {"bets": int(len(r)), "roi": round(float(r.mean()), 5)}
    if len(r) >= 30 and n_boot:
        days, inv = np.unique(d, return_inverse=True)
        num, den = np.bincount(inv, r), np.bincount(inv)
        idx = np.random.default_rng(seed).integers(0, len(days), (n_boot, len(days)))
        boot = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
        out["lo90"], out["hi90"] = round(float(np.percentile(boot, 5)), 5), round(float(np.percentile(boot, 95)), 5)
    return out


def by_band(ret, price, day) -> dict:
    band = pd.cut(price, BANDS, labels=BAND_LABELS, right=False)
    return {str(b): summary(ret[band == b], day[band == b]) for b in BAND_LABELS if (band == b).sum()}


#: 2024 = 2024/25; the seasons from 2025/26 were read by earlier pre-registered rules (development evidence)
PERIODS = {"2024": (2024, 2024), "2025_on_spent": (2025, 2099), "all": (2024, 2099)}


# --------------------------------------------------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------------------------------------------------

def market_frames(m: pd.DataFrame) -> dict[str, dict]:
    """Per market: for each selection, Betfair's early and closing prices, the result (won, push), the model's
    probability, and where each price is Betfair's ("BFE"/"BFEC", else blank)."""
    d = m[m["ft_home"].notna() & m["ft_away"].notna()].copy()
    gd = (d["ft_home"] - d["ft_away"]).to_numpy(float)
    goals = (d["ft_home"] + d["ft_away"]).to_numpy(float)
    out = {}
    one = {"H": gd > 0, "D": gd == 0, "A": gd < 0}
    def tag(cols, label):
        ok = np.all([np.isfinite(d[c].to_numpy(float)) for c in cols], axis=0)
        return np.where(ok, label, "")
    out["match_odds"] = {"d": d, "sels": {k: {"open": d[f"bfe_open_{k.lower()}"].to_numpy(float),
                                              "close": d[f"bfe_close_{k.lower()}"].to_numpy(float),
                                              "won": one[k].astype(float), "push": np.zeros(len(d)),
                                              "p": d[f"p_{k.lower()}"].to_numpy(float)} for k in one},
                         "open_src": tag(["bfe_open_h", "bfe_open_d", "bfe_open_a"], "BFE"),
                         "close_src": tag(["bfe_close_h", "bfe_close_d", "bfe_close_a"], "BFEC")}
    po = d["p_o25"].to_numpy(float)
    out["over_under_25"] = {"d": d, "sels": {
        "Over": {"open": d["bfe_open_o25"].to_numpy(float), "close": d["bfe_close_o25"].to_numpy(float),
                 "won": (goals > 2.5).astype(float), "push": np.zeros(len(d)), "p": po},
        "Under": {"open": d["bfe_open_u25"].to_numpy(float), "close": d["bfe_close_u25"].to_numpy(float),
                  "won": (goals < 2.5).astype(float), "push": np.zeros(len(d)), "p": 1 - po}},
        "open_src": tag(["bfe_open_o25", "bfe_open_u25"], "BFE"),
        "close_src": tag(["bfe_close_o25", "bfe_close_u25"], "BFEC")}
    # the handicap: open and close lines differ, so each moment settles on its own line
    ah = {}
    for when in ("open", "close"):
        line = d[f"{when}_ah_line"].to_numpy(float)
        res = ah_home_units(gd, line)
        ph = ah_probability(d["lam_h"].to_numpy(float), d["lam_a"].to_numpy(float), line)
        ah[when] = {"line": line, "res": res, "p_home": ph}
    out["asian_handicap"] = {"d": d, "ah": ah, "open_src": d["open_ah_src"].astype(str).to_numpy(),
                             "close_src": d["close_ah_src"].astype(str).to_numpy(),
                             "sels": {"Home": {"open": d["open_ah_h"].to_numpy(float),
                                               "close": d["close_ah_h"].to_numpy(float)},
                                      "Away": {"open": d["open_ah_a"].to_numpy(float),
                                               "close": d["close_ah_a"].to_numpy(float)}}}
    return out


def ah_probability(lh: np.ndarray, la: np.ndarray, line: np.ndarray, max_goals: int = 10) -> np.ndarray:
    """The model's (win, push, lose) fractions for the home side at the line, from independent Poisson goals with the
    Dixon-Coles rates (the low-score correction left out: a small approximation at these lines)."""
    from scipy.stats import poisson
    g = np.arange(max_goals + 1)
    out = np.full((len(lh), 3), np.nan)
    for k, (a, b, ln) in enumerate(zip(lh, la, line)):
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(ln)):
            continue
        pm = np.outer(poisson.pmf(g, a), poisson.pmf(g, b))
        diff = g[:, None] - g[None, :]
        parts = [ln] if abs((ln * 4) % 2) != 1 else [ln - 0.25, ln + 0.25]
        w = p = l_ = 0.0
        for pl in parts:
            r = diff + pl
            w += pm[r > 0].sum() / len(parts)
            p += pm[r == 0].sum() / len(parts)
            l_ += pm[r < 0].sum() / len(parts)
        s = w + p + l_
        out[k] = (w / s, p / s, l_ / s)
    return out


def venue_of(src: np.ndarray) -> np.ndarray:
    """'betfair' where the price is Betfair's, else 'other' (never scored)."""
    return np.where(pd.Series(src).astype(str).str.startswith("BFE").to_numpy(), "betfair", "other")


def scan(m: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    res = {"notes": {"commission_betfair": COMMISSION, "lay_costing": "symmetric book about the fair price",
                     "bootstrap": f"{n_boot} draws by match day, 90%"}}
    frames = market_frames(m)
    for mk in ("match_odds", "over_under_25"):
        f = frames[mk]
        d, sels = f["d"], f["sels"]
        day = d["match_date"].astype(str).to_numpy()
        season = d["season"].to_numpy()
        names = list(sels)
        out_mk = {}
        for when in ("open", "close"):
            src = f[f"{when}_src"]
            ven = venue_of(src)
            prices = [sels[s][when] for s in names]
            fairs = fair(prices)
            for venue in ("betfair",):
                com = COMMISSION
                for per, (lo_s, hi_s) in PERIODS.items():
                    rows = (ven == venue) & (season >= lo_s) & (season <= hi_s)
                    if rows.sum() < 200:
                        continue
                    key = f"{when}_{venue}_{per}"
                    block = {}
                    for s, fp in zip(names, fairs):
                        px, won, p = sels[s][when], sels[s]["won"], sels[s]["p"]
                        r = rows & np.isfinite(px) & (px > 1)
                        back = back_return(px[r], won[r], com)
                        lay_px = sym_lay(px[r], fp[r], ticks=venue == "betfair")
                        lay = lay_return(lay_px, won[r], com)
                        ev_b = p[r] * (px[r] - 1) * (1 - com) - (1 - p[r])
                        ev_l = (1 - p[r]) * (1 - com) - p[r] * (lay_px - 1)
                        block[s] = {
                            "back_all": summary(back, day[r], n_boot),
                            "back_by_band": by_band(back, px[r], day[r]),
                            "lay_all": summary(lay, day[r], n_boot),
                            "lay_by_band": by_band(lay, px[r], day[r]),
                            "model_back": {str(e): summary(np.where(ev_b > e, back, np.nan), day[r], n_boot)
                                           for e in EDGES},
                            "model_lay": {str(e): summary(np.where(ev_l > e, lay, np.nan), day[r], n_boot)
                                          for e in EDGES},
                        }
                    out_mk[key] = block
        # trading the move: entry at the open, greened at the close (same venue at both ends)
        vo, vc = venue_of(f["open_src"]), venue_of(f["close_src"])
        f_open = dict(zip(names, fair([sels[s]["open"] for s in names])))
        f_close = dict(zip(names, fair([sels[s]["close"] for s in names])))
        trade = {}
        for venue in ("betfair",):
            for per, (lo_s, hi_s) in PERIODS.items():
                rows = (vo == venue) & (vc == venue) & (season >= lo_s) & (season <= hi_s)
                if rows.sum() < 200:
                    continue
                block = {}
                for s in names:
                    o, c, p = sels[s]["open"], sels[s]["close"], sels[s]["p"]
                    r = rows & np.isfinite(o) & np.isfinite(c) & (o > 1) & (c > 1)
                    tk = venue == "betfair"
                    # back at the quoted back price, lay at the symmetric book's lay price, at each end
                    b2l = green(o[r], sym_lay(c[r], f_close[s][r], tk), 1)
                    l2b = green(sym_lay(o[r], f_open[s][r], tk), c[r], -1)
                    if venue == "betfair":                     # commission on each greened profit
                        b2l = np.where(b2l > 0, b2l * (1 - COMMISSION), b2l)
                        l2b = np.where(l2b > 0, l2b * (1 - COMMISSION), l2b)
                    # the model's disagreement with the open: back the selection the model prices shorter than the
                    # open (expects the price to come in), lay the one it prices longer
                    gap = p[r] * o[r] - 1                       # model's expected return at the open, before costs
                    block[s] = {"back_to_lay_all": summary(b2l, day[r], n_boot),
                                "back_to_lay_by_band": by_band(b2l, o[r], day[r]),
                                "lay_to_back_all": summary(l2b, day[r], n_boot),
                                "model_back_to_lay": {str(e): summary(np.where(gap > e, b2l, np.nan), day[r], n_boot)
                                                      for e in (0.05, 0.10)},
                                "model_lay_to_back": {str(e): summary(np.where(gap < -e, l2b, np.nan), day[r], n_boot)
                                                      for e in (0.05, 0.10)},
                                "close_moved_toward_model": round(float(np.nanmean(
                                    np.sign(1 / c[r] - 1 / o[r]) == np.sign(p[r] - 1 / o[r]))), 4)}
                trade[f"{venue}_{per}"] = block
        out_mk["trade_open_to_close"] = trade
        res[mk] = out_mk
    res["asian_handicap"] = scan_ah(frames["asian_handicap"], n_boot)
    return res


def scan_ah(f: dict, n_boot: int) -> dict:
    d = f["d"]
    day = d["match_date"].astype(str).to_numpy()
    season = d["season"].to_numpy()
    out = {}
    for when in ("open", "close"):
        ven = venue_of(f[f"{when}_src"])
        a = f["ah"][when]
        res, pmod = a["res"], a["p_home"]
        h, aw = f["sels"]["Home"][when], f["sels"]["Away"][when]
        fh, fa = fair([h, aw])
        for venue in ("betfair",):
            com = COMMISSION
            for per, (lo_s, hi_s) in PERIODS.items():
                rows = (ven == venue) & (season >= lo_s) & (season <= hi_s) & np.isfinite(h) & np.isfinite(aw) \
                    & np.isfinite(res[:, 0])
                if rows.sum() < 200:
                    continue
                block = {}
                for side, px, fp, (wi, li) in (("Home", h, fh, (0, 2)), ("Away", aw, fa, (2, 0))):
                    r = rows & (px > 1)
                    won, push = res[r, wi], res[r, 1]
                    back = back_return(px[r], won, com, push)
                    lay_px = sym_lay(px[r], fp[r], ticks=venue == "betfair")
                    lay = lay_return(lay_px, won, com, push)
                    pw, pp = pmod[r, wi], pmod[r, 1]
                    ev_b = pw * (px[r] - 1) * (1 - com) - (1 - pw - pp)
                    ev_l = (1 - pw - pp) * (1 - com) - pw * (lay_px - 1)
                    block[side] = {"back_all": summary(back, day[r], n_boot),
                                   "lay_all": summary(lay, day[r], n_boot),
                                   "model_back": {str(e): summary(np.where(ev_b > e, back, np.nan), day[r], n_boot)
                                                  for e in EDGES},
                                   "model_lay": {str(e): summary(np.where(ev_l > e, lay, np.nan), day[r], n_boot)
                                                 for e in EDGES}}
                out[f"{when}_{venue}_{per}"] = block
    return out


def longshot_lay_sensitivity(m: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Laying 10.0+ selections at Betfair's close (the favourite-longshot bias), against the lay price actually paid:
    the closing back price times 1.00 to 1.30. Also each lay's liability per unit stake, since the return is per
    backer's stake and the capital at risk is the liability."""
    d = m[m["ft_home"].notna()]
    gd = (d["ft_home"] - d["ft_away"]).to_numpy(float)
    day_all = d["match_date"].astype(str).to_numpy()
    season = d["season"].to_numpy()
    out = {}
    for sel, won, col in (("Away", gd < 0, "bfe_close_a"), ("Home", gd > 0, "bfe_close_h"),
                          ("Draw", gd == 0, "bfe_close_d")):
        px = d[col].to_numpy(float)
        for per, (lo_s, hi_s) in PERIODS.items():
            r = np.isfinite(px) & (px >= 10) & (season >= lo_s) & (season <= hi_s)
            if r.sum() < 30:
                continue
            w = won[r].astype(float)
            rec = {"bets": int(r.sum()), "win_rate": round(float(w.mean()), 5),
                   "implied": round(float(np.mean(1 / px[r])), 5),
                   "liability_per_stake_at_x1.10": round(float((px[r] * 1.1 - 1).mean()), 2)}
            for mk in (1.00, 1.05, 1.10, 1.20, 1.30):
                rec[f"lay_at_back_x{mk:.2f}"] = summary(lay_return(px[r] * mk, w, COMMISSION), day_all[r], n_boot)
            out[f"betfair_close_{per}_{sel}_10plus"] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from football import lab
    m = lab.latest(a.cache)
    log.info("matrix: %d rows, %d with a result", len(m), int(m["ft_home"].notna().sum()))
    res = scan(m, a.boot)
    res["longshot_lay_sensitivity"] = longshot_lay_sensitivity(m, a.boot)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    log.info("written %s", a.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
