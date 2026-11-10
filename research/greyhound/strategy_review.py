"""The strategy review on the new model (the owner, 10 Nov 2026: "Do a full review, open minded, on profitable trading
and win betting or lay betting strategies. Based on the new model.").

Every family of rule a pre-race price can drive, scored on Betfair's greyhound price files (the BSP, the morning and
pre-play weighted average prices and volumes, the place market), joined to a model's out-of-sample prices (the fast
loop's saved prices: batch 19, ``greyhound/research/pred/<name>.parquet``: fitted before 1 Jun 2026, 1 Jun to 5 Nov
2026 priced). Each rule is fixed here before its scores. **Development:** 1 Jun to 31 Aug 2026. **Holdout:** 1 Sep to
5 Nov 2026. **A pass:** the development return above zero and the holdout return's 90% interval (a bootstrap over
days) above zero, with at least ``MIN_BETS`` holdout bets. Commission 2% on net winnings (greyhound/costs.py). Every
rule is counted: about one in twenty passes a one-sided 90% bound by chance.

The families:

1. **Back at the SP** where the model's value at the Betfair SP clears a margin (executable as a limit-on-close back:
   the SP bet with its lowest acceptable price), and **lay at the SP** where the model's chance is that far above the
   SP's (a limit-on-close lay with its highest price). On the model's chance and on a market-plus-model blend
   (Benter's conditional logit of the BSP's and the model's log chances, fitted on the development days only).
2. **The morning move:** where the model's SP forecast is longer than the morning price (the morning WAP), lay in the
   morning; where it is shorter, back in the morning. Closed at the SP (greened: the same profit whatever wins) or
   held to the result. The morning WAP is an average of what traded, not a price on offer: each rule is read at the
   WAP and 1 and 2 ticks worse (a taker pays the spread). With and without a morning-volume floor.
3. **The pre-play move:** the same against the pre-play WAP (traded nearer the off; less to move, more to fill).
4. **The place market:** the model's chance through Harville's formula against the place SP, backed or laid at it.

The recorded books (6 Oct on) carry what the price files cannot (the book on offer, the queue): the limit-order
study, the trade-out study and the early lay read those, on the same saved prices, in the same fast-loop batch.

    python -m research.greyhound.strategy_review --pred greyhound/research/pred/stack_review.parquet --out out/review
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound.costs import COMMISSION as COM

log = logging.getLogger(__name__)

DEV_END = "2026-09-01"
MIN_BETS = 100
KEY = ["race_date", "race_time", "track", "trap"]
SP_MARGINS = (0.0, 0.05, 0.10, 0.20, 0.30)
SP_CAPS = (20.0, 100.0)
MOVE_MARGINS = (0.05, 0.10, 0.20, 0.30)
MOVE_VOLS = (0.0, 20.0)
MOVE_CAP = 30.0
HAIRCUTS = (0, 1, 2)
PLACE_MARGINS = (0.05, 0.10, 0.20)
N_BOOT = 2000


# --------------------------------------------------------------------------------------------------------------------
# The data
# --------------------------------------------------------------------------------------------------------------------

def load_prices(store, years) -> pd.DataFrame:
    """GB win runners with their price-file prices, and the place market's SP and result where it has one."""
    parts = []
    for y in years:
        t = store.get_parquet(f"greyhound_prices/prices_{y}.parquet")
        if t is not None and len(t):
            parts.append(t[t.country == "GB"])
    d = pd.concat(parts, ignore_index=True).dropna(subset=["trap"])
    d = d.drop_duplicates(["market"] + KEY, keep="last")
    vals = ["event_id", "bsp", "ppwap", "morningwap", "morning_vol", "pp_vol", "win_lose"]
    w = d[d.market == "win"][KEY + [c for c in vals if c in d]]
    p = d[d.market == "place"][KEY + ["bsp", "win_lose"]].rename(columns={"bsp": "bsp_place", "win_lose": "placed"})
    return w.merge(p, on=KEY, how="left")


def join(pred: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The model's runners on the price files: a market kept only when every runner joined and the file's result
    agrees with GBGB's on every runner (a join on the wrong race would score one race's prices on another's result)."""
    p = pred.copy()
    p["race_time"] = p.race_time.astype(str).str[:5]
    p["race_date"] = p.race_date.astype(str)
    m = p.merge(prices.rename(columns={"bsp": "bsp_file"}), on=KEY, how="inner")
    why = {"model_runners": int(len(p)), "model_races": int(p.race_id.nunique()), "joined_runners": int(len(m))}
    whole = m.groupby("race_id").trap.transform("size") == m.groupby("race_id").field.transform("first")
    m = m[whole]
    agree = (m.win_lose.isna() | (m.win_lose == m.won)).groupby(m.race_id).transform("all")
    m = m[agree].copy()
    m["bsp"] = m.bsp.where(m.bsp > 1, m.bsp_file)
    m = m[m.bsp > 1]
    m = m[m.groupby("race_id").bsp.transform("size") == m.groupby("race_id").field.transform("first")]
    why.update({"kept_runners": int(len(m)), "kept_races": int(m.race_id.nunique()),
                "morning_priced": round(float((m.morningwap > 1).mean()), 3),
                "preplay_priced": round(float((m.ppwap > 1).mean()), 3),
                "place_priced": round(float((m.bsp_place > 1).mean()), 3)})
    m["period"] = np.where(m.race_date < DEV_END, "dev", "holdout")
    return m.sort_values(["race_id", "trap"]).reset_index(drop=True), why


# --------------------------------------------------------------------------------------------------------------------
# Profit and loss, a unit at a time (the market's net winnings pay the commission)
# --------------------------------------------------------------------------------------------------------------------

def _net(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x * (1 - COM), x)


def back_hold(price, won) -> np.ndarray:
    """A unit backed at ``price``, held to the result: a return on the unit risked."""
    return _net(np.where(won == 1, price - 1, -1.0))


def lay_hold(price, won) -> np.ndarray:
    """A unit laid at ``price``, held to the result: a return on the liability (price - 1) risked."""
    return _net(np.where(won == 1, -(price - 1), 1.0)) / (price - 1)


def back_green(entry, close) -> np.ndarray:
    """Backed at ``entry``, laid at ``close`` for entry/close units: the same profit whatever wins, on the unit risked."""
    return _net(entry / close - 1)


def lay_green(entry, close) -> np.ndarray:
    """Laid at ``entry``, backed at ``close`` for entry/close units: the same profit whatever wins, on the liability."""
    return _net(1 - entry / close) / (entry - 1)


def ticks(x: np.ndarray, k: int, up: bool) -> np.ndarray:
    """``k`` ticks up or down Betfair's ladder from each price (on the ladder first)."""
    if k == 0:
        return np.asarray(x, float)
    from research.greyhound.limit_study import ticks_from
    return ticks_from(x, k, up)


# --------------------------------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------------------------------

def _boot_lower(pnl: np.ndarray, day: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """The 90% interval of the mean return, days resampled with their bets."""
    if not len(pnl):
        return (np.nan, np.nan)
    d = pd.DataFrame({"d": day, "p": pnl}).groupby("d").p.agg(["sum", "size"])
    s, n = d["sum"].to_numpy(), d["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    means = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def score(sel: pd.DataFrame, pnl: np.ndarray) -> dict:
    out = {}
    for per in ("dev", "holdout"):
        k = (sel.period == per).to_numpy()
        x = pnl[k]
        lo, hi = _boot_lower(x, sel.race_date.to_numpy()[k])
        out[per] = {"bets": int(k.sum()), "per_day": round(float(k.sum()) / max(sel.race_date[k].nunique(), 1), 1),
                    "return%": round(100 * float(np.mean(x)), 2) if len(x) else None,
                    "ci90%": [round(100 * lo, 2), round(100 * hi, 2)] if len(x) else None}
    d, h = out["dev"], out["holdout"]
    out["passes"] = bool(d["bets"] and h["bets"] >= MIN_BETS and d["return%"] > 0 and h["ci90%"][0] > 0)
    return out


# --------------------------------------------------------------------------------------------------------------------
# The families
# --------------------------------------------------------------------------------------------------------------------

def blend_chance(m: pd.DataFrame) -> pd.Series:
    """Benter's blend: the race softmax of a x log(BSP chance) + b x log(model chance), a and b fitted on the
    development days only (the conditional logit's likelihood of the winners)."""
    from scipy.optimize import minimize

    from greyhound.model import race_norm
    q = race_norm(1.0 / m.bsp, m.race_id)
    lq, lp = np.log(q.clip(1e-6)).to_numpy(), np.log(m.p_model.clip(1e-6)).to_numpy()
    codes = pd.factorize(m.race_id)[0]
    dev = (m.period == "dev").to_numpy()

    def softmax(a, b):
        z = a * lq + b * lp
        z = z - pd.Series(z).groupby(codes).transform("max").to_numpy()
        e = np.exp(z)
        return e / pd.Series(e).groupby(codes).transform("sum").to_numpy()

    won = m.won.to_numpy(float)

    def nll(w):
        p = softmax(w[0], w[1])
        return -float(np.log(np.clip(p[dev & (won == 1)], 1e-12, None)).sum())
    w = minimize(nll, np.array([1.0, 0.0]), method="Nelder-Mead").x
    out = pd.Series(softmax(w[0], w[1]), index=m.index)
    out.attrs["weights"] = {"bsp": round(float(w[0]), 4), "model": round(float(w[1]), 4)}
    return out


def sp_rules(m: pd.DataFrame, chance: pd.Series, tag: str) -> list[dict]:
    """Backs and lays at the SP on ``chance``'s value (limit-on-close)."""
    rows = []
    s, won, p = m.bsp.to_numpy(float), m.won.to_numpy(float), chance.to_numpy(float)
    back_val = p * (1 + (s - 1) * (1 - COM)) - 1               # a unit backed: the expected return
    lay_val = (1 - p) * (1 - COM) - p * (s - 1)                # a unit laid: the expected return on the stake
    for cap in SP_CAPS:
        for th in SP_MARGINS:
            k = (back_val > th) & (s <= cap)
            rows.append({"family": f"SP back ({tag})", "rule": f"value > {th:.2f}, SP <= {cap:g}",
                         **score(m[k], back_hold(s[k], won[k]))})
            k = (lay_val / np.maximum(s - 1, 1e-9) > th) & (s <= cap)
            rows.append({"family": f"SP lay ({tag})", "rule": f"value > {th:.2f} of liability, SP <= {cap:g}",
                         **score(m[k], lay_hold(s[k], won[k]))})
    return rows


def move_rules(m: pd.DataFrame, entry_col: str, label: str, vol_col: str | None) -> list[dict]:
    """Lay where the forecast SP is above the entry price by a margin, back where it is below; greened at the SP or
    held to the result; at the entry price and 1, 2 ticks worse."""
    rows = []
    e0, s, won = m[entry_col].to_numpy(float), m.bsp.to_numpy(float), m.won.to_numpy(float)
    f = m.pred_bsp.to_numpy(float)
    vol = m[vol_col].fillna(0).to_numpy(float) if vol_col else np.zeros(len(m))
    ok = (e0 > 1) & (e0 <= MOVE_CAP) & np.isfinite(f)
    r = np.log(np.where(ok, f, np.nan) / np.where(ok, e0, np.nan))
    for v in (MOVE_VOLS if vol_col else (0.0,)):
        for th in MOVE_MARGINS:
            for side in ("lay", "back"):
                k = ok & (vol >= v) & ((r > th) if side == "lay" else (r < -th))
                for h in HAIRCUTS:
                    e = ticks(e0[k], h, up=(side == "lay"))       # a lay takes a higher price, a back a lower
                    good = np.isfinite(e) & (e > 1)
                    sub = m[k][good]
                    e, sk, wk = e[good], s[k][good], won[k][good]
                    vtag = f", {label} volume >= {v:g}" if vol_col else ""
                    base = f"forecast {'above' if side == 'lay' else 'below'} the {label} price by {th:.2f} (log){vtag}"
                    green = lay_green(e, sk) if side == "lay" else back_green(e, sk)
                    hold = lay_hold(e, wk) if side == "lay" else back_hold(e, wk)
                    rows.append({"family": f"{label} move: {side}, closed at the SP", "rule": f"{base}, {h} ticks",
                                 **score(sub, green)})
                    rows.append({"family": f"{label} move: {side}, held", "rule": f"{base}, {h} ticks",
                                 **score(sub, hold)})
    return rows


def place_rules(m: pd.DataFrame) -> list[dict]:
    """The model's chances through Harville against the place SP (races with a whole place market, 2 or 3 placed)."""
    from research.greyhound.place_value import MAXN, harville
    d = m[m.bsp_place > 1].copy()
    g = d.groupby("race_id")
    d = d[(g.trap.transform("size") == g.field.transform("first")) & g.placed.transform("sum").isin([2, 3])
          & (g.trap.transform("size") <= MAXN)]
    if not len(d):
        return []
    d = d.sort_values(["race_id", "trap"])
    codes, idx = np.unique(d.race_id.to_numpy(), return_inverse=True)
    pos = d.groupby("race_id").cumcount().to_numpy()
    q = np.zeros((len(codes), MAXN))
    q[idx, pos] = d.p_model.to_numpy(float)
    k = d.groupby("race_id").placed.sum().reindex(codes).to_numpy()
    d["p_place"] = harville(q, k.astype(int))[idx, pos]
    s, y, p = d.bsp_place.to_numpy(float), d.placed.to_numpy(float), d.p_place.to_numpy(float)
    rows = []
    for th in PLACE_MARGINS:
        kb = p * (1 + (s - 1) * (1 - COM)) - 1 > th
        rows.append({"family": "place SP back (Harville on the model)", "rule": f"value > {th:.2f}",
                     **score(d[kb], back_hold(s[kb], y[kb]))})
        kl = ((1 - p) * (1 - COM) - p * (s - 1)) / np.maximum(s - 1, 1e-9) > th
        rows.append({"family": "place SP lay (Harville on the model)", "rule": f"value > {th:.2f} of liability",
                     **score(d[kl], lay_hold(s[kl], y[kl]))})
    return rows


def forecast_quality(m: pd.DataFrame) -> dict:
    """What the model knows before the market moves: its forecast of the SP against the morning and pre-play prices
    as forecasts of the SP (RMSE of log(SP), runners priced by all three), and the win chance against the BSP's."""
    from greyhound.model import logloss, race_norm
    k = (m.morningwap > 1) & (m.ppwap > 1)
    ls = np.log(m.bsp[k])
    out = {"runners_all_three": int(k.sum())}
    for name, col in (("model", "pred_bsp"), ("morning WAP", "morningwap"), ("pre-play WAP", "ppwap")):
        out[f"rmse_log_sp_{name}"] = round(float(np.sqrt(np.mean((np.log(m[col][k]) - ls) ** 2))), 4)
    # does the model add to the morning price? the SP regressed on both (development fit, holdout read)
    dev = k & (m.period == "dev")
    hold = k & (m.period == "holdout")
    X = lambda d: np.c_[np.ones(len(d)), np.log(d.morningwap), np.log(d.pred_bsp)]  # noqa: E731
    beta, *_ = np.linalg.lstsq(X(m[dev]), np.log(m.bsp[dev]), rcond=None)
    e_both = np.log(m.bsp[hold]) - X(m[hold]) @ beta
    beta_m, *_ = np.linalg.lstsq(X(m[dev])[:, :2], np.log(m.bsp[dev]), rcond=None)
    e_morn = np.log(m.bsp[hold]) - X(m[hold])[:, :2] @ beta_m
    out["holdout_rmse_log_sp_morning_only"] = round(float(np.sqrt(np.mean(e_morn ** 2))), 4)
    out["holdout_rmse_log_sp_morning_and_model"] = round(float(np.sqrt(np.mean(e_both ** 2))), 4)
    out["weights_morning_model"] = [round(float(beta[1]), 3), round(float(beta[2]), 3)]
    q = race_norm(1.0 / m.bsp, m.race_id)
    out["logloss_model"] = round(logloss(m.p_model, m.won), 5)
    out["logloss_bsp"] = round(logloss(q, m.won), 5)
    return out


def review(m: pd.DataFrame) -> dict:
    rows = []
    rows += sp_rules(m, m.p_model, "the model's chance")
    b = blend_chance(m)
    rows += sp_rules(m, b, "BSP + model blend")
    rows += move_rules(m, "morningwap", "morning", "morning_vol")
    rows += move_rules(m, "ppwap", "pre-play", "pp_vol")
    rows += place_rules(m)
    passes = [r for r in rows if r["passes"]]
    return {"rules": rows, "rules_tried": len(rows), "passes": len(passes),
            "expected_by_chance": round(0.05 * len(rows), 1), "blend_weights": b.attrs.get("weights"),
            "forecast": forecast_quality(m)}


def markdown(name: str, why: dict, out: dict) -> str:
    f = out["forecast"]
    lines = [f"## Strategy review: {name}", "",
             f"Runners kept {why['kept_runners']:,} in {why['kept_races']:,} races (of {why['model_runners']:,} "
             f"priced; a race kept only when every runner joined the price files and the result agrees). Morning WAP "
             f"on {why['morning_priced']:.0%} of runners, pre-play WAP {why['preplay_priced']:.0%}, place SP "
             f"{why['place_priced']:.0%}. Development to {DEV_END}, holdout after; commission {COM:.0%}.", "",
             f"Forecast of the SP (RMSE of log(SP), {f['runners_all_three']:,} runners with every price): the model "
             f"{f['rmse_log_sp_model']}, the morning WAP {f['rmse_log_sp_morning WAP']}, the pre-play WAP "
             f"{f['rmse_log_sp_pre-play WAP']}. Holdout, the morning WAP alone {f['holdout_rmse_log_sp_morning_only']}, "
             f"with the model {f['holdout_rmse_log_sp_morning_and_model']} (weights {f['weights_morning_model']}). "
             f"Win log-loss: the model {f['logloss_model']}, the BSP {f['logloss_bsp']}; the blend's weights "
             f"{out['blend_weights']}.", "",
             f"**{out['rules_tried']} rules, {out['passes']} pass** (about {out['expected_by_chance']} by chance).", "",
             "| Family | Rule | Dev bets | Dev return | Holdout bets | Holdout return (90%) | Pass |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(out["rules"], key=lambda r: -(r["holdout"]["ci90%"] or [-1e9])[0]):
        d, h = r["dev"], r["holdout"]
        ci = h["ci90%"] or ["", ""]
        lines.append(f"| {r['family']} | {r['rule']} | {d['bets']} | {d['return%']}% | {h['bets']} | "
                     f"{h['return%']}% ({ci[0]} to {ci[1]}) | {'yes' if r['passes'] else ''} |")
    return "\n".join(lines) + "\n"


def run(store, pred: pd.DataFrame, name: str, out_dir: Path) -> dict:
    years = sorted({int(str(d)[:4]) for d in pred.race_date.unique()})
    m, why = join(pred, load_prices(store, years))
    out = review(m)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"review_{name}.json").write_text(json.dumps({"join": why, **out}, indent=1, default=str))
    (out_dir / f"review_{name}.md").write_text(markdown(name, why, out))
    return {"report": f"review_{name}.md", "rules_tried": out["rules_tried"], "passes": out["passes"],
            "forecast": out["forecast"], "join": why}


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", required=True, help="a saved prices key in the store (greyhound/research/pred/...)")
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/review")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    pred = store.get_parquet(a.pred)
    name = Path(a.pred).stem
    print(json.dumps(run(store, pred, name, Path(a.out)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
