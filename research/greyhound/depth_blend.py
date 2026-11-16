"""Does our price add to the Betfair SP where the market is thin?

The owner, 11 Nov 2026: "Do not stop until we beat the market." The scoreboard so far is one blend for every race:
our chance and the SP's, weighted by a conditional logit fitted on the first part of the out-of-sample prices and
scored on the rest against the SP alone. On batch 26's trees it gives our price a weight of 0.135 and does not beat
the SP on the later half (0.42778 against 0.42758; reports/greyhound_model.md section 44).

One weight for every race assumes the SP is as good everywhere. It is not made the same way everywhere: a Saturday
night open race takes thousands of pounds before the off and a morning graded race a few hundred, and a price made
by little money carries more noise for a second opinion to average out. So the weight is fitted by the depth of the
race's market:

- ``realised``: the money matched on the race before the off (Betfair's price files, ``pp_vol`` summed over the
  field), in three bands cut at the fit part's terciles. Known only at the off, as the SP is: this asks whether the
  blend is a better chance than the SP, which is the scoreboard's question.
- ``expected``: the same bands on what the race could be expected to take before a penny is matched - the median
  money of the fit part's races at the same track in the same part of the day (morning, afternoon, evening; the
  track's median where that cell has fewer than ``MIN_CELL`` races). Known when the card is: this is the one a trade
  could be built on.

Each band has its own two weights (on the log of the SP's share and of ours, normalised in the race), fitted on the
days before ``split`` and scored on the days from it, runner by runner against the SP alone, paired by day.
Nothing is fitted on a scored day.

    python -m research.greyhound.depth_blend --pred out/pred_f_stack.parquet --split 2026-09-01 --out out/depth.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

MIN_CELL = 30
BANDS = ("thin", "middle", "deep")


def part_of_day(race_time: pd.Series) -> pd.Series:
    h = pd.to_numeric(race_time.astype(str).str[:2], errors="coerce")
    return pd.Series(np.select([h < 12, h < 18], ["morning", "afternoon"], "evening"), index=race_time.index)


def prepare(pred: pd.DataFrame, col: str = "p_model") -> pd.DataFrame:
    """The runners of races the SP prices whole, with the SP's share (``q``), ours (``p``, normalised over the same
    runners) and the race's money before the off (``depth``; NaN where the files have none)."""
    d = pred.copy()
    ok = (d.bsp > 1) & (d[col] > 0)
    d = d[ok.groupby(d.race_id).transform("all") & (d.groupby("race_id").won.transform("sum") >= 1)].copy()
    q = 1.0 / d.bsp.astype(float)
    d["q"] = q / q.groupby(d.race_id).transform("sum")
    p = d[col].astype(float)
    d["p"] = p / p.groupby(d.race_id).transform("sum")
    if "pp_vol" in d:
        v = pd.to_numeric(d.pp_vol, errors="coerce")
        d["depth"] = v.groupby(d.race_id).transform("sum").where(v.notna().groupby(d.race_id).transform("any"))
    else:
        d["depth"] = np.nan
    d["day"] = d.race_date.astype(str)
    d["part"] = part_of_day(d.race_time) if "race_time" in d else "all"
    return d


def expected_depth(d: pd.DataFrame, fit: pd.Series) -> pd.Series:
    """What each race could be expected to take: the median of the fit part's races in its track and part of day."""
    races = d[fit].drop_duplicates("race_id")[["track", "part", "depth"]].dropna(subset=["depth"])
    cell = races.groupby(["track", "part"]).depth.agg(["median", "size"])
    track = races.groupby("track").depth.median()
    allm = float(races.depth.median()) if len(races) else np.nan
    key = pd.MultiIndex.from_arrays([d.track.astype(str), d.part.astype(str)])
    m = cell["median"].where(cell["size"] >= MIN_CELL).reindex(key).to_numpy(float)
    t = track.reindex(d.track.astype(str)).to_numpy(float)
    return pd.Series(np.where(np.isfinite(m), m, np.where(np.isfinite(t), t, allm)), index=d.index)


def bands(x: pd.Series, fit: pd.Series, race: pd.Series) -> tuple[pd.Series, list[float]]:
    """Three bands of ``x`` cut at the terciles of the fit part's races; a race with no ``x`` has no band."""
    r = pd.DataFrame({"x": x[fit], "race": race[fit]}).drop_duplicates("race").x.dropna()
    if len(r) < 3:
        return pd.Series(np.nan, index=x.index, dtype=object), []
    lo, hi = (float(v) for v in r.quantile([1 / 3, 2 / 3]))
    b = np.select([x <= lo, x <= hi], [BANDS[0], BANDS[1]], BANDS[2]).astype(object)
    return pd.Series(np.where(x.notna(), b, None), index=x.index, dtype=object), [round(lo, 2), round(hi, 2)]


def _nll(w: np.ndarray, lq: np.ndarray, lp: np.ndarray, won: np.ndarray, starts: np.ndarray) -> float:
    z = w[0] * lq + w[1] * lp
    mx = np.maximum.reduceat(z, starts)
    lse = np.log(np.add.reduceat(np.exp(z - np.repeat(mx, np.diff(np.r_[starts, len(z)]))), starts)) + mx
    return float(-(won * z).sum() + (np.add.reduceat(won, starts) * lse).sum())


def fit_weights(d: pd.DataFrame) -> dict:
    """The conditional logit's two weights (SP, ours) on ``d``'s races."""
    from scipy.optimize import minimize
    d = d.sort_values("race_id", kind="stable")
    race = d.race_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    lq, lp, won = np.log(d.q.to_numpy(float)), np.log(d.p.to_numpy(float)), d.won.to_numpy(float)
    r = minimize(_nll, np.array([1.0, 0.0]), args=(lq, lp, won, starts), method="BFGS")
    return {"market": round(float(r.x[0]), 4), "model": round(float(r.x[1]), 4), "races": int(len(starts))}


def blend(d: pd.DataFrame, w: dict) -> pd.Series:
    z = w["market"] * np.log(d.q.astype(float)) + w["model"] * np.log(d.p.astype(float))
    e = np.exp(z - z.groupby(d.race_id).transform("max"))
    return e / e.groupby(d.race_id).transform("sum")


def _ll(p, won) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    w = np.asarray(won, float)
    return -(w * np.log(p) + (1 - w) * np.log(1 - p))


def score(d: pd.DataFrame, pb: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """The blend's log-loss a runner against the SP's, paired by day (negative: the blend is the better chance)."""
    if not len(d):
        return {"runners": 0}
    day = pd.DataFrame({"d": d.day.to_numpy(), "b": _ll(pb, d.won), "m": _ll(d.q, d.won), "o": _ll(d.p, d.won)}) \
        .groupby("d").agg(b=("b", "sum"), m=("m", "sum"), o=("o", "sum"), n=("b", "size"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(day), (n_boot, len(day)))
    B, M, N = (day[c].to_numpy()[idx].sum(1) for c in ("b", "m", "n"))
    diff = (B - M) / N
    n = float(day.n.sum())
    out = {"runners": int(n), "races": int(d.race_id.nunique()), "days": int(len(day)),
           "logloss_sp": round(float(day.m.sum() / n), 5), "logloss_ours": round(float(day.o.sum() / n), 5),
           "logloss_blend": round(float(day.b.sum() / n), 5),
           "diff": round(float((day.b.sum() - day.m.sum()) / n), 5),
           "interval_95": [round(float(np.quantile(diff, 0.025)), 5), round(float(np.quantile(diff, 0.975)), 5)],
           "z": round(float((day.b.sum() - day.m.sum()) / n / diff.std()), 2) if diff.std() > 0 else None}
    out["beats_the_sp"] = bool(out["interval_95"][1] < 0)
    return out


def banded(d: pd.DataFrame, fit: pd.Series, band: pd.Series) -> dict:
    """Each band's weights from the fit part, its score on the rest, and the bands' scored runners pooled."""
    out, parts = {"bands": {}}, []
    for b in BANDS:
        a, s = d[fit & (band == b)], d[~fit & (band == b)]
        if a.race_id.nunique() < 200:
            out["bands"][b] = {"skipped": "too few races to fit", "fit_races": int(a.race_id.nunique())}
            continue
        w = fit_weights(a)
        out["bands"][b] = {"weights": w}
        if len(s):
            pb = blend(s, w)
            out["bands"][b]["scored"] = score(s, pb)
            parts.append((s, pb))
    if parts:
        out["pooled"] = score(pd.concat([s for s, _ in parts]), pd.concat([p for _, p in parts]))
    return out


def run(pred: pd.DataFrame, split: str, col: str = "p_model") -> dict:
    d = prepare(pred, col)
    fit = d.day < split
    res = {"split": split, "from": str(d.day.min()), "to": str(d.day.max()), "chance": col,
           "runners": int(len(d)), "races": int(d.race_id.nunique()),
           "races_with_the_money": int(d.drop_duplicates("race_id").depth.notna().sum())}
    a, s = d[fit], d[~fit]
    if a.race_id.nunique() < 200 or not len(s):
        return dict(res, skipped="too few races either side of the split")
    w = fit_weights(a)
    res["one_blend"] = {"weights": w, "scored": score(s, blend(s, w))}
    if d.depth.notna().any():
        race_depth = d.drop_duplicates("race_id").depth.dropna()
        res["money_before_the_off"] = {q: round(float(race_depth.quantile(q)), 1) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        real, cuts = bands(d.depth, fit, d.race_id)
        res["realised"] = dict(banded(d, fit, real), cuts=cuts)
        exp = expected_depth(d, fit)
        eb, cuts = bands(exp, fit, d.race_id)
        res["expected"] = dict(banded(d, fit, eb), cuts=cuts)
        # how well the expectation ranks the money that came (scored part)
        r = d[~fit].drop_duplicates("race_id")
        ok = r.depth.notna() & exp[r.index].notna()
        if ok.sum() > 50:
            res["expected"]["rank_correlation_with_the_money"] = round(
                float(pd.Series(exp[r.index][ok]).rank().corr(r.depth[ok].rank())), 3)
    # the pre-off average price as the market, where the files have it for the whole field
    if "ppwap" in d:
        whole = (d.ppwap > 1).groupby(d.race_id).transform("all")
        e = d[whole & ~fit]
        if len(e):
            pw = 1.0 / e.ppwap.astype(float)
            pw = pw / pw.groupby(e.race_id).transform("sum")
            res["pre_off_average_price"] = {"runners": int(len(e)), "logloss": round(float(_ll(pw, e.won).mean()), 5),
                                            "logloss_sp_same_runners": round(float(_ll(e.q, e.won).mean()), 5)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="2026-09-01")
    ap.add_argument("--memory", action="store_true",
                    help="our chance with the residual memory (research/greyhound/residual_memory.py), fitted before the split")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    pred = pd.read_parquet(a.pred)
    col = "p_model"
    if a.memory:
        from research.greyhound import residual_memory as rm
        p = rm.misses(pred)
        prm = rm.fit(p[p.race_date.astype(str) < a.split])
        pred, col = rm.apply(p, prm), "p_model_m"
    res = run(pred, a.split, col)
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
