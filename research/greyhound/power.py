"""Power and futility for the greyhound gates (the owner, 20 Nov 2026: the discovery round, weakness 5; ideas
uncert-09 and uncert-06 in reports/greyhound_discovery_1120.md). A planning table, 0 trials: it labels a read
under-powered; it never cancels or re-times a registered read.

The deflated Sharpe gate (research/overfit.py ``dsr_check``) passes a rule whose Sharpe beats the best Sharpe
``n_trials`` rules with no edge would reach by luck, with 95% probability. Under the null spread of a Sharpe over n
periods (1 / (n - 1)) and near-normal returns that is a bar on the rule's t-statistic:

    t_required = E[max of n_trials standard normals] + 1.645

(``required_t``; ``passes`` is the exact condition for normal returns and reproduces ``dsr_check``). From a rule's
observed t over its days so far, the days it needs at the same effect, at half the effect, and the chance of a pass
by a date follow. A t read off a by-day interval already carries the day clustering; ``cluster_t`` gives it from
per-race returns (the design effect beside it: how much the races of a day move together).

    python -m research.greyhound.power --inputs research/greyhound/power_inputs_1120.json --out out/power
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

Z90 = 1.6448536269514722          # a two-sided 90% interval is the mean +/- Z90 standard errors


def required_t(n_trials: int, threshold: float = 0.95) -> float:
    """The t-statistic a rule needs to pass the deflated Sharpe at ``n_trials`` (large-sample, normal returns)."""
    from model.uncertainty import expected_max_sharpe
    return float(expected_max_sharpe(int(n_trials), 1.0) + stats.norm.ppf(threshold))


def passes(sharpe: float, n_obs: int, n_trials: int, threshold: float = 0.95) -> bool:
    """dsr_check's condition for normal returns (skew 0, kurtosis 3) and the null spread of the trials' Sharpes."""
    from model.uncertainty import expected_max_sharpe
    from research.overfit import MIN_OBS, null_sr_variance
    if n_obs < MIN_OBS:
        return False
    sr_star = expected_max_sharpe(int(n_trials), null_sr_variance(n_obs))
    z = (sharpe - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(1.0 + 0.5 * sharpe ** 2)
    return bool(stats.norm.cdf(z) >= threshold)


def t_from_interval(mean: float, lo: float, hi: float) -> float | None:
    """The t behind a mean and its two-sided 90% interval (the labs' by-day bootstrap)."""
    se = (hi - lo) / (2 * Z90)
    return float(mean / se) if se > 0 else None


def cluster_t(returns, days) -> dict:
    """Mean, its standard error with the days as clusters (the mean of the day sums over the mean day count: a ratio
    of means), the t, and the design effect against treating every return as independent."""
    r, d = np.asarray(returns, float), np.asarray(days)
    ok = np.isfinite(r)
    r, d = r[ok], d[ok]
    u, inv = np.unique(d, return_inverse=True)
    g = len(u)
    if g < 2 or len(r) < 2:
        return {"n": int(len(r)), "days": int(g), "mean": float(r.mean()) if len(r) else None, "se": None, "t": None,
                "design_effect": None}                    # one day: a mean, and no interval
    s, c = np.bincount(inv, r), np.bincount(inv).astype(float)
    mean = s.sum() / c.sum()
    resid = s - mean * c
    se = float(np.sqrt(g / (g - 1) * (resid ** 2).sum()) / c.sum())
    se_iid = float(r.std(ddof=1) / np.sqrt(len(r)))
    return {"n": int(len(r)), "days": int(g), "mean": float(mean), "se": se,
            "t": float(mean / se) if se > 0 else None,
            "design_effect": float((se / se_iid) ** 2) if se_iid > 0 else None}


def days_needed(t_now: float, days_now: int, n_trials: int, power: float = 0.8, effect: float = 1.0,
                min_days: int = 0) -> float | None:
    """Days for ``power`` of clearing ``required_t`` when the effect a day stays at ``effect`` x the one observed
    (t grows with the square root of the days). ``None`` when the observed effect is zero or negative."""
    if t_now is None or t_now <= 0 or days_now <= 0:
        return None
    per_day = effect * t_now / np.sqrt(days_now)
    need = ((required_t(n_trials) + stats.norm.ppf(power)) / per_day) ** 2
    return float(max(need, min_days))


def chance_by(t_now: float, days_now: int, days_at_read: int, n_trials: int, effect: float = 1.0) -> float | None:
    """The chance the t at ``days_at_read`` clears ``required_t`` if the effect a day is ``effect`` x the observed."""
    if t_now is None or days_now <= 0 or days_at_read <= 0:
        return None
    expect = effect * t_now / np.sqrt(days_now) * np.sqrt(days_at_read)
    return float(stats.norm.sf(required_t(n_trials) - expect))


def table(rows: list[dict], n_trials: int, counts=(1, 100, 436)) -> dict:
    """One line a candidate: its t, the bar, the days it needs, its chance at its read."""
    from research.overfit import MIN_OBS
    bar = required_t(n_trials)
    out = {"n_trials": int(n_trials), "t_required": round(bar, 3),
           "t_required_at_other_counts": {str(k): round(required_t(k), 3) for k in counts}, "rows": []}
    for r in rows:
        if r.get("daily"):                                   # one figure a day: the mean, its t and interval from them
            v = np.asarray(r["daily"], float)
            se = v.std(ddof=1) / np.sqrt(len(v))
            r = {**r, "mean": round(float(v.mean()), 3), "days": len(v),
                 "lo": round(float(v.mean() - Z90 * se), 3), "hi": round(float(v.mean() + Z90 * se), 3)}
        t = t_from_interval(r["mean"], r["lo"], r["hi"]) if r.get("lo") is not None else r.get("t")
        days = int(r["days"])
        md = MIN_OBS if r.get("period", "day") == "day" else 0
        row = {"name": r["name"], "source": r.get("source"), "period": r.get("period", "day"), "days": days,
               "trades": r.get("trades"), "mean_%": r["mean"], "ci90": [r.get("lo"), r.get("hi")],
               "t": None if t is None else round(t, 2), "fill": r.get("fill"),
               "days_for_80%_power": None, "days_at_half_effect": None, "chance_at_read": None,
               "read_days": r.get("read_days")}
        if t is not None:
            a, b = days_needed(t, days, n_trials, min_days=md), days_needed(t, days, n_trials, effect=0.5, min_days=md)
            row["days_for_80%_power"] = None if a is None else int(np.ceil(a))
            row["days_at_half_effect"] = None if b is None else int(np.ceil(b))
            if r.get("read_days"):
                c = chance_by(t, days, int(r["read_days"]), n_trials)
                short = r.get("period", "day") == "day" and int(r["read_days"]) < MIN_OBS
                row["chance_at_read"] = 0.0 if short else (None if c is None else round(c, 3))
                row["under_powered"] = bool(short or (c is not None and c < 0.5))
        out["rows"].append(row)
    return out


def markdown(res: dict) -> str:
    md = [f"Bar at {res['n_trials']:,} rules counted: t of at least {res['t_required']}. At other counts: "
          + ", ".join(f"{k}: {v}" for k, v in res["t_required_at_other_counts"].items()) + ".", "",
          "| Candidate | Fill | Days | Trades | Mean % | 90% | t | Days for 80% power | At half the effect | Days at its "
          "read | Chance of a pass then | Source |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "no data" if v is None else str(v)
    for r in res["rows"]:
        ci = "no data" if r["ci90"][0] is None else f"{r['ci90'][0]} to {r['ci90'][1]}"
        md.append(f"| {r['name']} | {f(r['fill'])} | {r['days']} | {f(r['trades'])} | {r['mean_%']} | {ci} | {f(r['t'])} | "
                  f"{f(r['days_for_80%_power'])} | {f(r['days_at_half_effect'])} | {f(r['read_days'])} | "
                  f"{f(r['chance_at_read'])} | {f(r['source'])} |")
    return "\n".join(md)


def main(argv=None) -> int:
    from research.greyhound.trial_count import greyhound_trials
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", default="out/power")
    a = ap.parse_args(argv)
    res = table(json.loads(Path(a.inputs).read_text())["rows"], greyhound_trials())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "power.json").write_text(json.dumps(res, indent=1))
    (out / "power.md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
