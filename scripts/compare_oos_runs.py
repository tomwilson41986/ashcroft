#!/usr/bin/env python3
"""Compare two walk-forward runs on the rows they share, pairing by runner.

Two evaluation runs over the same folds predict the same horses in the same
races, so the honest comparison is paired: for each runner, how much did the
error change. An unpaired difference of two means buries a real effect inside
the spread of the race prices themselves -- the field runs from 1.5 to 400, and
the variation between variants is a few percent of a log point.

The bootstrap resamples whole races rather than runners, because runners in a
race are not independent: normalising the book makes one runner's price a
function of every other's.

    python scripts/compare_oos_runs.py \\
        --base data/oos_predictions_v0_l2.csv \\
        --variant data/oos_predictions_v2_profit.csv \\
        --out reports/h2h_v2_profit.md

The decision rule this exists to serve: a variant replaces the default only if
its paired interval excludes zero in its favour AND neither Brier skill against
the market nor concordance is worse by more than its own interval. Rank-1 ROI
is reported and never decisive -- at twelve thousand races its interval is
about two points wide, which is larger than any difference worth having.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = ["race_date", "race_time", "track", "horse_name"]


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["race_date"] = pd.to_datetime(d["race_date"], errors="coerce")
    if "raceid" not in d.columns:
        d["raceid"] = (d["race_date"].dt.strftime("%Y-%m-%d") + "_"
                       + d["track"].astype(str) + "_" + d["race_time"].astype(str))
    d = d[(pd.to_numeric(d["bfsp"], errors="coerce") > 1.0)
          & (pd.to_numeric(d["predicted_bfsp"], errors="coerce") > 1.0)]
    return d


def paired(base: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    """Rows both runs predicted, with each one's error side by side."""
    cols = KEY + ["raceid", "bfsp", "predicted_bfsp", "won"]
    b = base[[c for c in cols if c in base.columns]].copy()
    v = variant[[c for c in cols if c in variant.columns]].copy()
    m = b.merge(v, on=KEY, suffixes=("_b", "_v"))
    if m.empty:
        raise SystemExit("the two runs share no rows; are they the same folds?")
    m["raceid"] = m["raceid_b"]
    m["err_b"] = np.abs(np.log(m["predicted_bfsp_b"] / m["bfsp_b"]))
    m["err_v"] = np.abs(np.log(m["predicted_bfsp_v"] / m["bfsp_v"]))
    m["delta"] = m["err_v"] - m["err_b"]        # negative = the variant is better
    return m


def race_bootstrap(values: np.ndarray, races: np.ndarray, n_boot: int = 500,
                   level: float = 0.9, seed: int = 0) -> tuple[float, float]:
    """CI for a mean, resampling whole races with replacement."""
    df = pd.DataFrame({"v": values, "r": races})
    g = df.groupby("r")["v"].agg(["sum", "size"])
    if len(g) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    sums, sizes = g["sum"].to_numpy(), g["size"].to_numpy()
    draws = rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)), size=n_boot)
    tot, n = draws @ sums, draws @ sizes
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(n > 0, tot / n, np.nan)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return float(np.nanquantile(means, lo)), float(np.nanquantile(means, hi))


def normalised_prob(d: pd.DataFrame, price_col: str, race_col: str = "raceid") -> np.ndarray:
    ip = 1.0 / pd.to_numeric(d[price_col], errors="coerce").to_numpy(dtype=float)
    book = pd.Series(ip).groupby(d[race_col].to_numpy()).transform("sum").to_numpy()
    return np.where(book > 0, ip / book, np.nan)


def skill(d: pd.DataFrame, price_col: str) -> dict:
    """Brier and log loss of the run's normalised probabilities, against the market's."""
    y = pd.to_numeric(d["won"], errors="coerce").fillna(0).to_numpy(dtype=float)
    q = np.clip(normalised_prob(d, price_col), 1e-9, 1 - 1e-9)
    m = np.clip(normalised_prob(d, "bfsp"), 1e-9, 1 - 1e-9)
    brier, brier_m = float(np.mean((q - y) ** 2)), float(np.mean((m - y) ** 2))
    return {
        "brier": brier,
        "brier_skill_vs_market": 1.0 - brier / brier_m,
        "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
    }


def concordance(d: pd.DataFrame, price_col: str, race_col: str = "raceid") -> float:
    """Share of within-race winner/loser pairs the price orders correctly."""
    ok = tot = 0
    for _, g in d.groupby(race_col):
        w = g[pd.to_numeric(g["won"], errors="coerce") > 0]
        l = g[~(pd.to_numeric(g["won"], errors="coerce") > 0)]
        if w.empty or l.empty:
            continue
        wp = w[price_col].to_numpy()[:, None]
        lp = l[price_col].to_numpy()[None, :]
        ok += int((wp < lp).sum()) + 0.5 * int((wp == lp).sum())
        tot += wp.size * lp.size
    return ok / tot if tot else float("nan")


def top_pick_roi(d: pd.DataFrame, price_col: str, commission: float = 0.05,
                 race_col: str = "raceid") -> tuple[float, float, float, int]:
    rank = d.groupby(race_col)[price_col].rank(method="first")
    top = d[rank == 1]
    y = pd.to_numeric(top["won"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ret = np.where(y > 0, (top["bfsp"].to_numpy() - 1) * (1 - commission), -1.0)
    lo, hi = race_bootstrap(ret, top[race_col].to_numpy())
    return float(ret.mean() * 100), lo * 100, hi * 100, len(top)


def by_rank(m: pd.DataFrame, ranks=(1, 2, 3, 8)) -> pd.DataFrame:
    """Paired error change at the ranks people actually look at."""
    m = m.copy()
    m["rank_b"] = m.groupby("raceid")["predicted_bfsp_b"].rank(method="first")
    rows = []
    for r in ranks:
        sel = m[m["rank_b"] >= r] if r == max(ranks) else m[m["rank_b"] == r]
        if sel.empty:
            continue
        lo, hi = race_bootstrap(sel["delta"].to_numpy(), sel["raceid"].to_numpy())
        rows.append({"rank": f"{r}+" if r == max(ranks) else str(r), "n": len(sel),
                     "err_base": sel["err_b"].mean(), "err_variant": sel["err_v"].mean(),
                     "delta": sel["delta"].mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def book(d: pd.DataFrame, price_col: str, race_col: str = "raceid") -> float:
    return float(d.groupby(race_col)[price_col].apply(lambda s: (1.0 / s).sum()).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--commission", type=float, default=0.05)
    a = ap.parse_args()

    b, v = load(a.base), load(a.variant)
    m = paired(b, v)

    lo, hi = race_bootstrap(m["delta"].to_numpy(), m["raceid"].to_numpy(), a.n_boot)
    delta = float(m["delta"].mean())
    verdict = ("the variant wins" if hi < 0 else
               "the base wins" if lo > 0 else
               "no difference the data can resolve")

    sb, sv = skill(b, "predicted_bfsp"), skill(v, "predicted_bfsp")
    cb, cv = concordance(b, "predicted_bfsp"), concordance(v, "predicted_bfsp")
    cm = concordance(b, "bfsp")
    rb, rblo, rbhi, nb = top_pick_roi(b, "predicted_bfsp", a.commission)
    rv, rvlo, rvhi, nv = top_pick_roi(v, "predicted_bfsp", a.commission)

    L = []
    L.append(f"# {os.path.basename(a.variant)} against {os.path.basename(a.base)}\n")
    L.append(f"{len(m):,} paired runners over {m['raceid'].nunique():,} races.\n")
    L.append("## Mean absolute log error, paired\n")
    L.append(f"- base **{m['err_b'].mean():.4f}**, variant **{m['err_v'].mean():.4f}**")
    L.append(f"- paired difference **{delta:+.4f}** (90% CI {lo:+.4f} to {hi:+.4f}) "
             f"— negative favours the variant")
    L.append(f"- **{verdict}**\n")
    L.append("## By the base model's rank in the race\n")
    # to_string, not to_markdown: the latter needs `tabulate`, which the CI
    # image does not carry, and a comparison script that cannot run in CI is
    # not much use.
    L.append("```")
    L.append(by_rank(m).round(4).to_string(index=False))
    L.append("```")
    L.append("\n## Against the market\n")
    L.append(f"| | base | variant |\n|---|---|---|")
    L.append(f"| Brier skill vs market | {sb['brier_skill_vs_market']:+.4f} | {sv['brier_skill_vs_market']:+.4f} |")
    L.append(f"| log loss | {sb['log_loss']:.5f} | {sv['log_loss']:.5f} |")
    L.append(f"| concordance | {cb:.5f} | {cv:.5f} |")
    L.append(f"| market concordance | {cm:.5f} | {cm:.5f} |")
    L.append(f"| implied book | {book(b, 'predicted_bfsp'):.4f} | {book(v, 'predicted_bfsp'):.4f} |")
    L.append("\n## Top pick, flat stakes\n")
    L.append(f"- base    {rb:+.2f}% ({rblo:+.2f} to {rbhi:+.2f}) on {nb:,} bets")
    L.append(f"- variant {rv:+.2f}% ({rvlo:+.2f} to {rvhi:+.2f}) on {nv:,} bets")
    L.append("\nReported, not decisive: at this sample the interval is wider than "
             "any difference worth acting on.\n")

    text = "\n".join(L)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
