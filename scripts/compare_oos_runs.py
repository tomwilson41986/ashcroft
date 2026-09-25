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

`--folds` cuts both runs to the same subset of walk-forward folds. The folds are
chronological, so `--folds 0-4` against `--folds 5-10` asks whether a difference
measured over the whole window is there in each half of it, or only in one. That
is a different question from the rule above -- half a sample has a wider interval
by construction, so what a split answers is whether the sign is stable, not
whether it is significant a second time.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = ["race_date", "race_time", "track", "horse_name"]


def parse_folds(spec: str | None) -> set[int] | None:
    """`0-4`, `5-10`, `3`, `0,2,5-7` -> the fold indices named. None for all."""
    if spec is None:
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(part))
        except ValueError:
            raise SystemExit(f"--folds {spec!r}: cannot read {part!r} as a fold "
                             f"or a range of folds")
    if not out:
        raise SystemExit(f"--folds {spec!r} names no folds")
    return out


def load(path: str, folds: set[int] | None = None) -> pd.DataFrame:
    """One run's predictions, optionally cut to a subset of its folds.

    The filter belongs here because every statistic downstream -- `skill`,
    `concordance`, `top_pick_roi`, `book` and, through the merged frame,
    `paired_skill_deltas` -- is computed from the frames this returns. Filtering
    once at the source is what keeps a half-sample report from mixing a
    half-sample error with a whole-sample concordance."""
    d = pd.read_csv(path)
    d["race_date"] = pd.to_datetime(d["race_date"], errors="coerce")
    if folds is not None:
        if "fold_idx" not in d.columns:
            raise SystemExit(
                f"--folds was given but {path} carries no fold_idx column, so "
                f"there is nothing to select on. Runs from evaluate_oos.py "
                f"record it; an older CSV may not.")
        d = d[pd.to_numeric(d["fold_idx"], errors="coerce").isin(folds)]
    if "raceid" not in d.columns:
        d["raceid"] = (d["race_date"].dt.strftime("%Y-%m-%d") + "_"
                       + d["track"].astype(str) + "_" + d["race_time"].astype(str))
    d = d[(pd.to_numeric(d["bfsp"], errors="coerce") > 1.0)
          & (pd.to_numeric(d["predicted_bfsp"], errors="coerce") > 1.0)]
    if d.empty:
        where = f" in folds {sorted(folds)}" if folds is not None else ""
        raise SystemExit(f"{path} has no usable rows{where}")
    return d


def fold_spans(d: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Each fold's first and last race date, as plain strings."""
    if "fold_idx" not in d.columns:
        return {}
    t = d[["fold_idx", "race_date"]].copy()
    t["fold_idx"] = pd.to_numeric(t["fold_idx"], errors="coerce")
    t = t.dropna(subset=["fold_idx", "race_date"])
    if t.empty:
        return {}
    g = t.groupby("fold_idx")["race_date"].agg(["min", "max"])
    return {int(k): (r["min"].strftime("%Y-%m-%d"), r["max"].strftime("%Y-%m-%d"))
            for k, r in g.iterrows()}


def assert_comparable_folds(base: pd.DataFrame, variant: pd.DataFrame) -> None:
    """Refuse a fold-subset comparison when fold k is not the same period in both.

    Selecting by fold index only means something if the index means the same
    thing in both runs. It does when they share a cached matrix and an
    `--eval-from`, and that is an assumption to check rather than trust:
    comparing fold 3 of an 11-fold run against fold 3 of a 22-fold one would
    produce a confident, well-formed number about two different periods.

    Only called when `--folds` is given. Without it the comparison is paired on
    the runner key and never reads the fold index, so differing geometry costs
    nothing there."""
    gb, gv = fold_spans(base), fold_spans(variant)
    if not gb or not gv:
        return
    bad = sorted(k for k in set(gb) | set(gv) if gb.get(k) != gv.get(k))
    if bad:
        detail = "; ".join(f"fold {k}: base {gb.get(k, '(absent)')} vs "
                           f"variant {gv.get(k, '(absent)')}" for k in bad[:5])
        raise SystemExit(
            f"the two runs' folds cover different dates, so selecting by fold "
            f"index would compare different periods -- {detail}"
            + (f" (and {len(bad) - 5} more)" if len(bad) > 5 else ""))


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
    """Share of within-race winner/loser pairs the price orders correctly.

    NOT the same statistic as ``model.diagnostics.concordance_index``, which
    orders every runner by finishing position. This one asks only whether the
    winner was priced shorter than each beaten horse, which is an easier
    question, so it reads higher: on run 10 this is 0.721 against the market's
    0.760, while the placing-based index is 0.657 against 0.682. Both say the
    market orders races better; quoting one where the other is expected makes
    two honest numbers look like a contradiction."""
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


def _race_terms(d: pd.DataFrame, price_col: str, race_col: str = "raceid"):
    """Per-race sums that Brier skill and concordance are built from.

    Both statistics are ratios of sums over races, so bootstrapping them means
    resampling races and re-summing -- not recomputing from scratch. Doing the
    per-race work once turns 500 bootstrap draws from minutes into milliseconds,
    which is what makes it affordable to put an interval on the two numbers the
    decision rule actually turns on."""
    y = pd.to_numeric(d["won"], errors="coerce").fillna(0).to_numpy(dtype=float)
    q = np.clip(normalised_prob(d, price_col, race_col), 1e-9, 1 - 1e-9)
    mk = np.clip(normalised_prob(d, "bfsp", race_col), 1e-9, 1 - 1e-9)
    races = d[race_col].to_numpy()

    se_model = pd.Series((q - y) ** 2).groupby(races).sum()
    se_market = pd.Series((mk - y) ** 2).groupby(races).sum()
    n = pd.Series(np.ones(len(d))).groupby(races).sum()

    ok, tot = {}, {}
    for rid, g in d.groupby(race_col):
        w = g[pd.to_numeric(g["won"], errors="coerce") > 0][price_col].to_numpy()
        l = g[~(pd.to_numeric(g["won"], errors="coerce") > 0)][price_col].to_numpy()
        if w.size == 0 or l.size == 0:
            ok[rid], tot[rid] = 0.0, 0.0
            continue
        cmp_ = w[:, None] < l[None, :]
        tie = w[:, None] == l[None, :]
        ok[rid] = float(cmp_.sum()) + 0.5 * float(tie.sum())
        tot[rid] = float(w.size * l.size)
    idx = se_model.index
    return {
        "idx": idx,
        "se_model": se_model.to_numpy(), "se_market": se_market.to_numpy(),
        "n": n.reindex(idx).to_numpy(),
        "ok": np.array([ok[i] for i in idx]), "tot": np.array([tot[i] for i in idx]),
    }


def paired_skill_deltas(base: pd.DataFrame, variant: pd.DataFrame, races: np.ndarray,
                        n_boot: int = 500, level: float = 0.9, seed: int = 0) -> dict:
    """Paired CIs for the two quantities the decision rule gates on.

    A variant only replaces the default if it does not lose Brier skill or
    concordance by more than its own interval, so those two need intervals.
    Reporting the point estimates alone would leave the rule unusable -- and
    "0.001 worse" is meaningless without knowing whether 0.001 is noise."""
    keep = pd.Index(np.unique(races))
    b = _race_terms(base[base["raceid"].isin(keep)], "predicted_bfsp")
    v = _race_terms(variant[variant["raceid"].isin(keep)], "predicted_bfsp")
    common = b["idx"].intersection(v["idx"])
    sel_b = b["idx"].get_indexer(common)
    sel_v = v["idx"].get_indexer(common)

    def brier_skill(t, sel, w):
        num = float((t["se_model"][sel] * w).sum() / (t["n"][sel] * w).sum())
        den = float((t["se_market"][sel] * w).sum() / (t["n"][sel] * w).sum())
        return 1.0 - num / den

    def conc(t, sel, w):
        tot = float((t["tot"][sel] * w).sum())
        return float((t["ok"][sel] * w).sum()) / tot if tot else np.nan

    ones = np.ones(len(common))
    point = {"brier_skill": brier_skill(v, sel_v, ones) - brier_skill(b, sel_b, ones),
             "concordance": conc(v, sel_v, ones) - conc(b, sel_b, ones)}

    rng = np.random.default_rng(seed)
    R = len(common)
    draws = rng.multinomial(R, np.full(R, 1.0 / R), size=n_boot).astype(float)
    ds, dc = [], []
    for w in draws:
        ds.append(brier_skill(v, sel_v, w) - brier_skill(b, sel_b, w))
        dc.append(conc(v, sel_v, w) - conc(b, sel_b, w))
    a = (1 - level) / 2
    out = dict(point)
    out["brier_skill_ci"] = (float(np.quantile(ds, a)), float(np.quantile(ds, 1 - a)))
    out["concordance_ci"] = (float(np.quantile(dc, a)), float(np.quantile(dc, 1 - a)))
    return out


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


def _label(path: str, other: str) -> str:
    """Name a run by its file, falling back to the directory when both files
    are called the same thing -- two artifacts unzipped side by side both hold
    `data/oos_predictions.csv`, and a report titled "x against x" names
    neither."""
    name, on = os.path.basename(path), os.path.basename(other)
    if name != on:
        return name
    parent = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    return f"{parent}/{name}" if parent else name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--commission", type=float, default=0.05)
    ap.add_argument("--note", default=None,
                    help="a line of context to put under the title, e.g. what the two runs are")
    ap.add_argument("--json-out", default=None,
                    help="also write the decision's numbers as JSON (the research loop's table)")
    ap.add_argument("--folds", default=None,
                    help="restrict both runs to these walk-forward folds, e.g. 0-4, "
                         "5-10, 3, or 0,2,5-7. Use it to ask whether a difference "
                         "measured over the whole window holds in each half of it")
    a = ap.parse_args()

    folds = parse_folds(a.folds)
    b, v = load(a.base, folds), load(a.variant, folds)
    if folds is not None:
        assert_comparable_folds(b, v)
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
    L.append(f"# {_label(a.variant, a.base)} against {_label(a.base, a.variant)}\n")
    if a.note:
        L.append(f"*{a.note}*\n")
    # The scope belongs in the header, not in whoever-reads-this's memory: a
    # report over half the window must not be readable as one over all of it.
    scope = f"**Folds {a.folds}.** " if a.folds else ""
    L.append(f"{scope}{len(m):,} paired runners over {m['raceid'].nunique():,} races, "
             f"{m['race_date'].min():%Y-%m-%d} to {m['race_date'].max():%Y-%m-%d}.\n")
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
    sk = paired_skill_deltas(b, v, m["raceid"].to_numpy(), a.n_boot)
    L.append("\n## Against the market\n")
    L.append(f"| | base | variant | variant − base (90% CI) |\n|---|---|---|---|")
    L.append(f"| Brier skill vs market | {sb['brier_skill_vs_market']:+.4f} | "
             f"{sv['brier_skill_vs_market']:+.4f} | {sk['brier_skill']:+.5f} "
             f"({sk['brier_skill_ci'][0]:+.5f} to {sk['brier_skill_ci'][1]:+.5f}) |")
    L.append(f"| log loss | {sb['log_loss']:.5f} | {sv['log_loss']:.5f} | |")
    L.append(f"| winner-vs-loser concordance | {cb:.5f} | {cv:.5f} | "
             f"{sk['concordance']:+.5f} "
             f"({sk['concordance_ci'][0]:+.5f} to {sk['concordance_ci'][1]:+.5f}) |")
    L.append(f"| ...the market's | {cm:.5f} | {cm:.5f} | |")
    L.append(f"| implied book (1.0 by construction since Step A) | "
             f"{book(b, 'predicted_bfsp'):.4f} | {book(v, 'predicted_bfsp'):.4f} |")
    L.append("\n## Top pick, flat stakes\n")
    L.append(f"- base    {rb:+.2f}% ({rblo:+.2f} to {rbhi:+.2f}) on {nb:,} bets")
    L.append(f"- variant {rv:+.2f}% ({rvlo:+.2f} to {rvhi:+.2f}) on {nv:,} bets")
    L.append("\nReported, not decisive: at this sample the interval is wider than "
             "any difference worth acting on.\n")

    # The rule, applied here rather than left to the reader.
    primary = hi < 0
    brier_ok = sk["brier_skill"] >= 0 or sk["brier_skill"] >= sk["brier_skill_ci"][0]
    conc_ok = sk["concordance"] >= 0 or sk["concordance"] >= sk["concordance_ci"][0]
    replaces = primary and brier_ok and conc_ok
    L.append("\n## Decision\n")
    L.append(f"- paired error interval excludes zero in the variant's favour: "
             f"**{'yes' if primary else 'no'}**")
    L.append(f"- Brier skill not worse by more than its own interval: "
             f"**{'yes' if brier_ok else 'no'}**")
    L.append(f"- concordance not worse by more than its own interval: "
             f"**{'yes' if conc_ok else 'no'}**")
    L.append(f"\n**{'The variant replaces the default.' if replaces else 'The default stands.'}**\n")

    text = "\n".join(L)
    print(text)
    if a.json_out:
        import json
        r1 = by_rank(m)
        r1 = r1[r1["rank"] == "1"] if "rank" in r1.columns else r1.iloc[0:0]
        summary = {
            "n": int(len(m)), "races": int(m["raceid"].nunique()),
            "err_base": float(m["err_b"].mean()), "err_variant": float(m["err_v"].mean()),
            "delta": delta, "delta_ci": [float(lo), float(hi)],
            "rank1_delta": float(r1["delta"].iloc[0]) if len(r1) else None,
            "brier_skill": float(sk["brier_skill"]), "brier_skill_ci": [float(x) for x in sk["brier_skill_ci"]],
            "concordance": float(sk["concordance"]), "concordance_ci": [float(x) for x in sk["concordance_ci"]],
            "top_pick_roi": [float(rb), float(rv)], "replaces": bool(replaces),
        }
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(summary, f, indent=2)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
