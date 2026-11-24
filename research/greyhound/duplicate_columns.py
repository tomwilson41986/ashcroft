"""Duplicate columns in a greyhound feature matrix (reports/greyhound_feature_review.md, "lastrace": several exact
duplicates of engine columns, e.g. lr_ae_last = gp_ae_l1). Lists every pair of columns that are exactly the same
(same missing rows, equal values at float32) or nearly so (|Pearson r| over their common rows above ``--threshold``,
0.999), with a suggested column to drop, as JSON, for an experiment spec's ``"drop"`` list.

The matrix is the fast loop's cache (greyhound/fastloop.py ``build_cache``: ``greyhound/feature_cache/<hash>/
features.parquet``, every engine block's columns), a local copy of it, and/or point-in-time tables
(``greyhound/pit2/<table>.parquet``) joined on race and dog.

    python -m research.greyhound.duplicate_columns --matrix features.parquet --tables lastrace.parquet --out dups.json
    python -m research.greyhound.duplicate_columns --cache --attach lastrace,optics --out dups.json   # runner (S3)

The fast loop's ``drop`` entries are PREFIXES (``f.startswith(pre)``): a suggested name that starts another column's
name would drop that one too; such names are listed in ``prefix_clash``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["race_id", "dog_id"]
THRESHOLD = 0.999
SAMPLE = 200_000
MIN_OVERLAP = 100
SCREEN = 50_000               # rows the all-pairs screen reads
SLACK = 0.002


def _table_prefixes() -> tuple[str, ...]:
    """The point-in-time tables' prefixes (a table's column is the one suggested to drop against an engine column)."""
    try:
        import importlib
        from greyhound import pit
        return tuple(importlib.import_module(f"greyhound.pit.{n}").PREFIX for n in pit.TABLES)
    except Exception:
        return ()


def _f32(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64).astype(np.float32)


def _suggest(a: str, b: str, cov: dict, tables: tuple[str, ...]) -> str:
    """The one to drop: a table's column against an engine column; else the less covered; else the later name."""
    ta, tb = a.startswith(tables), b.startswith(tables)
    if ta != tb:
        return a if ta else b
    if cov[a] != cov[b]:
        return a if cov[a] < cov[b] else b
    return max(a, b)


def find_duplicates(df: pd.DataFrame, threshold: float = THRESHOLD, sample: int = SAMPLE, seed: int = 0,
                    exclude: tuple[str, ...] = ()) -> dict:
    """The exact and near-duplicate pairs among ``df``'s numeric columns (keys and ``exclude`` left out)."""
    cols = [c for c in df.columns if c not in KEY and c not in exclude
            and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    n = len(df)
    cov = {c: round(float(df[c].notna().mean()), 4) if n else 0.0 for c in cols}
    tables = _table_prefixes()

    # exact: the same missing rows and the same values at float32, on every row (a hash a column, then compared)
    groups: dict = {}
    for c in cols:
        v = _f32(df[c])
        nan = np.isnan(v)
        h = hashlib.sha1(np.packbits(nan).tobytes() + np.where(nan, 0, v).astype(np.float32).tobytes()).hexdigest()
        groups.setdefault(h, []).append(c)
    exact = []
    for g in groups.values():
        for i in range(1, len(g)):
            exact.append((g[0], g[i]))

    # near: on a sample, the correlation of the mean-filled standardised columns screens; each candidate is then
    # read on its common rows
    rows = np.random.default_rng(seed).choice(n, size=min(sample, n), replace=False) if n > sample else np.arange(n)
    S = np.column_stack([_f32(df[c].iloc[rows]) for c in cols]) if cols else np.zeros((len(rows), 0), np.float32)
    ok = np.isfinite(S)
    cnt = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, S, 0).sum(axis=0) / np.maximum(cnt, 1)
        Z = np.where(ok, S - mean.astype(np.float32), np.float32(0.0))       # centred, float32 (memory)
        sd = np.sqrt((Z.astype(np.float64) ** 2).sum(axis=0) / np.maximum(cnt, 1))
    del S
    live = (sd > 0) & (cnt >= MIN_OVERLAP)
    constant = [c for c, l, k in zip(cols, live, cnt) if not l and k > 0]
    idx = np.flatnonzero(live)
    # the screen: every pair's correlation over its common rows, on the first ``SCREEN`` sampled rows (matrix sums
    # over the missing masks), loose by ``SLACK`` for float32 rounding; each candidate is read again on the sample
    m = min(SCREEN, len(rows))
    Zs = (Z[:m, idx] / sd[idx]).astype(np.float32)
    M = ok[:m, idx].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = M.T @ M
        sa = (Zs.T @ M) / nn                                        # [a, b]: the mean of a where both are present
        saa = ((Zs ** 2).T @ M) / nn
        sab = (Zs.T @ Zs) / nn
        R = (sab - sa * sa.T) / np.sqrt((saa - sa ** 2) * (saa - sa ** 2).T)
    R = np.where((nn >= MIN_OVERLAP) & np.isfinite(R), R, 0.0) if len(idx) else np.zeros((0, 0))
    ii, jj = np.nonzero(np.triu(np.abs(R) > threshold - SLACK, 1))
    pairs = {tuple(sorted(p)) for p in exact}
    near = []
    for a_i, b_i in zip(idx[ii], idx[jj]):
        a, b = cols[a_i], cols[b_i]
        both = ok[:, a_i] & ok[:, b_i]
        k = int(both.sum())
        if k < MIN_OVERLAP:
            continue
        x, y = Z[both, a_i].astype(float), Z[both, b_i].astype(float)
        if x.std() == 0 or y.std() == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > threshold and tuple(sorted((a, b))) not in pairs:
            near.append({"a": a, "b": b, "corr": round(r, 6), "overlap_rows": k,
                         "same_missing_rows": bool((ok[:, a_i] == ok[:, b_i]).all()),
                         "coverage_a": cov[a], "coverage_b": cov[b], "drop": _suggest(a, b, cov, tables)})
    ex = [{"a": a, "b": b, "corr": 1.0, "coverage_a": cov[a], "coverage_b": cov[b], "drop": _suggest(a, b, cov, tables)}
          for a, b in exact]
    drop: list[str] = []
    for p in sorted(ex, key=lambda p: (p["a"], p["b"])) + sorted(near, key=lambda p: -abs(p["corr"])):
        keep = p["b"] if p["drop"] == p["a"] else p["a"]
        if p["drop"] not in drop and keep not in drop:
            drop.append(p["drop"])
    every = list(df.columns)
    clash = {d: [c for c in every if c != d and c.startswith(d)] for d in drop}
    return {"rows": int(n), "sample_rows": int(len(rows)), "columns": len(cols), "threshold": threshold,
            "exact": ex, "near": sorted(near, key=lambda p: -abs(p["corr"])), "constant_in_sample": constant,
            "drop": drop, "prefix_clash": {d: v for d, v in clash.items() if v}}


def load(args) -> pd.DataFrame:
    df = None
    if args.matrix:
        df = pd.read_parquet(args.matrix)
    if args.cache:                                                # the runner: the fast loop's cache in S3
        from greyhound.fastloop import load_cache
        from sources.common import Store
        df, _ = load_cache(Store())
    if df is None:
        raise SystemExit("give --matrix or --cache")
    for p in args.tables or []:
        t = pd.read_parquet(p)
        cols = [c for c in t.columns if c not in KEY and c not in df.columns]
        j = df[KEY].merge(t[KEY + cols].drop_duplicates(KEY), on=KEY, how="left")
        for c in cols:
            df[c] = j[c].to_numpy()
    for name in [x for x in (args.attach or "").split(",") if x]:  # the runner: a stored table, by name
        from greyhound import pit
        pit.attach(df, name)
    return df


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--matrix", help="a feature matrix parquet (the fast loop's features.parquet)")
    ap.add_argument("--cache", action="store_true", help="load the fast loop's current cache from S3 (the runner)")
    ap.add_argument("--tables", nargs="*", help="point-in-time table parquets to join on race_id, dog_id")
    ap.add_argument("--attach", help="stored point-in-time tables by name, comma-separated (the runner)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--sample", type=int, default=SAMPLE)
    ap.add_argument("--keep-labels", action="store_true", help="also compare the result and price columns")
    ap.add_argument("--out", help="write the JSON here (else print it)")
    args = ap.parse_args(argv)
    df = load(args)
    exclude: tuple = ()
    if not args.keep_labels:
        from greyhound.fastloop import EXTRA_COLS, LABELS
        exclude = tuple(LABELS) + tuple(EXTRA_COLS) + ("t",)
    res = find_duplicates(df, args.threshold, args.sample, exclude=exclude)
    text = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    else:
        print(text)
    return res


if __name__ == "__main__":
    main()
