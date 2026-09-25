#!/usr/bin/env python3
"""Check a trained price model before it is published.

    python scripts/verify_model.py --model-dir data/models --served-dir /tmp/served \\
        --feature-cache .feature_cache --db horse_racing.db --start-date 2021-01-01 \\
        --out reports/model_verify.md

A retrain used to reach the repository on trust: the job finished, so the
artefact was committed, and the first thing to price with it was the 06:00 run.
This is the gate in between. It fails (exit 1) unless:

1. the artefact is servable: `assert_meta_is_servable` passes (no post-race
   input, a recorded objective and target), and the booster reads exactly the
   features its metadata lists;
2. the live path builds every feature it reads: each engine block with a feature
   in the model is switched on by `blocks_needed`, and each drop-in block
   (model/blocks) it reads is one the live path builds (`blocks.used_by`), from
   engine blocks it can switch on (the block's ENGINE);
3. on the matrix's last days it prices every runner (finite, positive) with a
   book of 1 in every race, and its log prices correlate with the served
   model's at 0.9 or better. Drop-in blocks are built on the matrix as the
   live path builds them (`blocks.attach_as_trained`). The new model has seen these days, so this is a
   test that it is sane, not that it is better: the walk-forward evaluation is
   where better is decided.

The report says what was checked and what was found either way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KEY = ["race_date", "race_time", "track", "horse_name"]
MIN_CORRELATION = 0.9


def _load(model_dir: str):
    import lightgbm as lgb
    from model.bfsp_model import attach_serving_rule
    meta = json.loads(Path(model_dir, "bfsp_model_meta.json").read_text())
    booster = lgb.Booster(model_file=str(Path(model_dir, "bfsp_model.lgb")))
    attach_serving_rule(booster, meta)
    return booster, meta


def check_servable(booster, meta) -> list[str]:
    from model.bfsp_model import assert_meta_is_servable
    problems = []
    try:
        assert_meta_is_servable(meta)
    except ValueError as exc:
        problems.append(f"not servable: {exc}")
    if booster.num_feature() != len(meta.get("feature_cols", [])):
        problems.append(f"the booster reads {booster.num_feature()} features, the metadata lists "
                        f"{len(meta.get('feature_cols', []))}")
    return problems


def check_live_path(cols: list[str]) -> list[str]:
    from model import blocks
    from model.bfsp_features import (FORM_WINDOW_FEATURES, FRESHNESS_FEATURES, INTENT_FEATURES,
                                     SHAPE_DRAW_FEATURES, SHAPE_FORM_FEATURES, blocks_needed)
    problems = []
    need = blocks_needed(cols)
    owners = {"race_shape": set(SHAPE_DRAW_FEATURES) | set(SHAPE_FORM_FEATURES),
              "intent": set(INTENT_FEATURES), "freshness": set(FRESHNESS_FEATURES),
              "form_windows": set(FORM_WINDOW_FEATURES), "shape_form": set(SHAPE_FORM_FEATURES)}
    for flag, feats in owners.items():
        if set(cols) & feats and not need.get(flag):
            problems.append(f"the model reads {flag} features but blocks_needed leaves {flag} off")
    for name in blocks.used_by(cols):
        try:
            mod = blocks.load(name)
        except (TypeError, KeyError) as exc:
            problems.append(f"the drop-in block {name} breaks the contract: {exc}")
            continue
        unknown = [f for f in getattr(mod, "ENGINE", ()) if f not in need]
        if unknown:
            problems.append(f"the drop-in block {name} reads the engine blocks {unknown}, which the live "
                            f"engine has no flag for")
    return problems


def check_prices(new, new_meta, served, served_meta, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    from model.bfsp_model import predict_prices
    problems, notes = [], []
    absent = [c for c in new_meta["feature_cols"] if c not in frame.columns]
    if absent:
        problems.append(f"the matrix lacks {len(absent)} of the model's features ({absent[:3]}...): "
                        f"the prices were not checked")
        return problems, notes
    out = predict_prices(new, frame.copy(), new_meta["feature_cols"], race_col="raceid")
    p = out["predicted_bfsp"].to_numpy(dtype=float)
    if not np.all(np.isfinite(p) & (p > 0)):
        problems.append(f"{int((~(np.isfinite(p) & (p > 0))).sum())} runners without a finite positive price")
    book = out.groupby("raceid")["predicted_bfsp"].apply(lambda s: float((1.0 / s).sum()))
    worst = float((book - 1.0).abs().max()) if len(book) else float("nan")
    if not worst < 1e-6:
        problems.append(f"a race book is {worst:.2e} away from 1")
    bsp = pd.to_numeric(out["bfsp"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(bsp) & (bsp > 1) & np.isfinite(p) & (p > 0)
    err_new = float(np.mean(np.abs(np.log(p[ok]) - np.log(bsp[ok]))))
    notes.append(f"{len(out):,} runners in {out['raceid'].nunique():,} races, "
                 f"{out['race_date'].min():%Y-%m-%d} to {out['race_date'].max():%Y-%m-%d}; "
                 f"worst book error {worst:.1e}; mean absolute log error against BSP: new {err_new:.4f} "
                 f"(in-sample)")
    if served is not None:
        missing = [c for c in served_meta["feature_cols"] if c not in frame.columns]
        if missing:
            notes.append(f"the served model reads {len(missing)} columns the matrix lacks; not compared")
        else:
            old = predict_prices(served, frame.copy(), served_meta["feature_cols"], race_col="raceid")
            q = old["predicted_bfsp"].to_numpy(dtype=float)
            both = np.isfinite(q) & (q > 0) & np.isfinite(p) & (p > 0)
            corr = float(np.corrcoef(np.log(p[both]), np.log(q[both]))[0, 1])
            err_old = float(np.mean(np.abs(np.log(q[ok & both]) - np.log(bsp[ok & both]))))
            notes.append(f"against the served model: correlation of log prices {corr:.4f}; "
                         f"the served model's error on the same rows {err_old:.4f}")
            if not corr >= MIN_CORRELATION:
                problems.append(f"log prices correlate {corr:.3f} with the served model's (< {MIN_CORRELATION})")
    return problems, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="data/models")
    ap.add_argument("--served-dir", default=None, help="the model now served, to compare against")
    ap.add_argument("--feature-cache", default=".feature_cache")
    ap.add_argument("--db", default="horse_racing.db")
    ap.add_argument("--start-date", default="2021-01-01")
    ap.add_argument("--days", type=int, default=14, help="price the matrix's last N days")
    ap.add_argument("--out", default="reports/model_verify.md")
    a = ap.parse_args()

    new, meta = _load(a.model_dir)
    served, served_meta = (None, None)
    if a.served_dir and Path(a.served_dir, "bfsp_model.lgb").exists():
        served, served_meta = _load(a.served_dir)

    problems = check_servable(new, meta) + check_live_path(meta.get("feature_cols", []))
    notes = [f"{len(meta.get('feature_cols', []))} features, target {meta.get('target')}, "
             f"best_iteration {meta.get('best_iteration')}, trained through {meta.get('trained_through')}"]
    if served_meta:
        gained = sorted(set(meta["feature_cols"]) - set(served_meta["feature_cols"]))
        lost = sorted(set(served_meta["feature_cols"]) - set(meta["feature_cols"]))
        notes.append(f"against the served model ({len(served_meta['feature_cols'])} features): "
                     f"+{len(gained)} ({', '.join(gained[:6])}{' ...' if len(gained) > 6 else ''}), "
                     f"-{len(lost)} ({', '.join(lost[:6])}{' ...' if len(lost) > 6 else ''})")

    from model import blocks, feature_cache
    key = feature_cache.cache_key(a.db, a.start_date)
    every = set(meta["feature_cols"]) | set((served_meta or {}).get("feature_cols", []))
    drop_in = blocks.used_by(every)
    built = {c for b in drop_in for c in blocks.features(b)}
    want = sorted(every - built)
    path = Path(a.feature_cache, f"features_{key}.parquet")
    if not path.exists():
        problems.append(f"no cached matrix under {a.feature_cache} for key {key}: the prices were not checked")
    else:
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(path).schema_arrow.names)
        reads: list[str] = []
        for b in drop_in:                      # a block without READS reads anything: the whole matrix
            r = getattr(blocks.load(b), "READS", None)
            reads += sorted(have) if r is None else list(r)
        cols = list(dict.fromkeys(c for c in KEY + ["raceid", "bfsp"] + want + reads if c in have))
        frame = pd.read_parquet(path, columns=cols)
        if drop_in:
            # on the whole matrix, as training built them: their windows need the history
            try:
                frame, _ = blocks.attach_as_trained(frame, drop_in)
                notes.append(f"drop-in blocks built on the matrix as the live path builds them: {', '.join(drop_in)}")
            except Exception as exc:                     # noqa: BLE001 - reported, and the gate fails
                problems.append(f"the drop-in blocks {drop_in} could not be built on the matrix: {exc}")
        last = pd.to_datetime(frame["race_date"]).max()
        frame = frame[pd.to_datetime(frame["race_date"]) > last - pd.Timedelta(days=a.days)].reset_index(drop=True)
        p2, n2 = check_prices(new, meta, served, served_meta, frame)
        problems += p2
        notes += n2

    ok = not problems
    lines = [f"# Model verification: {'PASS' if ok else 'FAIL'}", ""]
    lines += [f"- {n}" for n in notes]
    lines += ["", "## Problems" if problems else "No problems found."]
    lines += [f"- {p}" for p in problems]
    text = "\n".join(lines) + "\n"
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
