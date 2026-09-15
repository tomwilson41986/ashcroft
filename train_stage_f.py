"""
Two-stage model in the Benter / racing² framework form:

    Stage F  our fundamental model — market-free features, winner label,
             race-grouped softmax (conditional logit or LightGBM with the
             race-softmax objective), walk-forward
    Stage C  blend with the market: c ∝ exp(α ln f + γ ln π), fitted only on
             OUT-OF-FOLD fundamentals from earlier folds
    Metric   ΔR² = R²_combined − R²_market with a race-bootstrap CI,
             conditional calibration tables, per-segment ΔR²
    Guards   the walk-forward is purged and embargoed around every fold
             boundary (--gap-days / --embargo-days, Section VI.2.3) and
             runners the model cannot rate take the market's probability
             instead of an invented one (--min-runs / --min-kf-n /
             --min-ratable, Section IV.1). Both are on by default: without
             them the ΔR² that comes out is not a measurement.

Sources
    --source blandford   the Blandford / Timeform feed (parquet from
                         blandford_sync-style fetch, or the blandford_results
                         table in --db). Works without horse_racing.db.
    --source db          race_results + CustomMetricsEngine features filtered
                         to Stage F by model.feature_registry (+ blandford
                         features when the table is present). Needs the DB.

Usage
    python train_stage_f.py --source blandford --parquet data/blandford_2025_2026.parquet --model clogit
    python train_stage_f.py --source blandford --parquet ... --model lgbm --test-start 2025-07-01
    python train_stage_f.py --source db --db horse_racing.db --model lgbm --start-date 2022-01-01

Artifacts (data/models/stage_f/): {tag}_metrics.json, {tag}_oof.csv,
{tag}_features.csv (with stages), {tag}_coefs.csv or {tag}_booster.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model.connections import add_connection_features
from model.feature_registry import classify, stage_f_columns
from model.phase1 import build_stage_f_features, prepare_blandford_frame
from model.primitives import within_race_transforms
from model.stage_f import (EMBARGO_DAYS, PURGE_DAYS, ConditionalLogit, StageC, TemperatureScaler, apply_unratable_rule,
                           conditional_calibration, delta_r2, lgb_race_softmax, mcfadden_r2, purge_embargo_masks,
                           race_log_loss)
from model.uncertainty import ConfigurationLedger

log = logging.getLogger("train_stage_f")
OUT_DIR = Path(__file__).resolve().parent / "data" / "models" / "stage_f"

LGB_PARAMS = {"learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "lambda_l2": 10.0, "max_depth": 6, "verbose": -1, "num_threads": 0, "seed": 42}


# ---------------------------------------------------------------------------
# Feature blocks (all Stage-F legal)
# ---------------------------------------------------------------------------

def _aptitude_residuals(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Horse's lag-safe mean NMFP under today's conditions minus its overall mean
    (distance band, surface); an aptitude signal the market prices imperfectly."""
    d = d.sort_values(["horse_name", "race_date", "race_time"]).copy()
    dist = pd.to_numeric(d["distance_f"], errors="coerce")
    d["_dband"] = pd.cut(dist, [0, 6.5, 8.5, 10.5, 13.0, 40], labels=False)
    d["_surf"] = d["surface"].astype(str).str.lower().str.contains("weather|aw|poly|tapeta|fibre", regex=True).astype(int) if "surface" in d.columns else 0
    g = d.groupby("horse_name")["nmfp"]
    overall = g.transform(lambda s: s.shift(1).expanding().mean())
    feats = []
    for key, name in (("_dband", "dist_apt"), ("_surf", "surf_apt")):
        cond = d.groupby(["horse_name", key])["nmfp"].transform(lambda s: s.shift(1).expanding().mean())
        n_cond = d.groupby(["horse_name", key])["nmfp"].transform(lambda s: s.shift(1).expanding().count())
        d[name] = ((cond - overall) * n_cond / (n_cond + 2.0)).fillna(0.0)   # shrink thin conditional records
        feats.append(name)
    return d.drop(columns=["_dband", "_surf"]).sort_index(), feats


def build_blandford_features(bf: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, list[str]]:
    d = prepare_blandford_frame(bf, flat_only=True)
    d, kf = build_stage_f_features(d, kalman_fit_end=test_start)
    base = [c for c in ["kf_z", "kf_sd", "kf_vs_max", "tf_master_z", "tf_master_vs_max", "tfig_ewm_z", "perf_max3_z", "perf_min5_z",
                        "perf_iqm5_z", "nmfp_mean3_z", "nmfp_max_career_z", "lto_tfig_above_ability", "lto_vs_career", "sos_vs_today",
                        "dslr_ln", "runs_count_ln", "age_z", "draw_pct", "first_run"] if c in d.columns]
    d = d.rename(columns={"jockey": "jockey_name"})
    d, conn_feats = add_connection_features(d, entities=("trainer", "jockey_name"))
    d, apt_feats = _aptitude_residuals(d)
    d["is_female"] = d["gender"].astype(str).str.lower().isin(["f", "m"]).astype(float) if "gender" in d.columns else 0.0
    d["draw_x_lnN"] = d["draw_pct"] * np.log(d["n_runners"].astype(float))
    d["draw_x_sprint"] = d["draw_pct"] * (pd.to_numeric(d["distance_f"], errors="coerce") <= 6.5).astype(float)
    d["kf_sd_x_runs"] = d["kf_sd"] * d["runs_count_ln"]
    # Stage-C-only history (market-derived, previous runs): in-running low vs BSP, BSP advantage
    d = d.sort_values(["horse_name", "race_date", "race_time"])
    bsp = pd.to_numeric(d["betfair_win_sp"], errors="coerce"); ipm = pd.to_numeric(d["ip_min"], errors="coerce")
    ratio = (ipm / bsp).where((bsp > 1) & (ipm > 0)).clip(0.01, 1.0)
    g = d.groupby("horse_name")
    d["lr_ipmin_ratio"] = ratio.groupby(d["horse_name"]).shift(1)
    d["ipmin_ratio_3r"] = ratio.groupby(d["horse_name"]).transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    d["lr_pi_market"] = d["pi_market"].groupby(d["horse_name"]).shift(1)
    d = d.sort_index()
    d = within_race_transforms(d, ["trainer_sr_shrunk", "jockey_sr_shrunk", "tj_sr_shrunk", "trainer_form_30d", "kf_sd"], suffixes=("z",))
    feats = base + conn_feats + apt_feats + ["is_female", "draw_x_lnN", "draw_x_sprint", "kf_sd_x_runs", "trainer_sr_shrunk_z",
                                              "jockey_sr_shrunk_z", "tj_sr_shrunk_z", "trainer_form_30d_z", "kf_sd_z"]
    d.attrs["kalman"] = kf
    return d, feats


def build_db_features(db_path: str, start_date: str | None) -> tuple[pd.DataFrame, list[str]]:
    """race_results + custom metrics, Stage F columns only (+ Blandford feed features if loaded)."""
    from model.custom_metrics import CustomMetricsEngine
    from model.perf_figures import add_perf_figure_features, ensure_raceid
    from train_bfsp import ALL_FEATURE_COLS, load_data
    df = load_data(db_path, start_date=start_date)
    df = CustomMetricsEngine().calculate_all(df)
    df = add_perf_figure_features(ensure_raceid(df))
    df["won"] = (pd.to_numeric(df["placing_numerical"], errors="coerce") == 1).astype(float)
    bsp = pd.to_numeric(df["bfsp"], errors="coerce").where(lambda s: s > 1.0)
    df["pi_market"] = (1 / bsp) / (1 / bsp).groupby(df["raceid"]).transform("sum")
    feats = stage_f_columns([c for c in ALL_FEATURE_COLS if c in df.columns])
    try:
        from model.blandford_features import BLANDFORD_FEATURES, add_blandford_features
        df = add_blandford_features(df, db_path=db_path)
        feats += [c for c in BLANDFORD_FEATURES if c in df.columns and c not in ("LR_tf_ipmin_ratio", "tf_ipmin_3r", "LR_tf_bsp_adv")]
    except Exception as exc:  # table absent
        log.info("no blandford features: %s", exc)
    ok = df.groupby("raceid")["won"].transform("sum") == 1
    return df[ok & df["pi_market"].notna()].copy(), feats


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def fit_predict_clogit(Xtr, ytr, gtr, Xte, gte, l2: float, **_):
    m = ConditionalLogit(l2=l2).fit(Xtr, ytr, gtr)
    return m.predict_proba(Xte, gte), m


def fit_predict_lgbm(Xtr, ytr, gtr, Xte, gte, rounds: int = 1500, params: dict | None = None, valid_frac: float = 0.12, **_):
    import lightgbm as lgb
    params = {**LGB_PARAMS, **(params or {})}
    codes = pd.factorize(pd.Series(gtr), sort=False)[0]
    order = np.argsort(codes, kind="stable")
    Xs, ys, cs = Xtr[order], ytr[order], codes[order]
    # last valid_frac of races (by order of appearance = chronological) as early-stopping set
    G = cs.max() + 1; cut = int(G * (1 - valid_frac))
    tr_m = cs < cut; va_m = ~tr_m
    sizes_tr = np.bincount(cs[tr_m]); sizes_va = np.bincount(cs[va_m] - cut)
    fobj_tr, feval_tr = lgb_race_softmax(sizes_tr)
    _, feval_va = lgb_race_softmax(sizes_va)
    dtr = lgb.Dataset(Xs[tr_m], label=ys[tr_m], free_raw_data=False); dva = lgb.Dataset(Xs[va_m], label=ys[va_m], reference=dtr, free_raw_data=False)
    booster = lgb.train({**params, "objective": fobj_tr}, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["valid"],
                        feval=feval_va, callbacks=[lgb.early_stopping(100, verbose=False)])
    from model.stage_f import race_softmax
    return race_softmax(booster.predict(Xte, num_iteration=booster.best_iteration), gte), booster


# ---------------------------------------------------------------------------
# Walk-forward two-stage
# ---------------------------------------------------------------------------

# Confidence gate (Section IV.1). A horse with fewer than three prior runs has
# no career aggregate worth the name — the framework routes 0-2 runs to a
# debutant sub-model that does not exist yet, so until it does those runners
# take the market's price rather than an invented one.
DEFAULT_GATE = {"min_runs": 3, "min_kf_n": 1, "min_ratable": 2}


def _gate_col(d: pd.DataFrame, mask, col: str):
    """The gate input if the frame carries it, else None (gate ignores it)."""
    return d.loc[mask, col].values if col in d.columns else None


def walk_forward(d: pd.DataFrame, feats, model: str, test_start: str, n_folds: int, l2: float, rounds: int, min_train_races: int = 300,
                 stage_c_extra=None, gap_days: int = PURGE_DAYS, embargo_days: int = EMBARGO_DAYS,
                 gate: dict | None = DEFAULT_GATE):
    """Walk-forward Stage F + Stage C, purged and embargoed (Section VI.2.3).

    stage_c_extra: columns (positive, e.g. previous-run in-running low ratio) added to Stage C as log terms; NaN -> 1.
    gap_days:      training rows within this many days *before* a fold's first
                   test date are purged — they are the rows whose labels sit
                   inside the test period's trailing feature windows.
    embargo_days:  the fold's first days are dropped from the test side for the
                   mirror-image reason. See model.stage_f.purge_embargo_masks.
    gate:          kwargs for model.stage_f.apply_unratable_rule (min_runs,
                   min_kf_n, min_ratable); None disables the confidence gate.
                   A frame carrying neither runs_count nor kf_n cannot be
                   gated, and the rule reduces to the identity.
    """
    d = d.copy(); d["race_date"] = pd.to_datetime(d["race_date"])
    X = d[feats].astype(float).values; y = d["won"].values.astype(float); g = d["raceid"].values
    dates = np.sort(d["race_date"].unique())
    test_dates = dates[dates >= np.datetime64(test_start)]
    edges = np.linspace(0, len(test_dates), n_folds + 1).astype(int)
    d["f_oof"] = np.nan; d["c_oof"] = np.nan; d["fold"] = -1; d["ratable"] = np.nan
    fit_fn = fit_predict_lgbm if model == "lgbm" else fit_predict_clogit
    last_model = None
    n_purged = n_embargoed = n_unratable = n_skipped = 0
    for k in range(n_folds):
        te_dates = test_dates[edges[k]: edges[k + 1]]
        if len(te_dates) == 0:
            continue
        tr, te = purge_embargo_masks(d["race_date"], te_dates, gap_days=gap_days, embargo_days=embargo_days)
        n_purged += int(((d["race_date"] < te_dates[0]).values & ~tr).sum())
        n_embargoed += int((d["race_date"].isin(te_dates).values & ~te).sum())
        if d.loc[tr, "raceid"].nunique() < min_train_races or te.sum() == 0:
            continue
        t0 = time.time()
        p, last_model = fit_fn(X[tr], y[tr], g[tr], X[te], g[te], l2=l2, rounds=rounds)
        if gate is not None:
            p, ratable, race_ok = apply_unratable_rule(p, d.loc[te, "pi_market"].values, g[te],
                                                       runs_count=_gate_col(d, te, "runs_count"),
                                                       kf_n=_gate_col(d, te, "kf_n"), **gate)
            d.loc[te, "ratable"] = ratable.astype(float)
            n_unratable += int((~ratable).sum()); n_skipped += int(len(np.unique(g[te][~race_ok])))
        d.loc[te, "f_oof"] = p; d.loc[te, "fold"] = k
        prev = d["f_oof"].notna().values & (d["fold"].values < k) & (d["fold"].values >= 0)
        if d.loc[prev, "raceid"].nunique() >= min_train_races:
            ex_prev = {c_: d.loc[prev, c_].fillna(1.0).clip(1e-3, None).values for c_ in (stage_c_extra or [])}
            ex_te = {c_: d.loc[te, c_].fillna(1.0).clip(1e-3, None).values for c_ in (stage_c_extra or [])}
            c = StageC().fit(d.loc[prev, "f_oof"].values, d.loc[prev, "pi_market"].values, y[prev], g[prev], extra=ex_prev or None)
            # a race the gate skipped has NaN f; ConditionalLogit imputes NaN to the
            # column mean, so Stage C would silently invent a blend for it
            d.loc[te, "c_oof"] = np.where(np.isfinite(p), c.predict_proba(p, d.loc[te, "pi_market"].values, g[te],
                                                                          extra=ex_te or None), np.nan)
            alpha, gamma = c.alpha_, c.gamma_
            if stage_c_extra:
                log.info("   Stage C extra coefs: %s", dict(zip(c.names_[2:], np.round(c.coef_raw_[2:], 3))))
        else:
            alpha = gamma = np.nan
        log.info("fold %d: train %d races, test %d races, %.0fs; Stage C alpha=%.3f gamma=%.3f", k, d.loc[tr, "raceid"].nunique(),
                 len(np.unique(g[te])), time.time() - t0, alpha, gamma)
    d.attrs["validation"] = {"gap_days": int(gap_days), "embargo_days": int(embargo_days), "train_rows_purged": n_purged,
                             "test_rows_embargoed": n_embargoed, "gate": gate, "runners_unratable": n_unratable,
                             "races_skipped_unratable": n_skipped}
    log.info("purge/embargo %dd/%dd: %d train rows purged, %d test rows embargoed; gate: %d runners unratable, %d races skipped",
             gap_days, embargo_days, n_purged, n_embargoed, n_unratable, n_skipped)
    return d, last_model


def evaluate(d: pd.DataFrame) -> dict:
    te = d["f_oof"].notna()
    y = d.loc[te, "won"].values; g = d.loc[te, "raceid"].values; f = d.loc[te, "f_oof"].values; pi = d.loc[te, "pi_market"].values
    out = {"n_races_test": int(len(np.unique(g))), "n_runners_test": int(te.sum()),
           "r2_market": mcfadden_r2(pi, y, g), "r2_fundamental": mcfadden_r2(f, y, g),
           "logloss_market": race_log_loss(pi, y), "logloss_fundamental": race_log_loss(f, y),
           "delta_fundamental_vs_market": delta_r2(f, pi, y, g)}
    tc = te & d["c_oof"].notna()
    if tc.sum():
        yc = d.loc[tc, "won"].values; gc = d.loc[tc, "raceid"].values; c = d.loc[tc, "c_oof"].values; pc = d.loc[tc, "pi_market"].values
        out.update({"n_races_combined": int(len(np.unique(gc))), "r2_combined": mcfadden_r2(c, yc, gc), "logloss_combined": race_log_loss(c, yc),
                    "r2_market_combined_rows": mcfadden_r2(pc, yc, gc), "r2_fundamental_combined_rows": mcfadden_r2(d.loc[tc, "f_oof"].values, yc, gc),
                    "delta_combined_vs_market": delta_r2(c, pc, yc, gc),
                    "conditional_calibration": conditional_calibration(d.loc[tc, "f_oof"].values, pc, yc, gc).to_dict("records")})
        # segments: field size bands and months
        seg = {}
        n = d.loc[tc].groupby("raceid")["won"].transform("size").values
        for name, m in (("field_le_8", n <= 8), ("field_9_12", (n > 8) & (n <= 12)), ("field_13_plus", n > 12)):
            if len(np.unique(gc[m])) > 100:
                seg[name] = delta_r2(c[m], pc[m], yc[m], gc[m], n_boot=200)
        months = pd.to_datetime(d.loc[tc, "race_date"]).dt.to_period("M").astype(str).values
        for mo in sorted(set(months)):
            m = months == mo
            if len(np.unique(gc[m])) > 100:
                seg[mo] = delta_r2(c[m], pc[m], yc[m], gc[m], n_boot=200)
        out["segments"] = seg
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["blandford", "db"], default="blandford")
    ap.add_argument("--parquet", default=None); ap.add_argument("--db", default="horse_racing.db"); ap.add_argument("--start-date", default=None)
    ap.add_argument("--model", choices=["clogit", "lgbm"], default="clogit")
    ap.add_argument("--test-start", default="2025-07-01"); ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--l2", type=float, default=2.0); ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--max-races", type=int, default=None); ap.add_argument("--tag", default=None); ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--stage-c-extra", default=None, help="comma-separated extra Stage C log-inputs (e.g. lr_ipmin_ratio)")
    ap.add_argument("--gap-days", type=int, default=PURGE_DAYS, help="purge: training rows this many days before each fold are dropped")
    ap.add_argument("--embargo-days", type=int, default=EMBARGO_DAYS, help="embargo: test rows in the first days of each fold are dropped")
    ap.add_argument("--min-runs", type=int, default=DEFAULT_GATE["min_runs"], help="ratability gate: minimum prior career runs")
    ap.add_argument("--min-kf-n", type=int, default=DEFAULT_GATE["min_kf_n"], help="ratability gate: minimum Kalman observations")
    ap.add_argument("--min-ratable", type=int, default=DEFAULT_GATE["min_ratable"], help="skip races with fewer ratable runners")
    ap.add_argument("--no-gate", action="store_true", help="disable the unratable-runner rule (measurement becomes optimistic)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    if args.source == "blandford":
        if args.parquet:
            bf = pd.read_parquet(args.parquet)
        else:
            conn = sqlite3.connect(args.db)
            bf = pd.read_sql_query("SELECT * FROM blandford_results" + (f" WHERE meeting_date >= '{args.start_date}'" if args.start_date else ""), conn); conn.close()
        d, feats = build_blandford_features(bf, args.test_start)
    else:
        d, feats = build_db_features(args.db, args.start_date)
        d["race_date"] = pd.to_datetime(d["race_date"])
    if args.max_races:
        keep = d["raceid"].drop_duplicates().tail(args.max_races)
        d = d[d["raceid"].isin(keep)]
    log.info("features (%d): %s", len(feats), feats)
    log.info("frame: %d runners, %d races; build %.0fs", len(d), d["raceid"].nunique(), time.time() - t0)
    extra = [c for c in (args.stage_c_extra.split(",") if args.stage_c_extra else []) if c in d.columns]
    gate = None if args.no_gate else {"min_runs": args.min_runs, "min_kf_n": args.min_kf_n, "min_ratable": args.min_ratable}
    d, model = walk_forward(d, feats, args.model, args.test_start, args.folds, args.l2, args.rounds, stage_c_extra=extra or None,
                            gap_days=args.gap_days, embargo_days=args.embargo_days, gate=gate)
    res = evaluate(d)
    res.update({"model": args.model, "source": args.source, "features": feats, "test_start": args.test_start, "folds": args.folds,
                "kalman": d.attrs.get("kalman"), "validation": d.attrs.get("validation"), "runtime_s": round(time.time() - t0)})
    tag = args.tag or f"{args.source}_{args.model}"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # Section VI.2.6 wants an honest count of configurations tried, and a count
    # reconstructed at write-up time is always short. Every run logs itself.
    ledger = ConfigurationLedger(out / "configurations.json")
    ledger.record({"tag": tag, "model": args.model, "source": args.source, "n_features": len(feats), "l2": args.l2,
                   "rounds": args.rounds, "folds": args.folds, "test_start": args.test_start, "gap_days": args.gap_days,
                   "embargo_days": args.embargo_days, "gate": gate, "stage_c_extra": extra},
                  delta_r2=(res.get("delta_combined_vs_market") or {}).get("delta_r2"),
                  r2_combined=res.get("r2_combined"), n_races=res.get("n_races_test"))
    res["configurations_tried"] = ledger.n_trials
    json.dump(res, open(out / f"{tag}_metrics.json", "w"), indent=2, default=float)
    d.loc[d["f_oof"].notna(), ["raceid", "race_date", "horse_name", "won", "pi_market", "f_oof", "c_oof", "fold"]].to_csv(out / f"{tag}_oof.csv", index=False)
    classify(feats).to_csv(out / f"{tag}_features.csv", index=False)
    if args.model == "clogit" and model is not None:
        model.coef_table(feats).to_csv(out / f"{tag}_coefs.csv", index=False)
    elif model is not None:
        model.save_model(str(out / f"{tag}_booster.txt"))
        pd.DataFrame({"feature": feats, "gain": model.feature_importance("gain")}).sort_values("gain", ascending=False).to_csv(out / f"{tag}_importance.csv", index=False)
    v = res.get("validation") or {}
    print(f"\n=== {tag}: {res['n_races_test']} test races ===")
    print(f"purge {v.get('gap_days')}d / embargo {v.get('embargo_days')}d: {v.get('train_rows_purged')} train rows purged, "
          f"{v.get('test_rows_embargoed')} test rows embargoed | unratable {v.get('runners_unratable')} runners, "
          f"{v.get('races_skipped_unratable')} races skipped")
    print(f"R2_market {res['r2_market']:.4f} | R2_fundamental {res['r2_fundamental']:.4f}  (all test folds)")
    print(f"same rows as Stage C: R2_market {res.get('r2_market_combined_rows', float('nan')):.4f} | R2_fundamental {res.get('r2_fundamental_combined_rows', float('nan')):.4f} | R2_combined {res.get('r2_combined', float('nan')):.4f}")
    print(f"logloss market {res['logloss_market']:.4f} | fundamental {res['logloss_fundamental']:.4f} | combined {res.get('logloss_combined', float('nan')):.4f}")
    dc = res.get("delta_combined_vs_market", {})
    print(f"ΔR² combined vs market: {dc.get('delta_r2', float('nan')):+.5f}  90% CI ({dc.get('ci', (np.nan, np.nan))[0]:+.5f}, {dc.get('ci', (np.nan, np.nan))[1]:+.5f})  P(Δ<=0)={dc.get('p_delta_le_0', float('nan')):.2f}")
    for k, v in res.get("segments", {}).items():
        print(f"   {k:14s} ΔR² {v['delta_r2']:+.5f} CI ({v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}) races {v['n_races']}")
    print(f"configurations tried in this out-dir: {ledger.n_trials} (deflate any Sharpe by it — model.uncertainty.deflated_sharpe_ratio)")
    print(f"artifacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
