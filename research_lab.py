"""
Research lab CLI — entry point for the RESEARCH_FRAMEWORK.md toolkit.

    python research_lab.py score --predictions data/oos_predictions.csv
        Model-vs-market scoring report: log-loss / Brier skill vs Betfair SP,
        Murphy (reliability / resolution) decomposition, ECE, concordance,
        monthly model-market divergence.

    python research_lab.py abm --db horse_racing.db --date 2026-03-12 [--track Kempton --time 2.30]
        Simulate a race card (or one race) and print ABM probabilities.

    python research_lab.py abm-features --db horse_racing.db --from 2024-01-01 --out data/abm_features.parquet
        Batch-generate ABM features for training (then train_bfsp.py --abm-features ...).

    python research_lab.py abm-calibrate --db horse_racing.db --from 2025-01-01 --races 200
        Pattern-oriented ABC calibration of the interaction parameters.

    python research_lab.py effects --db horse_racing.db --from 2023-01-01
        Cross-classified ridge effects (horse / jockey / trainer / sire) on performance figures.

    python research_lab.py causal --db horse_racing.db --treatment first_time_headgear
        Doubly-robust intervention effect with E-value.

    python research_lab.py draw --db horse_racing.db --track Chester --dist 5
        GP-smoothed draw bias + Moran's I.

    python research_lab.py market --db horse_racing.db --from 2024-01-01
        Betfair price-movement diagnostics (steam deciles) + feature coverage.

    python research_lab.py bets --predictions data/oos_predictions.csv [--blend 0.5]
        Overlay tiers, cumulative overlay strategies with bootstrap CIs, overlays by
        price band, per-race rank performance (model vs market), disagreement analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys

import numpy as np
import pandas as pd

log = logging.getLogger("research_lab")


def _load_db(db_path: str, date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    where, params = [], []
    if date_from:
        where.append("race_date >= ?"); params.append(date_from)
    if date_to:
        where.append("race_date <= ?"); params.append(date_to)
    w = (" WHERE " + " AND ".join(where)) if where else ""
    df = pd.read_sql_query(f"SELECT * FROM race_results{w} ORDER BY race_date, race_time", conn, params=params)
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def _with_metrics(df: pd.DataFrame, perf: bool = True) -> pd.DataFrame:
    """Custom metrics (pace / style / speed columns) + perf-figure features."""
    from model.custom_metrics import CustomMetricsEngine
    from model.perf_figures import add_perf_figure_features
    df = CustomMetricsEngine().calculate_all(df)
    return add_perf_figure_features(df) if perf else df


# ---------------------------------------------------------------------------
def cmd_score(args):
    from model.diagnostics import divergence_by_period, market_implied_probs, race_normalise, scoring_report
    from model.perf_figures import ensure_raceid
    df = pd.read_csv(args.predictions)
    df = ensure_raceid(df)
    if args.date_from:
        df = df[pd.to_datetime(df["race_date"]) >= args.date_from]
    rep = scoring_report(df, p_col=args.p_col, y_col=args.y_col, race_col="raceid", price_col=args.price_col)
    rel = rep.pop("reliability")
    print("\n=== Model vs market (Betfair SP) ===")
    for k, v in rep.items():
        print(f"{k:32s} {v:.5f}" if isinstance(v, float) else f"{k:32s} {v}")
    print("\nReliability table (model):")
    print(rel.round(4).to_string(index=False))
    d = df.dropna(subset=[args.p_col, args.price_col]).copy()
    d = d[pd.to_numeric(d[args.price_col], errors="coerce") > 1]
    d["_pm"] = race_normalise(d, args.p_col, "raceid"); d["_pk"] = market_implied_probs(d, args.price_col, "raceid")
    drift = divergence_by_period(d, "_pm", "_pk", "raceid", "race_date", "MS")
    print("\nMonthly model-vs-market JS divergence (drift monitor):")
    print(drift.round(4).to_string())
    if args.json:
        json.dump({k: (v if not isinstance(v, tuple) else list(v)) for k, v in rep.items()}, open(args.json, "w"), indent=2, default=float)
        print(f"\nwritten {args.json}")


def cmd_bets(args):
    from model.bet_analysis import bet_report
    df = pd.read_csv(args.predictions)
    if args.date_from:
        df = df[pd.to_datetime(df["race_date"]) >= args.date_from]
    rep = bet_report(df, commission=args.commission, blend_lambda=args.blend, p_col=args.p_col, price_col=args.price_col)
    pd.set_option("display.width", 220)
    print(f"\n{rep['n_runners']} runners, {rep['n_races']} races; commission {args.commission:.0%}; returns per unit stake at BSP\n")
    titles = {
        "overlay_tiers": "=== Overlay tiers: edge = p_model / p_market - 1 (win_rate vs model_p vs market_p) ===",
        "cumulative_overlays": "=== Back everything with edge >= threshold (90% cluster-bootstrap CI by race); lay the mirror-image underlays ===",
        "blend_cumulative_overlays": f"=== Same, using the log-linear blend p = softmax({args.blend} ln p_model + {1 - args.blend:.2f} ln p_market) ===",
        "overlay_by_price_10": "=== Overlays (edge >= 10%) by BSP band, with the all-runners ROI in the band as reference ===",
        "model_ranks": "=== Model rank within race (1 = model's top pick) ===",
        "market_ranks": "=== Market rank within race (1 = favourite) ===",
        "disagreement": "=== Disagreement: the model's #1 by market rank, and the favourite by model rank ===",
        "rank_by_field": "=== Top pick by field size: model #1 vs favourite ===",
        "concordance_by_field": "=== Within-race concordance by field size ===",
    }
    for key, title in titles.items():
        if key in rep:
            print(title); print(rep[key].round(4).to_string(index=False)); print()


def cmd_ratings_eval(args):
    """Evaluate external rating sets (HRB Ratings Machine) against the OOS predictions."""
    from model.bet_analysis import prepare_bets
    from model.hrb_features import pivot_ratings
    from model.rating_eval import evaluate_rating_sets, knockoff_screen
    pred = prepare_bets(pd.read_csv(args.predictions), commission=0.05)
    conn = sqlite3.connect(args.db)
    hr = pd.read_sql_query("SELECT race_date, track, race_time_24, horse_norm, set_name, rating FROM hrb_ratings", conn); conn.close()
    if hr.empty:
        print("no rows in hrb_ratings (run hrb_ratings.py --fetch/--load first)"); return 1
    from betfair_prices import db_time_to_24h, normalise_horse, normalise_track
    pred["_k"] = pred["race_date"].astype(str) + "|" + pred["track"].map(normalise_track) + "|" + pred["race_time"].map(db_time_to_24h) + "|" + pred["horse_name"].map(normalise_horse)
    hr["_k"] = hr["race_date"].astype(str) + "|" + hr["track"].map(normalise_track) + "|" + hr["race_time_24"].astype(str) + "|" + hr["horse_norm"]
    hr["race_results_id"] = hr["_k"]
    wide = pivot_ratings(hr.rename(columns={"_k": "race_results_id"}).drop(columns=["race_results_id"], errors="ignore").assign(race_results_id=hr["_k"]))
    d = pred.merge(wide, left_on="_k", right_on="race_results_id", how="left")
    cols = [c for c in wide.columns if c.startswith("hrb_")]
    d = d[d["race_date"].isin(hr["race_date"].unique())]
    print(f"{len(d)} OOS runners on {d['raceid'].nunique()} races in the rating window; rating columns: {cols}")
    pd.set_option("display.width", 250)
    ev = evaluate_rating_sets(d, cols)
    print("\n=== Standalone and incremental value per rating set ===")
    print(ev.round(4).to_string(index=False))
    if len(cols) > 1:
        print("\n=== Model-X knockoffs across all sets (race z-scores) + model/market ===")
        print(knockoff_screen(d, cols).round(4).to_string(index=False))
    if args.out:
        ev.to_csv(args.out, index=False); print(f"written {args.out}")


def cmd_phase1(args):
    """Honest ΔR² test (racing² master framework Phase 1) on the Blandford/Timeform feed."""
    from model.phase1 import run_phase1
    from model.stage_f import ConditionalLogit
    if args.parquet:
        bf = pd.read_parquet(args.parquet)
    else:
        conn = sqlite3.connect(args.db)
        bf = pd.read_sql_query("SELECT * FROM blandford_results" + (f" WHERE meeting_date >= '{args.date_from}'" if args.date_from else ""), conn); conn.close()
    res = run_phase1(bf, test_start=args.test_start, n_folds=args.folds, l2=args.l2, flat_only=not args.all_codes)
    pd.set_option("display.width", 220)
    print(f"races total {res['n_races_total']} | test {res['n_races_test']} | kalman {dict((k, round(v, 2)) for k, v in res['kalman'].items())}")
    print(f"R2_market {res['r2_market']:.4f} | R2_fundamental {res['r2_fundamental']:.4f} | R2_combined {res.get('r2_combined', float('nan')):.4f}")
    for key in ("delta_fundamental_vs_market", "delta_combined_vs_market"):
        if key in res:
            print(key, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[key].items()})
    print(f"next-run MSE: kalman {res['next_run_mse_kalman']:.1f} | ewm270 {res['next_run_mse_ewm270']:.1f}")
    if "ordering" in res:
        print("ordering:", {k: round(v, 3) for k, v in res["ordering"].items()})
    print("\nConditional calibration (Benter Tables 3/4 format):"); print(res["conditional_calibration"].round(3).to_string(index=False))
    wf = res["frame"]; te = wf["race_date"] >= pd.Timestamp(args.test_start)
    m = ConditionalLogit(l2=args.l2).fit(wf.loc[~te, res["features"]].values, wf.loc[~te, "won"].values, wf.loc[~te, "raceid"].values)
    print("\nStage F coefficients (standardised):"); print(m.coef_table(res["features"]).round(3).to_string(index=False))
    if args.out:
        wf.drop(columns=["frame"], errors="ignore").to_parquet(args.out, index=False); print(f"written {args.out}")


def cmd_abm(args):
    from model.abm import simulate_race
    df = _load_db(args.db, args.date, args.date)
    if df.empty:
        print("no rows for that date"); return 1
    hist = _load_db(args.db, None, args.date)  # history for lag-safe style / perf features
    hist = _with_metrics(hist)
    day = hist[hist["race_date"] == pd.Timestamp(args.date)]
    if args.track:
        day = day[day["track"].str.lower() == args.track.lower()]
    if args.time:
        day = day[day["race_time"].astype(str).str.startswith(args.time)]
    pd.set_option("display.width", 220)
    for (t, rt), g in day.groupby(["track", "race_time"]):
        feats, res, _ = simulate_race(g, n_sims=args.sims, seed=args.seed, abilities=args.abilities)
        show = feats.merge(g[["horse_name", "official_rating", "stall", "bfsp"]], on="horse_name", how="left")
        print(f"\n{t} {rt} — {len(g)} runners, {g['dist_furlongs'].iloc[0]}f {g['going_description'].iloc[0]}")
        cols = ["horse_name", "stall", "official_rating", "bfsp", "abm_win_prob", "abm_win_prob_solo", "abm_pace_delta",
                "abm_place_prob", "abm_exp_beaten_l", "abm_p_led_early", "abm_p_trouble"]
        print(show.sort_values("abm_win_prob", ascending=False)[cols].round(3).to_string(index=False))
        print(f"pace: contest={feats['abm_pace_contest'].iloc[0]:.2f} collapse_p={feats['abm_pace_collapse_p'].iloc[0]:.2f} entropy={feats['abm_race_entropy'].iloc[0]:.2f}")


def cmd_abm_features(args):
    from model.abm import ABM_FEATURES, add_abm_features
    from model.abm.features import ABM_KEY_COLS
    df = _with_metrics(_load_db(args.db, None, args.date_to))
    target = df[df["race_date"] >= pd.Timestamp(args.date_from)] if args.date_from else df
    log.info("ABM features for %d rows / %d races", len(target), target.groupby(["race_date", "race_time", "track"]).ngroups)
    out = add_abm_features(target, n_sims=args.sims, seed=args.seed, abilities=args.abilities, n_jobs=args.jobs,
                           max_races=args.max_races)
    keep = out[ABM_KEY_COLS + ABM_FEATURES]
    if args.out.endswith(".parquet"):
        keep.to_parquet(args.out, index=False)
    else:
        keep.to_csv(args.out, index=False)
    print(f"written {args.out}: {len(keep)} rows, {keep['abm_win_prob'].notna().mean():.1%} with features")


def cmd_abm_calibrate(args):
    from model.abm import calibrate_interactions
    df = _with_metrics(_load_db(args.db, None, args.date_to))
    df = df[df["race_date"] >= pd.Timestamp(args.date_from)] if args.date_from else df
    res = calibrate_interactions(df, n_sims=args.sims, n_particles=args.particles, n_rounds=args.rounds,
                                 max_races=args.races, abilities=args.abilities, seed=args.seed)
    print("\nbest parameters:"); print(json.dumps(res["params"], indent=2))
    print(f"loss {res['loss']:.3f} on {res['n_races']} races")
    print("observed :", {k: round(v, 3) for k, v in res["observed"].items()})
    print("simulated:", {k: round(v, 3) for k, v in res["simulated"].items()})
    if args.out:
        json.dump({"params": res["params"], "loss": res["loss"], "observed": res["observed"], "simulated": res["simulated"]},
                  open(args.out, "w"), indent=2)
        res["particles"].to_csv(args.out.replace(".json", "_particles.csv"), index=False)
        print(f"written {args.out}")


def cmd_effects(args):
    from model.effects import CrossClassifiedRidge
    from model.perf_figures import add_perf_figure_features
    df = add_perf_figure_features(_load_db(args.db, args.date_from, args.date_to))
    ents = args.entities.split(",")
    m = CrossClassifiedRidge(ents, lambdas=args.ridge, prior_col="official_rating" if args.prior_or else None,
                             min_count=args.min_count).fit(df, "perf_lbs", eb_iterations=3)
    print(f"fitted on {int(df['perf_lbs'].notna().sum())} rated runs; sigma_e^2={m.sigma2_e_:.2f}; lambdas={ {k: round(v, 2) for k, v in m.lambdas.items()} }")
    for e in ents:
        print(f"\n=== {e}: top / bottom effects (lbs vs pooled mean) ===")
        print(m.summary(e, top=args.top).round(2).to_string())


def cmd_causal(args):
    from model.causal import first_time_flag, intervention_study
    from model.custom_metrics import CustomMetricsEngine
    df = CustomMetricsEngine().calculate_all(_load_db(args.db, args.date_from, args.date_to))
    if args.treatment not in df.columns:
        if args.treatment == "first_time_headgear":
            df["first_time_headgear"] = first_time_flag(df, "headgear")
        else:
            print(f"treatment column {args.treatment} not found"); return 1
    covs = [c for c in args.covariates.split(",") if c in df.columns]
    outcome = args.outcome
    if outcome == "won" and "won" not in df.columns:
        df["won"] = (pd.to_numeric(df["placing_numerical"], errors="coerce") == 1).astype(float)
    res = intervention_study(df, args.treatment, outcome, covs, method=args.method)
    print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in res.items()}, indent=2, default=float))


def cmd_draw(args):
    from model.spatial import morans_i, smooth_draw_bias, stall_adjacency
    df = _load_db(args.db, args.date_from, args.date_to)
    df["won"] = (pd.to_numeric(df["placing_numerical"], errors="coerce") == 1).astype(float)
    t = smooth_draw_bias(df, args.track, args.dist, min_runners=args.min_runners)
    if t.empty:
        print("no data"); return 1
    print(t.round(4).to_string(index=False))
    mi = morans_i(t["raw_rate"].values - t["expected_rate"].values, stall_adjacency(t["stall"].values))
    print(f"\nMoran's I on excess win rate by stall: I={mi['I']:.3f} (expected {mi['expected']:.3f}), p={mi['p_value']:.3f}")


def cmd_market(args):
    from betfair_prices import coverage_report
    from model.market_features import MARKET_FEATURES, add_market_features, load_market_frame, market_movement_summary
    print("=== betfair_prices coverage by month ==="); print(coverage_report(args.db).tail(24).to_string(index=False))
    df = _load_db(args.db, args.date_from, args.date_to)
    mk = load_market_frame(args.db, args.date_from)
    d = df.merge(mk, left_on="id", right_on="race_results_id", how="inner")
    if d.empty:
        print("no matched market rows in range (run betfair_prices.py --load --match)"); return 1
    print(f"\n{len(d)} runners with market data")
    print("\n=== Steam (morning -> BSP) deciles: is the move informative beyond the price? ===")
    print(market_movement_summary(d).round(4).to_string(index=False))
    f = add_market_features(df, market=mk)
    print("\nfeature coverage:"); print(f[MARKET_FEATURES].notna().mean().round(3).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score"); s.add_argument("--predictions", default="data/oos_predictions.csv")
    s.add_argument("--p-col", default="predicted_win_prob_norm"); s.add_argument("--y-col", default="won")
    s.add_argument("--price-col", default="bfsp"); s.add_argument("--from", dest="date_from", default=None)
    s.add_argument("--json", default=None); s.set_defaults(fn=cmd_score)

    for name, fn in (("abm", cmd_abm), ("abm-features", cmd_abm_features), ("abm-calibrate", cmd_abm_calibrate)):
        s = sub.add_parser(name); s.add_argument("--db", default="horse_racing.db")
        s.add_argument("--sims", type=int, default=500 if name != "abm-calibrate" else 150)
        s.add_argument("--seed", type=int, default=42); s.add_argument("--abilities", default="composite")
        s.set_defaults(fn=fn)
        if name == "abm":
            s.add_argument("--date", required=True); s.add_argument("--track", default=None); s.add_argument("--time", default=None)
        else:
            s.add_argument("--from", dest="date_from", default=None); s.add_argument("--to", dest="date_to", default=None)
        if name == "abm-features":
            s.add_argument("--out", default="data/abm_features.parquet"); s.add_argument("--jobs", type=int, default=1)
            s.add_argument("--max-races", type=int, default=None)
        if name == "abm-calibrate":
            s.add_argument("--races", type=int, default=200); s.add_argument("--particles", type=int, default=32)
            s.add_argument("--rounds", type=int, default=3); s.add_argument("--out", default="data/abm_calibration.json")

    s = sub.add_parser("effects"); s.add_argument("--db", default="horse_racing.db")
    s.add_argument("--from", dest="date_from", default=None); s.add_argument("--to", dest="date_to", default=None)
    s.add_argument("--entities", default="horse_name,jockey_name,trainer,stallion"); s.add_argument("--ridge", type=float, default=10.0)
    s.add_argument("--prior-or", action="store_true"); s.add_argument("--min-count", type=int, default=3); s.add_argument("--top", type=int, default=15)
    s.set_defaults(fn=cmd_effects)

    s = sub.add_parser("causal"); s.add_argument("--db", default="horse_racing.db")
    s.add_argument("--from", dest="date_from", default=None); s.add_argument("--to", dest="date_to", default=None)
    s.add_argument("--treatment", default="first_time_headgear"); s.add_argument("--outcome", default="won")
    s.add_argument("--covariates", default="official_rating,LR_RSR,LR3_RSR,preracehorsecareerRSR,career_runs,days_since_lr,horse_age,NFP,RB")
    s.add_argument("--method", default="aipw"); s.set_defaults(fn=cmd_causal)

    s = sub.add_parser("draw"); s.add_argument("--db", default="horse_racing.db"); s.add_argument("--track", required=True)
    s.add_argument("--dist", type=float, required=True); s.add_argument("--min-runners", type=int, default=8)
    s.add_argument("--from", dest="date_from", default=None); s.add_argument("--to", dest="date_to", default=None); s.set_defaults(fn=cmd_draw)

    s = sub.add_parser("market"); s.add_argument("--db", default="horse_racing.db")
    s.add_argument("--from", dest="date_from", default=None); s.add_argument("--to", dest="date_to", default=None); s.set_defaults(fn=cmd_market)

    s = sub.add_parser("ratings-eval"); s.add_argument("--predictions", default="data/oos_predictions.csv")
    s.add_argument("--db", default="horse_racing.db"); s.add_argument("--out", default=None); s.set_defaults(fn=cmd_ratings_eval)

    s = sub.add_parser("phase1"); s.add_argument("--db", default="horse_racing.db"); s.add_argument("--parquet", default=None)
    s.add_argument("--from", dest="date_from", default=None); s.add_argument("--test-start", default="2025-07-01")
    s.add_argument("--folds", type=int, default=8); s.add_argument("--l2", type=float, default=2.0); s.add_argument("--all-codes", action="store_true")
    s.add_argument("--out", default=None); s.set_defaults(fn=cmd_phase1)

    s = sub.add_parser("bets"); s.add_argument("--predictions", default="data/oos_predictions.csv")
    s.add_argument("--p-col", default="predicted_win_prob_norm"); s.add_argument("--price-col", default="bfsp")
    s.add_argument("--commission", type=float, default=0.05); s.add_argument("--blend", type=float, default=0.5)
    s.add_argument("--from", dest="date_from", default=None); s.set_defaults(fn=cmd_bets)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
