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

    python research_lab.py clv --predictions data/oos_predictions.csv --db horse_racing.db [--source snapshots]
        Closing-line value: does the BSP forecast beat the morning price? Walk-forward blend
        ln BSP ~ ln p_morning + ln p_pred, early-bet rule, realised CLV (= greened-up profit) by tier.

    python research_lab.py price-cal --predictions data/oos_predictions.csv [--save models/bsp_price_calibrator.json]
        Calibrate the BFSP forecast against realised BFSP: walk-forward quantile
        regression that removes the compression and rank bias in the raw price.

    python research_lab.py stake --predictions data/oos_predictions.csv [--bank-chart out.png]
        Market-blind staking: Kelly on the model's own probabilities (full to 1/20),
        bankroll path with drawdowns, per-rank flat / proportional / level-profit /
        Kelly staking, probability-shrinkage scan, predicted-vs-actual BFSP by rank.

    python research_lab.py bets --predictions data/oos_predictions.csv [--blend 0.5]
        Overlay tiers, cumulative overlay strategies with bootstrap CIs, overlays by
        price band, per-race rank performance (model vs market), disagreement analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
        "by_race_code": "=== Top pick by race code (segments under 200 bets pooled as (other)) ===",
        "by_race_type": "=== Top pick by race type ===",
        "by_race_class": "=== Top pick by race class ===",
        "by_going": "=== Top pick by going ===",
        "by_month": "=== Top pick by month ===",
        "concordance_by_race_type": "=== Within-race concordance by race type (gap = model - market) ===",
    }
    for key, title in titles.items():
        if key in rep:
            print(title); print(rep[key].round(4).to_string(index=False)); print()


def cmd_price_cal(args):
    from model.perf_figures import ensure_raceid
    from model.price_calibration import BSPPriceCalibrator, calibration_report, walk_forward_calibrate
    df = ensure_raceid(pd.read_csv(args.predictions))
    # Fit against the booster's own output, not the served price.
    #
    # The served price is already race-normalised, and the previous report
    # compared calibrated output with calibrated output because the raw column
    # was never exported -- which is how the calibrator came to look as though
    # it "added nothing". `predicted_bfsp_raw` is exported now; fall back to the
    # served column only for older prediction files that predate it.
    price_col = args.price_col
    if price_col is None:
        price_col = "predicted_bfsp_raw" if "predicted_bfsp_raw" in df.columns else "predicted_bfsp"
    print(f"calibrating {price_col} against realised bfsp")
    df = df[(df["bfsp"] > 1.0) & (df[price_col] > 1.0)]
    if args.date_from:
        df = df[pd.to_datetime(df["race_date"]) >= args.date_from]
    d = walk_forward_calibrate(df, price_col=price_col, n_folds=args.folds)
    rep = calibration_report(d, raw_col=price_col)
    pd.set_option("display.width", 200)
    print(f"\n{rep['n']:,} runners, {rep['races']:,} races scored out of sample "
          f"({args.folds} walk-forward folds; the first block is training only)\n")
    print("=== Forecast quality: raw model price vs calibrated ===")
    print(rep["summary"].round(4).to_string(index=False))
    print("\n=== Median forecast / actual BFSP, by the model's rank in the race ===")
    print(rep["by_rank"].round(4).to_string(index=False))
    if "coverage" in rep:
        print("\n=== Quantile coverage: share of runners whose BFSP came in at or below the forecast ===")
        print(rep["coverage"].round(2).to_string(index=False))
    if args.save:
        cal = BSPPriceCalibrator().fit(df, price_col=price_col)
        cal.save(args.save)
        print(f"\ncalibrator fitted on all {cal.meta_['rows']:,} rows and saved to {args.save}")
    if args.out:
        keep = [c for c in d.columns if c.startswith("bsp_forecast")]
        base = [c for c in ("raceid", "race_date", "horse_name", price_col,
                            "predicted_bfsp", "bfsp") if c in d.columns]
        d[list(dict.fromkeys(base)) + keep].to_csv(args.out, index=False)
        print(f"calibrated predictions written to {args.out}")


def cmd_stake(args):
    from model.staking import staking_report, bankroll_path
    df = pd.read_csv(args.predictions)
    if args.date_from:
        df = df[pd.to_datetime(df["race_date"]) >= args.date_from]
    rep = staking_report(df, commission=args.commission, max_rank=args.max_rank,
                         p_col=args.p_col, price_col=args.price_col)
    pd.set_option("display.width", 220)
    print(f"\n{rep['n_runners']:,} runners, {rep['n_races']:,} races; commission {args.commission:.0%}. "
          "Selection and stake sizing use model probabilities only; the market supplies the settlement price.\n")
    pr = rep["promise_vs_reality"]
    print("=== Full-Kelly promise vs reality ===")
    print(f"  runners backed (positive model edge) : {pr['bets']:,} of {pr['runners']:,} ({pr['pct_of_field_backed']:.1f}% of the field)")
    print(f"  mean full-Kelly stake                : {pr['mean_kelly_stake_pct']:.2f}% of bank")
    print(f"  model edge, median / mean / weighted : {pr['median_model_edge_pct']:+.1f}% / {pr['mean_model_edge_pct']:+.1f}% / {pr['stake_weighted_model_edge_pct']:+.1f}%")
    print(f"  realised edge, flat / stake-weighted : {pr['realised_edge_pct']:+.2f}% / {pr['stake_weighted_realised_edge_pct']:+.2f}%")
    print(f"  log growth promised / realised       : {pr['promised_log_growth']:+,.0f} / {pr['realised_log_growth']:+,.0f}\n")
    titles = {
        "kelly_ladder": "=== Kelly ladder: bank compounded race by race, start = 1.0 (log10_terminal_bank = 0 means break-even) ===",
        "kelly_by_rank": "=== Quarter-Kelly restricted to the model's n-th choice ===",
        "kelly_top_k": "=== Quarter-Kelly restricted to the model's top n ===",
        "rank_staking": "=== Model rank within race, flat stakes at BSP, and three staking plans ===",
        "topk": "=== Back the model's top k in every race ===",
        "shrinkage": "=== Flatten the probabilities (p ∝ p^λ, λ=0 is 1/N) and re-run quarter-Kelly ===",
        "price_forecast": "=== Predicted BFSP vs realised BFSP by model rank ===",
        "by_field": "=== Model #1 by field size ===",
        "by_price": "=== Model #1 by the model's own forecast price ===",
    }
    for key, title in titles.items():
        if key in rep and len(rep[key]):
            print(title); print(rep[key].round(4).to_string(index=False)); print()
    for f in rep.get("focus", []):
        if not f:
            continue
        print(f"=== Focus: {f['label']} ===")
        print(f"  {f['n']:,} bets, {f['win_rate_pct']:.1f}% winners at an average BSP of {f['avg_bsp']:.2f}")
        print(f"  flat ROI {f['roi_flat_pct']:+.2f}%  (90% race-bootstrap CI {f['roi_ci_lo_pct']:+.2f}% to {f['roi_ci_hi_pct']:+.2f}%, t = {f['t_stat']:+.2f})")
        print(f"  quarter-Kelly bank: 10^{f['kelly_log10_terminal_bank']:.2f}, worst drawdown {f['kelly_max_drawdown_pct']:.1f}%")
        if len(f["stability"]):
            print(f["stability"].round(3).to_string(index=False))
        print()
    if args.bank_chart:
        _bank_chart(rep, args.bank_chart, args.commission)
    if args.out_dir:
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        for key, val in rep.items():
            if isinstance(val, pd.DataFrame) and not key.startswith("_") and len(val):
                val.to_csv(os.path.join(args.out_dir, f"stake_{key}.csv"), index=False)
        print(f"tables written to {args.out_dir}")


def _bank_chart(rep, path, commission):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from model.staking import bankroll_path
    d = rep["_frame"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    for f in (1.0, 0.5, 0.25, 0.125, 0.05):
        p = bankroll_path(d, fraction=f)
        ax[0].plot(np.arange(len(p)), p["log_bank"].values / np.log(10), lw=1.2, label=f"{f:g} Kelly")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xlabel("races bet (chronological)")
    ax[0].set_ylabel("log10 bank   (0 = break-even, -2 = 1% of bank left)")
    ax[0].set_title("Kelly on model probabilities, settled at BSP")
    ax[0].legend(); ax[0].grid(alpha=.3)

    def curve(sub, label):
        s = sub.sort_values(["race_date", "race_time"]) if "race_time" in sub.columns else sub.sort_values("race_date")
        ax[1].plot(np.arange(len(s)), s["ret_back"].cumsum().values, lw=1.2, label=label)
    top = d[d["model_rank"] == 1]
    curve(top, f"model #1, all races (n={len(top):,})")
    big = top[top["field"] >= 12]
    curve(big, f"model #1, field >= 12 (n={len(big):,})")
    longp = top[top["model_price"] >= 8]
    curve(longp, f"model #1, forecast price >= 8 (n={len(longp):,})")
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("bets (chronological)")
    ax[1].set_ylabel("cumulative profit (units, 1 unit per bet)")
    ax[1].set_title("Flat stakes on the model's top pick"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"chart written to {path}")


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


def _live_clv_frame(live_dir: str, db_path: str) -> pd.DataFrame:
    """Morning predictions with the prices they were made at, joined to the result.

    This is the only route to an honest closing-line number. `race_results.odds`
    is the returned SP, so the database holds two closing prices and no early
    one; `scripts/db_info.py --odds-check` is what establishes that. The
    morning job writes the racecard price and an exchange snapshot beside each
    forecast, and those files -- once the races have run and the BSP is in the
    database -- are the first data that can answer whether betting earlier on
    this signal is worth anything.

    ``live_dir`` is a directory of the morning job's predictions CSVs
    (``aws s3 cp --recursive s3://<bucket>/predictions <dir>``); files are read
    recursively, so the S3 layout can be mirrored as-is.
    """
    import glob
    from betfair_prices import normalise_horse, normalise_track, race_time_to_24h

    paths = sorted(glob.glob(os.path.join(live_dir, "**", "*.csv"), recursive=True))
    if not paths:
        return pd.DataFrame()
    # race_time must stay text. "2.30" read as a number becomes 2.3, which
    # db_time_to_24h cannot parse -- it needs two digits after the point --
    # so every row would join on None and the report would be empty with
    # nothing to say why.
    text = {c: str for c in ("date", "race_date", "venue", "track", "race_time",
                             "runner_name", "horse_name", "market_id")}
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path, dtype=text))
        except Exception as e:                            # noqa: BLE001
            log.warning(f"skipping {path}: {e}")
    if not frames:
        return pd.DataFrame()
    pred = pd.concat(frames, ignore_index=True)
    if "date" in pred.columns and "race_date" not in pred.columns:
        pred["race_date"] = pred["date"]
    if "venue" in pred.columns and "track" not in pred.columns:
        pred["track"] = pred["venue"]
    if "runner_name" in pred.columns and "horse_name" not in pred.columns:
        pred["horse_name"] = pred["runner_name"]
    pred = pred.drop_duplicates(subset=["race_date", "track", "race_time", "horse_name"], keep="last")

    conn = sqlite3.connect(db_path)
    res = pd.read_sql_query(
        "SELECT race_date, track, race_time, horse_name, bfsp, placing_numerical "
        "FROM race_results WHERE bfsp > 1", conn)
    conn.close()

    def key(d, track_col, time_col, horse_col):
        return (d["race_date"].astype(str).str.slice(0, 10) + "|"
                + d[track_col].map(normalise_track) + "|"
                + d[time_col].map(race_time_to_24h).astype(str) + "|"
                + d[horse_col].map(normalise_horse))

    pred["_k"] = key(pred, "track", "race_time", "horse_name")
    res["_k"] = key(res, "track", "race_time", "horse_name")
    d = pred.merge(res[["_k", "bfsp", "placing_numerical"]].drop_duplicates("_k"),
                   on="_k", how="inner")
    if d.empty:
        return d
    d["won"] = pd.to_numeric(d["placing_numerical"], errors="coerce") == 1
    d["raceid"] = d["race_date"].astype(str).str.slice(0, 10) + "_" + d["track"].astype(str) + "_" + d["race_time"].astype(str)
    if "bf_total_matched" in d.columns:
        d["morning_vol"] = pd.to_numeric(d["bf_total_matched"], errors="coerce")
    return d


def cmd_clv(args):
    """Closing-line value of early betting: does the BSP forecast beat the morning price?"""
    from betfair_prices import db_time_to_24h, normalise_horse, normalise_track, race_time_to_24h
    from model.clv import clv_report, early_bet_rule, prepare_clv_frame, price_move_model, steam_predictability
    from model.perf_figures import ensure_raceid
    if args.source == "live":
        d = _live_clv_frame(args.live_dir, args.db)
        early = args.early_col if args.early_col != "morningwap" else "racecard_odds"
        have = int(pd.to_numeric(d.get(early, pd.Series(dtype=float)), errors="coerce").gt(1).sum()) if len(d) else 0
        print(f"{len(d):,} settled runners from {args.live_dir}; {have:,} carry {early}")
        if have < args.min_rows:
            print(f"\nNot enough yet: closing-line value needs {args.min_rows:,} runners with "
                  f"both an early price and a settled BSP, and nothing historic can supply "
                  f"them -- race_results.odds is the returned SP, a closing price (see "
                  f"`python scripts/db_info.py --odds-check`). This accrues forward from the "
                  f"first morning run that wrote prices.")
            return 0
        d = prepare_clv_frame(d, bsp_col="bfsp", early_col=early, pred_col="predicted_bfsp")
        return _clv_tables(d, args)

    pred = ensure_raceid(pd.read_csv(args.predictions))
    if args.morning_csv:
        mk = pd.read_csv(args.morning_csv)
    elif args.source == "snapshots":
        # earliest live snapshot per runner per day from the daily pipeline's betfair_odds table
        conn = sqlite3.connect(args.db)
        snap = pd.read_sql_query("SELECT race_date, venue, race_time, runner_name, best_back, total_matched, snapshot_at FROM betfair_odds "
                                 "WHERE best_back > 1", conn); conn.close()
        snap = snap.sort_values("snapshot_at").groupby(["race_date", "venue", "race_time", "runner_name"], as_index=False).first()
        # race_time here is Betfair's ISO marketStartTime, not a racecard time:
        # db_time_to_24h does not match it and returned None for every row, so
        # this join produced nothing. race_time_to_24h takes either form.
        mk = pd.DataFrame({"race_date": snap["race_date"].astype(str), "track": snap["venue"], "race_time_24": snap["race_time"].map(race_time_to_24h),
                           "horse_norm": snap["runner_name"].map(normalise_horse), "morningwap": snap["best_back"], "morning_vol": snap["total_matched"],
                           "snapshot_at": snap["snapshot_at"]})
    else:
        conn = sqlite3.connect(args.db)
        mk = pd.read_sql_query("SELECT race_date, track_bf AS track, race_time AS race_time_24, horse_norm, morningwap, ppwap, bsp AS bsp_file, morning_vol, pp_vol "
                               "FROM betfair_prices WHERE market_type = 'win'", conn); conn.close()
    mk["_k"] = mk["race_date"].astype(str) + "|" + mk["track"].map(normalise_track) + "|" + mk["race_time_24"].astype(str) + "|" + mk["horse_norm"]
    pred["_k"] = pred["race_date"].astype(str) + "|" + pred["track"].map(normalise_track) + "|" + pred["race_time"].map(db_time_to_24h) + "|" + pred["horse_name"].map(normalise_horse)
    d = pred.merge(mk.drop(columns=["race_date", "track", "race_time_24", "horse_norm"]), on="_k", how="inner")
    print(f"joined {len(d)} runners / {d['raceid'].nunique()} races with morning prices (of {len(pred)} predictions)")
    if d.empty:
        return 1
    d = prepare_clv_frame(d, bsp_col="bfsp", early_col=args.early_col, pred_col="predicted_bfsp")
    return _clv_tables(d, args)


def _clv_tables(d, args):
    """The report itself, shared by every source of an early price."""
    from model.clv import clv_report, early_bet_rule, price_move_model, steam_predictability
    d, coefs = price_move_model(d, n_folds=args.folds, extra_cols=[c for c in ["vol_share"] if c in d.columns])
    pd.set_option("display.width", 220)
    print("\n=== ln BSP ~ ln p_early + ln p_pred + ln n (walk-forward folds) — β2 > 0 means fundamentals predict the move ===")
    print(coefs.round(4).to_string(index=False))
    print("\nsteam predictability:", {k: round(v, 4) for k, v in steam_predictability(d).items()})
    bets = early_bet_rule(d, min_pred_clv=args.min_clv, commission=args.commission, max_early_odds=args.max_odds)
    rep = clv_report(d, bets, commission=args.commission)
    tab = rep.pop("by_pred_clv_tier")
    print("\n=== CLV report (back at the morning price, green up at BSP) ===")
    for k, v in rep.items():
        print(f"{k:28s} {v}")
    print("\nby predicted-CLV tier (all runners):"); print(tab.round(4).to_string(index=False))


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

    s = sub.add_parser("clv"); s.add_argument("--predictions", default="data/oos_predictions.csv"); s.add_argument("--db", default="horse_racing.db")
    s.add_argument("--morning-csv", default=None, help="alternative to the DB: csv with race_date, track, race_time_24, horse_norm, morningwap, morning_vol")
    s.add_argument("--source", choices=["files", "snapshots", "live"], default="files",
                   help="files: betfair_prices (historic CSVs); snapshots: earliest live betfair_odds snapshot per runner; "
                        "live: the morning job's own predictions CSVs, which carry the price at prediction time")
    s.add_argument("--live-dir", default="data/live_predictions", help="--source live: directory of the morning job's predictions CSVs")
    s.add_argument("--min-rows", type=int, default=1000, help="--source live: refuse to report on fewer settled runners than this")
    s.add_argument("--early-col", default="morningwap"); s.add_argument("--min-clv", type=float, default=0.10); s.add_argument("--max-odds", type=float, default=50.0)
    s.add_argument("--commission", type=float, default=0.05); s.add_argument("--folds", type=int, default=6); s.set_defaults(fn=cmd_clv)

    s = sub.add_parser("price-cal"); s.add_argument("--predictions", default="data/oos_predictions.csv")
    s.add_argument("--folds", type=int, default=6); s.add_argument("--save", default=None)
    s.add_argument("--out", default=None); s.add_argument("--from", dest="date_from", default=None)
    s.add_argument("--price-col", default=None,
                   help="Price to calibrate (default: predicted_bfsp_raw when the file has it, "
                        "which is the booster's output before the race book is normalised)")
    s.set_defaults(fn=cmd_price_cal)

    s = sub.add_parser("stake"); s.add_argument("--predictions", default="data/oos_predictions.csv")
    s.add_argument("--p-col", default="predicted_win_prob_norm"); s.add_argument("--price-col", default="bfsp")
    s.add_argument("--commission", type=float, default=0.05); s.add_argument("--max-rank", type=int, default=6)
    s.add_argument("--bank-chart", default=None); s.add_argument("--out-dir", default=None)
    s.add_argument("--from", dest="date_from", default=None); s.set_defaults(fn=cmd_stake)

    s = sub.add_parser("bets"); s.add_argument("--predictions", default="data/oos_predictions.csv")
    s.add_argument("--p-col", default="predicted_win_prob_norm"); s.add_argument("--price-col", default="bfsp")
    s.add_argument("--commission", type=float, default=0.05); s.add_argument("--blend", type=float, default=0.5)
    s.add_argument("--from", dest="date_from", default=None); s.set_defaults(fn=cmd_bets)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
