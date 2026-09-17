#!/usr/bin/env python3
"""Release notes for a trained BFSP model, from its summary and its metadata.

A release note is the one place a reader learns what the artefact is without
loading it, so it says the recipe out loud: which objective, whether rows were
weighted, what was predicted, how early stopping was decided and whether it
actually fired. The artefact this replaced said none of that -- it recorded a
params dict and no recipe, and a model trained on the race's own result looked
exactly like a model trained honestly.

    python scripts/release_notes.py [summary.json] [meta.json]
"""
import json
import os
import sys


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main():
    summary_path = sys.argv[1] if len(sys.argv) > 1 else "data/models/bfsp_training_summary.json"
    meta_path = (sys.argv[2] if len(sys.argv) > 2
                 else os.path.join(os.path.dirname(summary_path), "bfsp_model_meta.json"))
    s, m = _load(summary_path), _load(meta_path)

    wf = s.get("walk_forward_avg", {})
    owf = s.get("overall_walk_forward", {})
    fm = s.get("final_model", {})

    dr = s.get("data_range", {})
    if isinstance(dr, dict):
        data_range = f"{dr.get('min_date', '?')} to {dr.get('max_date', '?')}"
        total_rows = dr.get("total_rows", s.get("total_rows", "unknown"))
    else:
        data_range = str(dr)
        total_rows = s.get("total_rows", "unknown")

    n_features = m.get("n_features", s.get("n_features", "unknown"))

    print("## BFSP Model Training Summary")
    print()
    print(f"- **Data range**: {data_range}")
    print(f"- **Total rows**: {total_rows:,}" if isinstance(total_rows, int) else f"- **Total rows**: {total_rows}")
    print(f"- **Features**: {n_features}")
    print(f"- **Trained through**: {m.get('trained_through', 'unknown')}")
    print()

    if m:
        w = m.get("sample_weighting", {})
        weighting = w.get("type", "unknown")
        if weighting == "exponential_decay":
            weighting += f" (rate {w.get('rate')}/yr)"
        best, cap = m.get("best_iteration"), m.get("num_boost_round")
        stop = ("early-stopped" if m.get("early_stopped") else
                "hit the cap -- the holdout was still improving")
        print("### Recipe")
        print(f"- **Objective**: {m.get('objective', '?')} "
              f"(LightGBM `{m.get('lightgbm_objective', '?')}`)")
        print(f"- **Target**: {m.get('target', '?')}")
        print(f"- **Row weighting**: {weighting}")
        print(f"- **Early stopping**: last {m.get('holdout_days', '?')} days of the training "
              f"window, never the scored fold")
        print(f"- **Iterations**: {best} of {cap} — {stop}")
        print(f"- **Refit on every row**: {m.get('refit_on_full')}")
        print(f"- **Purge / embargo**: {m.get('purge_days', '?')}d / {m.get('embargo_days', '?')}d")
        print(f"- **Seed**: {m.get('params', {}).get('seed', '?')}")
        print(f"- **Feature code hash**: `{m.get('feature_code_hash') or 'unknown'}`")
        hm = m.get("holdout_metrics") or {}
        if hm:
            # `mae` is in the units of whatever was predicted, which is only
            # log(BFSP) for the default target -- so name the target with it
            # rather than calling every one of them a log MAE.
            n, mae = hm.get("n"), hm.get("mae")
            n_txt = f"{n:,}" if isinstance(n, int) else str(n or "?")
            mae_txt = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae or "?")
            print(f"- **Holdout**: {n_txt} rows from {m.get('holdout_start', '?')}, "
                  f"MAE {mae_txt} on {m.get('target', '?')}")
        top = [n for n, _ in (m.get("top_gain") or [])][:8]
        if top:
            print(f"- **Top features by gain**: {', '.join(top)}")
        print()

    print("### Walk-Forward Validation")
    if wf.get("n_folds"):
        print(f"- Folds: {wf['n_folds']}")
        print(f"- MAE (log): {wf.get('wf_log_mae', 'n/a')}")
        print(f"- R² (log): {wf.get('wf_log_r2', 'n/a')}")
        print(f"- MdAPE: {wf.get('wf_median_ape_pct', 'n/a')}%")
        print(f"- Correlation: {owf.get('overall_wf_correlation', 'n/a')}")
    else:
        print("- Not run in the trainer (`--wf-folds 0`). `evaluate_oos.py` is the")
        print("  evaluation: it fits the same recipe over the same folds and writes")
        print("  the reports, so running it here as well produced a second set of")
        print("  numbers nobody read, at about thirteen minutes a fold.")
    print()
    print("### Final Model (in-sample — not a performance claim)")
    print(f"- MAE (log): {fm.get('log_mae', 'n/a')}")
    print(f"- R² (log): {fm.get('log_r2', 'n/a')}")
    print(f"- MdAPE: {fm.get('median_ape_pct', 'n/a')}%")
    print(f"- Correlation: {fm.get('correlation', 'n/a')}")


if __name__ == "__main__":
    main()
