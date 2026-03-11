#!/usr/bin/env python3
"""Generate release notes from BFSP training summary JSON."""
import json
import sys


def main():
    summary_path = sys.argv[1] if len(sys.argv) > 1 else "data/models/bfsp_training_summary.json"
    with open(summary_path) as f:
        s = json.load(f)

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

    print("## BFSP Model Training Summary")
    print()
    print(f"- **Data range**: {data_range}")
    print(f"- **Total rows**: {total_rows:,}" if isinstance(total_rows, int) else f"- **Total rows**: {total_rows}")
    print(f"- **Features**: {s.get('n_features', 187)}")
    print(f"- **Walk-forward folds**: {wf.get('n_folds', 'n/a')}")
    print()
    print("### Walk-Forward Validation")
    print(f"- MAE (log): {wf.get('wf_log_mae', 'n/a')}")
    print(f"- R\u00b2 (log): {wf.get('wf_log_r2', 'n/a')}")
    print(f"- MdAPE: {wf.get('wf_median_ape_pct', 'n/a')}%")
    print(f"- Correlation: {owf.get('overall_wf_correlation', 'n/a')}")
    print()
    print("### Final Model")
    print(f"- MAE (log): {fm.get('log_mae', 'n/a')}")
    print(f"- R\u00b2 (log): {fm.get('log_r2', 'n/a')}")
    print(f"- MdAPE: {fm.get('median_ape_pct', 'n/a')}%")
    print(f"- Correlation: {fm.get('correlation', 'n/a')}")


if __name__ == "__main__":
    main()
