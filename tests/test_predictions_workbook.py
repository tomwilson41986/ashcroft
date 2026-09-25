"""scripts/predictions_workbook.py: a day's predictions as a workbook. Synthetic rows, mechanics only."""
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from predict_bfsp_today import OUTPUT_COLS, with_model_rank
from scripts import predictions_workbook as pw


def _day():
    rng = np.random.default_rng(0)
    rows = []
    for t, track in [("1.30", "Newmarket"), ("2.05", "Haydock"), ("12.50", "Newmarket")]:
        p = rng.dirichlet(np.ones(5))
        for k in range(5):
            rows.append(dict(race_date="2026-09-25", race_time=t, track=track, race_name="H", horse_name=f"{track}{t}{k}",
                             predicted_bfsp=float(1 / p[k]), predicted_win_prob_norm=float(p[k]), odds=float(1.1 / p[k]),
                             raceid=f"{track}{t}", number_of_runners=5))
    return pd.DataFrame(rows)


def test_the_rank_and_the_columns():
    out = with_model_rank(_day())
    assert out.groupby("raceid")["model_rank"].min().eq(1).all()
    first = out.loc[out.groupby("raceid")["predicted_bfsp"].idxmin()]
    assert (first["model_rank"] == 1).all()
    assert {"model_rank", "predicted_bfsp", "jockey_name", "odds"} <= set(OUTPUT_COLS)


def test_the_workbook(tmp_path):
    path = pw.build(with_model_rank(_day()), {"feature_cols": ["x"] * 615, "best_iteration": 6000},
                    str(tmp_path / "p.xlsx"), note="test")
    wb = load_workbook(path)
    assert wb.sheetnames == ["Read me", "Top picks", "All runners", "Newmarket", "Haydock"]
    top = wb["Top picks"]
    assert [top.cell(row=r, column=1).value for r in range(2, 5)] == ["12.50", "1.30", "2.05"]   # in running order
    assert str(top.cell(row=2, column=13).value).startswith("=IF(")                             # a formula, not a value
    allr = wb["All runners"]
    assert allr.max_row == 16 and str(allr.cell(row=2, column=allr.max_column).value).startswith("=IF(")
