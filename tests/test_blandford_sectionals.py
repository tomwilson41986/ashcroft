"""blandford_sectionals.py: the parsers and the within-race readings, on made-up rows (mechanics only)."""
import numpy as np
import pandas as pd

import blandford_sectionals as bs


def _atr_row(horse, pos, fin_pct, early, s21, s1f, racename="Newcastle", date="12-09-2025", time="18:30"):
    return {"Position": str(pos), "Dist Btn": 1.5, "Horse": horse, "SP": "'12/1'", "Time": time, "Racename": racename,
            "Date": date, "Type": "A Handicap", "Runners": "4 ran", "Distance": "6f", "Ground": "Standard",
            "Horse Finish %": str(fin_pct), "Early Speed (MPH)": str(early), "Mid Speed (MPH)": "38.0",
            "Final Speed (MPH)": "37.1", "Top Speed (MPH)": "41.0", "2f-1f": s21, "1f-Finish": s1f,
            "Start-5f": "12.10 Rear", "Finish": "1m 10.2s"}


def test_parse_atr_keeps_uk_and_irish_rows_and_reads_the_fields():
    rows = [_atr_row("what what what", 1, 101.2, 39.5, 11.9, 12.4),
            _atr_row("popova", 2, 99.1, 40.1, 12.1, 12.8, racename="Naas (IRE)"),
            _atr_row("zia horse", 3, 98.0, 38.0, 12.0, 12.7, racename="Zia Park (USA)")]
    d = bs.parse_atr(rows)
    assert len(d) == 2 and set(d.track) == {"Newcastle", "Naas"}
    r = d[d.horse_norm == "whatwhatwhat"].iloc[0]
    assert r.day == "2025-09-12" and r.race_time == "18:30" and r.pos == 1 and r.runners == 4
    assert abs(r.finish_pct - 101.2) < 1e-9 and abs(r.early_mph - 39.5) < 1e-9
    assert abs(r.seg_2f_1f - 11.9) < 1e-9 and abs(r.seg_1f_fin - 12.4) < 1e-9
    assert "Start-5f" in r.segments


def test_parse_rtv_reads_the_url_and_converts_feet():
    rows = [{"RaceURL": "https://www.racingtv.com/results/2025-05-17/newbury/1435", "Position": "4th",
             "horseName": "notable speech", "Finishing Speed % Value": "104.2%", "Finishing Speed % Pos": "2ND",
             "0-20MPH Value": "2.98S", "0-20MPH Pos": "5TH", "Top Speed Value": "41.72MPH", "Top Speed Pos": "3RD",
             "Stride Length Value": "24.5FT", "Stride Length Pos": "1ST", "Avg Frequency Value": "2.31", "Notes": "x"},
            {"RaceURL": "no url", "horseName": "nobody"}]
    d = bs.parse_rtv(rows)
    assert len(d) == 1
    r = d.iloc[0]
    assert r.day == "2025-05-17" and r.track_slug == "newbury" and r.race_time == "14:35" and r.pos == 4
    assert abs(r.fsp_pct - 104.2) < 1e-9 and r.fsp_rank == 2 and abs(r.zero20_s - 2.98) < 1e-9
    assert abs(r.stride_m - 24.5 * 0.3048) < 1e-9


def test_within_race_readings_are_against_the_field():
    h = pd.DataFrame({"race": ["a"] * 3 + ["b"] * 2, "pos": [1, 2, 3, 1, 2],
                      "f": [102.0, 100.0, 98.0, 101.0, 99.0], "e": [40.0, 39.0, 41.0, 38.0, 39.0],
                      "l": [24.0, 24.5, 25.0, 24.0, 24.0]})
    w = bs._within(h, h.race, "f", "e", False, "l", None)
    assert np.allclose(w.fsp_rel[:3], [2.0, 0.0, -2.0])
    assert w.early_pct[2] == 0.0 and w.early_pct[1] == 1.0          # the quickest early is in front (0)
    assert np.allclose(w.late_rel[:3], [-0.5, 0.0, 0.5])
    assert w.n_field[3] == 2 and np.isnan(w.fsp_rel[3])             # a field of two reads nothing
    assert np.isnan(w.sl_rel).all()
