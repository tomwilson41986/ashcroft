"""The NFP and lengths-beaten variants (model/blocks/form_variants.py), value by value.

Hand-built careers and races, each number worked out here from its definition.
tests/test_feature_blocks.py covers the lag and card rules. Mechanics only.
"""
import numpy as np
import pandas as pd
import pytest

from model.blocks import form_variants as fv
from model.shrinkage import shrunk_mean


def _race(day, runners, time="2.30", track="york", dist=8.0, trainer="T1"):
    """runners: (horse, placing or None for a non-finisher, lengths beaten or None)."""
    rows = []
    for h, p, lb in runners:
        rows.append(dict(race_date=pd.Timestamp(day), race_time=time, track=track, horse_name=h,
                         trainer=trainer, number_of_runners=len(runners), dist_furlongs=dist,
                         placing_numerical=np.nan if p is None else float(p),
                         LB=np.nan if lb is None else float(lb)))
    return rows


def _frame(races):
    return pd.DataFrame([r for race in races for r in race])


def _today(out, horse="x"):
    return out[out["horse_name"] == horse].sort_values("race_date").iloc[-1]


def test_nfp_with_a_non_finisher_read_as_last():
    # x: won a 5-runner race, pulled up in a 5-runner race, 3rd of 5; then runs again
    races = [
        _race("2024-01-01", [("x", 1, 0), ("a", 2, 1), ("b", 3, 2), ("c", 4, 3), ("d", 5, 4)]),
        _race("2024-02-01", [("x", None, None), ("a", 1, 0), ("b", 2, 1), ("c", 3, 2), ("d", 4, 3)]),
        _race("2024-03-01", [("x", 3, 2), ("a", 1, 0), ("b", 2, 1), ("c", 4, 3), ("d", 5, 4)]),
        _race("2024-04-01", [("x", None, None), ("a", None, None)]),          # today's card
    ]
    out = fv.build(_frame(races))
    t = _today(out)
    assert t["fv_nfp0_l1"] == pytest.approx(0.5)                 # 3rd of 5
    assert t["fv_nfp0_m3"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)   # the pulled-up run counts, as last
    assert t["fv_nfp0_car"] == pytest.approx(0.5)
    # the plain NFP skips it: (1 + 0.5) / 2 over the known runs, exponentially weighted in runs
    w1, w3 = 2 ** (-1 / 3), 2 ** (-3 / 3)                        # 1 and 3 runs back
    assert t["fv_nfp_e3"] == pytest.approx((w1 * 0.5 + w3 * 1.0) / (w1 + w3))
    # a card race has no result: its runners are unknown, not non-finishers
    m = fv.run_measures(_frame(races))
    assert np.isnan(m["nfp0"][-2:]).all()


def test_the_nfp_scaled_by_the_field_beaten():
    races = [
        _race("2024-01-01", [("x", 1, 0)] + [(f"r{i}", i + 2, i + 1) for i in range(19)]),   # won a 20-runner race
        _race("2024-02-01", [("x", 4, 3), ("a", 1, 0), ("b", 2, 1), ("c", 3, 2)]),           # last of 4
        _race("2024-03-01", [("x", None, None)]),
    ]
    t = _today(fv.build(_frame(races)))
    assert t["fv_nfp_fsw_car"] == pytest.approx((19 * 1.0 + 3 * 0.0) / 22)


def test_lengths_beaten_six_ways():
    # a mile race: x 3rd beaten 4 lengths; the beaten finishers' mean margin (1 + 4 + 7) / 3 = 4
    races = [
        _race("2024-01-01", [("w", 1, 0), ("s", 2, 1), ("x", 3, 4), ("l", 4, 7)], dist=8.0),
        _race("2024-02-01", [(h, None, None) for h in "wsxl"]),
    ]
    out = fv.build(_frame(races))
    x, w = _today(out, "x"), _today(out, "w")
    assert x["fv_lbc_l1"] == 4 and x["fv_lblog_l1"] == pytest.approx(np.log1p(4))
    assert x["fv_lbpf_l1"] == pytest.approx(4 / 8)
    assert x["fv_lbrel_l1"] == pytest.approx(4 / 4)
    assert x["fv_lbpp_l1"] == pytest.approx(4 / 2)                  # two places behind the winner
    assert x["fv_lbw_l1"] == 4
    # the winner: beaten by nothing, and its margin (1 length, the second's) as a negative
    assert w["fv_lbc_l1"] == 0 and w["fv_lbrel_l1"] == 0 and w["fv_lbpp_l1"] == 0
    assert w["fv_lbw_l1"] == pytest.approx(-1.0)


def test_a_tailed_off_run_is_capped():
    races = [_race("2024-01-01", [("w", 1, 0), ("x", 2, 45)]), _race("2024-02-01", [("x", None, None)])]
    t = _today(fv.build(_frame(races)))
    assert t["fv_lbc_l1"] == fv.LB_CAP and t["fv_lbw_l1"] == fv.LB_CAP
    assert t["fv_lblog_l1"] == pytest.approx(np.log1p(45))          # the log keeps the order without a cap


def test_a_debutant_gets_its_trainers_record_and_the_prior_fades_with_runs():
    # trainer T9's runners averaged NFP 0.75 on earlier days; its new horse x has not run
    history = [_race(f"2024-01-{d:02d}", [(f"t{d}", 1, 0), (f"o{d}", 2, 1), (f"p{d}", 3, 2)], trainer="T9")
               for d in range(1, 21)]
    for r in history:                 # only the winners are T9's
        for row in r[1:]:
            row["trainer"] = "O"
    card = _race("2024-02-01", [("x", None, None), ("y", None, None)], trainer="T9")
    out = fv.build(_frame(history + [card]))
    t = _today(out)
    tm, tn = 1.0, 0.0                 # T9: twenty winners, a year's half-life
    days_back = np.arange(12, 32)     # 2024-01-01..20 before 2024-02-01
    wts = 2.0 ** (-days_back / 365.0)
    prior = shrunk_mean(1.0 * wts.sum(), wts.sum(), 0.5, fv.TRAINER_PRIOR_K)
    assert t["fv_nfp_s3"] == pytest.approx(prior)                   # no runs: the prior itself
    assert t["fv_nfp_s8"] == pytest.approx(prior)
    # one win from one run: pulled toward the prior with the weight of 3 (or 8) runs
    more = _race("2024-02-01", [("x", 1, 0), ("y", 2, 1)], trainer="T9")
    again = _race("2024-03-01", [("x", None, None)], trainer="T9")
    t2 = _today(fv.build(_frame(history + [more, again])))
    assert 0.5 < t2["fv_nfp_s8"] < t2["fv_nfp_s3"] < 1.0
