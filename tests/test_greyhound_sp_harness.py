"""research/greyhound/sp_harness.py: settlement at the SP to the penny, the day-cluster reading, the cells."""

import numpy as np
import pandas as pd
import pytest

from research.greyhound import sp_harness as sh


def test_settlement_nets_a_race_and_takes_commission_off_a_gain_only():
    b = pd.DataFrame({"day": ["d1", "d1", "d1", "d2"], "race_id": [1, 1, 2, 3], "side": ["back", "back", "lay", "lay"],
                      "bsp": [3.0, 5.0, 4.0, 4.0], "won": [1, 0, 0, 1]})
    r = sh.settle(b, 0.02).set_index("race_id")
    # race 1: a back of 1 wins at 3.0 (+2), a back of 1 loses (-1): net +1 before commission, 0.98 after; 2 at risk
    assert r.loc[1, "gross"] == pytest.approx(1.0) and r.loc[1, "net"] == pytest.approx(0.98) and r.loc[1, "risk"] == 2
    # race 2: a lay risking 1 at 4.0 (stake 1/3) wins its stake; race 3: the same lay loses its liability, no commission
    assert r.loc[2, "net"] == pytest.approx(0.98 / 3) and r.loc[3, "net"] == pytest.approx(-1.0)


def test_the_reading_is_a_ratio_with_the_day_as_the_cluster():
    rt = sh.ratio_t(["a", "a", "b", "c"], [1.0, 1.0, -1.0, 0.5], [1.0, 1.0, 1.0, 1.0])
    assert rt["days"] == 3 and rt["ratio"] == pytest.approx(1.5 / 4) and rt["lo"] < rt["ratio"] < rt["hi"]
    one = sh.ratio_t(["a"], [1.0], [1.0])
    assert one["ratio"] == 1.0 and one["lo"] is None and sh.ratio_t([], [], [])["ratio"] is None
    days = np.repeat([f"d{i:02d}" for i in range(40)], 5)
    per = pd.DataFrame({"day": days, "race_id": range(len(days)), "risk": 1.0, "net": 0.02})
    out = sh.read(per, 100)
    assert out["roi_%"] == pytest.approx(2.0) and out["bets"] == 200 and out["days"] == 40 and out["n_trials"] == 100


def _frame():
    """Two priced six-runner races a day over three periods; dog ids repeat so later races hold experienced dogs."""
    rows, rid = [], 0
    for day in ("2018-03-01", "2018-03-02", "2023-03-01", "2023-03-02", "2023-03-03"):
        for _ in range(2):
            rid += 1
            for trap in range(1, 7):
                rows.append({"race_id": rid, "dog_id": trap + (100 if rid % 2 else 0), "day": day, "bsp": 6.0,
                             "bsp_p_norm": 1 / 6, "won": int(trap == 1)})
    f = pd.DataFrame(rows)
    f["runs_before"] = f.groupby("dog_id").cumcount()
    first = f.runs_before == 0
    f["cell"] = np.select([first, f.runs_before <= 2, f.runs_before <= 5], ["first_run", "runs_1_2", "runs_3_5"], "runs_6_plus")
    f["rival_of_first_run"] = first.groupby(f.race_id).transform("any") & ~first
    return f


def test_every_cell_and_side_is_read_on_every_period_and_a_pass_needs_both():
    res = sh.run(_frame(), 0.02, 10)
    assert [r["rule"] for r in res["rules"]] == [f"{s}_{c}" for c in sh.CELLS for s in sh.SIDES]
    first = next(r for r in res["rules"] if r["rule"] == "back_first_run")
    assert first["unseen"]["bets"] == 12 and first["development"]["bets"] == 0       # every first run is in 2018 here
    assert res["cells"]["first_run"]["unseen"]["ae_at_the_bsp"] == pytest.approx(1.0)
    assert all(r["passes_the_unseen_block"] is False or r["chosen_on_development"] for r in res["rules"])
    assert "Chosen on development" in sh.markdown({**res, "frame": {}})
    assert sh.cell_rows(_frame(), "rival_of_first_run").empty                       # a first-run race here holds only first runs
