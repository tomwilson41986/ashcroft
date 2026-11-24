"""research/greyhound/edge_models.py on a constructed panel (unit test only): every arm scored walk-forward, the
reading's clauses present, Benter's test at the decision book reported."""
import numpy as np

from research.greyhound import edge_models as em
from tests.test_greyhound_limit_study import synthetic_panel


def test_every_arm_is_scored():
    p = synthetic_panel(n_days=12, races=25)
    p = p.assign(back2=np.nan, back2_size=np.nan, back3=np.nan, back3_size=np.nan, back1_size=50.0,
                 minutes_to_off=4.0, trap=p.selection_id + 1)
    live = p[p["mark"] == p["mark"].iloc[0]].assign(mark="race_traded")
    rec = p[p["mark"] == "first_traded"]
    out = em.run(rec, live)
    assert len(out["scored_days"]) == 5 and set(out["arms"]) == set(em.ARMS)
    assert out["benter_at_decision"]["races"] > 0
    for r in out["arms"].values():
        if r["backs"]:
            assert set(r["reading"]) == {"clv_interval_above_zero", "calibrated_on_taken", "net_positive",
                                         "dsr_passed", "min_backs_met", "pass"}
