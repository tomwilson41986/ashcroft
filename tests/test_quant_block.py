"""The quant block (model/blocks/quant.py): the Sharpe ratio, signed chi and A/E of an entity's record against
the Betfair SP, and the horse's form read as a return series, on small hand-built histories with known answers.

Mechanics only; the block is scored on the real history by the research loop. The lag, card, order and READS
rules are checked for every block by tests/test_feature_blocks.py."""

import numpy as np
import pandas as pd
import pytest

from model.blocks import quant as q


def _runs(rows):
    """One runner per race unless said otherwise: (day, trainer, horse, placing, bsp, field)."""
    out = []
    for i, (day, trainer, horse, placing, bsp, field) in enumerate(rows):
        date = pd.Timestamp("2025-03-01") + pd.Timedelta(days=day)
        race = f"r{i}"
        out.append({"raceid": race, "race_date": date, "race_time": "2.30", "track": "York", "horse_name": horse,
                    "trainer": trainer, "jockey_name": "J", "stallion": "S", "placing_numerical": placing,
                    "bfsp": bsp, "total_dst_bt": "0" if placing == 1 else "2", "dist_furlongs": 8.0})
        for k in range(field - 1):                     # the rest of the field: equal prices, the others' chances
            out.append({"raceid": race, "race_date": date, "race_time": "2.30", "track": "York",
                        "horse_name": f"x{i}_{k}", "trainer": f"O{i}_{k}", "jockey_name": f"K{k}",
                        "stallion": f"Z{k}", "placing_numerical": (2 if placing == 1 else 1) + k,
                        "bfsp": bsp, "total_dst_bt": "1", "dist_furlongs": 8.0})
    return pd.DataFrame(out)


def test_the_entity_record_against_the_prices():
    # the trainer's six runs, each in a five-runner race at level prices (every chance 0.2): winners on days 0 and 2
    hist = [(d, "T", f"h{d}", 1 if d in (0, 2) else 3, 5.0, 5) for d in range(6)] + [(10, "T", "today", np.nan, np.nan, 1)]
    df = _runs(hist)
    df.loc[df["race_date"] == pd.Timestamp("2025-03-11"), "placing_numerical"] = np.nan
    v = q.run_values(df)
    key = q._codes([df["trainer"]])
    day = q.day_index(df)
    m = q.entity_metrics(key, day, v, np.inf)
    today = (df["horse_name"] == "today").to_numpy()
    wins, expected, var = 2.0, 6 * 0.2, 6 * 0.2 * 0.8
    assert m["z"][today][0] == pytest.approx((wins - expected) / np.sqrt(var))
    assert m["ae"][today][0] == pytest.approx((wins + q.K_AE) / (expected + q.K_AE))
    assert m["n"][today][0] == pytest.approx(6.0)
    r = np.array([4 * 0.95, -1, 4 * 0.95, -1, -1, -1])
    shrunk = (r.sum() + q.MU0 * q.K_SH) / (6 + q.K_SH)
    assert m["sh"][today][0] == pytest.approx(shrunk / r.std())


def test_a_days_own_runs_never_enter_its_record():
    # two runners of the yard on the same day read the same record: the day before's, not each other's
    hist = [(0, "T", "a", 1, 3.0, 4), (1, "T", "b", 1, 3.0, 4), (1, "T", "c", 5, 3.0, 6)]
    df = _runs(hist)
    m = q.entity_metrics(q._codes([df["trainer"]]), q.day_index(df), q.run_values(df), 365.0)
    b, c = (df["horse_name"] == "b").to_numpy(), (df["horse_name"] == "c").to_numpy()
    assert m["n"][b][0] == pytest.approx(m["n"][c][0]) and m["z"][b][0] == pytest.approx(m["z"][c][0])
    assert m["n"][b][0] == pytest.approx(2 ** (-1 / 365.0))


def test_decay_weights_the_chi_by_the_squared_weights():
    # one win a year ago and one loss yesterday: the variance term uses the weights squared
    hist = [(0, "T", "a", 1, 2.0, 2), (365, "T", "b", 2, 2.0, 2), (366, "T", "today", np.nan, np.nan, 1)]
    df = _runs(hist)
    df.loc[df["horse_name"] == "today", "placing_numerical"] = np.nan
    m = q.entity_metrics(q._codes([df["trainer"]]), q.day_index(df), q.run_values(df), 365.0)
    t = (df["horse_name"] == "today").to_numpy()
    w1, w2 = 2 ** (-366 / 365.0), 2 ** (-1 / 365.0)
    num = w1 * (1 - 0.5) + w2 * (0 - 0.5)
    den = np.sqrt((w1 ** 2 + w2 ** 2) * 0.25)
    assert m["z"][t][0] == pytest.approx(num / den)


def test_the_horse_series():
    # a horse's runs, oldest first: pounds beaten 4, 0 (won), pulled up (20), 2, 6; then today
    lbs = np.array([4.0, 0.0, 20.0, 2.0, 6.0, np.nan])
    lsp = np.log(np.array([10.0, 8.0, 6.0, 5.0, 4.0, np.nan]))
    horse = np.zeros(6, dtype=np.int64)
    day = np.arange(6, dtype=np.int64) * 10
    out = q.series_metrics(horse, day, np.arange(6), lbs, lsp)
    last5 = np.array([4.0, 0.0, 20.0, 2.0, 6.0])
    assert out["qm_hs_lbs_sd5"][5] == pytest.approx(last5.std())
    up = np.maximum(last5 - last5.mean(), 0)
    assert out["qm_hs_lbs_dn5"][5] == pytest.approx(np.sqrt((up ** 2).mean()))
    assert out["qm_hs_lbs_best5"][5] == 0.0
    assert out["qm_hs_lbs_dd10"][5] == pytest.approx(6.0)        # the last run, 6 lb, against the best, 0
    assert out["qm_hs_since_best10"][5] == 3.0                    # the win was four runs back: three since
    assert out["qm_hs_mkt_sd5"][5] == pytest.approx(lsp[:5].std())
    slope = np.polyfit(np.arange(5), lsp[:5], 1)[0]
    assert out["qm_hs_mkt_tr5"][5] == pytest.approx(slope)        # shortening: negative
    assert out["qm_hs_mkt_tr5"][5] < 0
    # two runs back is not enough for a volatility; the first run has no history at all
    assert np.isnan(out["qm_hs_lbs_sd5"][2]) and np.isnan(out["qm_hs_lbs_best5"][0])


def test_the_block_names_every_feature_and_fills_them():
    hist = [(d, f"T{d % 2}", f"h{d % 3}", 1 + (d % 4), 2.0 + d % 5, 4) for d in range(40)]
    df = _runs(hist)
    out = q.build(df)
    assert list(out.columns[-len(q.FEATURES):]) == q.FEATURES
    assert len(out) == len(df)
    for c in q.FEATURES:
        assert out[c].notna().any(), c
