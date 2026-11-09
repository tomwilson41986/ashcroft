"""research/greyhound/capacity.py: the lays sized to each liability, replayed and summarised."""
from datetime import timedelta

import pytest

from research.greyhound import capacity as cap, stream_replay as sr
from tests.test_greyhound_stream_replay import MID, T, delta, image


def test_lays_are_sized_to_the_liability_and_the_summary_reads_the_replay():
    base = [sr.SimOrder(MID, 7, "LAY", 5.0, 5.0, T + timedelta(seconds=1))]
    big = cap.at_liability(base, 100.0)
    assert big[0].size == 25.0 and not big[0].placed and base[0].size == 5.0       # 25 x (5 - 1) = 100
    lines = [image(T, [[5.0, 0]], [[5.2, 8]], [[5.0, 0]]), delta(T + timedelta(seconds=7), trd=[[4.9, 4]])]
    df = sr.replay(lines, big, through_cap=True)
    s = cap.summarise(df, big)
    assert s["liability_offered"] == 100.0 and s["liability_matched"] == pytest.approx(16.0)  # 4 x (5 - 1)
    assert s["share_matched"] == pytest.approx(0.16)
