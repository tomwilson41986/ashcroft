"""research/greyhound/take_close_review.py on a constructed ledger (unit test only)."""
import pandas as pd

from research.greyhound.take_close_review import review


def test_review_reads_the_drift():
    led = pd.DataFrame({"action": ["back_matched", "back_matched", "sp_lay"], "market_id": ["1", "2", "1"],
                        "selection_id": [5, 6, 5], "price": [4.0, 6.0, None], "matched": [10, 10, None],
                        "f_bsp": [3.6, 5.5, None], "mid": [3.8, 5.8, None], "back1": [4.0, 6.0, None],
                        "edge": [0.11, 0.09, None]})
    close = pd.DataFrame({"market_id": ["1", "2"], "selection_id": [5, 6], "bsp": [5.0, 5.0], "won": [0, 0]})
    b, out = review(led, close)
    assert out["drifted"] == 1 and out["shortened"] == 1 and out["backs"] == 2
    assert round(b.pnl.iloc[0], 4) == round(-10 + 10 * 3 / 4, 4) and round(b.pnl.iloc[1], 4) == round(-10 + 10 * 5 / 4, 4)
