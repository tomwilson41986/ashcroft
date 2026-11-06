"""greyhound/costs.py: one commission figure for every greyhound trader, model and study."""
from greyhound import costs, early_lay, live, model, tradeout
from research.greyhound import stream_lab


def test_one_figure_everywhere_and_the_override():
    c = costs.COMMISSION
    assert model.COM == tradeout.COMMISSION == stream_lab.COMM == early_lay.CONFIG["commission"] == c
    assert early_lay.commission({}) == c and costs.commission({"GREYHOUND_COMMISSION": "0.05"}) == 0.05
    assert costs.commission({"GREYHOUND_COMMISSION": "x"}) == c and costs.commission({"GREYHOUND_COMMISSION": "2"}) == c
    assert live.load_config()["commission"] == costs.commission()
