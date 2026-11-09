"""Paper-lines scan: cells, commission, resting fill share, the cross-market rule's sides."""

from __future__ import annotations

import pandas as pd

from research.open_close import football_paper_lines_scan as ps


def _line(mid, mt, runner, side="back", kind="cross", bb=2.0, bl=2.02, gc=0.01, pnl=1.0, stake=10.0, cp=60.0,
          filled=0.0, event="A v B"):
    return {"day": "2026-11-07", "market_id": mid, "market_type": mt, "event": event, "runner": runner,
            "selection_id": hash((mid, runner)) % 10_000, "kickoff_utc": "2026-11-07T15:00:00Z", "checkpoint_min": cp,
            "kind": kind, "side": side, "best_back": bb, "best_lay": bl, "green_close": gc, "pnl": pnl,
            "stake": stake, "filled": filled, "spread_ticks": 1, "close_back": bb, "close_lay": bl,
            "depth_back_gbp": 100.0}


def test_cell_commission_and_fill_share():
    df = pd.DataFrame([_line("1", "MATCH_ODDS", "A", gc=0.1), _line("2", "MATCH_ODDS", "A", gc=-0.1)])
    c = ps.cell(df)
    assert abs(c["greened_at_close"]["mean"] - (0.1 * 0.98 - 0.1) / 2) < 1e-9
    r = pd.DataFrame([_line("1", "MATCH_ODDS", "A", kind="rest", filled=10.0),
                      _line("2", "MATCH_ODDS", "A", kind="rest", filled=0.0)])
    assert ps.cell(r)["filled_share"] == 0.5


def test_cross_market_rule_backs_the_long_follower():
    rows = []
    # match odds: H 0.5, D 0.25, A 0.25 at mids; double chance prices 'Home or Draw' too long (chance 0.70 < 0.75)
    for r, p in (("A", 2.0), ("The Draw", 4.0), ("B", 4.0)):
        for side in ("back", "lay"):
            rows.append(_line("mo", "MATCH_ODDS", r, side=side, bb=p, bl=p))
    for r, p in (("Home or Draw", 1 / 0.70 * 1.0), ("Draw or Away", 1 / 0.55), ("Home or Away", 1 / 0.75)):
        for side in ("back", "lay"):
            rows.append(_line("dc", "DOUBLE_CHANCE", r, side=side, bb=p, bl=p))
    out = ps.cross_market(pd.DataFrame(rows))
    assert out["events_with_both_books"] == 1
    assert out["gap_over_1pt_at_60min"]["lines"] >= 1


def test_tradable_drops_empty_closing_books():
    ok = pd.DataFrame([_line("1", "MATCH_ODDS", "A"), dict(_line("2", "MATCH_ODDS", "A"), close_back=1.01,
                                                           close_lay=1000.0)])
    assert ps.tradable(ok).tolist() == [True, False]
