"""The cricket trader stays in Phase 0 until the owner says go at a gate (build spec 2, 8; CLAUDE.md Rules).

Nothing under cricket/ or its workflows may reach an order, a Betfair key or the live horse jobs; and the live horse
jobs carry nothing from cricket.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
CRICKET_WORKFLOWS = sorted(WORKFLOWS.glob("cricket-*.yml"))
FORBIDDEN_CODE = ("placeOrders", "place_orders", "betfair_client", "auto_trade", "from trading", "import trading",
                  "BETFAIR_LIVE_APP_KEY", "BETFAIR_APP_KEY", "BETFAIR_PASSWORD", "BETFAIR_USERNAME")
SECRET_SHAPES = (re.compile(r"AKIA[0-9A-Z]{16}"), re.compile(r"(?i)(password|secret|app_key)\s*=\s*['\"][^'\"]+"))


def test_cricket_code_has_no_route_to_an_order_or_a_betfair_key():
    files = list((ROOT / "cricket").rglob("*.py"))
    assert files
    for p in files:
        text = p.read_text()
        for bad in FORBIDDEN_CODE:
            assert bad not in text, f"{p.relative_to(ROOT)} mentions {bad}"
        for pat in SECRET_SHAPES:
            assert not pat.search(text), f"{p.relative_to(ROOT)} holds something shaped like a secret"


def test_cricket_workflows_take_no_betfair_secret():
    assert {p.name for p in CRICKET_WORKFLOWS} >= {"cricket-ingest.yml", "cricket-model.yml"}
    for p in CRICKET_WORKFLOWS:
        text = p.read_text()
        assert "BETFAIR" not in text, p.name
        assert "live-trade" not in text and "auto_trade" not in text, p.name


def test_the_live_horse_jobs_carry_nothing_from_cricket():
    for name in ("live-trade.yml", "live-record.yml"):
        assert "cricket" not in (WORKFLOWS / name).read_text().lower(), name
