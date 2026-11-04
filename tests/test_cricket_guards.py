"""The cricket trader stays on paper until the owner says go at a gate (build spec 2, 8; CLAUDE.md Rules).

- No cricket code names a betting method, the repo's horse trader, or the live application key.
- Betfair is reached only through cricket/market/betfair_read.py, whose whitelist holds read methods alone.
- Only the market recorder's workflow takes Betfair credentials, and only the delayed key.
- The live horse jobs carry nothing from cricket.
"""

import re
from pathlib import Path

from cricket.market.betfair_read import READ_METHODS

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
CRICKET_WORKFLOWS = sorted(WORKFLOWS.glob("cricket-*.yml"))
BETTING = ("placeOrders", "cancelOrders", "replaceOrders", "updateOrders", "place_orders")
NEVER = BETTING + ("auto_trade", "from trading", "import trading", "BETFAIR_LIVE_APP_KEY")
SECRET_SHAPES = (re.compile(r"AKIA[0-9A-Z]{16}"), re.compile(r"(?i)(password|secret|app_key)\s*=\s*['\"][^'\"]+"))
READER = ROOT / "cricket" / "market" / "betfair_read.py"


def test_cricket_code_has_no_route_to_an_order_or_the_live_key():
    files = list((ROOT / "cricket").rglob("*.py"))
    assert files
    for p in files:
        text = p.read_text()
        for bad in NEVER:
            assert bad not in text, f"{p.relative_to(ROOT)} mentions {bad}"
        for pat in SECRET_SHAPES:
            assert not pat.search(text), f"{p.relative_to(ROOT)} holds something shaped like a secret"
        if p != READER:
            assert "betfair_client" not in text and "_api_call" not in text, (
                f"{p.relative_to(ROOT)}: Betfair only through cricket/market/betfair_read.py")


def test_the_reader_whitelist_reads_only():
    assert READ_METHODS and all(m.startswith("list") for m in READ_METHODS)
    assert "listClearedOrders" not in READ_METHODS and "listCurrentOrders" not in READ_METHODS


def test_only_the_recorder_workflow_takes_betfair_credentials_and_only_the_delayed_key():
    assert {p.name for p in CRICKET_WORKFLOWS} >= {"cricket-ingest.yml", "cricket-model.yml", "cricket-record.yml"}
    for p in CRICKET_WORKFLOWS:
        text = p.read_text()
        assert "BETFAIR_LIVE_APP_KEY" not in text, p.name
        assert "live-trade" not in text and "auto_trade" not in text and "--live" not in text, p.name
        if p.name != "cricket-record.yml":
            assert "BETFAIR" not in text, p.name
    rec = (WORKFLOWS / "cricket-record.yml").read_text()
    assert "secrets.BETFAIR_APP_KEY" in rec and "python -m cricket.market.recorder" in rec


def test_the_live_horse_jobs_carry_nothing_from_cricket():
    for name in ("live-trade.yml", "live-record.yml"):
        assert "cricket" not in (WORKFLOWS / name).read_text().lower(), name
