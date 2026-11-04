"""The traders' dashboard served by the web app: /traders (the page) and /traders.json (what it re-reads). The figures
here are a small hand-made test input, never shown as real."""
from datetime import date

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import web.app as webapp  # noqa: E402
from dashboard.traders import LocalSource, build  # noqa: E402


@pytest.fixture
def data(tmp_path):
    return build(LocalSource(tmp_path), date(2026, 11, 4), cache=False)


def test_the_page_and_its_json_come_from_the_latest_build(monkeypatch, data):
    monkeypatch.setattr(webapp, "get_traders", lambda: data)
    c = TestClient(webapp.app)
    r = c.get("/traders")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<title>racing² Traders</title>" in r.text and r.headers["cache-control"] == "no-store"
    # the page re-reads traders.json relative to itself: /traders -> /traders.json
    assert 'fetch("traders.json' in r.text
    j = c.get("/traders.json")
    assert j.status_code == 200 and j.json()["today"] == "2026-11-04"
    assert [s["id"] for s in j.json()["sports"]] == ["horse_racing", "greyhounds", "football", "cricket"]


def test_no_build_yet_says_no_data_found(monkeypatch):
    monkeypatch.setattr(webapp, "get_traders", lambda: None)
    c = TestClient(webapp.app)
    r = c.get("/traders")
    assert r.status_code == 503 and "No data found" in r.text
    assert c.get("/traders.json").status_code == 503
