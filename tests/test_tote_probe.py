"""The Tote probe's readers on made-up pages (code paths only; nothing here is a Tote page or reported)."""

import json

import pytest

import tote_probe as tp

PAGE = """<html><head><title>Racecards | Tote</title>
<script id="__NEXT_DATA__" type="application/json">{"props": {"pageProps": {"meetings": [{"races": [
 {"raceId": 7, "pools": [{"poolType": "WIN", "poolTotal": 1234.5, "approxDividends": [3.2]}]}]}]}}}</script>
<script src="/_next/static/chunks/main.js"></script>
<script>fetch("/api/v2/racecards?date=today"); var g = "https://api.example.test/graphql"; var img = "/img/a.png";</script>
</head><body>
<a href="/racecards/kempton/2026-10-06/1400">Kempton 2:00</a>
<a href="https://tote.co.uk/results">Results</a>
<a href="https://othersite.test/racecards">elsewhere</a>
<a href="/about-us">About</a>
</body></html>"""


def test_the_page_data_is_read():
    data = tp.embedded_json(PAGE)
    assert list(data) == ["next_data"]
    keys = tp.interesting_keys(data["next_data"])
    assert "props.pageProps.meetings" in keys
    assert "props.pageProps.meetings[0].races[0].pools[0].poolTotal" in keys
    assert "props.pageProps.meetings[0].races[0].pools[0].approxDividends" in keys


def test_api_addresses_and_links():
    apis = tp.api_candidates(PAGE)
    assert "https://tote.co.uk/api/v2/racecards?date=today" in apis
    assert "https://api.example.test/graphql" in apis
    assert not any(a.endswith(".png") for a in apis)
    links = tp.site_links(PAGE)
    assert links == ["https://tote.co.uk/racecards/kempton/2026-10-06/1400", "https://tote.co.uk/results"]


def test_the_region_block_is_recognised():
    assert tp.blocked("<html><head><title>Unavailable in Region</title></head></html>")
    assert not tp.blocked(PAGE)


class _Response:
    def __init__(self, url, text, status=200, ctype="text/html"):
        self.url, self.text, self.status_code = url, text, status
        self.content = text.encode()
        self.headers = {"content-type": ctype}
        self.ok = status < 400

    def json(self):
        return json.loads(self.text)


def _fake_site(robots):
    pages = {
        "https://tote.co.uk/robots.txt": _Response("https://tote.co.uk/robots.txt", robots, ctype="text/plain"),
        "https://tote.co.uk/": _Response("https://tote.co.uk/", PAGE),
        "https://tote.co.uk/racecards": _Response("https://tote.co.uk/racecards", PAGE),
        "https://tote.co.uk/api/v2/racecards?date=today": _Response(
            "https://tote.co.uk/api/v2/racecards?date=today", '{"races": [{"pools": []}]}', ctype="application/json"),
    }
    asked = []

    def get(self, url, headers=None, timeout=None, allow_redirects=True):
        asked.append(url)
        return pages.get(url) or _Response(url, "not here", status=404)
    return get, asked


def test_a_run_keeps_to_robots_and_writes_its_summary(tmp_path, monkeypatch):
    get, asked = _fake_site("User-agent: *\nDisallow: /results\nDisallow: /racecards/\n")
    monkeypatch.setattr(tp.requests.Session, "get", get)
    s = tp.Probe(tmp_path, max_requests=30, pause=0).run()
    assert not any(u.startswith("https://tote.co.uk/results") for u in asked)
    assert not any(u.startswith("https://tote.co.uk/racecards/") for u in asked)
    skipped = {p["url"] for p in s["pages"] if p.get("skipped")}
    assert "https://tote.co.uk/results" in skipped
    assert s["api"]["https://tote.co.uk/api/v2/racecards?date=today"]["status"] == 200
    assert (tmp_path / "summary.json").exists() and (tmp_path / "robots.txt").read_text().startswith("User-agent")
    assert len(asked) <= 30


def test_a_refused_robots_file_stops_the_probe(tmp_path, monkeypatch):
    get, asked = _fake_site("")
    monkeypatch.setattr(tp.requests.Session, "get",
                        lambda self, url, **kw: _Response(url, "no", status=403) if url.endswith("robots.txt")
                        else get(self, url, **kw))
    s = tp.Probe(tmp_path, max_requests=30, pause=0).run()
    assert s["robots_disallow_all"] is True
    assert all(p.get("skipped") == "robots.txt" for p in s["pages"])


@pytest.mark.parametrize("url,name", [("https://tote.co.uk/", "tote.co.uk.html"),
                                      ("https://tote.co.uk/racecards/a b", "tote.co.uk_racecards_a_b.html")])
def test_file_names(url, name):
    assert tp.safe_name(url, ".html") == name


# --- the API stage (made-up config, scripts and answers; the key is a placeholder) ---------------------------------
KEY = "PLACEHOLDERKEY123"
CONFIG = ('window.config = {sportsbook: {apiUrl: "https://sb.example.test/", apiKey: "other"}, '
          f'tote: {{apiUrl: "https://api.production.racing.tote.co.uk/", apiKey: "{KEY}"}}}};')
INDEX = ('const m=__vite__mapDeps(["assets/results-AbC1.js","assets/header-x.js"]);'
         'let u=`${a.tote.apiUrl}race-card/cards/today?key=${a.tote.apiKey}`;import("./event-results.lazy-Q9.js")')
RESULTS_JS = 'let t=`${c.tote.apiUrl}race-card/results/today?key=${c.tote.apiKey}`,v=`${c.tote.apiUrl}x/${id}?key=`'


def test_the_api_address_and_key_are_read_from_the_config():
    assert tp.api_config(CONFIG) == ("https://api.production.racing.tote.co.uk/", KEY)
    assert tp.api_config("window.config = {}") == (None, None)


def test_the_key_is_redacted():
    assert tp.redact(f"https://h/race-card/pool/9?key={KEY}&x=1", None) == "https://h/race-card/pool/9?key=REDACTED&x=1"
    assert KEY not in tp.redact(f'{{"apiKey": "{KEY}"}}', KEY)


def test_path_templates_and_pools_of_one_race():
    assert tp.path_templates(RESULTS_JS) == ["race-card/results/today?key=${c.tote.apiKey}", "x/${id}?key="]
    pools = [{"id": 1, "poolType": {"name": "Exacta"}, "raceId": 7}, {"id": 2, "poolType": {"name": "Win"}, "raceId": 7},
             {"id": 3, "poolType": {"name": "Win"}, "raceId": 8}, {"id": 4, "poolType": {"name": "Placepot"}, "raceId": 7}]
    assert [p["id"] for p in tp.pools_of_one_race(pools)] == [2, 1]
    assert tp.shape({"a": [{"b": 1}, {"b": 2}], "c": "x" * 50}) == {"a": ["len 2", {"b": 1}], "c": "str(50)"}


def test_an_api_run_keeps_the_key_out_of_every_file(tmp_path, monkeypatch):
    api = "https://api.production.racing.tote.co.uk/"
    answers = {
        "https://tote.co.uk/robots.txt": _Response("", "User-agent: *\nDisallow: /account\n", ctype="text/plain"),
        "https://tote.co.uk/config.js": _Response("", CONFIG, ctype="text/javascript"),
        "https://tote.co.uk/": _Response("", '<script type="module" src="/assets/index-Zz.js"></script>'),
        "https://tote.co.uk/assets/index-Zz.js": _Response("", INDEX, ctype="text/javascript"),
        "https://tote.co.uk/assets/results-AbC1.js": _Response("", RESULTS_JS, ctype="text/javascript"),
        f"{api}race-card/cards/today?key={KEY}": _Response("", '[{"races": [{"raceId": 7}]}]', ctype="application/json"),
        f"{api}race-card/pools/today?key={KEY}": _Response(
            "", '{"pools": [{"id": 2, "poolType": {"name": "Win"}, "raceId": 7, "status": "OPEN"},'
                ' {"id": 1, "poolType": {"name": "Exacta"}, "raceId": 7}]}', ctype="application/json"),
        f"{api}race-card/pool/2?key={KEY}": _Response("", '{"id": 2, "total": 100}', ctype="application/json"),
        f"{api}race-card/pool/1?key={KEY}": _Response("", '{"id": 1, "total": 40}', ctype="application/json"),
        f"{api}race-card/results/today?key={KEY}": _Response("", '[{"dividends": []}]', ctype="application/json"),
    }
    asked = []

    def get(self, url, headers=None, timeout=None, allow_redirects=True):
        asked.append(url)
        r = answers.get(url) or _Response(url, "not here", status=404)
        r.url = url
        return r
    monkeypatch.setattr(tp.requests.Session, "get", get)
    s = tp.ApiProbe(tmp_path, max_requests=30, pause=0).run()
    assert s["api_url"] == api and s["key_found"] is True
    got = {e["path"]: e.get("status") for e in s["endpoints"]}
    assert got["race-card/cards/today?key=REDACTED"] == 200
    assert got["race-card/pool/2?key=REDACTED"] == 200 and got["race-card/pool/1?key=REDACTED"] == 200
    assert got["race-card/results/today?key=REDACTED"] == 200
    assert s["pool_types_today"] == {"Win": 1, "Exacta": 1}
    assert "https://tote.co.uk/assets/results-AbC1.js" in asked
    for f in tmp_path.iterdir():                       # every file: pages, answers, config and summary
        assert KEY not in f.read_text(), f.name
    assert len(asked) <= 30
