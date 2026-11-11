"""The desk's control layer (docs/agent_desk.md): the registry, the envelope, the risk kernel, the accounting, the
warden and the research ledger. Every control is shown to fire; nothing here touches the network or S3."""

import json
import re
from pathlib import Path

import pytest

from desk import SPORTS, account, envelope, registry, risk, trials, warden

ROOT = Path(__file__).resolve().parents[1]
FULL_ENV = {"DESK_BANKROLL": "1000", "DESK_DAILY_LOSS": "100", "DESK_SPORT_SHARE": "horse_racing=0.5,greyhounds=0.5",
            "DESK_RESERVE": "0", "DESK_PILOT_BUDGET": "500", "DESK_MAX_MARKET_EXPOSURE": "80"}
LIMITS = {"max_order": 20, "max_day_turnover": 200, "max_day_loss": 50, "max_market_exposure": 40,
          "max_orders_per_minute": 3, "price_collar": 0.2, "max_book_age_s": 5, "min_stake": 1}


def desk_strategy(**over) -> dict:
    gate = {"metric": "clv", "min_days": 30, "min_bets": 300, "interval": 0.9, "dsr": 0.95, "read": "forward"}
    d = {"id": "horse-test", "sport": "horse_racing", "name": "t", "stage": "pilot", "hypothesis": "h",
         "entry_point": "python -m desk.x", "workflow": "desk-x.yml", "order_ref_prefix": "dk1",
         "stop_keys": ["desk/STOP", "trading/STOP"], "kill_switch_fails": "safe", "ledger_tag": "t-1110",
         "gate": gate, "gate_sha256": registry.gate_hash(gate), "limits": dict(LIMITS)}
    d.update(over)
    return d


# --- the registry ----------------------------------------------------------------------------------------------------

def test_the_registry_loads_with_no_error_and_every_sport_has_its_folder():
    strategies, errs = registry.load()
    assert errs == []
    assert len(strategies) == len(list(registry.STRATEGIES.glob("*/*.json"))) >= 8
    assert {p.name for p in registry.STRATEGIES.iterdir() if p.is_dir()} == set(SPORTS)


def test_every_trader_that_can_place_an_order_is_registered_with_the_prefix_its_code_uses():
    by = {s.id: s for s in registry.load()[0]}
    src = lambda p: (ROOT / p).read_text()
    assert 'ref=f"ash{self.day:%m%d}' in src("trading/session.py") and by["horse-morning-clv"].order_ref_prefix == "ash"
    models = re.search(r'MODELS = (\{[^}]*\})', src("morning/live.py")).group(1)
    assert json.loads(models) == {"steamer": by["horse-morning-steamer"].order_ref_prefix,
                                  "quick_return": by["horse-morning-quick-return"].order_ref_prefix}
    assert f'REF = "{by["greyhound-early-lay"].order_ref_prefix}"' in src("greyhound/early_lay.py")
    assert f'REF = "{by["greyhound-quote-mid"].order_ref_prefix}"' in src("greyhound/quote_mid.py")
    assert 'ref = f"gh-' in src("greyhound/live.py") and by["greyhound-win-paper"].order_ref_prefix == "gh-"
    assert 'ref=f"fb-' in src("football/live.py") and by["football-match-odds"].order_ref_prefix == "fb-"


def test_the_registry_covers_every_trader_on_the_dashboard():
    text = (ROOT / "dashboard" / "traders.py").read_text()
    block = text[text.index("TRADERS = ["):text.index("\n]\n", text.index("TRADERS = ["))]
    on_dashboard = set(re.findall(r'Trader\("([^"]+)"', block))
    registered = {d for s in registry.load()[0] for d in s.dashboard_ids}
    assert on_dashboard and on_dashboard <= registered, on_dashboard - registered


def test_the_registry_switches_match_the_workflows():
    for s in registry.load()[0]:
        if not s.desk_run:                       # an owner's trader runs from its workflow; a desk one may have none yet
            assert (ROOT / ".github" / "workflows" / s.workflow).exists(), (s.id, s.workflow)
        if not (ROOT / ".github" / "workflows" / s.workflow).exists():
            continue
        wf = (ROOT / ".github" / "workflows" / s.workflow).read_text()
        if s.live_switch:
            assert s.live_switch in wf, (s.id, s.live_switch)


def test_the_gate_thresholds_are_the_owners():
    from research.overfit import DSR_THRESHOLD, MIN_OBS
    assert registry.MIN_DSR == DSR_THRESHOLD and registry.MIN_DAYS == MIN_OBS


@pytest.mark.parametrize("over,err", [
    ({"order_ref_prefix": "ashX"}, "would adopt"),
    ({"kill_switch_fails": "open"}, "fail safe"),
    ({"gate_sha256": "0" * 64}, "gate changed"),
    ({"limits": {"max_order": 5}}, "needs limits"),
    ({"ledger_tag": None}, "ledger tag"),
    ({"sport": "darts"}, "sport"),
    ({"stage": "live"}, "stage"),
])
def test_a_desk_strategy_that_breaks_a_rule_is_refused(over, err):
    errs = registry.check_one(desk_strategy(**over))
    assert any(err in e for e in errs), errs


@pytest.mark.parametrize("field_,value,err", [("dsr", 0.9, "dsr"), ("min_days", 20, "min_days"),
                                              ("interval", 0.8, "interval"), ("read", "history", "forward")])
def test_a_gate_looser_than_the_owners_rules_is_refused(field_, value, err):
    d = desk_strategy()
    d["gate"][field_] = value
    d["gate_sha256"] = registry.gate_hash(d["gate"])
    assert any(err in e for e in registry.check_one(d))


def test_prefixes_that_could_not_be_told_apart_are_refused(tmp_path):
    for i, p in (("horse-a", "dka"), ("horse-b", "dka2")):
        (tmp_path / "horse_racing").mkdir(exist_ok=True)
        (tmp_path / "horse_racing" / f"{i}.json").write_text(json.dumps(desk_strategy(id=i, order_ref_prefix=p)))
    _, errs = registry.load(tmp_path)
    assert any("could not be told apart" in e for e in errs)


def test_an_order_belongs_to_the_longest_prefix():
    a = registry._strategy(desk_strategy(id="a", order_ref_prefix="dk"))
    b = registry._strategy(desk_strategy(id="b", order_ref_prefix="dkb"))
    assert registry.owner_of("dkb-1", [a, b]).id == "b" and registry.owner_of("dk-1", [a, b]).id == "a"
    assert registry.owner_of("zz", [a, b]) is None and registry.owner_of(None, [a, b]) is None


# --- the envelope ----------------------------------------------------------------------------------------------------

def test_no_envelope_means_paper():
    env, why = envelope.from_env({})
    assert env is None and "paper only" in why


@pytest.mark.parametrize("key,value", [("DESK_BANKROLL", "lots"), ("DESK_DAILY_LOSS", "-5"), ("DESK_DAILY_LOSS", "5000"),
                                       ("DESK_SPORT_SHARE", "horse_racing=0.8,greyhounds=0.5"),
                                       ("DESK_SPORT_SHARE", "darts=0.1"), ("DESK_SPORT_SHARE", "horse_racing"),
                                       ("DESK_RESERVE", "")])
def test_a_malformed_envelope_means_paper(key, value):
    env, why = envelope.from_env({**FULL_ENV, key: value})
    assert env is None and "paper only" in why


def test_a_sound_envelope_is_read():
    env, why = envelope.from_env(FULL_ENV)
    assert why == "" and env.daily_loss == 100 and env.sport_daily_loss("horse_racing") == 50
    assert env.sport_daily_loss("cricket") == 0


# --- the accounting --------------------------------------------------------------------------------------------------

def test_the_worst_case_of_a_market():
    P = account.Position
    assert account.worst_case([P("m", 1, 40, -10)]) == -10                  # a GBP10 back at 5.0
    assert account.worst_case([P("m", 1, -40, 10)]) == -40                  # a GBP10 lay at 5.0
    assert account.worst_case([P("m", 1, 40, -10), P("m", 1, -40, 10)]) == 0  # backed and laid at the same price
    assert account.worst_case([P("m", 1, 40, -10), P("m", 2, 30, -10)]) == -20  # neither wins
    assert account.worst_case([]) == 0


def test_orders_are_put_to_their_strategy_and_the_sp_lay_counts_its_liability():
    strategies = registry.load()[0]
    orders = [{"marketId": "1.1", "selectionId": 7, "side": "BACK", "orderType": "LIMIT", "sizeMatched": 10,
               "averagePriceMatched": 5.0, "customerOrderRef": "ash11107"},
              {"marketId": "1.1", "selectionId": 7, "side": "LAY", "orderType": "MARKET_ON_CLOSE", "bspLiability": 40,
               "customerOrderRef": "ash11107x"},
              {"marketId": "1.2", "selectionId": 3, "side": "BACK", "sizeMatched": 5, "averagePriceMatched": 3.0,
               "customerOrderRef": "mes11103"},
              {"marketId": "1.2", "selectionId": 3, "side": "BACK", "sizeMatched": 5, "averagePriceMatched": 3.0},
              {"marketId": "1.3", "selectionId": 3, "side": "BACK", "sizeMatched": 5, "averagePriceMatched": 3.0,
               "customerOrderRef": "rogue-1"}]
    by = account.by_strategy(orders, strategies)
    assert len(by["horse-morning-clv"]) == 2 and len(by["horse-morning-steamer"]) == 1
    assert len(by[account.UNATTRIBUTED]) == 1 and len(by[account.UNKNOWN]) == 1
    worst = account.market_worst(orders)
    assert worst["1.1"] == -10            # the back's stake: the SP lay's winnings are not counted
    assert worst["1.2"] == -10


# --- the risk kernel -------------------------------------------------------------------------------------------------

def kernel(env_vars=FULL_ENV, funds=1000.0, **over):
    s = registry._strategy(desk_strategy(**over))
    other = registry._strategy(desk_strategy(id="horse-other", order_ref_prefix="dk2"))
    env, _ = envelope.from_env(env_vars)
    k = risk.RiskKernel([s, other], env, clock=lambda: 1000.0)
    k.state.funds_available = funds
    return k


def intent(**over):
    d = dict(strategy_id="horse-test", market_id="1.1", selection_id=7, side="BACK", price=5.0, size=10.0,
             best_price=5.0, book_age_s=1.0)
    d.update(over)
    return risk.Intent(**d)


def test_a_sound_intent_passes_whole():
    v = kernel().check(intent())
    assert v.ok and v.size == 10.0 and v.reason == ""


@pytest.mark.parametrize("make,want", [
    (lambda: (kernel(env_vars={}), intent()), "paper only"),
    (lambda: (kernel(stage="paper"), intent()), "no money"),
    (lambda: (kernel(stage="legacy", approved={"by": "o", "on": "d", "source": "s"}), intent()), "not a desk"),
    (lambda: (kernel(stage="retired"), intent()), "not a desk"),
    (lambda: (kernel(limits={"max_order": 5}), intent()), "limits not set"),
    (lambda: (kernel(), intent(book_age_s=30)), "stale"),
    (lambda: (kernel(), intent(price=7.0)), "collar"),
    (lambda: (kernel(), intent(price=1.0, best_price=1.0)), "price"),
    (lambda: (kernel(), intent(strategy_id="nobody")), "not in the registry"),
    (lambda: (kernel(funds=None), intent()), "funds not read"),
])
def test_each_refusal_fires(make, want):
    k, it = make()
    v = k.check(it)
    assert not v.ok and want in v.reason, v.reason


def test_a_stop_refuses_everything():
    k = kernel()
    k.state.stopped = "trading/STOP"
    assert "stopped" in k.check(intent()).reason


def test_the_order_rate_limit():
    k = kernel()
    for _ in range(3):
        k.state.note_sent(intent(market_id="9.9"), now=999.0)
    assert "order rate" in k.check(intent()).reason


@pytest.mark.parametrize("it,setup,size,why", [
    (dict(size=50.0), None, 20.0, "the order's cap"),
    (dict(), lambda k: k.state.turnover.__setitem__("horse-test", 195.0), 5.0, "day turnover"),
    (dict(), lambda k: k.state.settled.__setitem__("horse-test", -45.0), 5.0, "the strategy's day loss"),
    (dict(), lambda k: k.state.settled.__setitem__("horse-other", -46.0), 4.0, "horse_racing's day loss"),
    (dict(side="LAY"), None, 10.0, ""),                                     # GBP40 liability: the strategy's cap
    (dict(side="LAY", size=20.0), None, 10.0, "the strategy's market exposure"),
    (dict(), lambda k: setattr(k.state, "funds_available", 6.0), 6.0, "the funds"),
])
def test_each_cut(it, setup, size, why):
    k = kernel()
    if setup:
        setup(k)
    v = k.check(intent(**it))
    assert v.size == pytest.approx(size, abs=0.01) and why in v.reason, (v.size, v.reason)


def test_the_desks_market_exposure_counts_every_desk_strategy():
    k = kernel(env_vars={**FULL_ENV, "DESK_DAILY_LOSS": "500"})     # the day loss wide, so the market binds
    k.state.positions["horse-other"].append(account.Position("1.1", 7, -75.0, 25.0))   # another strategy laid it
    v = k.check(intent(side="LAY", size=10.0))      # GBP40 more liability: GBP115 if it wins, over the GBP80
    assert v.size == pytest.approx(1.25, abs=0.01) and "desk's market exposure" in v.reason


def test_the_reserve_for_the_horse_rule_is_kept():
    k = kernel(env_vars={**FULL_ENV, "DESK_RESERVE": "995"})
    v = k.check(intent())
    assert v.size == 5.0 and "reserve" in v.reason
    k = kernel(env_vars={**FULL_ENV, "DESK_RESERVE": "999.5"})
    assert "minimum stake" in k.check(intent()).reason


def test_the_pilot_budget():
    k = kernel(env_vars={**FULL_ENV, "DESK_PILOT_BUDGET": "12"})
    k.state.turnover["horse-other"] = 5.0
    assert k.check(intent()).size == pytest.approx(7.0)


def test_below_the_minimum_after_cuts_sends_nothing():
    k = kernel()
    k.state.turnover["horse-test"] = 199.5
    assert not k.check(intent()).ok


# --- the warden ------------------------------------------------------------------------------------------------------

def test_the_warden_finds_an_unknown_reference_and_a_day_loss():
    strategies = registry.load()[0]
    env, _ = envelope.from_env(FULL_ENV)
    orders = [{"marketId": "1.3", "selectionId": 3, "side": "BACK", "sizeMatched": 5, "averagePriceMatched": 3.0,
               "customerOrderRef": "rogue-1"}]
    cleared = [{"profit": -60.0, "customerOrderRef": "el-1110-x"}, {"profit": -45.0, "customerOrderRef": "ash11107"}]
    findings, fig = warden.assess(strategies, env, "", orders, cleared, {"exposure": -15.0})
    kinds = {f.kind for f in findings if f.level == "breach"}
    assert kinds == {"unknown-reference", "day-loss", "sport-day-loss"}
    assert fig["settled_total"] == -105.0 and fig["settled_by_sport"] == {"greyhounds": -60.0, "horse_racing": -45.0}
    sport = next(f for f in findings if f.kind == "sport-day-loss")
    assert "greyhounds" in sport.message and set(sport.stops) == {("capture", "greyhound/STOP")}


def test_without_an_envelope_the_warden_checks_the_registry_alone():
    strategies = registry.load()[0]
    findings, _ = warden.assess(strategies, None, "no envelope", [], [{"profit": -1e6, "customerOrderRef": "ash1"}], {})
    assert [f.kind for f in findings] == ["no-envelope"]


def test_a_sport_whose_only_switch_is_the_accounts_is_stopped_by_it():
    stops = warden._sport_stops("horse_racing", registry.load()[0])
    assert ("pred", "trading/STOP") in stops and ("capture", "trading/STOP") in stops
    assert ("pred", "morning/STOP") in stops


def test_the_warden_writes_stops_only_when_told(monkeypatch):
    calls = []
    monkeypatch.setenv("ULTRA_BETTING_S3_BUCKET", "pred-b")
    monkeypatch.setenv("CAPTURE_BUCKET", "cap-b")
    done = warden.write_stops(warden.ALL_STOP, "why", put=lambda b, k, body: calls.append((b, k)))
    assert sorted(calls) == [("cap-b", "trading/STOP"), ("pred-b", "trading/STOP")] and len(done) == 2
    src = (ROOT / "desk" / "warden.py").read_text()
    assert src.count("write_stops(") == 2 and "if breaches and a.enforce:" in src


def test_the_warden_only_reads():
    with pytest.raises(ValueError):
        warden._paged(object(), "placeOrders", {}, "x")
    for p in (ROOT / "desk").glob("*.py"):
        for word in ("placeOrders", "cancelOrders", "replaceOrders", "updateOrders"):
            assert word not in p.read_text(), (p.name, word)


# --- the research ledger ---------------------------------------------------------------------------------------------

LINE = {"when": "2026-11-10T09:30Z", "kind": "research", "tag": "new-1110", "run": 1, "result": "r"}


def test_the_ledger_passes_against_itself_and_a_good_line_can_be_added():
    text = trials.LEDGER.read_text()
    assert trials.check(text, text) == []
    assert trials.check(text + json.dumps(LINE) + "\n", text) == []


@pytest.mark.parametrize("edit,err", [
    (lambda t: t.replace(t.splitlines()[0], t.splitlines()[0].replace('"verdict"', '"verdikt"'), 1), "append-only"),
    (lambda t: "\n".join(t.splitlines()[1:]), "append-only"),
    (lambda t: t + json.dumps({**LINE, "when": "10 Nov"}) + "\n", "UTC"),
    (lambda t: t + json.dumps({**LINE, "kind": "preregistration", "question": "q", "design": "d",
                               "tag": json.loads(t.splitlines()[0])["tag"]}) + "\n", "tag already used"),
    (lambda t: t + json.dumps({**LINE, "kind": "preregistration"}) + "\n", "needs question"),
    (lambda t: t + json.dumps({**LINE, "kind": "preregistration", "question": "q"}) + "\n", "needs design"),
    (lambda t: t + json.dumps({**LINE, "kind": "trial", "question": "q", "design": "d", "reading": "r"}) + "\n",
     "needs strategy"),
    (lambda t: t + "{not json\n", "not JSON"),
])
def test_the_ledger_refuses(edit, err):
    text = trials.LEDGER.read_text()
    errs = trials.check(edit(text), text)
    assert any(err in e for e in errs), errs


def test_the_ledgers_house_form_passes():
    """Follow-up lines reuse their test's tag (ledger lines 32, 40, 41: clv-forward-live registered, amended,
    deployed); a preregistration carries its question and design, the reading inside the design (lines 642, 645)."""
    text = trials.LEDGER.read_text()
    tag = json.loads(text.splitlines()[0])["tag"]
    follow = {**LINE, "tag": tag, "kind": "preregistration-amendment"}
    prereg = {**LINE, "tag": "prereg-1110", "kind": "preregistration", "question": "q", "design": "d, read as r"}
    assert trials.check(text + json.dumps(follow) + "\n" + json.dumps(prereg) + "\n", text) == []


def test_trials_are_counted_conservatively():
    lines = [{"kind": "research"}, {"kind": "trial", "sport": "greyhounds"}, {"kind": "trial", "sport": "football"},
             {"kind": "decision"}]
    assert trials.count_trials(lines, "greyhounds") == 2 and trials.count_trials(lines) == 3


def test_the_backlog_is_sound_and_every_idea_is_sourced():
    files = sorted((ROOT / "desk" / "backlog").glob("*.jsonl"))
    assert {p.stem for p in files} == set(SPORTS)
    ids = []
    for p in files:
        for ln in p.read_text().splitlines():
            d = json.loads(ln)
            assert d["sport"] == p.stem and d["idea"] and d["source"] and trials.WHEN.match(d["added"]), d
            assert d["status"] in ("open", "blocked", "taken", "done", "dropped"), d
            ids.append(d["id"])
    assert len(ids) == len(set(ids))


# --- the line held: the live horse trader never depends on the desk ---------------------------------------------------

def test_no_legacy_trader_imports_the_desk():
    paths = [ROOT / "auto_trade.py", ROOT / "betfair_recorder.py"]
    for pkg in ("trading", "model", "morning", "greyhound", "football"):
        paths += list((ROOT / pkg).rglob("*.py"))
    for p in paths:
        assert not re.search(r"^\s*(from desk\b|import desk\b)", p.read_text(), re.M), p


def test_the_warden_needs_nothing_but_its_slim_install():
    """desk-warden.yml installs requests and python-dotenv alone: the warden's chain imports no model or research
    code (an import of research.overfit pulls the model package and scikit-learn)."""
    for name in ("__init__", "envelope", "registry", "account", "warden"):
        src = (ROOT / "desk" / f"{name}.py").read_text()
        assert not re.search(r"^\s*(from|import)\s+(research|model|pandas|numpy|trading)\b", src, re.M), name
        assert not re.search(r"\n\s+(from|import)\s+(research|model)\b", src), name
