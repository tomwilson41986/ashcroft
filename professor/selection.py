"""The Professor's selection (config.json; the backtest in reports/professor_rule.md): among the horses the syndicate
holds, advise each with a bookmaker price that clears the closing model's expected BSP by `overlay_bar`, at most one
a race, the horses with the highest expected win probability first, at most `max_tips` a day; ranks 1..`free_tips`
are the free tips.

Two ways of reading the day:
- `card()`: the day's card at a point in time (the dry run, the replay, the 10:00 post): every qualifier ranked by
  p_win, capped. This is the rule the backtest set.
- `alerts()`: the live loop, where the fills arrive one by one and a tip goes out at once: a horse qualifies as it
  arrives when its p_win clears `min_p_win_alert` and the day's cap is not reached (the trader fills its best
  expected value first, so the first arrivals are the best; the floor stands in for the rank). The sequential rule's
  own backtest is a follow-up (the first weeks' record scores it beside the ranked card).

The price advised is a bookmaker's when the bookmakers quote the horse (the owner, 23 Oct: "really we want to find
horses that are with bookmaker prices"): the highest price at least `min_books` major bookmakers offer (the
third-highest major quote), with the bookmakers at or above it named. When no bookmaker quotes the race (the scrape
failed, or the race is not priced yet), the exchange's best back at the fill is advised instead and the tip says so
(`price_source` exchange); with `require_books` on, such a horse is not advised at all.

A random share of the qualifiers (`withheld_share`) is withheld from publication: settled and reported like a tip,
never published, so the published impact on our own BSP can be measured (the design's withheld fifth). The draw is a
hash of day, market and selection, so it never changes within a day and cannot be gamed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from professor.candidates import Candidate

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("min_p_win_alert", 0.15)
    cfg.setdefault("min_alert_size", 50.0)
    cfg.setdefault("withheld_share", 0.2)
    cfg.setdefault("require_books", False)
    cfg.setdefault("max_tips", 5)
    cfg.setdefault("free_tips", 2)
    cfg.setdefault("max_per_race", 1)
    cfg.setdefault("min_books", 3)
    cfg.setdefault("max_price", 51.0)
    return cfg


@dataclass
class Tip:
    cand: Candidate
    advised: float                 # the price advised (a bookmaker's when there is one, else the exchange's)
    price_source: str              # 'books' or 'exchange'
    advised_books: float | None    # the price at least min_books major bookmakers offer
    bookmakers: list[str]          # the major bookmakers at or above it
    advised_exchange: float        # the exchange's best back at the first fill
    exchange_size: float
    overlay: float                 # advised / expected BSP - 1
    rank: int = 0
    tier: str = "private"          # 'free' or 'private'
    withheld: bool = False
    quotes: dict = field(default_factory=dict)   # bookmaker -> price, every major quote on the horse
    bog: dict = field(default_factory=dict)      # bookmaker -> bool where known
    published_utc: str | None = None
    slug: str = ""

    def to_row(self) -> dict:
        d = self.cand.to_row()
        d.update({k: v for k, v in asdict(self).items() if k not in ("cand", "quotes", "bog")})
        d["quotes"] = json.dumps(self.quotes)
        d["bog"] = json.dumps(self.bog)
        return d


def slug_of(c: Candidate) -> str:
    name = "".join(ch for ch in c.runner.lower() if ch.isalnum() or ch == " ").strip().replace(" ", "-")
    return f"{c.off_uk.replace(':', '')}-{''.join(ch for ch in c.venue.lower() if ch.isalnum())}-{name}"


def withheld(day: str, market_id: str, selection_id: int, share: float, salt: str = "professor") -> bool:
    if share <= 0:
        return False
    h = hashlib.sha1(f"{salt}|{day}|{market_id}|{selection_id}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < share


def books_price(quotes: dict[str, float], major: list[str], min_books: int) -> tuple[float | None, list[str]]:
    """The highest price at least `min_books` major bookmakers offer, and who offers at least it."""
    prices = sorted(((float(p), b) for b, p in quotes.items() if b in major and p and p > 1), reverse=True)
    if len(prices) < min_books:
        return None, []
    price = prices[min_books - 1][0]
    return price, [b for p, b in prices if p >= price - 1e-9]


def qualify(c: Candidate, cfg: dict, quotes: dict[str, float] | None, bog: dict | None = None,
            exchange_price: float | None = None) -> tuple[Tip | None, str]:
    """One horse against the rule. Returns (tip, reason): the tip when it qualifies, else why not."""
    if not (c.exp_bsp == c.exp_bsp) or c.exp_bsp <= 1:
        return None, "no expected BSP"
    ex_price = float(exchange_price or c.best_back or c.first_price)
    adv_books, books = (None, [])
    if quotes:
        adv_books, books = books_price(quotes, cfg["major_bookmakers"], int(cfg["min_books"]))
    if adv_books is not None:
        advised, source = adv_books, "books"
    elif cfg.get("require_books") or quotes:
        # the race is quoted and this horse is not with enough bookmakers, or bookmakers are required
        return None, f"fewer than {cfg['min_books']} major bookmakers quote it" if quotes else "no bookmaker quotes"
    else:
        advised, source = ex_price, "exchange"
    if advised > float(cfg["max_price"]):
        return None, f"over the maximum price ({advised:.1f})"
    overlay = advised / c.exp_bsp - 1.0
    if overlay < float(cfg["overlay_bar"]):
        return None, f"under the overlay bar ({overlay:+.1%})"
    if c.p_win < float(cfg.get("min_p_win", 0.0)):
        return None, f"under the win-probability floor ({c.p_win:.3f})"
    tip = Tip(cand=c, advised=float(advised), price_source=source, advised_books=adv_books, bookmakers=books,
              advised_exchange=ex_price, exchange_size=float(c.back_size or 0.0), overlay=float(overlay),
              quotes={b: float(p) for b, p in (quotes or {}).items() if p and p > 1}, bog=dict(bog or {}),
              slug=slug_of(c))
    return tip, "qualifies"


def card(cands: list[Candidate], cfg: dict, quotes_by_horse: dict[tuple, dict] | None = None,
         bog_by_horse: dict[tuple, dict] | None = None) -> tuple[list[Tip], list[dict]]:
    """The day's card at a point in time: every qualifier ranked by p_win, one a race, capped, tiered, the withheld
    marked. Returns the tips (published and withheld) and a log of every candidate's fate."""
    quotes_by_horse, bog_by_horse = quotes_by_horse or {}, bog_by_horse or {}
    tips, fates = [], []
    for c in cands:
        key = (c.market_id, c.selection_id)
        tip, why = qualify(c, cfg, quotes_by_horse.get(key), bog_by_horse.get(key))
        fates.append({"market_id": c.market_id, "selection_id": c.selection_id, "runner": c.runner, "venue": c.venue,
                      "off_uk": c.off_uk, "p_win": c.p_win, "exp_bsp": c.exp_bsp, "first_price": c.first_price,
                      "fate": why})
        if tip is not None:
            tips.append(tip)
    tips.sort(key=lambda t: (-t.cand.p_win, -t.overlay))
    # the withheld fifth is drawn among the qualifiers before the ranking, so the published ranks are the best of the
    # rest and the withheld are a random sample of qualifiers at every rank (the design's experiment)
    held, rest = [], []
    for t in tips:
        t.withheld = withheld(t.cand.day, t.cand.market_id, t.cand.selection_id, float(cfg.get("withheld_share", 0)))
        (held if t.withheld else rest).append(t)
    if int(cfg.get("max_per_race", 1)) == 1:
        seen, kept = set(), []
        for t in rest:
            if t.cand.race in seen:
                continue
            seen.add(t.cand.race)
            kept.append(t)
        rest = kept
    rest = rest[: int(cfg["max_tips"])]
    for i, t in enumerate(rest, start=1):
        t.rank = i
        t.tier = "free" if i <= int(cfg["free_tips"]) else "private"
    for t in held[: int(cfg["max_tips"])]:
        t.rank, t.tier = 0, "withheld"
    return rest + held[: int(cfg["max_tips"])], fates


def alerts(cands: list[Candidate], tips_so_far: list[Tip], cfg: dict, quotes_by_horse: dict[tuple, dict] | None = None,
           bog_by_horse: dict[tuple, dict] | None = None) -> tuple[list[Tip], list[dict]]:
    """The live loop's step: the new fills against the rule in arrival order, the day's cap and the one-a-race rule
    counted with the tips already out (withheld ones count against the cap: they are tips the record scores)."""
    quotes_by_horse, bog_by_horse = quotes_by_horse or {}, bog_by_horse or {}
    done = {(t.cand.market_id, t.cand.selection_id) for t in tips_so_far}
    out_tips = [t for t in tips_so_far if not t.withheld]
    races = {t.cand.race for t in out_tips} if int(cfg.get("max_per_race", 1)) == 1 else set()
    n = len(out_tips)
    new, fates = [], []
    for c in cands:
        key = (c.market_id, c.selection_id)
        if key in done:
            continue
        if c.first_utc is None:
            continue
        if n >= int(cfg["max_tips"]):
            fates.append({"runner": c.runner, "fate": "the day's cap is reached"})
            continue
        if c.race in races:
            fates.append({"runner": c.runner, "fate": "a tip is out in the race"})
            continue
        if c.p_win < float(cfg.get("min_p_win_alert", 0.15)):
            fates.append({"runner": c.runner, "fate": f"under the alert floor ({c.p_win:.3f})"})
            continue
        if float(c.back_size or 0) < float(cfg.get("min_alert_size", 0)):
            fates.append({"runner": c.runner, "fate": f"too little money at the price (GBP{c.back_size:.0f})"})
            continue
        tip, why = qualify(c, cfg, quotes_by_horse.get(key), bog_by_horse.get(key))
        fates.append({"runner": c.runner, "fate": why})
        if tip is None:
            continue
        if withheld(c.day, c.market_id, c.selection_id, float(cfg.get("withheld_share", 0))):
            tip.withheld, tip.rank, tip.tier = True, 0, "withheld"      # scored, never published, no rank taken
            new.append(tip)
            continue
        n += 1
        tip.rank = n
        tip.tier = "free" if n <= int(cfg["free_tips"]) else "private"
        races.add(c.race)
        new.append(tip)
    return new, fates


def tips_json(tips: list[Tip]) -> list[dict]:
    return [t.to_row() for t in tips]
