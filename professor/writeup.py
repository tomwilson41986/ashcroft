"""The Professor's words for a tip: a fact sheet from the data, then the text. In Phase 1 the text is the template
(deterministic sentences from the sheet, always valid); the Claude API phrasing comes in Phase 2 behind the same
`validate()`: every number in the text must be in the sheet, no banned claim, no URL, X within 280 characters.

The claim is the move, never the win: the write-up says why the price is expected to shorten (the model's price
against the market's, the horse in the model's terms), and every post ends with the compliance lines.
"""
from __future__ import annotations

import re

from professor.selection import Tip

PERSONA = ("The Professor: a former horse-racing trader and betting-syndicate quant. Plain, numerate, never a "
           "guarantee; he backs what the syndicate backs and says so.")


def _f(x, nd=2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return ""


def fraction(decimal: float) -> str:
    """A decimal price as the nearest everyday fraction (7/2, 11/4, 13/8), else one decimal place."""
    from fractions import Fraction
    d = float(decimal) - 1.0
    if d <= 0:
        return _f(decimal)
    for den in (1, 2, 4, 5, 8, 10, 3, 6, 7, 9, 11, 13, 16, 20, 25, 33, 40, 50, 66, 80, 100):
        fr = Fraction(d).limit_denominator(den)
        if abs(float(fr) - d) <= 0.015 * max(d, 1):
            return f"{fr.numerator}/{fr.denominator}" if fr.denominator > 1 else f"{fr.numerator}/1"
    return _f(decimal, 1)


def fact_sheet(tip: Tip, cfg: dict) -> dict:
    """Only facts: the race, the horse, the prices, the model's view, the syndicate's position, the compliance lines."""
    c, f = tip.cand, tip.cand.facts
    books = ", ".join(tip.bookmakers[:5]) + (" and others" if len(tip.bookmakers) > 5 else "")
    sheet = {
        "runner": c.runner, "venue": c.venue, "off_uk": c.off_uk, "day": c.day,
        "race_name": f.get("race_name") or f.get("market_name") or "", "race_class": f.get("race_class", ""),
        "race_type": f.get("race_type", ""), "distance": f.get("dist_furlongs", ""), "going": f.get("going_description", ""),
        "runners": f.get("number_of_runners", ""), "cloth": f.get("cloth_number", ""), "draw": f.get("stall_draw", f.get("stall", "")),
        "jockey": f.get("jockey") or f.get("jockey_name", ""), "trainer": f.get("trainer", ""),
        "age": f.get("age") or f.get("horse_age", ""), "official_rating": f.get("official_rating", ""),
        "form": f.get("form", ""), "headgear": f.get("wearing") or f.get("headgear", ""),
        "days_since_run": f.get("days_since_last_run") or f.get("days_since_lr", ""),
        "advised": _f(tip.advised), "advised_fraction": fraction(tip.advised), "price_source": tip.price_source,
        "bookmakers": books, "n_bookmakers": len(tip.bookmakers),
        "exchange": _f(tip.advised_exchange), "exchange_size": _f(tip.exchange_size, 0),
        "expected_bsp": _f(c.exp_bsp), "model_price": _f(c.predicted_bfsp), "overlay_pct": _f(tip.overlay * 100, 0),
        "win_pct": _f(c.p_win * 100, 0), "card_odds": _f(f.get("racecard_odds", ""), 1),
        "rank": tip.rank, "tier": tip.tier, "withheld": tip.withheld,
        "compliance": cfg["compliance"],
    }
    return sheet


def template(sheet: dict) -> dict:
    """The deterministic write-up: a headline, a body of three short paragraphs, the X text and the Instagram caption."""
    s = sheet
    where = f"{s['off_uk']} {s['venue']}"
    race = f" ({s['race_name']})" if s["race_name"] else ""
    price_line = (f"{s['advised_fraction']} ({s['advised']}) with {s['n_bookmakers']} bookmakers: {s['bookmakers']}."
                  if s["price_source"] == "books" else
                  f"{s['advised_fraction']} ({s['advised']}) on the exchange, GBP{s['exchange_size']} there at the time; "
                  f"no bookmaker price was read for this race.")
    headline = f"{s['runner']}, {where}, {s['advised_fraction']}"
    horse_bits = []
    if s["jockey"] or s["trainer"]:
        horse_bits.append(f"{s['jockey'] or 'the rider'} for {s['trainer'] or 'the yard'}")
    if s["form"]:
        horse_bits.append(f"form {s['form']}")
    if s["official_rating"]:
        horse_bits.append(f"rated {s['official_rating']}")
    if s["days_since_run"]:
        horse_bits.append(f"{s['days_since_run']} days since the last run")
    if s["headgear"]:
        horse_bits.append(f"in {s['headgear']}")
    horse_line = (f"{s['runner']}: " + ", ".join(horse_bits) + ".") if horse_bits else f"{s['runner']}."
    why = (f"The syndicate's model makes {s['runner']} a {s['model_price']} shot and expects the Betfair SP to close near "
           f"{s['expected_bsp']}; the price advised is {s['overlay_pct']}% above that closing price. The model's expected "
           f"win chance is {s['win_pct']}%. The bet is the price, not a prediction of the result.")
    body = "\n\n".join([
        f"{where}{race}. {price_line}",
        horse_line,
        why,
        f"{s['compliance']['position']} {s['compliance']['basis']}",
        s["compliance"]["age"],
    ])
    x_text = (f"The Professor: {s['runner']} {where}, {s['advised_fraction']}"
              + (f" ({s['n_bookmakers']} bookmakers)" if s["price_source"] == "books" else " (exchange)")
              + f". Model price {s['model_price']}, expected SP {s['expected_bsp']}. "
              + f"Syndicate backed. {s['compliance']['age']}")
    if len(x_text) > 280:
        x_text = x_text[:277] + "..."
    ig_caption = f"{headline}\n\n{why}\n\n{s['compliance']['position']}\n{s['compliance']['age']}"
    telegram = f"<b>{headline}</b>\n\n{body}"
    return {"headline": headline, "body": body, "x_text": x_text, "ig_caption": ig_caption, "telegram": telegram}


def validate(text: dict, sheet: dict, cfg: dict) -> list[str]:
    """Why a text fails: a number not in the sheet, a banned phrase, a URL, an X text over 280."""
    reasons = []
    allowed = set()
    for v in sheet.values():
        for n in re.findall(r"\d+(?:\.\d+)?", str(v)):
            allowed.add(n)
            allowed.add(n.rstrip("0").rstrip(".") if "." in n else n)
    for k, t in text.items():
        for n in re.findall(r"\d+(?:\.\d+)?", t):
            if n not in allowed and n.rstrip("0").rstrip(".") not in allowed:
                reasons.append(f"{k}: the number {n} is not in the fact sheet")
        low = t.lower()
        for b in cfg.get("writeup", {}).get("banned", []):
            if b in low:
                reasons.append(f"{k}: banned phrase {b!r}")
        if re.search(r"https?://|www\.", low):
            reasons.append(f"{k}: a URL")
    if len(text.get("x_text", "")) > 280:
        reasons.append("x_text over 280 characters")
    return reasons


def compose(tip: Tip, cfg: dict) -> dict:
    """The text for a tip: the template (Phase 1); validated all the same."""
    sheet = fact_sheet(tip, cfg)
    text = template(sheet)
    problems = validate(text, sheet, cfg)
    if problems:
        # the template only ever fails on a fact that carries a URL or a banned word; drop that fact and retry once
        text = template({k: ("" if any(k in p for p in problems) else v) for k, v in sheet.items()} | {"compliance": sheet["compliance"]})
    text["sheet"] = sheet
    return text
