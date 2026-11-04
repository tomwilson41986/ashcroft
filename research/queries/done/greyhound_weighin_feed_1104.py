"""The greyhounds.today weigh-in record, 7 Oct to 4 Nov (the owner, 4 Nov: "use the weigh-in for the tracks we have it";
read-only). Per day and track from the recorder's poll log: the cards seen, the runners, how many carried a weight at
the day's last poll, and when weights first appeared (UTC). Then one card's shape (keys, a runner with a weight), so
the live model can read it."""
import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.getcwd())
from sources.common import Store                         # noqa: E402

store = Store()
PREFIX = "greyhounds_today"
first, last = date(2026, 10, 7), date(2026, 11, 4)
rows = []
d = first
while d <= last:
    raw = store.get(f"{PREFIX}/{d:%Y-%m-%d}/polls.jsonl")
    if not raw:
        rows.append((f"{d:%Y-%m-%d}", "-", "no poll log"))
        d += timedelta(days=1)
        continue
    per = {}
    for line in raw.decode().splitlines():
        try:
            p = json.loads(line)
        except ValueError:
            continue
        for t, r in (p.get("tracks") or {}).items():
            if not isinstance(r, dict) or "error" in r:
                continue
            s = per.setdefault(t, {"cards": 0, "runners": 0, "weighted": 0, "first_weight": None, "polls": 0})
            s["polls"] += 1
            s["cards"] = max(s["cards"], r.get("cards", 0))
            s["runners"] = max(s["runners"], r.get("runners", 0))
            s["weighted"] = max(s["weighted"], r.get("weighted", 0))
            if r.get("weighted", 0) and not s["first_weight"]:
                s["first_weight"] = p["utc"][11:16]
    for t, s in sorted(per.items()):
        if s["cards"]:
            rows.append((f"{d:%Y-%m-%d}", t, f"cards {s['cards']}, runners {s['runners']}, weighted {s['weighted']}, "
                                            f"first weight {s['first_weight']} UTC, polls {s['polls']}"))
    if not any(s["cards"] for s in per.values()):
        rows.append((f"{d:%Y-%m-%d}", "-", f"poll log present, no cards at any track ({len(per)} tracks listed)"))
    d += timedelta(days=1)
print("== the weigh-in record by day and track")
for r in rows:
    print(" | ".join(r))

# one card's shape: the latest day with files
for back in range(0, 30):
    day = last - timedelta(days=back)
    files = sorted(store.listing(f"{PREFIX}/{day:%Y-%m-%d}/"))
    cards = [f for f in files if f.endswith(".json") and "polls" not in f]
    if cards:
        print(f"\n== {len(cards)} card files on {day}; the last three: {cards[-3:]}")
        key = cards[-1]
        key = key[key.index(PREFIX):] if PREFIX in key else key
        card = json.loads(store.get(key))

        def shape(x, depth=0, width=6):
            pad = "  " * depth
            if isinstance(x, dict):
                for k in list(x)[:25]:
                    v = x[k]
                    if isinstance(v, (dict, list)):
                        print(f"{pad}{k}: {type(v).__name__}{'[' + str(len(v)) + ']' if isinstance(v, list) else ''}")
                        if depth < 4:
                            shape(v, depth + 1)
                    else:
                        print(f"{pad}{k}: {str(v)[:60]!r}")
            elif isinstance(x, list) and x:
                shape(x[0], depth, width)
        shape(card)
        break
else:
    print("no data found: no card files in the last 30 days")
