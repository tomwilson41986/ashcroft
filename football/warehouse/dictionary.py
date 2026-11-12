"""The data dictionary, generated from the table contracts (so it cannot drift from the code).

    python -m football.warehouse dictionary --out docs/football_data_dictionary.md
"""

from __future__ import annotations

from pathlib import Path


def specs() -> list:
    """Every table's TableSpec: bronze, silver, then each gold module's (and the link stage's) in build order."""
    from football.warehouse import gold
    from football.warehouse.bronze import BRONZE_INDEX
    from football.warehouse.schema import BRONZE_RAW, SILVER
    out = [BRONZE_RAW, BRONZE_INDEX, *SILVER]
    mods = gold.order(gold.load())
    try:
        import importlib
        mods.append(importlib.import_module("football.warehouse.link"))
    except ModuleNotFoundError:
        pass
    for m in mods:
        out += list(getattr(m, "SPECS", ()))
    return out


def scope_of(name: str) -> str:
    from football.warehouse import gold
    for m in gold.load():
        if any(s.name == name for s in m.SPECS):
            return "one file a (year, month) partition" if m.SCOPE == "part" else "a single file"
    return "one file a (year, month) partition" if name.startswith(("bronze/", "silver/")) else "a single file"


def markdown() -> str:
    lines = ["# racing² football data warehouse - data dictionary", "",
             "Generated from the table contracts in `football/warehouse/` by "
             "`python -m football.warehouse dictionary`. Do not edit by hand.", "",
             "Times are UTC. A price is Betfair's decimal price. `handicap` is 0.0 where a runner has none. "
             "Tables live under `sources/football/warehouse/<table>/` in the capture bucket.", "",
             "| Table | Rows are | Key | Stored as |", "|---|---|---|---|"]
    all_specs = specs()
    for s in all_specs:
        first = s.doc.split(". ")[0].rstrip(".")
        lines.append(f"| `{s.name}` | {first} | {', '.join(s.key)} | {scope_of(s.name)} |")
    for s in all_specs:
        lines += ["", f"## `{s.name}`", "", s.doc, "", f"Key: {', '.join(f'`{k}`' for k in s.key)}. "
                  f"Stored as {scope_of(s.name)}, sorted by {', '.join(f'`{k}`' for k in (s.sort or s.key))}.", "",
                  "| Column | Type | Required | Meaning |", "|---|---|---|---|"]
        for c in s.columns:
            doc = c.doc.replace("|", "\\|")
            lines.append(f"| `{c.name}` | {c.type} | {'yes' if not c.nullable else ''} | {doc} |")
    return "\n".join(lines) + "\n"


def write(path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown())
    return str(p)
