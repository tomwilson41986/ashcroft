#!/usr/bin/env python3
"""A day's predictions as a workbook: every runner, the top of each race, one sheet per meeting.

    python scripts/predictions_workbook.py --predictions out/predictions.csv \\
        --meta data/models/bfsp_model_meta.json --out reports/predictions_2026-09-25.xlsx

    # beside another model's prices for the same day (e.g. the 06:00 record)
    python scripts/predictions_workbook.py --predictions out/predictions.csv --label 944 \\
        --compare 0600.csv --compare-label 615 --meta ... --out ...

Reads the CSV predict_bfsp_today.py --output-csv writes (the card's race and
runner fields, the model's predicted BSP, its win probability normalised
within the race, and its rank in the race). The model's outputs are values;
the columns derived from them in the workbook are formulas, so they follow
if a price is edited. The early-price rule's margin is one input cell on the
Read me sheet; every runner's rule price reads it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
HEAD_FILL = PatternFill("solid", start_color="1F3864")
TOP_FILL = PatternFill("solid", start_color="E2EFDA")
THIN = Side(style="thin", color="BFBFBF")
INPUT_FONT_COLOR = "0000FF"
#: the early-price rule, fixed before it was first tested (reports/clv_betfair_2026q1.md):
#: back where ln(Betfair morning price / predicted BSP) >= RULE_MARGIN, then close at BSP
RULE_MARGIN = 0.2
RULE_HEAD = "Rule: back at Betfair ≥"

#: (header, source column, number format, width); None source = a formula column
RUNNER_COLS = [
    ("Time", "race_time", "@", 7), ("Course", "track", "@", 14), ("Race", "race_name", "@", 34),
    ("Class", "race_class", "@", 6), ("Furlongs", "dist_furlongs", "0.0", 8), ("Going", "going_description", "@", 12),
    ("Runners", "number_of_runners", "0", 8), ("Horse", "horse_name", "@", 24), ("Jockey", "jockey_name", "@", 18),
    ("Trainer", "trainer", "@", 20), ("Draw", "stall", "0", 6), ("OR", "official_rating", "0", 6),
    ("Age", "horse_age", "0", 5), ("Days since run", "days_since_lr", "0", 9),
    ("Model rank", "model_rank", "0", 8), ("Predicted BSP", "predicted_bfsp", "0.00", 10),
    ("Win probability", "predicted_win_prob_norm", "0.0%", 10), ("Card odds (HRB)", "odds", "0.00", 10),
]


def _style_header(ws, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"


def _runner_sheet(ws, df: pd.DataFrame, rule_ref: str) -> None:
    cols = [c for c in RUNNER_COLS if c[1] in df.columns]
    headers = [h for h, *_ in cols] + ["Card odds vs model", RULE_HEAD]
    ws.append(headers)
    letter = {h: get_column_letter(i + 1) for i, h in enumerate(headers)}
    for i, row in enumerate(df.itertuples(index=False), start=2):
        rec = row._asdict()
        for j, (h, src, fmt, _) in enumerate(cols, start=1):
            v = rec.get(src)
            if pd.isna(v):
                v = None
            elif src in ("race_time", "race_class"):
                v = str(v)
            cell = ws.cell(row=i, column=j, value=v)
            cell.number_format = fmt
            cell.font = Font(name=FONT)
        # how much longer (positive) or shorter the card's odds are than the model's price
        k = len(cols) + 1
        if "Card odds (HRB)" in letter and "Predicted BSP" in letter:
            o, p = letter["Card odds (HRB)"], letter["Predicted BSP"]
            ws.cell(row=i, column=k, value=f'=IF(AND(ISNUMBER({o}{i}),{o}{i}>1,{p}{i}>0),{o}{i}/{p}{i}-1,"")')
        cell = ws.cell(row=i, column=k)
        cell.number_format = "+0%;-0%;0%"
        cell.font = Font(name=FONT)
        # the lowest Betfair price at which the early-price rule backs the runner
        if "Predicted BSP" in letter:
            p = letter["Predicted BSP"]
            ws.cell(row=i, column=k + 1, value=f'=IF(AND(ISNUMBER({p}{i}),{p}{i}>0),{p}{i}*EXP({rule_ref}),"")')
        cell = ws.cell(row=i, column=k + 1)
        cell.number_format = "0.00"
        cell.font = Font(name=FONT)
    _style_header(ws, len(headers))
    for j, (_, _, _, w) in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.column_dimensions[get_column_letter(len(headers) - 1)].width = 11
    ws.column_dimensions[get_column_letter(len(headers))].width = 11
    last = ws.max_row
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{last}"
    if "Model rank" in letter and last > 1:
        r = letter["Model rank"]
        ws.conditional_formatting.add(f"A2:{get_column_letter(len(headers))}{last}",
                                      FormulaRule(formula=[f"${r}2=1"], fill=TOP_FILL))


def _top_sheet(ws, df: pd.DataFrame, rule_ref: str) -> None:
    """The model's first and second choice in every race, and how far apart they are."""
    ws.append(["Time", "Course", "Race", "Runners", "Top pick", "Jockey", "Trainer", "Predicted BSP",
               "Win probability", "Second choice", "Second's BSP", "Second's probability",
               "Probability gap", "Card odds (HRB)", RULE_HEAD])
    key = ["race_date", "track", "race_time"]
    for _, race in df.groupby(key, sort=False):
        race = race.sort_values("predicted_bfsp")
        a = race.iloc[0]
        b = race.iloc[1] if len(race) > 1 else None
        i = ws.max_row + 1
        vals = [str(a.get("race_time", "")), a.get("track"), a.get("race_name"), a.get("number_of_runners"),
                a.get("horse_name"), a.get("jockey_name"), a.get("trainer"), a.get("predicted_bfsp"),
                a.get("predicted_win_prob_norm"),
                None if b is None else b.get("horse_name"), None if b is None else b.get("predicted_bfsp"),
                None if b is None else b.get("predicted_win_prob_norm"), None, a.get("odds")]
        for j, v in enumerate(vals, start=1):
            ws.cell(row=i, column=j, value=None if (v is not None and not isinstance(v, str) and pd.isna(v)) else v)
        ws.cell(row=i, column=13, value=f'=IF(AND(ISNUMBER(I{i}),ISNUMBER(L{i})),I{i}-L{i},"")')
        ws.cell(row=i, column=15, value=f'=IF(AND(ISNUMBER(H{i}),H{i}>0),H{i}*EXP({rule_ref}),"")')
        for j, fmt in ((4, "0"), (8, "0.00"), (9, "0.0%"), (11, "0.00"), (12, "0.0%"), (13, "0.0%"), (14, "0.00"),
                       (15, "0.00")):
            ws.cell(row=i, column=j).number_format = fmt
        for j in range(1, 16):
            ws.cell(row=i, column=j).font = Font(name=FONT)
    _style_header(ws, 15)
    for j, w in enumerate([7, 14, 34, 8, 24, 18, 20, 10, 10, 24, 10, 10, 10, 10, 11], start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.auto_filter.ref = f"A1:O{ws.max_row}"


def _compare_sheet(ws, df: pd.DataFrame, other: pd.DataFrame, label: str, other_label: str,
                   other_note: str) -> None:
    """Each race's top pick under this model and under another model's prices for the same day, side by side.

    The other model's pick is its shortest price among the runners still declared here, so a race that lost
    runners since the other file was written compares like with like."""
    key = ["track", "race_time", "horse_name"]
    m = df.merge(other[key + ["predicted_bfsp"]].rename(columns={"predicted_bfsp": "_other"}), on=key, how="left")
    head = ["Time", "Course", "Race", "Runners (now)", f"{label} top pick", f"{label} BSP", f"Its {other_label} BSP",
            f"{other_label} top pick (runners still declared)", f"{other_label} BSP", "Same pick?",
            f"{label} vs {other_label} on the {label}'s pick"]
    ws.append(head)
    for _, g in m.groupby(["race_date", "track", "race_time"], sort=False):
        a = g.sort_values("predicted_bfsp").iloc[0]
        b = g.sort_values("_other").iloc[0] if g["_other"].notna().any() else None
        i = ws.max_row + 1
        vals = [str(a["race_time"]), a["track"], a.get("race_name"), int(len(g)), a["horse_name"],
                float(a["predicted_bfsp"]), None if pd.isna(a["_other"]) else float(a["_other"]),
                None if b is None else b["horse_name"], None if b is None else float(b["_other"])]
        for j, v in enumerate(vals, start=1):
            ws.cell(row=i, column=j, value=v)
        ws.cell(row=i, column=10, value=f'=IF(E{i}=H{i},"yes","no")')
        ws.cell(row=i, column=11, value=f'=IF(AND(ISNUMBER(F{i}),ISNUMBER(G{i})),F{i}/G{i}-1,"")')
        for j, fmt in ((4, "0"), (6, "0.00"), (7, "0.00"), (9, "0.00"), (11, "+0%;-0%;0%")):
            ws.cell(row=i, column=j).number_format = fmt
        for j in range(1, 12):
            ws.cell(row=i, column=j).font = Font(name=FONT)
    last = ws.max_row
    _style_header(ws, len(head))
    ws.auto_filter.ref = f"A1:K{last}"
    ws.cell(row=last + 2, column=1, value="Races with the same top pick").font = Font(name=FONT, bold=True)
    ws.cell(row=last + 2, column=5,
            value=f'=COUNTIF(J2:J{last},"yes")&" of "&COUNTA(J2:J{last})').font = Font(name=FONT)
    if other_note:
        ws.cell(row=last + 3, column=1, value=other_note).font = Font(name=FONT, italic=True, size=9)
    for j, w in enumerate([7, 16, 34, 9, 24, 9, 10, 24, 9, 8, 12], start=1):
        ws.column_dimensions[get_column_letter(j)].width = w


def _readme(ws, df: pd.DataFrame, meta: dict, day: str, note: str, label: str = "") -> str:
    """The summary and the key; returns the absolute reference of the rule's margin cell."""
    rows = [
        (f"Predictions for {day}", None),
        (None, None),
        ("Runners", len(df)),
        ("Races", int(df.groupby(["track", "race_time"]).ngroups)),
        ("Meetings", int(df["track"].nunique())),
        (None, None),
        ("The model", label or None),
        ("Features", len(meta.get("feature_cols", []))),
        ("Trained through", meta.get("trained_through")),
        ("Boosting rounds", meta.get("best_iteration")),
        ("Target", meta.get("target")),
        ("Feature code hash", meta.get("feature_code_hash") or meta.get("feature_hash")),
        (None, None),
        ("How to read it", None),
        ("Predicted BSP", "The model's forecast of the Betfair Starting Price, from pre-race information only. "
                          "The prices in each race make a book of exactly 100%."),
        ("Win probability", "1 / predicted BSP: the model's chance for the horse, normalised within the race."),
        ("Model rank", "1 = the model's shortest price in the race (highlighted green)."),
        ("Card odds (HRB)", "The odds printed on the horseracebase card when it was fetched: a bookmaker "
                            "forecast, not the Betfair morning price the early-price rule was tested against."),
        ("Card odds vs model", "Card odds / predicted BSP - 1. Positive: the card is longer than the model."),
        (None, None),
        ("The early-price rule", None),
        ("Margin (log)", RULE_MARGIN),
        (RULE_HEAD, "Predicted BSP × e^margin: at 0.2, a Betfair price at least 1.22 times the model's (the model "
                    "at least 22% shorter). The rule was fixed before it was first tested "
                    "(reports/clv_betfair_2026q1.md): back at Betfair's morning volume-weighted price, where at "
                    "least £100 has been matched in the morning, and close the position at BSP (lay at BSP to "
                    "level the book). It is a trade on the price, not a bet held to the result. The margin (blue) "
                    "is the one input: change it to see the prices at another threshold."),
        (None, None),
        ("Note", note),
    ]
    margin_row = next(r for r, (k, _) in enumerate(rows, start=1) if k == "Margin (log)")
    for r, (k, v) in enumerate(rows, start=1):
        a = ws.cell(row=r, column=1, value=k)
        b = ws.cell(row=r, column=2, value=v)
        a.font = Font(name=FONT, bold=k in (f"Predictions for {day}", "The model", "How to read it",
                                            "The early-price rule", "Note"),
                      size=14 if r == 1 else 10)
        b.font = Font(name=FONT, color=INPUT_FONT_COLOR if r == margin_row else None)
        b.alignment = Alignment(wrap_text=True, vertical="top", horizontal="left")
    ws.cell(row=margin_row, column=2).number_format = "0.00"
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 100
    return f"'{ws.title}'!$B${margin_row}"


def build(pred: pd.DataFrame, meta: dict, out: str, note: str = "", label: str = "",
          other: pd.DataFrame | None = None, other_label: str = "", other_note: str = "") -> Path:
    df = pred.copy()
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")
    t = pd.to_datetime(df["race_time"].astype(str).str.replace(".", ":", regex=False), format="%H:%M", errors="coerce")
    # the card's times are 12-hour without am/pm (1.30 is 13:30): afternoon for anything before 11
    hours = t.dt.hour.where(t.dt.hour >= 11, t.dt.hour + 12)
    df["_sort"] = hours * 60 + t.dt.minute
    df = df.sort_values(["_sort", "track", "model_rank"], kind="mergesort").drop(columns="_sort")
    day = str(df["race_date"].iloc[0]) if len(df) else ""

    wb = Workbook()
    wb.active.title = "Read me"
    rule_ref = _readme(wb.active, df, meta, day, note, label)
    _top_sheet(wb.create_sheet("Top picks"), df, rule_ref)
    if other is not None:
        name = f"{label or 'model'} vs {other_label or 'other'}"[:31]
        _compare_sheet(wb.create_sheet(name), df, other, label or "model", other_label or "other", other_note)
    _runner_sheet(wb.create_sheet("All runners"), df, rule_ref)
    for track, g in df.groupby("track", sort=False):
        name = str(track)[:31].replace("/", "-")
        _runner_sheet(wb.create_sheet(name), g, rule_ref)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return Path(out)


def read_other(path: str) -> pd.DataFrame:
    """Another model's predictions for the day: predict_bfsp_today.py's CSV, or the 06:00 record kept in S3
    (predictions/<day>.csv: venue and runner_name for track and horse_name)."""
    o = pd.read_csv(path)
    o = o.rename(columns={c: n for c, n in (("venue", "track"), ("runner_name", "horse_name")) if n not in o.columns})
    o["race_time"] = o["race_time"].astype(str)
    return o.drop_duplicates(["track", "race_time", "horse_name"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--label", default="", help="The model's name in the workbook, e.g. 944")
    ap.add_argument("--compare", default="", help="Another model's predictions for the same day, set beside these")
    ap.add_argument("--compare-label", default="", help="Its name, e.g. 615")
    ap.add_argument("--compare-note", default="", help="A line under the comparison: where its prices came from")
    a = ap.parse_args()
    pred = pd.read_csv(a.predictions)
    pred["race_time"] = pred["race_time"].astype(str)
    meta = json.loads(Path(a.meta).read_text())
    other = read_other(a.compare) if a.compare else None
    print(build(pred, meta, a.out, a.note, a.label, other, a.compare_label, a.compare_note))


if __name__ == "__main__":
    main()
