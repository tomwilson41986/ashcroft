#!/usr/bin/env python3
"""A day's predictions as a workbook: every runner, the top of each race, one sheet per meeting.

    python scripts/predictions_workbook.py --predictions out/predictions.csv \\
        --meta data/models/bfsp_model_meta.json --out reports/predictions_2026-09-25.xlsx

Reads the CSV predict_bfsp_today.py --output-csv writes (the card's race and
runner fields, the model's predicted BSP, its win probability normalised
within the race, and its rank in the race). The model's outputs are values;
the columns derived from them in the workbook are formulas, so they follow
if a price is edited.
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


def _runner_sheet(ws, df: pd.DataFrame) -> None:
    cols = [c for c in RUNNER_COLS if c[1] in df.columns]
    headers = [h for h, *_ in cols] + ["Card odds vs model"]
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
    _style_header(ws, len(headers))
    for j, (_, _, _, w) in enumerate(cols, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.column_dimensions[get_column_letter(len(headers))].width = 11
    last = ws.max_row
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{last}"
    if "Model rank" in letter and last > 1:
        r = letter["Model rank"]
        ws.conditional_formatting.add(f"A2:{get_column_letter(len(headers))}{last}",
                                      FormulaRule(formula=[f"${r}2=1"], fill=TOP_FILL))


def _top_sheet(ws, df: pd.DataFrame) -> None:
    """The model's first and second choice in every race, and how far apart they are."""
    ws.append(["Time", "Course", "Race", "Runners", "Top pick", "Jockey", "Trainer", "Predicted BSP",
               "Win probability", "Second choice", "Second's BSP", "Second's probability",
               "Probability gap", "Card odds (HRB)"])
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
        for j, fmt in ((4, "0"), (8, "0.00"), (9, "0.0%"), (11, "0.00"), (12, "0.0%"), (13, "0.0%"), (14, "0.00")):
            ws.cell(row=i, column=j).number_format = fmt
        for j in range(1, 15):
            ws.cell(row=i, column=j).font = Font(name=FONT)
    _style_header(ws, 14)
    for j, w in enumerate([7, 14, 34, 8, 24, 18, 20, 10, 10, 24, 10, 10, 10, 10], start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.auto_filter.ref = f"A1:N{ws.max_row}"


def _readme(ws, df: pd.DataFrame, meta: dict, day: str, note: str) -> None:
    rows = [
        (f"Predictions for {day}", None),
        (None, None),
        ("Runners", len(df)),
        ("Races", int(df.groupby(["track", "race_time"]).ngroups)),
        ("Meetings", int(df["track"].nunique())),
        (None, None),
        ("The model", None),
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
        ("Note", note),
    ]
    for r, (k, v) in enumerate(rows, start=1):
        a = ws.cell(row=r, column=1, value=k)
        b = ws.cell(row=r, column=2, value=v)
        a.font = Font(name=FONT, bold=k in (f"Predictions for {day}", "The model", "How to read it", "Note"),
                      size=14 if r == 1 else 10)
        b.font = Font(name=FONT)
        b.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 100


def build(pred: pd.DataFrame, meta: dict, out: str, note: str = "") -> Path:
    df = pred.copy()
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")
    t = pd.to_datetime(df["race_time"].astype(str).str.replace(".", ":", regex=False), format="%H:%M", errors="coerce")
    # the card's times are 12-hour without am/pm (1.30 is 13:30): afternoon for anything before 11
    hours = t.dt.hour.where(t.dt.hour >= 11, t.dt.hour + 12)
    df["_sort"] = hours * 60 + t.dt.minute
    df = df.sort_values(["_sort", "track", "model_rank"], kind="mergesort").drop(columns="_sort")
    day = str(df["race_date"].iloc[0]) if len(df) else ""

    wb = Workbook()
    _readme(wb.active, df, meta, day, note)
    wb.active.title = "Read me"
    _top_sheet(wb.create_sheet("Top picks"), df)
    _runner_sheet(wb.create_sheet("All runners"), df)
    for track, g in df.groupby("track", sort=False):
        name = str(track)[:31].replace("/", "-")
        _runner_sheet(wb.create_sheet(name), g)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return Path(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    pred = pd.read_csv(a.predictions)
    meta = json.loads(Path(a.meta).read_text())
    print(build(pred, meta, a.out, a.note))


if __name__ == "__main__":
    main()
