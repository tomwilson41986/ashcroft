"""Shared loader for the times and pace audits (14 Nov 2026): GBGB runs by year as parquet (sources/gbgb.py's runs
tables), from the folder in ``GBGB_RUNS_DIR`` (default ``data/gbgb``). Cleaned: duplicates of (race, dog) dropped;
trials, handicaps and hurdles flagged; GBGB's going outside +-150 flagged as an error."""
import os
import numpy as np, pandas as pd
SRC = os.getenv("GBGB_RUNS_DIR", "data/gbgb")
YEARS = tuple(int(y) for y in os.getenv("GBGB_RUNS_YEARS", "2025,2026").split(","))
L = 0.08


def load():
    d = pd.concat([pd.read_parquet(f"{SRC}/runs_{y}.parquet") for y in YEARS], ignore_index=True)
    n0 = len(d)
    d = d.drop_duplicates(["race_id", "dog_id"])
    d["day"] = pd.to_datetime(d.race_date)
    d["cls"] = d.race_class.astype(str)
    d["fam"] = d.cls.str.extract(r"^([A-Z]+)")[0]
    d["trial"] = d.cls.str.startswith("T")
    d["hcap"] = d.handicap.fillna(False) | (d.cls == "HP")
    d["hurdle"] = d.race_type.eq("Hurdles") | d.fam.eq("H")
    d["key"] = d.track + "|" + d.distance_m.round().astype(int).astype(str)
    d["going_ok"] = d.going.between(-150, 150)
    d.attrs["n_raw"] = n0
    return d


def graded_winners(d):
    m = (d.position == 1) & ~d.trial & ~d.hcap & ~d.hurdle & d.run_time.notna() & d.going_ok
    return d[m].copy()
