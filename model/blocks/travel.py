"""Travel: how far the yard has sent the horse, against how far it usually sends them.

A trainer who ships a horse three hundred miles, or across the Irish Sea, to a course
the yard rarely visits has chosen the race; the market hears about it through the
day, and a forecast of the BSP made at 06:00 can anticipate it. No other block reads
where a yard is or where it runs. The yard's base is the centre of its runners'
courses on earlier days (decayed, half-life 365 days); a course is placed by its
coordinates (COURSES: the 86 British and Irish courses, to a few km). All of it is
on the 06:00 card (course, trainer, declarations) or from earlier days, and no result
is read:

    tv_km             great-circle km from today's course to the yard's base, once the
                      yard has five runners' worth of history
    tv_usual          the yard's usual trip: its earlier runners' tv_km, decayed the same way
    tv_km_vs_usual    tv_km less tv_usual
    tv_km_ratio       log((tv_km + 10) / (tv_usual + 10))
    tv_raid           1 when today's course is in the other country (Britain or Ireland)
                      from the one most of the yard's runners ran in, else 0
    tv_track_share    the yard's share of its runners that ran at today's course (decayed)
    tv_n_meeting      the yard's runners at today's meeting
    tv_km_single      tv_km when the horse is the yard's only runner at the meeting, else 0
    tv_horse_km       km from the course of the horse's last run (an earlier day) to today's

A course missing from COURSES has no position: its runners get NaN for every distance,
and its runs do not move a yard's base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import _codes, asof_decayed_mean, day_index, race_key

FEATURES = ["tv_km", "tv_usual", "tv_km_vs_usual", "tv_km_ratio", "tv_raid", "tv_track_share", "tv_n_meeting",
            "tv_km_single", "tv_horse_km"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "trainer"]
READS_RESULTS = False
HALFLIFE_DAYS = 365.0
MIN_RUNNERS = 5.0           # decayed runners before a yard has a base

#: Course (normalised: lower case, parenthetical and non-letters removed) -> (lat, lon, 1 for Ireland).
COURSES = {
    # Great Britain
    "aintree": (53.477, -2.951, 0), "ascot": (51.412, -0.673, 0), "ayr": (55.460, -4.614, 0),
    "bangor": (52.998, -2.915, 0), "bath": (51.417, -2.409, 0), "beverley": (53.846, -0.459, 0),
    "brighton": (50.832, -0.114, 0), "carlisle": (54.869, -2.917, 0), "cartmel": (54.199, -2.953, 0),
    "catterick": (54.381, -1.636, 0), "chelmsford": (51.764, 0.449, 0), "cheltenham": (51.921, -2.061, 0),
    "chepstow": (51.652, -2.683, 0), "chester": (53.187, -2.899, 0), "doncaster": (53.518, -1.106, 0),
    "epsom": (51.316, -0.257, 0), "exeter": (50.628, -3.599, 0), "fakenham": (52.826, 0.826, 0),
    "ffoslas": (51.749, -4.230, 0), "fontwell": (50.858, -0.635, 0), "goodwood": (50.893, -0.757, 0),
    "hamilton": (55.782, -4.033, 0), "haydock": (53.476, -2.628, 0), "hereford": (52.070, -2.730, 0),
    "hexham": (54.962, -2.113, 0), "huntingdon": (52.335, -0.174, 0), "kelso": (55.605, -2.425, 0),
    "kempton": (51.418, -0.402, 0), "leicester": (52.600, -1.098, 0), "lingfield": (51.176, -0.017, 0),
    "ludlow": (52.382, -2.724, 0), "marketrasen": (53.384, -0.321, 0), "musselburgh": (55.946, -3.050, 0),
    "newbury": (51.397, -1.302, 0), "newcastle": (55.012, -1.668, 0), "newmarket": (52.246, 0.378, 0),
    "newtonabbot": (50.535, -3.595, 0), "nottingham": (52.946, -1.106, 0), "perth": (56.421, -3.432, 0),
    "plumpton": (50.933, -0.059, 0), "pontefract": (53.694, -1.334, 0), "redcar": (54.612, -1.071, 0),
    "ripon": (54.124, -1.506, 0), "salisbury": (51.049, -1.847, 0), "sandown": (51.375, -0.359, 0),
    "sedgefield": (54.648, -1.462, 0), "southwell": (53.066, -0.919, 0), "stratford": (52.183, -1.723, 0),
    "taunton": (51.000, -3.083, 0), "thirsk": (54.227, -1.329, 0), "towcester": (52.128, -0.989, 0),
    "uttoxeter": (52.899, -1.856, 0), "warwick": (52.285, -1.594, 0), "wetherby": (53.930, -1.370, 0),
    "wincanton": (51.052, -2.407, 0), "windsor": (51.486, -0.628, 0), "wolverhampton": (52.603, -2.133, 0),
    "worcester": (52.198, -2.228, 0), "yarmouth": (52.617, 1.726, 0), "york": (53.943, -1.089, 0),
    # Ireland (Down Royal and Downpatrick race under the Irish authority)
    "ballinrobe": (53.630, -9.237, 1), "bellewstown": (53.690, -6.352, 1), "clonmel": (52.358, -7.690, 1),
    "cork": (52.130, -8.640, 1), "curragh": (53.163, -6.824, 1), "downroyal": (54.504, -6.143, 1),
    "downpatrick": (54.320, -5.719, 1), "dundalk": (54.024, -6.428, 1), "fairyhouse": (53.490, -6.533, 1),
    "galway": (53.298, -8.992, 1), "gowranpark": (52.627, -7.061, 1), "kilbeggan": (53.365, -7.494, 1),
    "killarney": (52.045, -9.505, 1), "laytown": (53.683, -6.237, 1), "leopardstown": (53.268, -6.198, 1),
    "limerick": (52.576, -8.766, 1), "listowel": (52.444, -9.487, 1), "naas": (53.214, -6.660, 1),
    "navan": (53.638, -6.702, 1), "punchestown": (53.197, -6.625, 1), "roscommon": (53.628, -8.178, 1),
    "sligo": (54.269, -8.466, 1), "thurles": (52.684, -7.817, 1), "tipperary": (52.475, -8.149, 1),
    "tramore": (52.158, -7.146, 1), "wexford": (52.339, -6.483, 1),
}
ALIASES = {"chelmsfordcity": "chelmsford", "bangorondee": "bangor", "kemptonpark": "kempton",
           "sandownpark": "sandown", "haydockpark": "haydock", "epsomdowns": "epsom", "lingfieldpark": "lingfield",
           "greatyarmouth": "yarmouth", "stratfordonavon": "stratford", "fontwellpark": "fontwell",
           "hamiltonpark": "hamilton", "catterickbridge": "catterick", "thecurragh": "curragh", "mallow": "cork",
           "royalwindsor": "windsor"}


def course_name(track: pd.Series) -> pd.Series:
    """'Newmarket (Rowley)' -> 'newmarket', 'Chelmsford City' -> 'chelmsford'."""
    t = (track.fillna("").astype(str).str.lower().str.replace(r"\(.*?\)", "", regex=True)
         .str.replace(r"[^a-z]", "", regex=True))
    return t.replace(ALIASES)


def km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (NaN where any coordinate is)."""
    r = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * r / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2)
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _name_codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    day = day_index(df)
    course = course_name(df["track"])
    table = pd.DataFrame.from_dict(COURSES, orient="index", columns=["lat", "lon", "ire"])
    lat = course.map(table["lat"]).to_numpy(float)
    lon = course.map(table["lon"]).to_numpy(float)
    ire = course.map(table["ire"]).to_numpy(float)
    tr = _name_codes(df["trainer"])

    blat, n_base = asof_decayed_mean(tr, day, lat, tr, day, HALFLIFE_DAYS)
    blon, _ = asof_decayed_mean(tr, day, lon, tr, day, HALFLIFE_DAYS)
    tv_km = np.where(n_base >= MIN_RUNNERS, km(lat, lon, blat, blon), np.nan)
    usual, n_usual = asof_decayed_mean(tr, day, tv_km, tr, day, HALFLIFE_DAYS)
    tv_usual = np.where(n_usual >= MIN_RUNNERS, usual, np.nan)
    ire_share, n_ire = asof_decayed_mean(tr, day, ire, tr, day, HALFLIFE_DAYS)
    raid = np.where((n_ire >= MIN_RUNNERS) & np.isfinite(ire), ((ire_share > 0.5) != (ire > 0.5)).astype(float),
                    np.nan)

    placed = (tr >= 0) & np.isfinite(lat)
    yard_course = np.where(placed, _codes(tr, course.to_numpy()), -1).astype(np.int64)
    ones = np.ones(n)
    _, n_here = asof_decayed_mean(yard_course, day, ones, yard_course, day, HALFLIFE_DAYS)
    _, n_all = asof_decayed_mean(tr, day, np.where(np.isfinite(lat), 1.0, np.nan), tr, day, HALFLIFE_DAYS)
    share = np.where(placed & (n_all > 0), n_here / (n_all + 1.0), np.nan)

    meeting = pd.factorize(df["race_date"].astype(str) + "|" + course, sort=True)[0]
    n_meet = pd.Series(ones).groupby([meeting, tr]).transform("sum").to_numpy(float)
    n_meet = np.where(tr >= 0, n_meet, np.nan)

    # the horse's last course: its previous run on an earlier day, in a fixed order within the day
    h = _name_codes(df["horse_name"])
    race = pd.factorize(race_key(df), sort=True)[0]
    o = np.lexsort((race, day, h))
    hs, ds = h[o], day[o]
    earlier = np.r_[False, (hs[1:] == hs[:-1]) & (ds[1:] > ds[:-1])] & (hs >= 0)
    # a horse's previous row may be a second run on that same earlier day: carry the day's last run
    plat, plon = np.full(n, np.nan), np.full(n, np.nan)
    plat[o] = np.where(earlier, np.r_[np.nan, lat[o][:-1]], np.nan)
    plon[o] = np.where(earlier, np.r_[np.nan, lon[o][:-1]], np.nan)
    # a second run on the same day as the first takes the first's previous course, not the first's own
    same_day = np.r_[False, (hs[1:] == hs[:-1]) & (ds[1:] == ds[:-1])]
    if same_day.any():
        pl, pn = plat[o], plon[o]
        for i in np.flatnonzero(same_day):
            pl[i], pn[i] = pl[i - 1], pn[i - 1]
        plat[o], plon[o] = pl, pn

    new = pd.DataFrame({
        "tv_km": tv_km,
        "tv_usual": tv_usual,
        "tv_km_vs_usual": tv_km - tv_usual,
        "tv_km_ratio": np.log((tv_km + 10.0) / (tv_usual + 10.0)),
        "tv_raid": raid,
        "tv_track_share": share,
        "tv_n_meeting": n_meet,
        "tv_km_single": np.where(np.isfinite(tv_km), np.where(n_meet == 1, tv_km, 0.0), np.nan),
        "tv_horse_km": km(lat, lon, plat, plon),
    }, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
