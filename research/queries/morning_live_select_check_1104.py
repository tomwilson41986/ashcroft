"""The live selection (morning/select.py: the 06:00 card put into the history) against the research's (the day's rows
of the results table), on past days whose 06:00 card is archived, and the selections for today (4 Nov 2026)."""
import io, os, sys, time
from datetime import date, timedelta
import boto3, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research/strat")
import betfair_prices as bp
from research.paper import morning_edges as ME, morning_features as MF
from morning import select as S

b = os.environ.get("PRED_BUCKET", "ashcroft")
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION", "eu-west-2"))
class G:
    def __call__(self, k): return s3.get_object(Bucket=b, Key=k)["Body"].read()
    def ls(self, p): return [o["Key"] for o in s3.list_objects_v2(Bucket=b, Prefix=p).get("Contents", [])]
get = G()
t0 = time.time()
hist = MF.L.load(); print(f"history {len(hist):,} rows ({time.time()-t0:.0f}s)")
print("table last day:", hist.race_date.max(), "race_time sample:", hist[hist.race_date == hist.race_date.max()].race_time.unique()[:12].tolist())
print("table jockeys with a bracket:", int(hist.jockey_name.fillna("").str.contains(r"\(").sum()),
      hist[hist.race_date >= "2026-10-01"].jockey_name.head(5).tolist(), "claim", hist.jockeys_claim.value_counts().head(4).to_dict())
booster, thr, meta = ME.load_model(); print("threshold", thr)
full = MF.build(hist); print(f"full build {time.time()-t0:.0f}s")
def key(df):
    return set(zip(df["model"], df["track"].map(bp.normalise_track), df["race_time"].map(bp.race_time_to_24h),
                   df["horse_name"].map(bp.normalise_horse)))
rows = []
for d in [date(2026, 10, 27) + timedelta(i) for i in range(9)]:
    day = d.isoformat()
    card = S.read_card(get, day)
    if card is None:
        print(day, "no card"); continue
    t1 = time.time()
    fr = S.history_with_card(hist, card, day)
    a = S.select(fr, day, booster, thr)
    x = full[full.race_date == day].copy()
    if len(x):
        x["pred"] = booster.predict(x[MF.CLEAN])
        r = pd.concat([x[MF.rule_qr(x)].assign(model="quick_return"),
                       x[(x.pred >= thr) & (x.field >= 5)].assign(model="steamer")])
    else:
        r = pd.DataFrame(columns=a.columns)
    ka, kr = key(a), key(r)
    for m in ("quick_return", "steamer"):
        A = {k for k in ka if k[0] == m}; R = {k for k in kr if k[0] == m}
        rows.append({"day": day, "model": m, "live": len(A), "research": len(R), "both": len(A & R),
                     "live_only": len(A - R), "research_only": len(R - A)})
    # diagnostics on the matched runners: code and the features
    cd = S.card_rows(card, day, hist)
    tab = hist[hist.race_date == day]
    k1 = cd.track.map(bp.normalise_track) + "|" + cd.race_time.map(bp.race_time_to_24h) + "|" + cd.horse_name.map(bp.normalise_horse)
    k2 = tab.track.map(bp.normalise_track) + "|" + tab.race_time.astype(str).map(bp.race_time_to_24h) + "|" + tab.horse_name.map(bp.normalise_horse)
    m = cd.assign(k=k1.to_numpy()).merge(tab.assign(k=k2.to_numpy()), on="k", suffixes=("_c", "_t"))
    if len(m):
        print(f"{day}: card {len(cd)} runners, {len(m)} matched to the table; code agrees {np.mean(m.code_c == m.code_t):.3f}; "
              f"claim agrees {np.mean(m.jockeys_claim_c.astype(float) == m.jockeys_claim_t.fillna(0).astype(float)):.3f}; "
              f"jockey agrees {np.mean(m.jockey_name_c == m.jockey_name_t):.3f}; trainer {np.mean(m.trainer_c == m.trainer_t):.3f}; "
              f"dslr {np.mean(m.days_since_lr_c.astype(float) == m.days_since_lr_t.astype(float)):.3f}; ({time.time()-t1:.0f}s)")
        bad = m[m.code_c != m.code_t][["race_name_c", "going_description_c", "code_c", "code_t"]].drop_duplicates().head(8)
        if len(bad): print(bad.to_string())
        badj = m[m.jockey_name_c != m.jockey_name_t][["jockey_name_c", "jockey_name_t"]].drop_duplicates().head(5)
        if len(badj): print(badj.to_string())
    # where a selection differs, why
    for mdl in ("quick_return", "steamer"):
        lo = [k for k in ka - kr if k[0] == mdl][:3]; ro = [k for k in kr - ka if k[0] == mdl][:3]
        if lo or ro: print(f"  {mdl} live-only {lo} research-only {ro}")
print(pd.DataFrame(rows).to_string())
t = pd.DataFrame(rows).groupby("model")[["live", "research", "both"]].sum(); t["agree"] = t.both / t[["live", "research"]].max(axis=1)
print(t.to_string())
day = "2026-11-04"; card = S.read_card(get, day)
if card is not None:
    sel = S.select(S.history_with_card(hist, card, day), day, booster, thr)
    print(f"\n{day}: {len(sel)} selections"); print(sel.groupby("model").size().to_string())
    print(sel.sort_values(["race_time", "track"]).head(40).to_string())
