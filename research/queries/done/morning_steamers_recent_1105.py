"""How the steamer model would have done on 3 and 4 Nov 2026 (after the replay window), traded as the live model
trades: selections from that day's 06:00 card (morning/select.py), backed WIN at the recorded 10:00 UK book down to 95%
of the best back, the live trader's horses left out, held to the result. The paper session stakes to win GBP100;
P&L is halved to the live GBP50. Commission 2% (the account) and 5% on each market's net."""
import os, sys
from datetime import date
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
hist = MF.L.load(); print("table last day", hist.race_date.max())
booster, thr, _ = ME.load_model()
rows = []
for d in ("2026-11-03", "2026-11-04"):
    card = S.read_card(G(), d)
    if card is None:
        print(d, "no card"); continue
    sel = S.select(S.history_with_card(hist, card, d), d, booster, thr)
    sel = sel[sel.model == "steamer"].assign(strategy="steam_am")
    print(f"\n{d}: {len(sel)} steamer selections")
    bets, D, live = ME.run_market(date.fromisoformat(d), "win", sel, None)
    if D is None:
        print("  no recorded books for the day"); continue
    if bets.empty:
        print("  no ledger rows"); continue
    st = ME.settle(bets[bets.event == "back"], D)
    starts = {m.market_id: m for m in D.markets}
    for r in bets.itertuples():
        m = starts.get(str(r.market_id))
        s_ = st[(st.market_id == r.market_id) & (st.selection_id == r.selection_id)] if len(st) else st
        rows.append({"day": d, "off": bp.utc_iso_to_uk_hhmm(m.start.strftime("%Y-%m-%dT%H:%M:%S.000Z")) if m else "",
                     "venue": m.venue if m else "", "horse": r.label, "event": r.event,
                     "best_back": r.best_back, "price": r.price, "stake": round(r.matched / 2, 2),
                     "bsp": s_.bsp.iloc[0] if len(s_) else np.nan,
                     "result": s_.result.iloc[0] if len(s_) else "",
                     "pnl": round(float(s_.pnl.iloc[0]) / 2, 2) if len(s_) else 0.0, "market_id": r.market_id})
R = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(R.drop(columns="market_id").to_string())
for d, g in R.groupby("day"):
    m = g[g.event == "back"]
    pm = m.groupby("market_id")["pnl"].sum()
    c2 = float((pm - 0.02 * pm.clip(lower=0)).sum()); c5 = float((pm - 0.05 * pm.clip(lower=0)).sum())
    e = float((1 / m.price).sum()) if len(m) else 0.0
    print(f"{d}: {len(g)} selections, {len(m)} backed, {int((g.event=='skipped_live').sum())} the live trader's; "
          f"staked GBP{m.stake.sum():.2f}; winners {int((m.result=='WINNER').sum())} vs {e:.2f} expected; "
          f"P&L GBP{c2:+.2f} at 2% (GBP{c5:+.2f} at 5%)")
