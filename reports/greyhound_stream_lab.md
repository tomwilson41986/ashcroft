# Greyhound stream lab: registration, 5 Nov 2026

The owner, 5 Nov 2026: "just keep an open mind. Our goal is to build a profitable strategy, i dont care which one it
is". The rules below are fixed before any of their results is seen. They are scored every night on the day's
recorded Exchange Stream (`greyhound/stream_record.py`, from 5 Nov) by `research/greyhound/stream_lab.py`
(`greyhound-stream-lab.yml`, 22:35 UTC), and the days are pooled.

**Fills.** Every rule takes the price on offer at a checkpoint before the scheduled off. A back takes the best back
price and a lay the best lay price, each with at least GBP2 shown there. No queue is assumed. Each trade is closed at
the Betfair SP: a back at B greened by a lay at the SP makes B / SP - 1 a unit staked, a lay at L makes 1 - L / SP.
Commission is 2% on a gain.

**Checkpoints.** 180, 120, 60, 30 and 10 s before the scheduled off. Decisions are taken at 60, 30 and 10 s.

**Rules** (win markets):

| Family | Rule | Parameters |
|---|---|---|
| gap | back where best back / Betfair's projected SP (near) - 1 > m; lay where 1 - best lay / projected SP > m | m in 3%, 6%, 10%; at 60, 30, 10 s |
| mom | back what has shortened by x or more since 180 s (mid); lay what has drifted by x or more | x in 5%, 10%; at 60, 30, 10 s |
| rev | back the drifters; lay the shorteners (the same moves) | as mom |
| fav_lay_10 | lay the favourite at 10 s | control: it should lose about the spread |

That is 18 gap rules, 24 momentum and reversal rules and 1 control: 43 rules.

**What counts.** A rule is a candidate only after 10 recorded days, with its pooled mean P&L's 90% lower bound (a
bootstrap over races) above zero. With 43 rules, about two clear that bound by chance, so a candidate then goes to a
fresh forward run of its own before any money. Nothing here places an order.

**Amendment, 5 Nov 2026, before any rule has 10 days** (the owner: "wire in" the overfitting checks). A candidate
also needs its deflated Sharpe by race (Bailey and Lopez de Prado 2014; `research/overfit.py`) at or above 0.95,
deflated by every rule the lab has scored: the 43 here and any added later, kept in `s3://$CAPTURE_BUCKET/
stream_lab/trials.json`. The 90% bound alone lets about two of 43 through by luck; the deflated Sharpe allows for
the count. The rules, fills and the 10-day bar are unchanged.

