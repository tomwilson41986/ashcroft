# Cricket research log - the search for an edge

The owner, 4 Nov 2026: "keep working in loops until you define a true edge and profitable strategy". This log keeps
that search honest. Testing many ideas on the same markets until one looks good is how a false edge is made, so:

## Protocol (fixed 4 Nov 2026, before any in-play result was seen)

1. **Data.** Betfair BASIC cricket, the owner's bundle (Jul-Oct 2026): the last traded price each minute and each
   market's definition and result; no ladder, no volume. Cricsheet for ball-by-ball and results.
2. **Holdout.** Markets whose scheduled start is on or after 2026-10-01 are the holdout: no statistic is computed
   on them until a strategy is frozen (its rule, thresholds and staking written here), then once. A strategy that
   fails the holdout is not retuned on it.
3. **Discovery.** Markets before 2026-10-01. Each hypothesis is written here before it is run, with its primary
   test; anything else is labelled exploratory.
4. **Profit is net.** Every bet pays 5% commission on net market winnings and is priced one tick worse than the
   last traded price (BASIC does not show the price was there to take). Standard errors resample whole matches.
5. **Every try is recorded**, the failures included, with the count of tries so far.
6. **A true edge** is: positive net return on the holdout, with the discovery result's 95% interval above zero,
   and a mechanism that makes sense. Anything less is reported as not found.

## Tries

| # | Date | Hypothesis | Data | Result | Decision |
|---|---|---|---|---|---|
| 1 | 4 Nov | The pre-match model adds to Betfair's close (Benter) | 218 matches, Jul-Oct | model weight -0.08 (s.e. 0.45); model 0.6106 vs market 0.5226 log loss | no edge (`benter_prematch.md`) |
| 2 | 4 Nov | The chase model adds to Betfair's price at the innings break | 123 matches, Jul-Oct | weight 0.02 (s.e. 0.27); model 0.6102 vs market 0.5331 | no edge (`innings_break_probe.md`) |

| 3 | 4 Nov | H1: lay in-play longshots (first minute under 0.20), one a market | 690 bets, discovery | ROI -4.1% (95% -7.6% to -0.7%) | fail: no favourite-longshot bias to lay |
| 4 | 4 Nov | H2: back the side after a 0.10 jump against it, a minute on | 1,236 bets, discovery | ROI -6.0% (-14.7% to +2.7%) | fail |
| 4x | 4 Nov | exploratory: H2 at 0.05 / 0.20; momentum at 0.05 / 0.10 / 0.20; backing H1's longshots | discovery | -4.8%, -23.1%; -1.0%, -3.5%, -10.0%; +3.3% (-13.2% to +20.3%) | nothing clears zero |

Calibration in play (discovery, a checkpoint every 10 minutes, both runners): every band within about two points of
its implied probability. Under 0.05 implied 0.025, won 0.037 (1,269 runner-minutes, 493 markets); 0.95-1.0 implied
0.975, won 0.963. Seen in discovery, so only a candidate: it can be tested on the holdout once, frozen first.

Tries 1 and 2 used every month, before this protocol existed: October is not untouched for those two
questions, and neither may be revisited as a holdout test.

## Registered before running (4 Nov 2026)

**Panel** (`cricket/market/inplay_panel.py`): 1,130 two-runner Match Odds markets, 701 discovery, 429 holdout;
317,609 market-minutes. A usable price: the market open at the minute, both runners traded within the last 5 minutes
(H1) or the last minute (H2).

**H1 - in-play favourite-longshot bias.** Longshots in play win less often than their price implies, so laying them
pays. *Primary rule:* in each market, the first minute at which a runner's implied probability is under 0.20 with a
usable price, lay that runner one tick above its last traded price, GBP 10 liability. One bet a market. Commission 5%
on the market's net winnings. *Pass in discovery:* ROI on liability above zero with the match-resampled 95% interval
above zero. *Exploratory, reported:* the calibration curve by implied-probability band; backing instead of laying.

**H2 - overreaction to big moves.** After a sharp move (a wicket or a boundary in the price), the market overshoots,
so backing the side the move went against pays. *Primary rule:* a jump is a change of at least 0.10 in runner A's
implied probability between consecutive minutes, both open with prices under a minute old. One minute after the jump
minute (bet delay and settling), if still open and fresh, back the side whose probability fell, one tick below its
last traded price, GBP 10. Every jump a bet; commission on each market's net. *Pass in discovery:* as H1.
*Exploratory, reported:* momentum (back the side that gained), thresholds 0.05 and 0.20.

**H5 - the first-innings runs line, pre-match** (registered 4 Nov, before running). Betfair's line markets are even
money on the total against the line (the line is the last traded value: the market's median). The simulator's
pre-match runs distribution (the walk-forward 2026 model, the toss known, 1,000 paths) may place the median better.
*Primary rule:* each matched in-scope event with a `1_INNING_20_OVR_LINE` close; p = the simulator's P(total >
line); if p >= 0.55 back over at the line plus one run, if p <= 0.45 back under at the line minus one run; GBP 10 at
even money, 5% commission on winnings; settled on Cricsheet's first-innings total. Discovery: markets before
2026-10-01. *Pass:* as H1. *Exploratory:* the 6-over line on the same rule; the model's and the line's median
errors.

| 5 | 4 Nov | H5: the simulator's runs distribution against the first-innings lines (edge 0.05) | discovery: 83 20-over, 101 6-over markets | 20-over: ROI +3.2% (68 bets; 95% -19.7% to +26.2%), line MAE 26.9 vs model 29.5; 6-over: +0.6% (62 bets; -24.5% to +25.8%), MAE 11.1 vs 11.0 | fail: no edge from the model |

Exploratory, from try 5: only 39.6% of first-innings 6-over totals went over the closing line (101 markets). The
last in-play line trade was checked as a stand-in result for events Cricsheet lacks: it is within half a run at the
median but on the wrong side of the close in 6.2% of 96 markets, too often to settle on. Cricsheet results only.

**H6 - the powerplay line is set too high** (registered 4 Nov, before running). Backers favour overs, so 6-over
lines close above the median. *Primary rule:* back UNDER at the closing line minus one run, GBP 10 even money, 5%
commission on winnings. *Test 1 (discovery, a market family not yet looked at):* every matched in-scope event's
`2_INNING_6_OVR_LINE` before 2026-10-01, settled on Cricsheet's second-innings runs in overs 1-6 (the chase's runs
if it ended sooner). *Test 2 (holdout, once):* both 6-over lines, October 2026. *Pass:* test 1's ROI interval above
zero, then test 2's ROI above zero. *Exploratory:* the 10- and 15-over lines.

| 6 | 4 Nov | H6 test 1: under on the second-innings 6-over line at the pre-match close | discovery: 8 markets with a pre-match close | ROI +46% (8 bets; 95% -2.5% to +95%) | inconclusive: too few markets (the chase's lines trade in play, not before the off) |

**H6 test 1b** (registered 4 Nov, before running): the same rule priced where a chase line actually trades: the
`2_INNING_6_OVR_LINE` last traded value standing 5 minutes after the innings break (cricket/model/innings_break.py's
break: the earliest in-play suspension of a first-innings market), traded within the 10 minutes before. Discovery
markets before 2026-10-01; settled on Cricsheet's chase runs in overs 1-6. *Pass:* ROI interval above zero.
| 7 | 4 Nov | H6 test 1b: under on the second-innings 6-over line priced at the break | discovery: 65 markets | 35.4% over the line, mean result minus line -4.6 runs; UNDER at line - 1 wins 61.5%, ROI +20.0% (95% -4.0% to +41.1%); exploratory OVER at line + 1: -40.0% (-61.0% to -19.0%) | not passed (lower bound below zero), but the same direction as the first-innings 6-over lines: the lead to pursue |
| 7x | 4 Nov | exploratory scan of every line type, discovery | 1st 6/10/15/20-over at the close; 2nd 10/15/20 at the break | 6-over: see 5 and 7; 20-over 50.6% over (83), no bias; 10/15/20 too few (3-27) | the effect is the powerplay lines' |

## Frozen: strategy S1, powerplay unders (4 Nov 2026, before the holdout was read)

- Markets: `1_INNING_6_OVR_LINE` priced at its last pre-match trade; `2_INNING_6_OVR_LINE` priced at its last trade
  standing 5 minutes after the innings break (traded within the 10 minutes before). Matched in-scope men's T20s.
- Bet: UNDER at the price's line minus one run, GBP 10, even money, 5% commission on winnings.
- Settled on Cricsheet: the innings' runs in overs 1-6.
- Holdout: markets scheduled on or after 2026-10-01, read once now. Discovery did not clear its bar (the chase
  family's interval reached -4.0%), so a positive holdout counts as support, not proof: the edge is only declared
  with the full BASIC history's markets as well.
| 8 | 4 Nov | S1 on the October holdout, once | 29 markets (14 first-innings at the close, 15 chase at the break) | unders won 48.3%; ROI -5.9% (95% -39.5% to +27.8%), P&L -GBP 17 on GBP 290 | not confirmed: S1 is not an edge on this evidence (the holdout is small; it is not retuned) |

Settlement stand-in for events Cricsheet lacks (`cricket/research/line_proxy.py`), checked on all 190 Cricsheet-settled
6-over lines (both innings, every month): the line's last in-play trade before the market's first in-play suspension,
used only when that trade is within a minute of the suspension and at least 2 runs from the bet's line: 64 of 64
agree with Cricsheet on whether the under won (all 190 unfiltered: 94.2%).

**H7 - S1 on events Cricsheet does not cover** (registered 4 Nov, before running). S1's rule unchanged, on every
6-over line market (both innings) of the bundle's events not matched to Cricsheet (women's, domestic and other
leagues: no line-market test has used them), every month, settled by the stand-in where it applies (others are
dropped and counted). *Pass:* ROI interval above zero. *Exploratory:* men's and women's apart.
| 9 | 4 Nov | H7: S1 on events Cricsheet does not cover, settled by the stand-in | 94 markets settled (163 not settleable, 152 no S1 price), every month | 48.9% over; unders won 51.1%; ROI -0.4% (95% -19.1% to +18.2%); exploratory first-innings +9.5%, chase -15.7%, men +3.1%, women -10.6% | fail: with try 8, S1 is rejected. The discovery pattern did not recur |

**H8 - late money: the drift into the close carries information the close does not** (registered 4 Nov, before
running). *Primary test:* every two-runner Match Odds market with a winner and both runners' prices traded at 60
minutes before the scheduled off and at the close, discovery months; logistic regression of runner A winning on
logit(close p) and the drift logit(close p) - logit(t-60 p); the drift's weight significantly away from zero (z beyond
1.96). If it is, the strategy (back the runner the money came for, or against, by the sign found) goes to the
holdout once. *Exploratory:* the t-15 drift.

**H9 - the tie is overpriced** (registered 4 Nov, before running). *Primary rule:* every `TIED_MATCH` market with a
closing price for "Yes", discovery months: lay "Yes" at one tick above the close, GBP 10 liability, 5% commission;
settled on Betfair's result. *Pass:* ROI interval above zero.

| 10 | 4 Nov | H8: the drift from 60 minutes out to the close predicts beyond the close | 611 markets, discovery | drift weight 0.43 (s.e. 0.37, z 1.14); exploratory 15-minute drift z 1.65 | fail |
| 11 | 4 Nov | H9: lay the tie at one tick above the close | 242 markets, discovery: 3 ties, mean implied 4.86% | ROI +31.4% (95% +6.7% to +66.6%), passing the bar, but all of it from 12 markets whose tie closed at 1.02-8.0 (implying 12-98% for a 1% event: stray trades, not prices to take). The 225 at 30 or more: 1 tie in 79 at 30-50 (ROI +0.8%), 2 in 146 above 50 (-0.2%) | not an edge: the pass is an artefact; a filter chosen after seeing it would not be a test |

Independent check for H9 (Cricsheet, no Betfair data): 130 ties in 12,597 20-over T20s with a result, 2016-2026,
1.03% (men 1.03%, women 1.04%; 0.17%-2.66% a year). Betfair's realistic tie prices (50-75, 1.3-2.0%) are only a
little above it: after 5% commission and a tick, nothing is left.

## Where the search stands (4 Nov 2026, 11 tries)

No edge found. Betfair's cricket prices are well calibrated pre-match, at the innings break and minute by minute in
play; the model adds nothing to them; the powerplay-under pattern did not recur on the holdout or on 94 independent
events; the tie market's only apparent edge was stray prices. What the evidence leaves open, each needing data this
bundle does not hold:

1. **More markets.** Four months is too few for the structural tests (lines, ties, minor leagues): the full BASIC
   history (2016-2026, free, the owner's download) would give thousands.
2. **The order books of thin markets.** BASIC holds trades, not what was on offer. Whether thin markets (ties, lines,
   minor leagues) leave prices a resting order could take, the stray tie prices among them, needs the read-only
   recorder (`cricket-record.yml`) running on a UK or Irish host.
3. **Speed.** The one in-play edge the evidence does not touch is reacting to a ball before the price does: a fast
   feed against each market's bet delay. The recorder measures the delay; the feed costs money.
