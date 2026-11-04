# Cricket - Phase 0 market conditions, pre-match (Benter against Betfair's close)

Betfair: `betfair_historic/markets_cricket_2026.parquet` (BASIC: last traded prices, no ladder). Model: walk-forward pre-match probabilities (`cricket/eval/prematch.parquet`).

Markets 1,133; matched to Cricsheet 218; unmatched 915 (no Cricsheet match in scope that day: 904; more than one candidate (a series): 10; no close: 1). In scope are men's 20-over T20s decided without D/L that Cricsheet holds: most unmatched markets are women's, legends and domestic leagues it does not cover.

## Benter regression (all matched matches)

| term | weight | s.e. | z |
|---|---:|---:|---:|
| logit(model p) | -0.079 | 0.445 | -0.18 |
| logit(market p) | 1.223 | 0.267 | 4.58 |

Model weight significantly positive (z above 1.96): **no**

## Model against the close, same matches

| probability | log loss |
|---|---:|
| market close (overround removed) | 0.5226 |
| model (walk-forward, pre-match) | 0.6106 |

Model minus market: +0.0880 (95% interval +0.0508 to +0.1240, 218 matches). Correlation of the two logits: 0.79.

Matched matches by competition: Vitality Blast Men 47; Major League Cricket 21; ICC Men's T20 World Cup Sub Regional Asia Qualifier A 20; ICC Men's T20 World Cup Sub Regional Europe Qualifier B 15; ICC Men's T20 World Cup Africa Sub Regional Qualifier C 15; ICC Men's T20 World Cup Sub Regional Asia Qualifier B 13; Lanka Premier League 12; European T20 Premier League 11; ECA Men's European Cup 8; Caribbean Premier League 7; Asian Games Men's Cricket Competition 6; CSA Pro20 5.

Walk-forward and strategy: not run. They fit on earlier years, and the matched markets cover 2026 only.

## Unmatched markets (first 20)

| market | event | time | why |
|---|---|---|---|
| 1.259351773 | Durham W v Surrey W | 2026-06-23T09:30:00.000Z | 0 candidate matches |
| 1.259519057 | Sweden v Portugal | 2026-07-01T07:30:00.000Z | 2 candidate matches |
| 1.259562807 | Gloucestershire W v Leicestershire Foxes W | 2026-07-03T13:30:00.000Z | 0 candidate matches |
| 1.259562814 | Sussex Sharks W v Northamptonshire W | 2026-07-03T14:00:00.000Z | 0 candidate matches |
| 1.259562823 | Derbyshire Falcons W v Kent W | 2026-07-03T17:30:00.000Z | 0 candidate matches |
| 1.259562830 | Glamorgan W v Middlesex W | 2026-07-03T13:30:00.000Z | 0 candidate matches |
| 1.259563240 | Yorkshire W v Lancashire Thunder W | 2026-07-01T13:30:00.000Z | 0 candidate matches |
| 1.259563961 | The Blaze W v Surrey W | 2026-07-03T13:30:00.000Z | 0 candidate matches |
| 1.259568084 | England W v South Africa W | 2026-07-02T17:30:00.000Z | 0 candidate matches |
| 1.259581568 | Hermes-Dvs v Vcc | 2026-07-04T09:00:00.000Z | 0 candidate matches |
| 1.259581571 | Rcc v Voc Rotterdam | 2026-07-04T09:00:00.000Z | 0 candidate matches |
| 1.259581574 | SV Kampong Cricket v HCC | 2026-07-04T09:00:00.000Z | 0 candidate matches |
| 1.259581577 | Vra v Hbs | 2026-07-04T09:00:00.000Z | 0 candidate matches |
| 1.259581779 | Bengaluru Blasters v Hubli Tigers | 2026-07-01T08:30:00.000Z | 0 candidate matches |
| 1.259584113 | Namibia v Vidarbha Cricket Association | 2026-07-02T07:30:00.000Z | 0 candidate matches |
| 1.259586499 | Anurag Nalgonda Knights v Anvita Khammam Aces | 2026-07-01T13:45:00.000Z | 0 candidate matches |
| 1.259586502 | Hyderabad E-Champions v Warangal Warriors | 2026-07-01T08:45:00.000Z | 0 candidate matches |
| 1.259606033 | Sofia Stars v Young Stars Bulgaria | 2026-07-01T06:30:00.000Z | 0 candidate matches |
| 1.259606037 | Karimnagar Diamonds v Ranga Reddy Risers | 2026-07-02T08:45:00.000Z | 0 candidate matches |
| 1.259606043 | Coastal Kings Mangaluru v Shivamogga Yodhas | 2026-07-02T13:30:00.000Z | 0 candidate matches |
