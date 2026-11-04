# Cricket model v1-2026-11-03 - model card

Built 2026-11-04T11:38:55Z by `python -m cricket.model.production`. Paper trading only: no live key.

- Scope: men's 20-over six-ball T20 (match types T20, IT20), decided, no D/L, from 2012.
- Training: 2,258,238 deliveries from 9,688 matches, to 2026-11-03 (Cricsheet).
- Next-ball model: XGBoost multinomial over 0, 1, 2, 3, 4, 6, W, extra; 95 rounds.
- Ratings: Dirichlet c = 60.0, decay r = 0.88 a match, as of 2026-11-04.

## Calibration

- chase: stack, fitted on data/cricket/store/cricket/eval/model_v1_preds.parquet: 32,699 rows, seasons 2023-2026
- first_innings: stack, fitted on data/cricket/store/cricket/eval/model_v1_first_innings_preds.parquet: 5,598 rows, seasons 2023-2026

The calibrators were fitted on the walk-forward per-season models' out-of-sample output, not on this model's own: a model trained on more data may need less (or other) correction. In the evaluation the raw simulator scored better than the calibrated one in both innings (scoring rose faster than layers fitted on earlier seasons follow); every quote carries both (`p_raw`, `p`).

## Held-out scores (walk-forward evaluation)

- chase: 24,271 rows, seasons 2024-2026: calibrated log loss 0.3814, ECE 0.0681; raw simulator 0.3693, ECE 0.0331; baseline 0.4014, ECE 0.0866 (`data/cricket/store/cricket/eval/model_v1_preds.parquet`)
- first innings: 4,180 rows, seasons 2024-2026: calibrated log loss 0.6314, ECE 0.1004; raw simulator 0.6093, ECE 0.0339; baseline 0.6389, ECE 0.0976 (`data/cricket/store/cricket/eval/model_v1_first_innings_preds.parquet`)

## Leaves out

- Batting order beyond the crease is the listed XI order; future bowlers are the XI average.
- No free hits, no runs on wicket balls, no wickets on wides or no-balls.
- No bowler type, matchups, weather or toss effects.
- Not checked against Betfair prices: the Benter combination and the strategy test wait on price history.
