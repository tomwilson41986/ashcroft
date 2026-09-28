# Model verification: PASS

- 993 features, target race_xent, best_iteration 12600, trained through 2026-09-22
- against the served model (615 features): +380 (bk_jk_mkt, bk_jk_rides_on_horse, bk_jk_same, bk_jk_upgrade, bk_tr_mkt, bk_tr_mkt_trend ...), -2 (damsire_runners, sire_runners)
- drop-in blocks built on the matrix as the live path builds them: bookings, connection_windows, debut_market, exposure, form_lines, form_variants, handicap_angles, head_to_head, kalman, race_relative, race_relative_wide, time_figure, travel
- 5,225 runners in 525 races, 2026-09-09 to 2026-09-22; worst book error 2.2e-16; mean absolute log error against BSP: new 0.2944 (in-sample)
- against the served model: correlation of log prices 0.9823; the served model's error on the same rows 0.3125

No problems found.
