# Model verification: PASS

- 615 features, target logit_norm_prob, best_iteration 6000, trained through 2026-09-22
- against the served model (535 features): +80 (fw_ae_car, fw_ae_l1, fw_ae_m3, fw_ae_m5, fw_ae_w10, fw_ae_w3 ...), -0 ()
- 5,225 runners in 525 races, 2026-09-09 to 2026-09-22; worst book error 2.2e-16; mean absolute log error against BSP: new 0.2779 (in-sample)
- against the served model: correlation of log prices 0.9930; the served model's error on the same rows 0.3543

No problems found.
