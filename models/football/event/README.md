racing² event models, fitted on the Wyscout public match events (Pappalardo et al. 2019, CC BY 4.0, doi:10.6084/m9.figshare.c.4415000).

- `xt_grid_16x12.csv`: expected threat, 12 rows (y, 0-100 across the pitch) by 16 columns (x, 0-100 towards the opponent's goal).
- `xg_logistic.json`: the logistic xG (dist as log(1 + metres), angle in radians, binary features), penalties apart.

Built by `football/event_models.py`.
