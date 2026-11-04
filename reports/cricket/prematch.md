# Cricket - pre-match probabilities, walk-forward (the model side of the Benter test)

Source: `data/cricket/store/cricket/tables`. Built by `python -m cricket.model.prematch`. The state before the first ball, after the toss; each year from a model, rounds and prior from before it. The constant is the share of matches won by the side batting first in every in-scope match before that year.

| year | matches | rounds | model log loss | constant log loss | constant | model ECE |
|---|---:|---:|---:|---:|---:|---:|
| 2016 | 484 | 44 | 0.6768 | 0.6910 | 0.492 | 0.0445 |
| 2017 | 385 | 47 | 0.6988 | 0.6926 | 0.477 | 0.0595 |
| 2018 | 586 | 65 | 0.6671 | 0.6926 | 0.478 | 0.0471 |
| 2019 | 872 | 52 | 0.6635 | 0.6903 | 0.479 | 0.0424 |
| 2020 | 380 | 78 | 0.6654 | 0.6912 | 0.474 | 0.0364 |
| 2021 | 834 | 82 | 0.6712 | 0.6914 | 0.473 | 0.0464 |
| 2022 | 898 | 72 | 0.6668 | 0.6956 | 0.473 | 0.0524 |
| 2023 | 850 | 95 | 0.6555 | 0.6902 | 0.478 | 0.0126 |
| 2024 | 1,149 | 95 | 0.6366 | 0.6931 | 0.475 | 0.0295 |
| 2025 | 1,076 | 95 | 0.6491 | 0.6916 | 0.477 | 0.0361 |
| 2026 | 1,005 | 95 | 0.6283 | 0.6912 | 0.477 | 0.0339 |
| all | 8,519 | | 0.6569 | 0.6920 | | 0.0204 |

Model minus constant: -0.0351 (95% interval -0.0396 to -0.0309, one row a match).
The test that matters is against Betfair's close (cricket/model/benter.py): the constant only shows the model knows something before the first ball.
