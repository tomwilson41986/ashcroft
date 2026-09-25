# Adding a feature, testing it, retraining and deploying, in minutes

The first real-data answer on a new idea now arrives in about fifteen minutes
(measured), and several ideas can be tested at once. Timings are measured on
the runs named; the two marked *estimate* will be replaced by the next real
retrain and served-recipe run.

| Step | Before (25 Sep morning) | Now | Measured on |
|---|---|---|---|
| Add a feature | Edit the metrics engine; every change rebuilt the 11–17 min feature matrix | One file in `model/blocks/`; no rebuild | — |
| Test it for leaks | Write a leak test by hand | Automatic: `pytest tests/test_feature_blocks.py -k <name>`, 16 s | — |
| First real-data answer | 2h32m (six fits in a row) | **10–15 min** (quick recipe, one job per fold) | research-loop runs 48, 49 |
| Several ideas at once | One per run | Up to six variants against one base in one run | run 49: 6 variants, 14 min |
| Confirm on the served recipe | 2–3 h | 25–45 min, *estimate* (one fold's fit: ~20 min at 3,000 rounds serially) | — |
| Retrain for serving | 1h21m | ~30 min at 3,000 rounds, *estimate*: one fit (28 min), no rebuild, no stage 2 | train run 31 (before) |
| Check and publish | By hand | `scripts/verify_model.py` gate, then commit | — |
| Deploy | Merge a PR by hand, before 06:00 | Merge the PR (the 06:00 run reads the default branch) | — |

## 1. Add a feature: one file

```bash
python scripts/new_block.py going_pref --prefix gp_ --title "Going preference against a baseline."
```

This writes `model/blocks/going_pref.py` from a template that already passes
every test. A block is:

- `FEATURES`: the columns it adds, all with your prefix;
- `build(df) -> df`: adds them. It may read the race records and the metrics
  engine's columns. **Earlier days only**: build every statistic from days
  before the row's own. The template shows the two lag-safe patterns: a
  per-horse window (`_Days`) and a cell mean (`asof_decayed_mean`, shrunk with
  `model/shrinkage.py`);
- a docstring saying what it measures.

Nothing registers it. The engine never imports `model/blocks`, so editing a
block never invalidates the cached feature matrix.

## 2. Test it (seconds, on your machine)

```bash
pytest tests/test_feature_blocks.py -k going_pref
```

The tests run every block through the whole metrics engine on a synthetic
history and check:
- the contract;
- that scrambling or blanking a day's results (the 06:00 card) moves none of
  that day's features;
- that later days do move;
- that row order does not matter;
- that no feature is empty.

A block that reads its own race fails here, before it costs a CI minute. Add a
hand-worked value test beside it (as in `tests/test_form_variants.py`,
`tests/test_time_figure.py`).

## 3. Evaluate it on real data (~15 minutes)

Edit `research/loop.json` and push:

```json
{"tag": "iter36-going-pref", "recipe": "quick", "bfsp": true, "outcome": false,
 "bfsp_base_blocks": "form_windows,form_variants",
 "variants": [{"name": "gp", "blocks": "going_pref"},
              {"name": "gp_no_x", "blocks": "going_pref", "drop": "gp_x"}]}
```

- **`recipe: quick`** screens about seven times faster a fit. On iteration 29
  it reproduced the served recipe's verdict: −0.0066 against −0.0073, the
  same decision.
- **`bfsp_base_blocks`** is what every arm is measured against (the served
  features plus these blocks). The base is fitted once and cached.
- **`variants`**: each is the base plus its own `blocks`, less its `drop`
  prefixes or `withhold` blocks, with its own `args` (for example
  `"--num-boost-round 6000"`). Up to about six run at once.
- To measure what a family is worth, withhold it:
  `{"name": "no_lb", "args": "--drop-feature-list research/lists/family_lb.txt"}`.

The run page and the artifact `research-<tag>-<run>` carry `compare.md`.
It opens with one line per variant: the paired error with its 90% CI, rank 1,
Brier skill against the base, and the decision. Then come the full
comparisons and the early-price trade for each arm.

**The decision rule.** A variant replaces the base only if its paired error
interval excludes zero in its favour, and neither Brier skill nor concordance
is worse by more than its own interval. With `"replicate_base": true` the base
is fitted twice to show the noise floor. Fits are deterministic, so it is zero.

## 4. Confirm on the served recipe (25–45 minutes, estimate)

The same `loop.json` with `"recipe": "served"`, plus `"bfsp_variant_args":
"--num-boost-round 6000"` if the promotion comes with the round cap. The
screen ranks blocks; this run decides.

## 5. Promote a block into the engine

The live 06:00 path builds features with the engine, not from `model/blocks`.
`scripts/verify_model.py` refuses a model that reads a drop-in feature until
the block is promoted. To promote one:
- import its `build` in `model/custom_metrics.py` and call it after the
  existing blocks, behind a flag like `form_windows`;
- add its served features to `model/bfsp_features.py`;
- add the flag to `blocks_needed`.

This is the one change that rebuilds the matrix (once).

## 6. Retrain (about 30 minutes, estimate)

Dispatch **Train BFSP Model** (`train-bfsp.yml`) on the branch with:
- `mode: train_only`;
- `fast: true` (the default): restores the cached matrix and skips the unused
  stage-2 model;
- `fixed_rounds`: the round count the evaluation's folds reached, or empty to
  choose it on a holdout (twice the time);
- `extra_args` as needed;
- `publish: true` to commit the artefact.

Before anything is committed, the job runs `scripts/verify_model.py`. It
checks that the model is servable and reads only features the live path
builds, and that on the last fortnight it prices every runner with a book of
1 per race and correlates 0.9+ with the model served now. A failure stops
the job. The artefact is still uploaded for inspection, but nothing is
committed.

## 7. Deploy (minutes)

The 06:00 run serves `data/models/` from the default branch. Merging the PR
that carries the verified artefact is the deploy, and it takes effect from the
next 06:00 run. Under the forward test's pre-registration a model changes only
between days, never mid-day. Record the change in
`reports/preregistration_clv_forward.md` and the ledger.

## Where the time went, and what fixed it

1. **Six fits in a row** (2h21m of iteration 31's 2h32m) → one job per
   (arm, fold) in parallel. The wall clock is now one fold's fit.
2. **The base refitted every iteration** → cached under a key covering the
   matrix, the arguments, the folds, the fitting code and the library. Fits
   are deterministic, so the cached base is the base. Five CPU models agreed
   to 8.9e-16.
3. **The served recipe's cost for a screen** → the quick recipe: 0.131 s a
   round against 0.296 s, and a third of the rounds.
4. **Every feature change rebuilt the matrix** → drop-in blocks, computed on
   the cached matrix.
5. **Training rebuilt the matrix and fitted twice** → it now restores the
   matrix, fits once at a known round count, and skips stage 2.
6. **Nothing checked an artefact before it was committed** → the
   verification gate.
