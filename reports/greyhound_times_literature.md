# Greyhound race times: standards, going, sectionals, class, figures and expected time

racing² - research note, 14 November 2026. Scope: UK greyhound race times from GBGB results (run time, calculated time, first sectional, position, beaten lengths, grade, distance, trap, prizes, weight, comment codes).

**How to read this note.** Web sources were read on 14 Nov 2026 and are treated as untrusted data. Every external claim carries its URL. Statements marked *(inference)* are my arithmetic or reasoning, not a source. Statements marked *(unverified)* could not be confirmed against a primary source. Repository facts cite a file path.

**What this note does not repeat.** The earlier note (`greyhound_literature.md` in the same scratchpad) already covers Benter's two-stage model, Bolton & Chapman, Lessmann et al., Chen et al. (1994), Schumaker & Johnson (2008), Lyons (2016), the Betfair greyhound tutorial's "relative to the track and distance median of winners" normalisation, and the exchange-microstructure literature. The repository's own `reports/greyhound_model.md` section 41 (11 Nov 2026) already records an expert survey (Timeform greyhound, GBGB rules 139 and 151, Greyhound Stats UK, form writers, GRV) and builds the expected-time model `greyhound/pit/xtime.py`. This note builds on both and concentrates on what is **not yet** in the model.

---

## 1. Where we already are (repository, not web)

| Item | State in the repo | Source |
|---|---|---|
| Standard times | Rolling 28/91/365-day medians of winners' calculated time per track and trip (`tn_`); a moving standard per track, trip and hurdles inside the expected-time regression (`xt_`) | `greyhound/times.py`, `reports/greyhound_model.md` 41.3 |
| Meeting variant | Empirical-Bayes shrink of the night's mean winner deviation, `variant = mean_dev * n/(n+k)`, k = within/between variance (`tn_`); in `xt_` a variant from every runner's error against its expected time, in percent | `greyhound/times.py`, 41.3 |
| Cross-track linking | "linked across tracks by the dogs that move (the level is in percent, so it carries)" | 41.3 |
| Expected time | `100*ln(time) = standard + variant + dog level (Kalman) + preference + trap + grade + adjustments + trouble + noise`, each of whole race, sectional, run home | 41.3 |
| Out-of-sample error | Whole race 3.83 lengths before the race (2026), 3.67 after the meeting variant, 3.12 within the race; sectional 1.11 / 1.04 / 0.87 | 41.5 |
| Value to the price | Adding `xt_` cut RMSE of log(BSP) by 0.0067 (z -22.1) in the live stack; our log-loss 0.43686 vs BSP 0.42859 | 41.6, 41.7 |
| Known gaps listed by the repo | No within-meeting drift of the going; no racing manager's seeding (railer/middle/wide); GBGB's own going allowance read by no time feature; handicaps never set apart; run home charged the whole going allowance and the sectional none (engine figures) | 41.1, 41.3 "Not in it yet" |
| GBGB data rules | `calculated = run_time + going/100 + 0.08 * handicap_metres`, tolerance 0.02 s; going beyond +/-150 hundredths treated as a typing error | `greyhound/data.py` lines 102-108, 165-168 |
| Class ladder | A grade of A is about 1.2 lengths in winners' calculated time (1 Jul 2025 to 5 Nov 2026); OR1 -8.5, A1 -4.6, A5 +0.3 lengths vs track median | `greyhound/data.py` lines 15-30 |

So the high-value remaining work is: (a) the going (GBGB's allowance as information, distance scaling, within-meeting drift); (b) standards that survive surface and timing changes; (c) sectional pars per trap and the going split between sectional and run home; (d) class and field strength, including retrospective "form held up" re-rating; (e) the error distribution and race simulation around the expected time.

---

## 2. GBGB definitions (what the numbers mean)

**Calculated time and lengths.** Where photo-timing is used, distances between greyhounds are calculated from recorded times "at the rate of 1 length per 0.08 of a second"; for handicaps the published calculated time is the recorded or calculated time plus 0.08 s for each metre of handicap allotment from scratch; where qualification depends on times the calculated times are used ([GBGB Rule 139](https://rules.gbgb.org.uk/section-7-trials-meetings-and-race-meetings/rule-139-beaten-distance-calculated-time)).

**Timing.** All trials and races are timed by photo- or ray-timing apparatus, with a licensed timekeeper's hand time as a check. The winning time runs from when the front of the trap reaches 45 degrees from the perpendicular until the winner's nose reaches the line. Rule 138 specifies only the winning time; it does not mention sectionals or precision ([GBGB Rule 138](https://rules.gbgb.org.uk/section-7-trials-meetings-and-race-meetings/rule-138-timing-timekeeper/)).

**Going allowance.** The racing manager states the going allowance as a plus or minus figure against normal, and its quantification must appear in all trial and race records ([GBGB Rule 151](https://rules.gbgb.org.uk/section-7-trials-meetings-and-race-meetings/rule-151-records-of-trials-and-races-going-allowances/)). Units are hundredths of a second, with "N" for normal ([Timeform racecard guide](https://www.timeform.com/greyhound-racing/beginners-guide/introduction/how-to-read-a-racecard-1282020); [Romford race card guide](https://romfordgreyhoundstadium.co.uk/race-card-guide)). Practitioners call a hundredth a "spot" (e.g. Oxford "going adjusted times were 18 spots slower" before new traps, [Greyhound Stats UK](https://greyhoundstats.co.uk/track_comparison_all.php)).

**Sign convention: the web sources conflict; settle it on our data.**
- Towcester's guide says positive figures mean a slower surface and gives a race with going +10 and winning time 29.54, the +10 being added to give the calculated time ([Towcester](https://towcester-racecourse.co.uk/?p=32612)). Adding time for a slow surface would make a slow night look slower still, so the text and the arithmetic disagree *(inference)*.
- Timeform's greyhound guide gives "a 29.36 win on ground rated 0.18 fast" adjusted to 29.18 ([Timeform](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)), which subtracts time for fast ground: the opposite of what a correction should do *(inference)*. This is Timeform's own allowance, not GBGB's.
- Our data: GBGB's calculated time equals run time plus going/100 within 0.02 s (`greyhound/data.py` 105-108). For that to be a correction, a positive figure must mean a fast surface *(inference)*.
- **Check to run:** regress the night's raw winners' deviation from the standard on GBGB's going figure. A negative slope near -1 (positive going on nights that ran fast) confirms "+ = fast, calc = time + going". The slope's size also measures how fully the racing manager corrects (section 4.2).

**Sectional.** On UK cards the sectional (split) is the time from leaving the traps to crossing the winning line the first time ([Timeform racecard guide](https://www.timeform.com/greyhound-racing/beginners-guide/introduction/how-to-read-a-racecard-1282020); [Romford](https://romfordgreyhoundstadium.co.uk/race-card-guide)). Its length therefore depends on where the traps sit relative to the line: e.g. 75 m from Nottingham's 500 m traps to the line, splits around 4.88 s ([Greyhound Star](https://greyhoundstar.co.uk/?p=15168), via search summary, *unverified* figures). Races whose traps are past the line (e.g. Towcester 400 m and 330 m) cross it once and have no split ([Towcester](https://towcester-racecourse.co.uk/?p=32612)). Splits are recorded in hundredths ([Towcester](https://towcester-racecourse.co.uk/?p=32612)). Whether every runner's sectional is measured by the timing beam or only the leader's (others derived) was **not found** in the rules: check on our data (ties, rounding, whether non-leaders' splits ever precede the leader's).

**Track changes move standards.** Greyhound Stats UK rebased Oxford from 18 Mar 2023 (new traps for 450/943 m) and 20 Apr 2023 (253/650 m), and Central Park from 2 Jul 2023 after track works and re-measured distances; before that Central Park's going was returned 15, 40, 55 and 80 spots slow for 2, 4, 6 and 8 bends. It lists new traps, photo-timing changes, track works and changes to going allowances as things that move going-adjusted times ([Greyhound Stats UK](https://greyhoundstats.co.uk/track_comparison_all.php)).

---

## 3. Practitioner systems: what each does

### 3.1 Beyer speed figures (US thoroughbreds)
- **Parallel time chart:** a table assigning a figure to every final time at every distance. It descends from Taulbot (1959); Kovitz (1963) adjusted for velocity because adding a fixed time at all distances distorts the numbers; Beyer's anchor was that 1:13 for 6f equalled 1:26.1 for 7f at Calder ([Wikipedia](https://en.wikipedia.org/wiki/Beyer_Speed_Figure)).
- **Beaten lengths:** about 2 points a length in routes and 3 in sprints ([America's Best Racing](https://www.americasbestracing.net/gambling/2020-horse-racing-speed-figures-explained)), i.e. a length is worth more at short trips, consistent with a percent-of-time scale *(inference)*.
- **Variant:** first from pars (the day's average deviation from class pars), later from **projections**: for each race, the winning figure that best fits the top finishers' recent figures and margins; the variant is the average gap between projection and result ([DRF, Ask Beyer](https://www.drf.com/news/ask-beyer-track-variant-measures-inherent-speed-racing-surface); [Wikipedia](https://en.wikipedia.org/wiki/Beyer_Speed_Figure)). Beyer dropped pars because a par ignores how far the winner beat the field.
- **Practice:** separate variants for sprints and routes; a separate pace variant ("Always make a separate pace variant"); split the day's variant when conditions change; mark uncertain figures; check them when horses from the day race back ([DRF, tips on making figs](https://www.drf.com/news/tips-making-your-own-figs)). Shippers between tracks are the main tool to realign circuits; purse changes shift class pars ([DRF](https://www.drf.com/news/making-beyer-speed-figs-tricky-business)).
- **Evidence:** top-figure horse won about 35% at a slight loss in 1974 (secondary quote, *unverified*, [Wikipedia](https://en.wikipedia.org/wiki/Beyer_Speed_Figure)). Rosenbloom (2003), *Omega* 31(5): 339-348, built win probabilities from Beyer numbers and reported them better than the market model by an SPRT-style test, with positive returns only when long shots were excluded (abstract and secondary summary only, [EconPapers](https://econpapers.repec.org/RePEc:eee:jomega:v:31:y:2003:i:5:p:339-348)). Balakrishnan & Bernard: the 1992 introduction of Beyers in the DRF raised betting volume but showed no significant change in market efficiency ([CLS Blue Sky summary](https://clsbluesky.law.columbia.edu/2021/09/02/public-information-and-capital-flows-evidence-from-a-betting-market/)). Lesson: a published figure is priced in quickly; the edge is in a better figure than the public's.

### 3.2 Ragozin "Sheets" and Thoro-Graph
- Lower is better (0 can win the Kentucky Derby, 20 belongs in claimers). Ragozin figures "incorporate wind, weight carried and, most importantly, ground loss." Thoro-Graph, founded by a former Ragozin employee, is built similarly and runs a few points lower ([America's Best Racing](https://www.americasbestracing.net/gambling/2020-horse-racing-speed-figures-explained)). Ragozin's contribution was reading the **pattern** of figures (form cycles, the "bounce") ([search summary of TDN obituary](https://thoroughbreddailynews.com/speed-figure-pioneer-len-ragozin-dies/), page itself blocked: *unverified*).
- No public formula was found for either (Thoro-Graph's site returned empty; *unverified*).
- **For us:** ground loss maps to running wide (comment codes "W", "VW", "Crd"); the pattern idea maps to modelling the trajectory of a dog's figures (already partly in the Kalman level, `fm_` trend).

### 3.3 Timeform horse timefigures
- Standards per course and distance from horse merit, age and weight, excluding records, updated annually; time differences expressed **per 60 seconds** so a deficit means the same at every trip; going solved mathematically from the day's races (no per-furlong allowance), needing six or seven races with at least four by horses of established merit; wind by vector analysis; a poor timefigure does not prove a horse bad because of pace ([Timeform timefigures explained](https://www.timeform.com/horse-racing/features/timefigures/timeform_computer_timefigures_explained)).
- Conversion: `lb = 1500 * (race time - standard time) / race time` (25 lb a second at 60 s). Times are not normally distributed, so records and averages are unsuitable for standards; only 20% to 40% of Flat races are truly run (about 40% at 5f, half that at 2 miles), so with seven races there is about a 92% chance that at least one is truly run; the going is set from the race giving the **minimum** required value, i.e. the best-run race. Timeform rejects rigid class pars in favour of form handicapping ([Timeform, analysis of overall times](https://www.timeform.com/racing/articles/the-timeform-knowledge-analysis-of-overall-times-2262015)).
- **For us:** the greyhound analogue of "truly run" is a clean run (no trouble codes). A meeting variant estimated from clean runs only, or with trouble-coded runs down-weighted, follows the same logic.

### 3.4 Timeform greyhound ratings
- Expected time per dog from its previous eight races at the same number of bends (races plus trials if fewer than eight); all expected times anchored on a dog rated 100 on normal going (e.g. Doncaster 483 m: 100 = 29.69 s); 0-140 scale (60 low grade, 120+ open class); ratings comparable across tracks but grades are not (an A1 winner at Sunderland about 80 vs about 105 needed for A1 at Hove); a length roughly 0.07-0.08 s ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)).
- Going: preliminary allowance from expected vs actual after trouble, **adjusted by race distance**; the final allowance adjusts for inside/outside bias (wide runners beating expectation on clean runs while inside runners fall short) and for early vs late meeting differences; sectional ratings on the same scale with sectionals **adjusted proportionally** for the going; beaten dogs can be rated above the winner after trouble ([same](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)).

### 3.5 Racing Post Topspeed (horses)
- Race times and a going allowance against RP standard times pegged to a 100-rated horse at 9st on good ground, adjusted for weight; same scale as RPRs. Detail is from forum posts only ([The Racing Forum](https://theracingforum.co.uk/forums/topic/racing-post-speed-figures); [standard times thread](https://theracingforum.co.uk/forums/topic/rp-standard-times-list/page/2)) - *unverified*. No Racing Post greyhound rating method was found ("no data found").

### 3.6 Hong Kong Jockey Club speed figures
- Par times from up to four years of results on good or good-to-firm only, per class, distance, track and surface; variant from the first three finishers in each race against their class pars (up to 30 values a 10-race card); "there can be more than one variant per meeting"; weight carried adjusted; ground lost to interference or running wide estimated from video (three wide round two bends about 3.5 lengths, about 0.5 s), deliberately conservative; Class 1 standard = 124 at every distance, about one point per 0.1 s ([HKJC](https://www.hkjc.com/english/formguide/what_SpeedFigures.asp)).
- **For us:** using the first three, not only the winner, triples the variant's sample per race; HKJC explicitly allows several variants a meeting.

### 3.7 TrackMaster / Equibase
- Raw speed rating from a parallel time chart, plus an **Intertrack Variant** updated annually for speed differences between tracks, plus a Daily Track Variant ([AQHA guide](https://www.aqha.com/en/news-and-publications/e-books/free-ebooks/horse-race-tip-speed-ratings)). Equibase's formula is proprietary ("no data found").

### 3.8 Australian and Irish greyhound bodies
- **GRNSW time-based grading:** each dog's finishing times over its last six starts, the two slowest dropped, rated from a national Time Bands Table across all tracks and distances ([GRNSW](https://www.grnsw.com.au/racing/grading/time-based-grading)). The table's construction is in linked PDFs not read (*unverified*).
- **GRV:** grades by weighted points for placings by track category and race type, not time ([GRV grading guidelines](https://www.grv.org.au/wp-content/uploads/2025/11/GRV-Grading-Guidelines-December-2025-.pdf), via search summary).
- **Racing Queensland:** a Time Standard per track, class and distance as the benchmark for grade penalties ([RQ grading procedures](https://www.racingqueensland.com.au/getmedia/f5975b64-8eaf-4169-8756-1a911b386cbb/Grading-Procedures-April-2025.pdf), via search summary, *unverified* detail).
- **GRI (Ireland):** grades are 0.20 s time bands at the 525-yard standard (A1 28.70-28.89, A2 28.90-29.09 ... A10 30.50-30.70; open <=28.49); a raced dog moves to the grade of its time "winning or estimated"; back-graded after three unsuccessful races at a distance category ([GRI grading rules](https://www.grireland.ie/globalassets/images/grading/national-grading-system-rules--tables.pdf)). Track ratings (from 1 Sep 2015) put each track in a group with a fixed fast/slow allowance that grows with distance: e.g. group 6 is 0.20 slow at sprints, 0.40 middle, 0.50 long up to 800 yards, 0.60 beyond; group 1 (Cork) 0.05/0.10/0.10/0.15 fast; stewards can rate a track differently on a night ([GRI track ratings](https://www.grireland.ie/About-GRI/news/gri-track-ratings/)).
- **For us:** two independent bodies scale inter-track and night allowances with distance; a seconds-only variant for the whole card is wrong for mixed-trip cards. The Irish 0.20 s band (2.5 lengths) per grade is a ready prior for class pars at a 4-bend trip in Ireland, about twice our measured 1.2 lengths per A grade in the UK (`greyhound/data.py`), which is plausible because bands are thresholds on time, not mean winners' times *(inference)*.

### 3.9 Betfair Data Scientists tutorials (Australia)
- FastTrack tutorial: run times z-scored within track-distance over the full 2018-2021 data (look-ahead), no track-condition adjustment, logistic regression on five 365-day features; model top pick won 32.96% of 2,491 test races against 41.75% for the SP favourite; "This model is not profitable" ([FastTrack tutorial](https://betfair-datascientists.github.io/modelling/fasttrackTutorial)).
- Topaz tutorial: `runTimeNorm = (rolling 365-day median of daily median winning time / resultTime).clip(0.9, 1.1)`, no split or condition normalisation; LightGBM and others; test log loss about 0.745 ([Topaz tutorial](https://betfair-datascientists.github.io/modelling/topazTutorial)).
- **For us:** these are well behind our `xt_` model; they confirm only that the simple median-of-winners normalisation is the public baseline.

### 3.10 Greyhound Stats UK time conversion
- Converts going-adjusted times between British and Irish tracks and trips with the same number of bends only; conversions relate to current track conditions; updated when enough data arrive after a change ([Greyhound Stats UK](https://greyhoundstats.co.uk/track_comparison_all.php)). Method not published ("no data found").

---

## 4. Methods by topic, with formulas and how they apply

### 4.1 Track and distance standard times

**What the leaders do.** Timeform: per course and distance, excluding records, annual update, ability- and weight-adjusted ([Timeform](https://www.timeform.com/horse-racing/features/timefigures/timeform_computer_timefigures_explained)). HKJC: four years, good ground only, per class ([HKJC](https://www.hkjc.com/english/formguide/what_SpeedFigures.asp)). Beyer: a parallel time chart ties trips at a track together ([Wikipedia](https://en.wikipedia.org/wiki/Beyer_Speed_Figure)). TrackMaster: an annual intertrack variant ([AQHA](https://www.aqha.com/en/news-and-publications/e-books/free-ebooks/horse-race-tip-speed-ratings)). Shippers realign circuits ([DRF](https://www.drf.com/news/making-beyer-speed-figs-tricky-business)).

**Recommended formulation (one model, all runners, not winners only).** A crossed-random-effects model on log time:

    100 ln(calc_time_ijm) = S[track, trip] + G[grade at track-trip] + V[meeting] + A[dog] + T[track, trip, trap] + x'b + e

fitted by REML (lme4-style crossed random effects, [Bates et al. 2015, JSS 67(1)](https://www.jstatsoft.org/article/view/v067i01)) or as the repo's walk-forward equivalent. The standard is `S` read at a reference grade, so it is not biased by a track's class mix or field sizes (the repo found 2-4 runner fields 0.32-0.51% faster: `reports/greyhound_model.md` 41.5). Because `A[dog]` is shared across tracks, dogs that move link the standards (the "shippers" principle).

**Small samples.** Shrink each `S[track, trip]` toward a geometry prior, not toward zero:
- `S_prior[track, trip] = track_speed[track] + f(trip, bends)`, where `f` is a pooled parallel-time curve of log time on log distance and bends, and `track_speed` a track random effect. A new or rarely run trip then borrows from the same track's other trips (Beyer/Kovitz parallel time) and from all tracks' curve. The empirical-Bayes weight is `n/(n + s2_within/s2_between)` as the repo already uses for the variant ([Efron & Morris 1975, JASA 70: 311-319](https://www.medicine.mcgill.ca/epidemiology/hanley/bios602/MultilevelData/EfronMorrisJASA1975.pdf) is the classic demonstration of this shrinkage).
- Times are skewed (trouble only slows a dog), so fit with a robust loss (Huber or Student-t) or on clean runs, not a plain mean: Timeform notes times "are not normally distributed" ([Timeform](https://www.timeform.com/racing/articles/the-timeform-knowledge-analysis-of-overall-times-2262015)).

**Surface and timing changes.** Rolling windows (the repo's 28/91/365) trade bias for noise and never know when a change happened. Better:
- **Change-point detection** on each track-trip's daily residual series (mean residual of all runners after dog and grade terms), e.g. PELT ([Killick, Fearnhead & Eckley 2012, JASA 107: 1590-1598](https://arxiv.org/abs/1101.1438)), with a penalty set so known events are found: Oxford 18 Mar 2023 and 20 Apr 2023, Central Park 2 Jul 2023 ([Greyhound Stats UK](https://greyhoundstats.co.uk/track_comparison_all.php)). A detected break resets `S` to the post-break level with the geometry prior; the pre-break runs stay usable through their own (pre-break) standard. Detect on data available at the time only (lag-safe), so a break is recognised after a few meetings, as Greyhound Stats does "as soon as sufficient data are available".
- Alternatively a **local-level state-space standard** (random walk per track-trip plus occasional large jumps, e.g. a t-distributed state noise) gives the same behaviour continuously.
- A break that hits all trips at a track at once is a timing or surface change; one confined to a trip is a trap or distance change. Test both (new traps changed Oxford's 450/943 m on a different date from 253/650 m).

**Evidence.** No published head-to-head of standard-setting methods for greyhounds was found ("no data found"). The test is ours: out-of-sample error of the expected time in lengths, as section 41.5 already reports.

### 4.2 Going allowances and meeting variants

**GBGB's allowance as information, not as truth.** GBGB's calculated time already includes the racing manager's allowance (Rule 151). Our variant on calculated time then measures what the manager missed. Two refinements:
1. **Calibrate GBGB's allowance.** Fit `raw_dev_m = -beta * going_m/100 + u_m` across meetings, where `raw_dev_m` is the meeting's mean raw-time deviation from expectation (from the mixed model). `beta = 1` means GBGB corrects fully on average; `beta < 1` that it under-corrects; per-track `beta` (random slope) captures racing managers who habitually return "N". Then use `raw_time - beta_track * going/100` as the prior centre and shrink our data-variant toward it rather than toward the calculated time. *(inference; this design is ours.)*
2. **Use the allowance as a feature.** The repo notes "GBGB's going allowance is read by no time feature" (`reports/greyhound_model.md` 41.1). Feed the declared going (known at the meeting) and the dog's past runs' going to the level filter, so runs on nights where GBGB's figure was large and our variant disagreed carry less weight.

**Distance scaling.** GBGB returns one allowance a race, but practitioners scale by trip: Central Park's old returns grew from 15 to 80 spots across 2 to 8 bends ([Greyhound Stats UK](https://greyhoundstats.co.uk/track_comparison_all.php)); GRI's track groups grow with distance ([GRI](https://www.grireland.ie/About-GRI/news/gri-track-ratings/)); Timeform adjusts its preliminary allowance by race distance ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)). Model the meeting effect as `V_m * (trip time)` (percent, as `xt_` does) **plus** a per-bend term `W_m * bends`, both random effects, and test whether the per-bend term earns its place. A slow surface may cost more on bends than on straights; no source measured this ("no data found").

**More than one variant a meeting (within-meeting drift).** HKJC allows several variants a meeting ([HKJC](https://www.hkjc.com/english/formguide/what_SpeedFigures.asp)); Beyer splits the day's variant when conditions change ([DRF](https://www.drf.com/news/tips-making-your-own-figs)); Timeform adjusts for early vs late in the meeting ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)). Implementation:

    V_m(r) = V_m + D_m * (r - r_mid)/n_races + smooth_m(r)

with `D_m` a random slope across the race number (or minutes after the first race), shrunk to zero; or a Kalman smoother on race-level residuals within the meeting (state = surface speed, random-walk noise fitted on history). For **features** of a later race the variant of a past run may use the whole of that past meeting (it is in the past); for a **live** prediction during the meeting, only the races already run are known: a filtered (not smoothed) drift gives the in-meeting going for later races, which is information the morning card does not have. *(inference)*
- Also check whether GBGB's own going figure changes between races in a meeting in our data. If it does, the racing manager already splits it and our drift should be read on top.

**Use more runners, but clean ones.** The repo's `tn_` variant uses winners; `xt_` uses every runner's error. Timeform's "truly run" idea ([Timeform](https://www.timeform.com/racing/articles/the-timeform-knowledge-analysis-of-overall-times-2262015)) and its correction for trouble before setting the allowance ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)) suggest weighting each runner's residual by the inverse of its predicted error variance (larger for trouble codes, debutants and dogs with uncertain levels). The variant is then a precision-weighted mean, shrunk by its total precision. This is the Beyer "projection" variant done statistically.

**Trap bias on the night.** Timeform adjusts the allowance for inside/outside bias ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)). Add a meeting-by-trap-group random effect (rails 1-2, middle 3-4, wide 5-6), shrunk heavily: it soaks up a rail that is slow on a watered night and stops it being read as six dogs' loss of form.

### 4.3 Sectional pars and over/under par

**Definition matters.** The UK sectional is traps to the line first time, so its distance is track-geometry specific and is absent for races starting past the line ([Romford](https://romfordgreyhoundstadium.co.uk/race-card-guide); [Towcester](https://towcester-racecourse.co.uk/?p=32612)). The run-in is `calc_time - sectional`.

**Pars.**
- Per track, trip and trap: `SP[track, trip, trap] = SP[track, trip] + delta[track, trip, trap]`, delta shrunk to a pooled trap profile. Wide traps may cover more ground to the line on some layouts; the earlier note found the trap barely predicts first-bend position (Butler et al.), so expect small deltas; test it.
- **Going split.** Timeform adjusts sectionals proportionally for the going ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)). The repo's engine charges the run home the whole allowance and the sectional none (`reports/greyhound_model.md` 41.1). Fix: `sec_adj = sec * (1 + v_m)`, `runin_adj = runin * (1 + v_m)`, with `v_m` the meeting's percent variant; or estimate separate sectional and run-in variants (Beyer's "separate pace variant", [DRF](https://www.drf.com/news/tips-making-your-own-figs)) with a correlated bivariate random effect.
- **Pace shape (finishing speed).** Rowlands' finishing speed percentage for horses: `FS% = 100 * T * d / (D * t)` with T and D the overall time and distance, t and d the section's; read against a course-distance par set from fast overall times; the compromise grows more than linearly with the gap from par ([Timeform, introduction to sectional timing part 2](https://www.timeform.com/horse-racing/features/sectionals/an_introduction_to_sectional_timing_part_2)). The greyhound analogue: `runin_speed / race_speed` against the track-trip par. Use it as a **context** variable for the run's figure: a dog that led easily at a slow early pace and one that came from behind through traffic differ in what the overall time says. This is Timeform's "a poor timefigure does not prove a horse is bad" ([Timeform](https://www.timeform.com/horse-racing/features/timefigures/timeform_computer_timefigures_explained)).

**Over/under par per dog.** `sec_vs_par = (sec_adj - SP_expected[track, trip, trap]) / 0.08` in lengths, both for the dog's history (early-pace level, already `xt_sec`) and for the **race shape forecast**: the expected leader at the line first time and the gap to the next (already `xt_sec_vs_in` etc.). What is new: a race-level pace pressure index (number of dogs whose expected sectional is within, say, 1 length of the fastest) as a modifier of the expected run-in, because early pressure costs late speed in horses (Rowlands) and plausibly in dogs; no greyhound study was found ("no data found").

### 4.4 Class-adjusted times

**Class pars.** HKJC pars per class, track and distance ([HKJC](https://www.hkjc.com/english/formguide/what_SpeedFigures.asp)); Beyer's pars per class with purse drift ([DRF](https://www.drf.com/news/making-beyer-speed-figs-tricky-business)); Irish grades are 0.20 s bands ([GRI](https://www.grireland.ie/globalassets/images/grading/national-grading-system-rules--tables.pdf)). In our model the class par is `G[grade at track-trip]`; keep it a **random effect within track-trip** shrunk to a pooled ladder (a grade at Sunderland is not a grade at Hove: [Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)), and let it drift (purse and kennel-population changes move pars, [DRF](https://www.drf.com/news/making-beyer-speed-figs-tricky-business)).

**Field strength from prior form (Beyer projection).** Replace the par for a race by the projection from the runners' own pre-race levels: `proj_race = weighted mean of A_hat[dog] for the field` (weights = precision). A race's residual against its projection, not against its class par, is what identifies the variant. Beyer adopted this because pars ignore the field ([DRF, Ask Beyer](https://www.drf.com/news/ask-beyer-track-variant-measures-inherent-speed-racing-surface)). `xt_` already does this implicitly by including dog levels; the explicit extra feature is `field_strength = mean (or top-3 mean) of the rivals' expected times`, which helps predict pace and trouble.

**Post-race form held up (retrospective re-rating).** Beyer: "Look for corroborating evidence when the horses from tough-to-do days race back" ([DRF](https://www.drf.com/news/tips-making-your-own-figs)); DRF's anecdote that races well above par produce later winners ([DRF](https://www.drf.com/news/fast-races-come-future-winners), anecdotal, two horses). Statistical version: a **smoother**, not a filter. Re-estimate each past meeting's variant using all runs up to today (each runner's later runs inform how good it was, which informs how fast the surface was). Then for a race on day d, recompute each runner's historic figures with the variants smoothed through day d-1. This is lag-safe because only information before d is used, and it reduces the error of the variant on low-runner nights. Expected value: highest for runs from small meetings and for dogs' early career. A Rauch-Tung-Striebel smoother over the meeting variants, or simply refitting the mixed model on all data before d, does it. *(inference: the design is ours; the principle is Beyer's.)*

**Prize money.** Within opens (OR, OR1-OR3), prize money separates class better than the grade label. Use log(first prize) as a covariate in `G` for open races only. Silverman (below) used class within race "profiles". No greyhound study tested prize money in a time model ("no data found").

### 4.5 Speed figures comparable across tracks and distances

- **Scale choice.** Timeform per-60-seconds scaling ([Timeform](https://www.timeform.com/horse-racing/features/timefigures/timeform_computer_timefigures_explained)) and Beyer's velocity-adjusted parallel chart ([Wikipedia](https://en.wikipedia.org/wiki/Beyer_Speed_Figure)) both say: a deficit should be read in proportion to the race's duration. Our `100 ln(time)` is the same idea.
- **Fit the exponent rather than assume it.** Let the spread of ability scale as `trip_time^b`: `b = 1` is percent, `b = 0` is seconds. Estimate `b` from dogs that run several trips (their level difference between trips) and from the between-dog SD of residuals by trip. If `b` is clearly below 1 (e.g. stayers' times spread less than in proportion), a percent figure overrates stayers' gaps *(inference; untested)*.
- **Publishable figure.** For a human-readable racing² Greyhound Figure: `GF = 100 + k * (S_ref - (100 ln(calc) - V_m - G_ref))`, with `k` chosen so 1 point = 1 length at a reference 4-bend trip (about 3.7 lengths per 1% of a 29.5 s race per `reports/greyhound_model.md` 41.5). HKJC uses 1 point per 0.1 s anchored at Class 1 = 124 ([HKJC](https://www.hkjc.com/english/formguide/what_SpeedFigures.asp)); Timeform anchors 100 on a fixed-merit dog per track and trip ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)). Only the internal log-time model matters for the price; the figure is for reports.
- **Trouble-adjusted figure.** HKJC adds ground lost (three wide round two bends about 3.5 lengths) conservatively ([HKJC](https://www.hkjc.com/english/formguide/what_SpeedFigures.asp)); Timeform can rate a beaten dog above the winner after trouble ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)). The repo prices interference codes, capped at 1.5% (41.3). Extension: price "wide/very wide" codes as ground lost (they describe path, not interference) but only for the **figure** fed to the level, not as a regressor that the level can absorb (the failure mode in 41.5 point 1).

### 4.6 Expected time per dog per race, and other ways to use times

- **Already strong:** `xt_` (section 1). Extensions with the highest expected value:
  1. **Error distribution, not a point.** Times are skewed and trouble-driven ([Timeform](https://www.timeform.com/racing/articles/the-timeform-knowledge-analysis-of-overall-times-2262015)). Model each dog's time as a two-part mixture: a clean-run Normal (or t) around the expected time with the dog's own SD, plus a trouble component with probability from trap, early pace vs neighbours and the dog's crowding history. Simulate the race jointly with a **shared meeting shock** and a **shared race shock** (pace), not independent draws (the current `et_p_fastest` and `ab_p_best` draw independently: `greyhound/etime.py`, `greyhound/ability.py`). Win probability then comes from the simulation and goes to the Benter second stage.
  2. **Rank-ordered dynamic model as a cross-check.** Glickman & Hennessy's stochastic rank-ordered logit: each competitor's ability a Gaussian random walk, performances extreme-value, filtered with an approximate Bayesian update ([Glickman & Hennessy 2015](https://www.glicko.net/research/multicompetitor.pdf)). It uses order only, so it is immune to timing errors and going mistakes; disagreement with the time model flags bad times.
  3. **Bayesian hierarchical regression of speed by race profile.** Silverman (2012), *Journal of Prediction Markets* 6(1): 1-13, HKJC 2005-2009, 36,006 runs: speed in m/s with coefficients varying by race profile (course x track x class x distance, 73 profiles), Gibbs sampler; top predicted speed won 21.63%; a conditional logit with the public odds gave coefficients 6.42 (odds) and 4.05 (predicted speed); flat-staking positive-EV horses lost 12.15% ([Silverman 2012](https://www.ubplj.org/index.php/jpm/article/download/590/628)). Lesson: per-profile coefficients are sensible, but a time model alone does not beat the market; it is an input to the second stage.
  4. **Mixed model of log finish time in another sport.** Wilson & Wilson (2024), Bayesian linear mixed model of log finish time with athlete and course random effects plus weather, checked by posterior predictive distributions per race ([arXiv 2405.09865](https://arxiv.org/abs/2405.09865)). Direct structural template for our crossed model and for posterior predictive checks of each meeting.
  5. **Projected pace shape.** From expected sectionals and the trap order, forecast the first-bend order and who is impeded; then add the expected cost of trouble to each dog's expected time. The earlier note's first-bend model (Butler et al.) is the input.

---

## 5. The lengths-per-second convention (0.08 s): check

- **GBGB defines it.** Distances are calculated from recorded times at 1 length per 0.08 s ([Rule 139](https://rules.gbgb.org.uk/section-7-trials-meetings-and-race-meetings/rule-139-beaten-distance-calculated-time)). So in GBGB data a beaten distance is a time gap divided by 0.08 and rounded to the margin notation. Converting back with 0.08 recovers the recorded gap up to rounding. **For reconstructing times, 0.08 is exactly right and must not vary by distance** *(inference from the rule)*.
- **As a physical distance it varies.** Timeform calls a length "roughly 0.07 to 0.08 seconds" ([Timeform greyhound](https://www.timeform.com/GREYHOUND-RACING/beginners-guide/betting/ratings-and-sectionals-explained-1382020)). Average speeds from Timeform's own examples: Doncaster 483 m in 29.69 s is 16.3 m/s, Sheffield 500 m under 28.50 s is above 17.5 m/s *(inference: arithmetic)*. At those speeds 0.08 s is 1.30-1.40 m of track. Dogs reach full speed (slightly above 20 m/s) after about 80-100 m ([search summary of UTS tracking work](https://www.mdpi.com/2076-2615/11/9/2687), full text blocked, *unverified*), so speed at the finish differs from the average and by trip; no source gave finishing speed by distance ("no data found").
- **Horse racing comparison.** The BHA's scale varies by code and going (e.g. 4 lengths a second over jumps on heavy, 5 on good) but not by distance, which the Racing Post criticised; it recommended publishing times instead ([Racing Post, 7 Dec 2017](https://racingpost.com/news/how-long-is-a-length-its-a-bit-like-a-piece-of-string-axONz6O9xUeK)). Greyhound racing already publishes times, so the margin is a time unit.
- **Where it does matter for us:**
  1. **Ratings across trips.** One length (0.08 s) is 0.50% of a 16 s sprint and 0.18% of a 45 s stayers' race *(arithmetic)*. Reading a beaten distance in percent (as `xt_` does) is the right equivalent of Beyer's "3 points a length in sprints, 2 in routes".
  2. **Handicap starts.** GBGB adds 0.08 s **per metre** of start (Rule 139). At 16.3 m/s a metre takes about 0.061 s, so a dog given a 10 m start is charged about 0.8 s against about 0.61 s of real running time: its calculated time overstates its slowness by about 0.19 s (about 2.4 lengths) *(inference; untested: the start also changes the dog's path and acceleration)*. **Check:** in handicaps, regress the dog's calculated-time residual (against its level from graded races) on its handicap metres; a positive slope near 0.019 s per metre confirms the bias. Until checked, keep handicaps out of the level and standards, as the repo intends.
  3. **Margin rounding.** Short head, head and neck are below 1/4 length (0.02 s); `sources/gbgb.py lengths()` maps them through `_FRAC`. Rounding noise is under 0.01-0.02 s for close finishes but larger for long margins if GBGB rounds big gaps to whole lengths (not checked: "no data found" on GBGB's rounding rule).

---

## 6. Academic work on race-time prediction (additions to the earlier note)

| Paper | What it did | Result | Use for us |
|---|---|---|---|
| Rosenbloom (2003), *Omega* 31(5): 339-348 ([EconPapers](https://econpapers.repec.org/RePEc:eee:jomega:v:31:y:2003:i:5:p:339-348)) | Win probabilities from Beyer figures vs the market model, SPRT-style comparison | Reported better than the market; positive returns only with a long-shot constraint (abstract and secondary summary; full text not read) | A time figure can carry information the market misses at the favourite end; test our edge by price band |
| Balakrishnan & Bernard ([summary](https://clsbluesky.law.columbia.edu/2021/09/02/public-information-and-capital-flows-evidence-from-a-betting-market/)) | Diff-in-diff on the 1992 introduction of Beyers in the DRF | More money bet, no significant efficiency change | Public figures (Timeform greyhound ratings on every UK card) are priced; ours must differ from them |
| Silverman (2012), *J. Prediction Markets* 6(1) ([PDF](https://www.ubplj.org/index.php/jpm/article/download/590/628)) | Hierarchical Bayesian regression of speed by race profile, HKJC, 36,006 runs | Top speed won 21.63%; combined with odds still lost 12.15% | Speed model is a stage-one input only |
| Glickman & Hennessy (2015) ([PDF](https://www.glicko.net/research/multicompetitor.pdf)) | Dynamic rank-ordered logit, random-walk abilities, approximate filter | Applied to 268 skiers over 103 events | Order-only rating as a robustness check against bad times |
| Wilson & Wilson (2024) ([arXiv](https://arxiv.org/abs/2405.09865)) | Bayesian LMM of log finish time, athlete and course random effects, weather | Courses differ strongly; rain slows; posterior predictive checks | Template for the crossed model and its per-meeting checks |
| Efron & Morris (1975) ([PDF](https://www.medicine.mcgill.ca/epidemiology/hanley/bios602/MultilevelData/EfronMorrisJASA1975.pdf)) | James-Stein shrinkage of early-season batting averages | Shrunk estimates beat raw averages | Justifies shrinking standards, variants and new dogs' levels |
| Killick et al. (2012) ([arXiv](https://arxiv.org/abs/1101.1438)) | PELT exact change-point detection in linear time | Orders of magnitude faster than exact alternatives | Detecting surface, trap and timing changes in standards |

Searched and **not found** ("no data found"): any peer-reviewed greyhound race-time model with going or meeting variants; any study of within-meeting drift in greyhound times; any published evaluation of GBGB going allowances; a published method for Racing Post greyhound ratings, Greyhound Recorder ratings, Topaz ratings or a US "Best Bets" greyhound rating; a greyhound sectional-pace (finishing speed) study.

---

## 7. Ranked actions (expected value for us, highest first)

Each is a registered experiment in the fast loop, scored as in `reports/greyhound_model.md` 41.4: out-of-sample error of the expected time in lengths, then RMSE of log(BSP) and log-loss against the live stack, and the blend weight against the BSP.

1. **Retrospective (smoothed) meeting variants and "form held up".** Re-estimate past meetings' variants with all runs before the prediction day; recompute dogs' historic figures from them. Source principle: Beyer's race-back check ([DRF](https://www.drf.com/news/tips-making-your-own-figs)). Why first: the variant is the largest single source of figure error after the dog (whole race 3.83 to 3.67 lengths from the variant alone, 41.5), and a smoother uses far more evidence than a same-night filter. Risk: leakage if any run on or after day d enters; enforce with the existing lag tests.
2. **GBGB going allowance calibrated and used as a prior and a feature** (section 4.2), with the sign check of section 2. Cheap, uses a field we already hold and do not read.
3. **Within-meeting drift and live in-meeting variant** (section 4.2). For the live price this is new information on the night: after 3-4 races, the filtered drift updates every later race's expected times. Sources: HKJC, Beyer, Timeform.
4. **Distance-scaled and per-bend variant; go split proportionally between sectional and run-in** (sections 4.2, 4.3). Sources: Central Park returns, GRI groups, Timeform.
5. **Change-point standards** replacing pure rolling windows (section 4.1), validated on Oxford 2023 and Central Park 2023.
6. **Race simulation with shared meeting and race shocks and a trouble mixture** (section 4.6, item 1), feeding the Benter second stage. Fixes the independence assumption in `et_p_fastest` and `ab_p_best`.
7. **Handicap-start check** of 0.08 s per metre (section 5) and, if confirmed, a corrected handicap time so handicap runs can enter dogs' levels.
8. **Fit the scaling exponent b** (section 4.5) instead of assuming percent.
9. **Precision-weighted, clean-run variant and night trap-bias term** (section 4.2).
10. **Prize money as a class covariate in opens; field-strength feature** (section 4.4).
11. **Order-only dynamic rating** (Glickman-Hennessy) as a cross-check and bad-time detector (section 4.6).

Caution from the evidence: every published time model that was scored against the market lost money alone (Silverman; the Betfair tutorials; Schumaker in the earlier note). The scoreboard is the blend weight against the BSP (0.040 now, 41.7), not the time error.

---

## 8. Could not verify or access

- **GBGB sign convention:** sources conflict (Towcester's text vs its arithmetic; Timeform's example). Settle on our data (section 2).
- **Whether GBGB sectionals are measured per runner or only for the leader:** not stated in Rules 138, 139 or 151.
- **GBGB margin rounding rule and the meaning of "Dis":** not found.
- **Ragozin and Thoro-Graph methods:** only a secondary article read; TDN obituary blocked (403), Thoro-Graph's site rendered empty; no formulas.
- **Racing Post Topspeed:** forum posts only.
- **TrackMaster/Equibase intertrack variant:** described only in an AQHA guide; formula proprietary.
- **GRNSW time bands table, GRV and Racing Queensland procedures:** PDFs not read in full; details from search summaries.
- **Rosenbloom (2003):** abstract and secondary summary only (ScienceDirect and ResearchGate returned 403).
- **Balakrishnan & Bernard:** blog summary only; paper not read.
- **UTS greyhound tracking (speed profile, about 20 m/s after 80-100 m):** MDPI page blocked (403); the figure is from a search summary.
- **Greyhound Star Nottingham split example (75 m, about 4.88 s):** from a search summary; page not read.
- **Greyhound Stats UK conversion method:** not published.
- **Paulick Report "Figure Makers" series:** blocked (403).
