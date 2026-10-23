# The Professor: an automated tipster on the Ashcroft model (design, 23 Oct 2026)

The owner asked (23 Oct 2026) for a design: a public tipster persona, "The Professor", a former horse-racing
trader and betting-syndicate quant, that tips the Ashcroft model's horses. This is the design and the evidence it
rests on; nothing here is built yet.


The owner wants a public tipster persona, "The Professor" (a former racing trader and syndicate quant), that
publishes the Ashcroft model's selections every day at 10:00 UK once the morning's Betfair trades are in: two
free tips a day on X and Instagram, every tip (free and private, up to 10 a day) in a Telegram members channel,
a detailed write-up per horse, a maximum price of 50/1, every advised price available with three or more major
bookmakers, and Stripe for membership payments. TikTok was asked for and is dropped (the owner, 23 Oct): its
Community Guidelines ban "facilitation or marketing of gambling", unaudited API clients can only post privately,
and its audit rules reject "a tool to upload content to accounts you manage".

Nothing of this exists yet: no bookmaker-odds code, no social or payment code, no per-horse narrative. What does
exist and is reused: the live trader's ledger (which horses we backed, at what price, the closing model's expected
BSP and expected CLV), the 04:00 UTC predictions file and the morning card it keeps in S3, the live market record
(runner, jockey, trainer, draw, OR, form string), the nightly results load (industry SP, BSP, placings), the SMTP
email helpers, the research-query workflow for backtests, and the Netlify site with S3-backed functions.

### The decisions this plan rests on (the owner, 23 Oct 2026)

| Question | Decision |
|---|---|
| TikTok | Dropped for now; revisit via an audited third-party scheduler |
| Odds feed | Scrape Oddschecker and Racing Post's odds comparison (no paid API); a third source, "market form", to be named by the owner (no site of that name was found) |
| Launch | Proof first: 4-6 weeks of free, publicly proofed tips with Stripe built but closed; then open paid membership |
| Volume | Up to 10 tips a day; the free two are ranks 1-2 |
| Selection | Price-based: horses we expect to win often AND to shorten, i.e. a bookmaker price above the BFSP we expect; a high strike rate wanted |

### What the evidence says the tips can honestly claim

The model's proven edge is the price move: every live back is laid off at the Betfair SP, and the live record
(30 Sep to 22 Oct) is scored as CLV. Held to the result, the backtests of the 22%+ overlay rule made +8 to +9% at
level stakes with a confidence interval touching zero, and the long prices carry the best CLV (21-50 +9.5%, 50-100
+26.7% live). So the record is published two ways, as the proofing norm expects: ROI to the advised bookmaker
price (what a follower gets, with and without Best Odds Guaranteed) and the beat over the Betfair SP (CLV), which
is the claim the model supports. Success claims wait for the proofed record (ASA/CAP: substantiated, period and
stakes stated, no guarantees or income claims; 18+ and a safer-gambling line on every post; the ASA's May 2026
Oddschecker ruling treats organic posts as ads and rejects a self-declared age gate). The Professor discloses that
the syndicate holds positions on every tip before it is published. Posts carry no URLs on X (a URL post costs
$0.20 against $0.015) and no affiliate links anywhere (those would need "#ad" and platform permission regimes).

## The selection rule

Among the horses the syndicate holds at 09:55 UK (the ledger's matched backs), advise each horse whose best price
quoted by three or more major bookmakers clears the closing model's expected BSP by `overlay_bar`, at most one a
race, ranked by expected win probability `p_win = 1/E[BSP]` (the closing model's book, the best winner forecast we
have); cap at 10, the top two free. The ledger already carries both inputs: `p_pool` (= 1/E[BSP]) and `edge`.

Parameters set by a backtest before anything is published (Phase 0): `overlay_bar` in {0, 0.05, 0.10, 0.15,
0.20}, `max_tips` in {2, 5, 10}, `max_per_race` in {1, all}, `min_p_win` in {0, 0.05, 0.08, 0.12}. Decision rule
written in the query's header: the cell with the highest ROI at the proxy price whose October holdout ROI at the
BSP is positive and whose free-tip (ranks 1-2) strike rate is at least 20%.

`research/queries/professor_rule_1023.py` (read-only, run by research-query.yml, moved to `done/` after; a line in
`reports/research_ledger.jsonl`; findings in `reports/professor_rule.md`). Two populations, held to the result at
1 point level:
- A. The live rule's own horses, 10-22 Oct: `live_orders` (`event='back'`, matched > 0, the first fill a horse)
  joined to `betfair_live_marks` at `clock_09:00` / `clock_10:00` (`back1` as the bookmaker-price proxy, also with
  5% and 10% haircuts) and `final` (`bsp`), and to `race_results` through `betfair_live_markets.race_results_id`.
- B. The history, 1 Aug - 22 Oct: `predictions/<day>.csv` from S3 (the `preds()` loop in
  `research/queries/done/benter_blend_1020.py:57`) joined to `betfair_prices` (`morningwap`, `bsp`, `win_lose`) and
  `race_results`; candidates rebuilt as `trading/strategy.py _plan_closing_clv` does it with
  `model/race_book.py race_expected_clv` and `data/models/closing_model_novol.json` (ev >= 0.03, price >=
  predicted BFSP, price <= 200); parameters set on Aug-Sep, confirmed on Oct.
- Per cell: tips a day, strike rate, points and ROI at the proxy price, ROI at BSP and CLV, longest losing run,
  max drawdown, the same for ranks 1-2 alone, by price band (<4, 4-8, 8-16, 16-51), a race-bootstrap 90% interval
  (`race_book.ratio_interval`); and ranking by `p_win` against ranking by overlay and by `edge`.

**Phase 0 outcome (23 Oct, reports/professor_rule.md)**: the backtest set `overlay_bar` 0.10, `min_p_win` 0, one a race,
`max_tips` 5 (ranks 6-10 return nothing; the owner's 10 stays the ceiling), the ranking by expected win probability confirmed
(by the overlay or the expected CLV the tips are the long prices: strike 5%, -12%). The expected return at the Betfair
morning price is +5 to +10% implied (+11 to +19% realised, intervals spanning zero); at the Betfair price at 10:00 UK on the
live record -6%; the industry SP loses 29-55%. What a bookmaker's price at 10:00 returns is the proofing period's question.

## Package `professor/` (one module a concern, like `football/` and `greyhound/`)

```
professor/
  __init__.py      the persona, the rule, the owner's decisions (23 Oct 2026)
  __main__.py      python -m professor --publish|--select|--odds|--settle|--proof|--reconcile|--replay
  config.json      every parameter (below)
  requirements.txt anthropic, playwright, requests-oauthlib, Pillow (not in the root requirements)
  store.py         S3 keys under professor/, read/write, the post log, out/professor/<day>/, the STOP switch
  candidates.py    ledger + predictions + card + catalogue -> Candidate rows
  odds.py          scraper orchestration, the 3-books rule, the snapshot (evidence)
  scrapers/base.py, oddschecker.py, racingpost.py   (a third Source slot for "market form")
  selection.py     the bar, the rank, the caps, free/private tiers
  writeup.py       fact sheet, Claude call, validation, templated fallback, compliance lines
  render.py        Pillow tip cards 1080x1080 and 1080x1350, the results card; bundled OFL fonts in assets/
  telegram.py, xapi.py, instagram.py   thin API clients
  publish.py       orders the channels; idempotent through posts.json
  settle.py        nightly results, the running record, the results post
  proof.py         proof JSON for the site
  members.py       Stripe reconcile, Telegram kicks, Instagram token refresh
```

**store.py**: bucket `ULTRA_BETTING_S3_BUCKET` (default `ashcroft`), prefix `professor/`. Keys:
`tips/<day>/{candidates.csv,tips.json,writeups.json,posts.json}`, `odds/<day>/snapshot_<HHMM>.csv` plus
`odds/<day>/raw/<source>_<HHMM>.html.gz`, `cards/<day>/<slug>.png`, `results/<day>/settled.csv`,
`results/record.json`, `proof/summary.json`, `proof/days/<day>.json`, `members/<customer_id>.json`,
`members/tokens/<token>.json`, `members/log/<day>.jsonl`, `members/ig_token.json`, `STOP` (no publishing while it
or `trading/STOP` exists). Reuse `ultra_betting.data.s3` read/write helpers.

**candidates.py**: `ledger_positions(day)` from `auto_trade.read_ledger()` (event `back`, matched > 0; one row a
market/selection: stake, avg_price, the first fill's `p_pool`, `edge`, `move`, first_ts); `race_facts(day)` from
the latest `racecards/<day>_*.csv` (written by `pipeline/predict.py` `card_sink`: race_name, class, type,
furlongs, going, runners, jockey, trainer, stall, OR, age, days_since_lr, headgear) joined to
`betfair_live/<day>/markets*.csv.gz` (`betfair_recorder._read_day_files`, `MARKET_FIELDS`: cloth, draw, form
string, wearing) by `betfair_prices.normalise_track/normalise_horse` and race time, as
`betfair_recorder.match_markets` does; `market_now(day)` the last open book from `books*.csv.gz` (REMOVED runners
dropped); `build(day, now)` drops races off within `min_minutes_to_off`.

**odds.py / scrapers/**: `Quote(source, bookmaker, venue, race_hhmm, horse, price, fetched_utc)`; each source in
its own try (a failure is a status line, never a stop); `fetch_html(url, use_browser=False)` with a desktop UA, a
2 s pause between pages, 3 retries with backoff, Playwright Chromium when a site needs it; bookmaker names mapped
through `bookmaker_aliases`; `snapshot()` writes the CSV and the raw pages (the proofing evidence);
`advised_price(quotes, major, min_books=3)` returns the highest price quoted by at least three major bookmakers
within one source (the third-highest major quote) and the bookmakers at or above it, or None. The scrapers run
only on the UK server (GitHub's runners are Cloudflare-blocked for these sites); the risk that a site blocks or
changes stays and is why a failed source does not stop the run and why the raw pages are kept.

**selection.py**: filter advised <= 51.0, overlay = advised/E[BSP] - 1 >= `overlay_bar`, p_win >= `min_p_win`,
one a race (highest p_win); rank by p_win; cap `max_tips`; ranks 1-2 tier `free`, the rest `private`.

**config.json** (the backtest sets the first three; the rest are the owner's decisions): `overlay_bar`,
`min_p_win`, `max_per_race`, `max_price` 51.0, `max_tips` 10, `free_tips` 2, `min_books` 3,
`min_minutes_to_off` 60, `scrape_at` 09:35, `select_at` 09:55, `publish_at` 10:00, `major_bookmakers` (bet365,
William Hill, Paddy Power, Betfair Sportsbook, Sky Bet, Ladbrokes, Coral, BetVictor, Unibet, Betfred, 888sport,
BoyleSports), `bookmaker_aliases`, `stake_points` 1, `bog` true, `commission_bsp` 0.02, `writeup` (model
`claude-sonnet-5-5`, effort low, max_tokens 1200, banned phrases: guarantee, can't lose, sure thing, income,
risk-free, easy money), `compliance` (the 18+/BeGambleAware line; the price-basis statement; the position
disclosure), `members.grace_days` 3.

**writeup.py**: `fact_sheet(tip)` holds only facts (race, horse, form, prices: advised and its bookmakers, Betfair
now, our expected BSP, overlay %, expected win %; the position disclosure); `compose(sheet)` calls the Claude API
(Python SDK, the persona as a cached system block, JSON output of headline / body / x_text / ig_caption, effort
low; check `stop_reason == "refusal"`); `validate(text, sheet)` rejects any number not in the sheet, any banned
phrase, any URL, an X text over 280 with the compliance line; one retry with the reasons, then `template(sheet)`,
deterministic sentences that always pass. Publishing never depends on the API.

**render.py**: `tip_card` (venue/time, horse, advised price and the bookmakers at it, "expected to shorten" line,
compliance footer, timestamp), `day_card` (the list, portrait for Instagram), `results_card`.

**publish.py**: `plan(day, tips, texts, cards, paid_open)` builds ordered posts keyed
`<channel>:<tier>:<day>:<slug>`: the members channel first at 10:00:00 UK (every tip, card plus caption, the
write-up as a second message; only when `PROFESSOR_PAID_OPEN=yes`), then the free channel (the two free tips), X
(one post a free tip, the second a reply to the first; card attached, no URL), Instagram (one carousel: day card
plus the two free tip cards; images fetched by Meta from the site's card route). `run()` skips keys already in
posts.json, records each API response (the published timestamp is the proof the tip preceded the off), and a
channel's failure never stops the others. `wait_until()` reuses `auto_trade.uk_time_on`.

**settle.py**: `settle_day(conn, day)` joins tips.json to `race_results` (as `match_markets` does) and
`betfair_live_marks` final `bsp`: result, industry SP, BSP, `pnl_advised` (1 pt level, non-runner void), `pnl_bog`
(max(advised, SP)), `pnl_bsp` (less commission), `clv = advised/bsp - 1`; `update_record()` keeps every settled
tip in `results/record.json` with totals by tier: n, strike rate, points, ROI to advised / BOG / BSP, mean CLV,
longest losing run, max drawdown, by month; `results_post()` posts the results card to both Telegram channels, X
and Instagram, key `results:<day>`. Runs for the last 3 days so a late result row settles the next night.

**proof.py**: `proof/summary.json` (headline numbers, cumulative points by day, the last 30 days of tips with
advised/BSP/result, the period-and-stakes statement, the compliance lines) and `proof/days/<day>.json`. Site:
`frontend/professor.html` + `frontend/professor.js` read `/api/professor/proof` and `/api/professor/day/<date>`,
two routes added to `frontend/netlify/functions/api.js` with its `readS3Json`.

**members.py**: `reconcile(now)` reads every `members/<id>.json` with a Telegram user id, asks Stripe for the
subscription's status, keeps active/trialing and past_due within the grace days, else `banChatMember` then
`unbanChatMember(only_if_banned=True)` and logs it; emails the summary through `send_email_report`.
`refresh_ig_token()` refreshes the long-lived Instagram token when under 10 days remain.

**Telegram / X / Instagram clients**: Bot API `sendPhoto`, `sendMessage`, `createChatInviteLink(member_limit=1,
expire_date)`, ban/unban; X v1.1 media upload + v2 post with OAuth 1.0a (pay-per-use, about $1.35 a month at three
image posts a day); Instagram Graph API with Instagram Login on a Business account (container -> publish; image,
carousel; publishing to our own account needs no App Review; 100 posts a day).

## Membership (Stripe -> Telegram) on the Netlify site

Netlify functions beside `frontend/netlify/functions/api.js` (Node, the same S3 pattern; `netlify.toml` redirects
for `/professor/cards/*`, `/professor/checkout`, `/professor/join`, `/professor/stripe`, `/professor/telegram`):
- `professor-card.js`: streams `professor/cards/<day>/<slug>.png` from S3 (Meta fetches images from a public URL).
- `professor-checkout.js`: creates a Stripe Checkout session in subscription mode (`STRIPE_PRICE_ID`,
  success URL `/professor/join?session_id=...`).
- `professor-stripe.js`: verifies the webhook signature; on `checkout.session.completed` writes
  `members/<customer_id>.json` (email, subscription id, status, a 32-hex token, telegram_user_id null) and
  `members/tokens/<token>.json`; `invoice.paid`, `invoice.payment_failed`, `customer.subscription.deleted` update
  the status.
- `professor-join.js`: the success page with the button `https://t.me/<bot>?start=<token>`.
- `professor-telegram.js`: the bot's webhook (checked by `X-Telegram-Bot-Api-Secret-Token`): on `/start <token>`
  for an active member not yet joined, a single-use one-hour invite link to the members channel and the user id
  written to the record; `/manage` sends a Customer Portal link.
The checkout button stays hidden and `PROFESSOR_PAID_OPEN` stays `no` through the proofing period. Stripe's
restricted-business list does not name tipping (only "sports forecasting with a prize"); ask Stripe support in
writing before going live, describing an information service that takes no bets and pays no prizes.

## Timing and runners

| Job | UK time | Cron (UTC) | Runner |
|---|---|---|---|
| predict.yml (exists) | 05:00 BST / 04:00 GMT | `0 4 * * *` | ubuntu-latest |
| live-trade.yml (exists) | 06:15 | `15 5 * * *` | `BETFAIR_RUNNER` |
| professor-publish.yml: scrape 09:35, select 09:55, publish 10:00 | 09:35 | `35 8 * * *` and `35 9 * * *` with a UK-hour guard (`TZ=Europe/London date +%H` == 09), the pattern of live-trade's "Choose the session" | `PROFESSOR_RUNNER` |
| professor-settle.yml (settle, proof, results post) | 23:15 / 22:15 | `15 22 * * *` (after daily-results 21:30 and the live settle 21:45) | ubuntu-latest |
| professor-members.yml (reconcile, IG token) | 07:20 / 06:20 | `20 6 * * *` | ubuntu-latest |

`BETFAIR_RUNNER` is held by the trading job from 05:15 UTC to the last race, so a 09:35 job on that label would
queue until the evening: the owner registers a second runner instance on the same UK server (label `professor`,
variable `PROFESSOR_RUNNER`, its own venv), which gives the scrapers the UK egress without waiting. Horses backed
after 09:55 are not advised that day.

Workflow files carry the repo's long header (the persona, the owner's decisions of 23 Oct, the runner reasoning):
- `professor-publish.yml`: `if: vars.PROFESSOR_RUNNER != ''`, concurrency `professor-publish`, timeout 60, the
  uv venv block from live-trade.yml, `pip install -r professor/requirements.txt`, `playwright install chromium`
  (cached), `python -m professor --publish --at 10:00 [--dry-run]`, `out/professor/` as an artifact, the
  failure email as daily-results.yml does; posts only with `PROFESSOR_LIVE=yes` (variable).
- `professor-settle.yml`: ubuntu-latest, Python 3.11, the DB from S3 (`scripts/fetch_data.py --db-only`),
  `python -m professor --settle --days 3 && python -m professor --proof`.
- `professor-members.yml`: `python -m professor --reconcile`.

Secrets to add (CLAUDE.md's table): `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `X_API_KEY`, `X_API_SECRET`,
`X_ACCESS_TOKEN`, `X_ACCESS_SECRET`, `IG_ACCESS_TOKEN` (seed), `STRIPE_SECRET_KEY`, and the test set
(`TELEGRAM_TEST_CHANNEL_ID`, `X_TEST_*`, `IG_TEST_*`). Variables: `PROFESSOR_RUNNER`, `PROFESSOR_LIVE`,
`PROFESSOR_PAID_OPEN`, `TELEGRAM_FREE_CHANNEL_ID`, `TELEGRAM_MEMBERS_CHANNEL_ID`, `IG_USER_ID`,
`STRIPE_PRICE_ID`, `SITE_URL`. Netlify environment: the Stripe keys and webhook secret, the bot token and webhook
secret, the channel id, and an S3 key allowed to write `professor/members/*` (the site's key reads only now).
Accounts the owner opens: X developer (pay-per-use), an Instagram Business account and Meta app, a Telegram bot
and two channels (free, members), a Stripe account and price, the second runner.

## Phasing

- **Phase 0 (week 1)**: the research query and `reports/professor_rule.md`; `professor/config.json` with the
  chosen bar and caps; `professor/__init__.py`; a CLAUDE.md section.
- **Phase 1 (weeks 1-2), proofing starts**: store, candidates, odds + the two scrapers, selection, the templated
  write-up, render, telegram, publish (free Telegram channel only), settle, proof + the site page and routes;
  `professor-publish.yml`, `professor-settle.yml`; the second runner. Every tip goes to the free channel and the
  proof page, timestamped, with its odds snapshot.
- **Phase 2 (weeks 3-4)**: the Claude write-ups, `xapi`, `instagram`, `professor-card.js`, the X and Instagram
  legs, the results post.
- **Phase 3 (weeks 4-6)**: Stripe functions, `members.py`, `professor-members.yml`, the members-channel leg
  behind `PROFESSOR_PAID_OPEN`; Stripe test mode with a GBP0 coupon for the owner's own run-through. Paid opens
  when the proofing period ends (the owner's call on the record).

## Verification

- Phase 0: the query runs green on research-query.yml; the report shows the grid for both populations and the
  October holdout; the chosen cell is traceable to its ledger line.
- Phase 1: `tests/test_professor_rule.py` (`advised_price` picks the third-highest major quote and returns None
  under three; `select` honours the bar, the cap, one a race, the ranks and tiers; a REMOVED runner is dropped);
  `tests/test_professor_odds.py` (saved HTML fixtures under `tests/fixtures/professor/` parse to the expected
  quotes; a source that raises leaves the other's quotes and a status line); `tests/test_professor_workflows.py`
  (crons, the UK-hour guard, runner label, the `PROFESSOR_LIVE` gate, concurrency, no Betfair secrets in the
  publish job), in the style of `tests/test_data_capture_workflows.py`. Dry runs: `python -m professor --replay
  --day 2026-10-22` from the cloud session writes `out/professor/2026-10-22/` (candidates, tips, cards) with no
  posting; then `professor-publish.yml` dispatched with `dry_run=true` on the UK runner proves the scrapers reach
  the sites from there (raw pages land in `professor/odds/<day>/raw/`); then one run with `--test-channels` to the
  private test channel; then `PROFESSOR_LIVE=yes`. Settle: `--settle --days 3 --dry-run` on the first three proofed
  days, the BSP ROI and CLV checked by hand against the ledger's own `clv`; the proof page shows the same numbers.
- Phase 2: `validate()` tests (a number not in the sheet, a banned phrase, a URL, over-length all fail; the
  template always passes); a replay with and without write-ups gives the same tips.json apart from the text; the
  test X and Instagram accounts receive the posts; re-running a day posts nothing (prefilled posts.json).
- Phase 3: Stripe test mode end to end with the Stripe CLI forwarding to the deployed function: checkout ->
  member record in S3 -> join page -> `/start` -> single-use invite -> the test channel; a cancelled test
  subscription is kicked by the next reconcile and logged; a webhook with a bad signature returns 400 and writes
  nothing.

## Open items for the owner

1. Name the third odds source ("market form"): the `Source` slot takes it without other changes.
2. Confirm with Stripe support in writing that a betting-advice subscription (no bets, no prizes) is accepted.
3. Open the accounts listed above and register the second UK runner.
4. The membership price (a config value; nothing in the design depends on it).

## Appendix: what was found about the platforms and the odds sources (23 Oct 2026)

### Bookmaker odds for UK/Irish racing

No free official per-bookmaker feed exists. The owner chose to scrape Oddschecker and Racing Post's odds
comparison from the UK server. The paid alternatives, should a scraper be blocked:

| Source | Per-bookmaker prices | Books | BOG flag | Cost | Caveat |
|---|---|---|---|---|---|
| The Racing API | yes, in the racecard (`odds[]` per runner, bookmaker, decimal, each-way terms) | "20+" UK/IE | no | plans not published on the site (reportedly from about GBP25 a month) | redistribution terms to confirm; cards refreshed every 3 min |
| OddsRelay | yes, with a Best Odds Guaranteed feed | 60+ UK/IE incl. bet365 | yes | GBP19-99 a month for own use | showing prices to subscribers needs an Enterprise contract (from GBP750 a month) |
| PuntersEdge | yes | about 5 UK books, median 4 a race; no bet365, Betfair Sportsbook, Unibet, BetVictor | no | about A$49 a month | explicitly allows showing prices to paying subscribers |
| The Odds API | no horse racing | | | | ruled out |
| Oddschecker | no public API; parent sells B2B feeds | all | shown on site | | Cloudflare-fronted; scraping is a terms risk for an unattended job, which is why a failed source never stops the run and the raw pages are kept |

None of them documents when the morning prices first appear; UK books price the day's cards from about 08:00 to
10:00, so the job scrapes at 09:35 and again at 09:55 for the candidates' races.

### Platforms

- **X**: pay-per-use since Feb 2026 ($0.015 a post, $0.20 with a URL; media upload billing not stated); no
  subscription tiers for new developers. Organic gambling content is allowed; paid partnerships for gambling stay
  prohibited in Europe and need disclosures in the UK. The ASA has held X's self-declared age gate insufficient.
- **Instagram**: the Graph API with Instagram Login on a Business account; publishing to one's own account needs
  no App Review; JPEG at a public URL, carousels to 20, Reels; 100 API posts a day. Meta's Community Standards have
  no gambling section (organic tips are not prohibited); boosted or branded gambling content needs written
  permission.
- **TikTok**: Community Guidelines (effective 24 Sep 2026) "do not allow the facilitation or marketing of
  gambling"; the ads policy names "insights, stats, or betting strategies" as gambling information; unaudited API
  clients post `SELF_ONLY`, and the audit rejects "a utility tool to help upload contents to the account(s) you or
  your team manages". Dropped.
- **Telegram**: Bot API `createChatInviteLink(member_limit=1, expire_date)` for one-off joins; `banChatMember`
  then `unbanChatMember(only_if_banned=True)` to remove a lapsed member. Star subscriptions exist (30-day periods,
  paid in Stars, payout through Fragment to a TON wallet with a 21-day hold, undisclosed commission) and are not
  used; Telegram Ads prohibit "tips, odds, forecasts", so the channel cannot be advertised there.
- **Stripe**: Checkout in subscription mode, webhooks (`checkout.session.completed`, `invoice.paid`,
  `invoice.payment_failed`, `customer.subscription.deleted`), the Customer Portal. The restricted-business list
  (22 Sep 2026) prohibits "sports forecasting or odds-making with a monetary or material prize" and, for Japan
  only, "consultation and advisory or prediction services relating to gambling"; tipping is otherwise unnamed.
  Written confirmation from Stripe support before launch.

### UK rules for a tipster

- A tipster that takes no bets needs no Gambling Commission licence; affiliates are regulated through the
  operators they promote (no affiliate links in this design).
- ASA/CAP (AdviceOnline, 22 May 2025): results must be proofed with an independent body before the off; past
  results stated with the period, total stakes and whether at SP or advised prices, not cherry-picked; no
  "guaranteed" or income claims, not "investment"; the tipped price must stay available "for a significant amount
  of time". The Cyan Blue Odds (Oddschecker) ruling of 27 May 2026 treated two organic Instagram posts as ads and
  found an "18+" bio insufficient; the ASA has monitored gambling posts since 11 Jun 2026. Every post therefore
  carries the 18+ and safer-gambling line, and no presenter or imagery with strong appeal to under-18s.
- Proofing norms (Smart Betting Club, Tipstrr, Checkatipster): the advised price recorded at publication and
  quoted only when three or more proper bookmakers offer it; ROI to advised prices and separately to SP/BSP; BOG
  disclosed (it overstates a record by about 2-3%); strike rate, average odds, longest losing run, drawdown, sample
  size.

## The service, redesigned for profit (the owner's ask, 23 Oct, after the backtest)

The backtest's lesson is about time and price, not about the horses. The trader's own first price returns +27%
implied on the live record; the Betfair price at 10:00 returns -6%; the industry SP loses 29-55%. A 10:00 bulletin
at bookmaker prices sells the followers the move after it has happened. The service below sells them the move.

### 1. The product: early alerts at the exchange price

- **The alert, not the bulletin.** A tip is published the moment the trader's first back on a qualifying horse
  fills (06:15 to about 09:00 UK; the GBP4,000 is spent by 08:38-10:27), to the members' Telegram at once. The
  10:00 post on X and Instagram is the day's card: the two free tips, 60-90 minutes after the members had them.
- **The advised price is the Betfair exchange price**, the best back at the alert with the money at it (both on
  record from the trader's own book), and the bookmaker prices at or above it beside it ("7.0 on the exchange,
  GBP140 there; 13/2 with bet365, Hills, PP"). The exchange price is real, timestamped, available to anyone with
  an account, never restricted, and it is the price the whole record is already scored on: the proofing problem
  of the first design (no bookmaker price on record) goes away. Bookmaker prices stay as the second route, with
  the 3-books rule and BOG, for the followers who have accounts.
- **The selection as set** (`professor/config.json`): the trader's horses, 10% over the expected BSP, one a race,
  by win probability, 5 a day, the top two free. One addition: the alert fires only while the exchange still shows
  at least `min_alert_size` (GBP50) at or above the advised price, so a member can take it.
- **Staking**: 1 point level, a 100-point bank, the worst run (12-27 tips) and drawdown (20-30 points) of the
  backtest stated up front; to-win staking as an option in the write-up, never Kelly.
- **The write-up** says why the price will move (the model's price against the market's, the horse's form in the
  model's terms, the connections' record), never why the horse will win: the claim is the move.

### 2. Why this is what the syndicate can sell without hurting itself

- **The tips are the short end of the book, where the trader's own edge is smallest** (under 5.0 the live CLV is
  +0.5%; 21-200 is +10-27%). Publishing the short-priced qualifiers gives away the least and keeps the long end,
  which is the trading profit, private.
- **Publishing shortens our own horses' BSP.** Every back is laid at the SP, so a tip that moves the close helps
  the position already held. This is a real, measurable second income and a conflict to disclose on every post
  ("the syndicate holds every horse before it is advised"). It is measured, not assumed: in the proofing period
  a random fifth of the qualifying horses is withheld from publication, and the BSP move of the published against
  the withheld, matched by rank and price, is the published impact (the same experiment shows whether the tips
  cost the trader's later top-ups their price).
- **The CLV record is the sales pitch**, and it is the one no other tipster can show: a page with the advised
  price against the BSP on every tip, updated nightly from the same data the trading settles on.

### 3. Capacity: the constraint that sets the price

Followers' money moves the price. A horse the rule picks has a median of about GBP500 traded all morning and
GBP150 or less a tenth of the time; 100 members backing GBP25 each is GBP2,500 on the alert. So:

- **Membership is capped**, at 100 seats to begin, raised only as the record shows the advised price still
  standing a minute after the alert (the alert records the book; the first minute's trades are in the market
  record). A waiting list is the marketing.
- **The price follows the cap**: GBP49 a month at 100 seats (GBP4,900 a month, about GBP59,000 a year gross) is
  the working figure; the free tier (two tips a day, an hour late) is the funnel. Costs are about GBP100 a month
  (X about GBP2, the Claude API GBP10-30, the site and runner already paid for), so the margin is the revenue.
- **No affiliate links in year one.** They need "#ad" and a bookmaker's permission regime, they pay best on losing
  players, and they tie the brand to firms that will restrict its best followers.

### 4. The money, honestly

| Line | Working figure | What it rests on |
|---|---|---|
| Membership | GBP4,900 a month at 100 seats and GBP49 | the record after the proofing period; churn of 10-15% a month is normal, so the funnel must refill 10-15 seats a month |
| The syndicate's own book | +1-2% CLV on the tipped horses, about GBP20-60 a day on today's stakes | the withheld-fifth experiment; nil until measured |
| Costs | about GBP100 a month | the design's accounts |
| The followers' return | about 0 to +10% ROI at the exchange price after commission, 23% strike at 5-6 | the backtest; the first weeks' record settles it |

A service at this size is worth about a fifth of the trading book's current run rate. It is worth building because
the cost is near nil, the brand and the proofed CLV record compound, and the cap can rise with the liquidity the
record shows.

### 5. The sequence

1. **Weeks 1-6, free and proofed.** The alerts to a free Telegram channel at the moment of the fill; the two free
   tips to X and Instagram at 10:00; the withheld fifth; the proof page from day one with the CLV, the ROI at the
   advised exchange price and at the best bookmaker price, and the share of tips whose price still stood a minute
   on. Collect the waiting list.
2. **Decision at week 6**, on the record: the implied ROI at the advised price positive, the price standing a
   minute on for most tips, and the published impact measured. If the record carries it, the paid channel opens at
   100 seats; the free channel keeps the two late tips. If not, the service stays free and the brand keeps the
   content, which costs nothing.
3. **From then**: raise the cap only on liquidity evidence; add the write-up archive and the monthly letter to the
   paid tier; revisit affiliate income only with CPA deals and the labels, and only once the record is a year old.

### What changes in the design above

- `professor-publish.yml` becomes a session on the UK runner from 06:15 UK beside the trader (the pattern of the
  recorder: it reads the trader's ledger copy in S3 every minute and alerts on new fills that qualify), with the
  10:00 card as a step in the same session; `scrape_at` is a loop from 07:00 (bookmaker prices as they appear).
- `selection.py` gains `min_alert_size` and the withheld fifth (a seeded random draw by day and market, recorded
  in tips.json as `withheld: true`, settled and reported like a tip, never published).
- The advised price field is `advised_exchange` (with `exchange_size`) and `advised_books` (price, bookmakers);
  the proof page scores both.
