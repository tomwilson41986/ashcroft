# Self-hosted runners on the UK server

Betfair refuses logins from GitHub's own runners, so every Betfair job runs on the UK server. Each job needs its own
runner, because a runner takes one job at a time and these jobs run all day.

Each sport is kept apart (the owner, 3 Nov 2026):
- horse racing on the live runner, and nothing else there;
- all football on `football`;
- greyhounds on `paper-models`;
- the other sports' nightly reads on `data`.

The guard tests fail if anything but horse racing runs on the live runner, or a football process appears outside
`football-trade.yml`.

| Runner label | Workflow | Runs | Key |
|---|---|---|---|
| `vars.BETFAIR_RUNNER` (exists) | `live-trade.yml`, `live-record.yml`, `betfair-prices.yml`, `tote-probe.yml`, `runner-diagnose.yml` | horse racing only: the live trader, its records, the UK/IE and other horse racing price files, the Tote | live (trader only) |
| `football` | `football-trade.yml` | 05:40 UTC for 23.5 h: all football. The trader, every Betfair match recorded and paper-traded, the football book records, the line-ups collector | live for the trader only, with `FOOTBALL_LIVE=yes`; delayed for the rest |
| `paper-models` | `paper-models.yml`, `greyhound-prices.yml` | greyhounds. By day (05:45 UTC to 21:30 UK): the paper trader, book recorders, early lay and the racing census. By night (22:35 to 05:15 UTC): Betfair's greyhound price files | live for the greyhound trader and book recorder; delayed for the rest |
| `data` | `other-data.yml` | 22:30 UTC nightly, about 75 minutes: every sport's market census, tennis odds, Betfair Historic Data | delayed |

## Set up the three new runners (on the server, as the live runner's user)

0. **Merge PR #184 first.** GitHub runs scheduled workflows only from the default branch
   (`claude/scrape-horse-racing-results-vaHKc`). Until then `football-trade.yml` does not exist there, and
   `paper-models.yml` there still starts the football trader on paper.
1. Get a registration token. In GitHub, go to the repository's **Settings > Actions > Runners > New self-hosted
   runner** and copy the token from the "Configure" step. Each token is valid for one hour and can register all three
   runners.
2. Put `scripts/setup_runner.sh` on the server, then run it once for each label. The repository is private, so use
   whichever way in works. One option is the live runner's own checkout, after a live job has run on the merged branch:
   `ls ~/actions-runner*/_work/ashcroft/ashcroft/scripts/setup_runner.sh`. Another is to paste the file into
   `~/setup_runner.sh` with an editor.

   ```bash
   bash ~/setup_runner.sh football
   bash ~/setup_runner.sh paper-models
   bash ~/setup_runner.sh data
   ```

   Each run does the following:
   - downloads runner v2.338.2 and checks it against GitHub's SHA-256;
   - asks for the token;
   - registers the runner with only its own label;
   - installs it as its own service.

   It leaves the live runner alone.
3. Check in **Settings > Actions > Runners** that all three runners show **Idle**. The labels must read
   `football`, `paper-models` and `data`, with no `self-hosted`.
4. Start the jobs. Either wait for the schedules, or go to **Actions > Football trading > Run workflow** and
   **Actions > Paper models > Run workflow**.

### Two things the script guards

- **Only its own label.** The runners are registered with `--no-default-labels`. If `BETFAIR_RUNNER` is
  `self-hosted`, a new runner with the default labels could otherwise take the live horse trader's job.
- **No shared runner.** Each workflow stops if its label is the same as the live runner's. `football-trade.yml` also
  stops if its label is the same as the paper models' runner.

## Real football stakes

`football-trade.yml` stays on paper, on the delayed key, until the repository variable `FOOTBALL_LIVE` is `yes`.
Set it at **Settings > Secrets and variables > Actions > Variables**.

With it set, the trader takes the live key and bets GBP2 a bet, GBP2 a match and GBP20 a day (`live_limits` in
`football/live_config.json`). Every other match stays on paper on the same feed.

To stop new bets within a minute, create the S3 object `football/STOP`. To stop everything, cancel the run.

## Capacity

Before the first real run, a simulated day of 3,000 markets through the feed used:
- about 205 MB of peak memory;
- about 30 Betfair calls a minute;
- under 6 seconds per loop.

The script prints free memory and disk before it installs. The football job writes up to about 750,000 book rows a
day under `~/.ashcroft/football/exchange/` and copies them to S3. Prune old local days if the disk is small.

## Remove a runner

```bash
cd ~/actions-runner-football && sudo ./svc.sh stop && sudo ./svc.sh uninstall && ./config.sh remove --token <a removal token>
```

The removal token comes from the same GitHub page, from the runner's **Remove** option.
