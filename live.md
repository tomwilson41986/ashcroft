# Ultra Betting — Autonomous Pipeline

## System Specification v2.0

**Date:** March 2026

---

## 1. What This System Does

Three things, every day, without you touching it:

1. **Morning** — Generate predictions for today's racing
2. **Pre-race** — Compare predictions to live Betfair prices, place bets where the rules say to
3. **Evening** — Settle bets, calculate P&L, email you a summary with a cumulative performance graph

Plus a Claude Code MCP server for when you want to manually check prices, override something, or investigate.

---

## 2. System Architecture

```
                         ┌──────────────────────────┐
                         │     GitHub Actions        │
                         │     (cron scheduler)      │
                         └─────┬──────┬──────┬──────┘
                               │      │      │
                    06:00 GMT  │      │      │  20:00 GMT
                    ┌──────────┘      │      └──────────┐
                    ▼                 ▼                  ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │  PREDICT     │  │  EXECUTE     │  │  SETTLE      │
            │              │  │              │  │              │
            │ Pull data    │  │ Pull preds   │  │ Pull results │
            │ Run model    │  │ Pull prices  │  │ Calc P&L     │
            │ Write preds  │  │ Apply rules  │  │ Build report │
            │ → S3         │  │ Place bets   │  │ Email summary│
            │              │  │ Log to S3    │  │ Update graph │
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                  │
                   ▼                 ▼                  ▼
            ┌─────────────────────────────────────────────────┐
            │                   AWS S3                         │
            │  s3://ultra-betting/                             │
            │  ├── predictions/{date}.csv                      │
            │  ├── executions/{date}.csv                       │
            │  ├── settlements/{date}.csv                      │
            │  ├── daily_pnl/{date}.json                       │
            │  ├── cumulative_pnl.json                         │
            │  └── reports/{date}.html                         │
            └─────────────────────────────────────────────────┘
                   │
                   │ (shared client library)
                   ▼
            ┌─────────────────────────────────────────────────┐
            │         Claude Code MCP Server                   │
            │         (on-demand, manual interaction)           │
            │                                                   │
            │  "What's the price on X?"                         │
            │  "Place £10 on Y"                                 │
            │  "Show me today's executions"                     │
            │  "Why didn't we bet on Z?"                        │
            └─────────────────────────────────────────────────┘
```

---

## 3. Shared Core Library: `ultra_betting`

Everything — the pipeline jobs, the MCP server — imports from the same Python package. This keeps the Betfair client, guardrails, and data models in one place.

```
ultra-betting/
├── src/
│   └── ultra_betting/
│       ├── __init__.py
│       │
│       ├── betfair/
│       │   ├── __init__.py
│       │   ├── client.py            # betfairlightweight wrapper
│       │   ├── auth.py              # Cert-based login + session keep-alive
│       │   ├── markets.py           # Market discovery helpers
│       │   ├── pricing.py           # Price fetching + formatting
│       │   ├── execution.py         # Bet placement + cancellation
│       │   └── settlement.py        # Settled bet retrieval
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── predict.py           # Ashcroft / XGBoost prediction runner
│       │   └── features.py          # Feature engineering (your 19 metric families)
│       │
│       ├── rules/
│       │   ├── __init__.py
│       │   ├── engine.py            # Rule engine — reads rules.yaml, outputs bet/no-bet
│       │   ├── edge.py              # Edge calculation (predicted vs market price)
│       │   └── staking.py           # Stake sizing (Kelly, fractional Kelly, fixed)
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── s3.py                # S3 read/write helpers
│       │   └── schemas.py           # Pydantic models for predictions, executions, settlements
│       │
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── daily_report.py      # HTML email report builder
│       │   ├── pnl.py               # P&L calculation + cumulative tracking
│       │   └── charts.py            # Matplotlib/Plotly cumulative P&L chart
│       │
│       ├── guardrails.py            # Max stake, max bets, circuit breakers
│       ├── audit.py                 # Structured JSON logging
│       └── config.py                # Configuration loading
│
├── pipeline/
│   ├── predict.py                   # GitHub Actions entry point: morning predictions
│   ├── execute.py                   # GitHub Actions entry point: pre-race execution
│   ├── settle.py                    # GitHub Actions entry point: evening settlement
│   └── shared.py                    # Common pipeline setup (logging, S3, auth)
│
├── mcp_server/
│   ├── __init__.py
│   ├── __main__.py                  # MCP stdio entry point
│   ├── server.py                    # Tool registration
│   └── tools/
│       ├── auth.py
│       ├── markets.py
│       ├── betting.py
│       ├── management.py
│       └── pipeline.py              # "Show today's predictions", "Why no bet on X?"
│
├── config/
│   ├── credentials.json             # (gitignored — injected via GH secrets)
│   ├── guardrails.yaml              # Stake limits, allowed markets, circuit breakers
│   └── rules.yaml                   # BETTING RULES — your model logic goes here
│
├── tests/
│   ├── test_rules_engine.py
│   ├── test_guardrails.py
│   ├── test_execution.py
│   └── conftest.py
│
├── .github/
│   └── workflows/
│       ├── predict.yml              # Cron: 06:00 GMT
│       ├── execute.yml              # Cron: configurable per-meeting timing
│       └── settle.yml               # Cron: 20:00 GMT
│
├── CLAUDE.md
├── pyproject.toml
└── README.md
```

---

## 4. Rules Engine (Pluggable)

The betting logic lives in `config/rules.yaml` and is interpreted by the rules engine. You define the rules, the engine applies them mechanically.

```yaml
# config/rules.yaml
# ---
# This file defines ALL betting logic.
# The pipeline reads it. No betting logic is hardcoded anywhere else.
#
# You fill this in with your model's rules.
# Examples of what goes here:

# What does the model output?
model_output:
  type: "probabilities"          # or "predicted_bsp", "both"
  # ...

# When do we bet?
entry_conditions:
  # ...

# Which side?
side_logic:
  # ...

# How much?
staking:
  method: "fractional_kelly"     # or "fixed", "percentage_bank"
  # ...

# When to execute?
timing:
  # ...

# What persistence type?
persistence:
  # ...
```

The rules engine (`src/ultra_betting/rules/engine.py`) works like this:

```python
class RulesEngine:
    """
    Pure function: (predictions, live_prices, rules_config) → list[BetInstruction]

    No side effects. No API calls. Just applies rules to data.
    The pipeline calls this, then passes the BetInstructions to the execution layer.
    """

    def __init__(self, rules_config: dict):
        self.config = rules_config

    def evaluate(
        self,
        predictions: list[Prediction],
        live_prices: dict[str, MarketPrices]
    ) -> list[BetInstruction]:
        """
        For each prediction:
        1. Find the corresponding live market/selection
        2. Calculate edge (however your rules define it)
        3. Check entry conditions
        4. Calculate stake
        5. Return BetInstruction if all conditions met, skip otherwise
        """
        instructions = []
        for pred in predictions:
            edge = self._calculate_edge(pred, live_prices)
            if self._meets_entry_conditions(pred, edge):
                stake = self._calculate_stake(pred, edge)
                instructions.append(BetInstruction(
                    market_id=pred.market_id,
                    selection_id=pred.selection_id,
                    side=self._determine_side(pred, edge),
                    stake=stake,
                    price=self._determine_price(pred, edge),
                    persistence=self.config["persistence"],
                    reasoning=f"Edge: {edge}, Pred: {pred.predicted_value}"
                ))
        return instructions
```

You define the rules. The engine applies them. The pipeline executes the output. Clean separation.

---

## 5. GitHub Actions Workflows

### 5.1 Predict (Morning)

```yaml
# .github/workflows/predict.yml
name: Daily Predictions
on:
  schedule:
    - cron: '0 6 * * *'   # 06:00 UTC daily
  workflow_dispatch:         # Manual trigger

jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[pipeline]"

      - name: Run predictions
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: eu-west-2
        run: python -m pipeline.predict

      # Output: s3://ultra-betting/predictions/{date}.csv
```

### 5.2 Execute (Pre-Race)

```yaml
# .github/workflows/execute.yml
name: Execute Bets
on:
  schedule:
    # Run every 30 mins during UK racing hours
    - cron: '*/30 10-20 * * *'   # 10:00–20:00 UTC
  workflow_dispatch:
    inputs:
      dry_run:
        description: 'Dry run mode'
        type: boolean
        default: false

jobs:
  execute:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[pipeline]"

      - name: Write Betfair certs
        run: |
          mkdir -p ~/.betfair/certs
          echo "${{ secrets.BETFAIR_CERT }}" > ~/.betfair/certs/betfair.crt
          echo "${{ secrets.BETFAIR_KEY }}" > ~/.betfair/certs/betfair.key

      - name: Execute bets
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: eu-west-2
          BETFAIR_USERNAME: ${{ secrets.BETFAIR_USERNAME }}
          BETFAIR_PASSWORD: ${{ secrets.BETFAIR_PASSWORD }}
          BETFAIR_APP_KEY: ${{ secrets.BETFAIR_APP_KEY }}
          DRY_RUN: ${{ inputs.dry_run || 'false' }}
        run: python -m pipeline.execute

      # Output: s3://ultra-betting/executions/{date}.csv
```

**Execution pipeline logic:**

```
1. Load today's predictions from S3
2. For each prediction, check: has this already been executed today? (idempotency)
3. For unexecuted predictions, check: is this market open? Is the race >N mins away?
4. Fetch live prices from Betfair
5. Run rules engine: predictions + prices → bet instructions
6. Apply guardrails to each instruction
7. Execute via Betfair API
8. Log execution to S3
```

The 30-minute cron handles the timing problem — races go off throughout the day, so the pipeline runs repeatedly, skipping already-executed bets and only acting on markets that are currently open and within the timing window defined in your rules.

### 5.3 Settle (Evening)

```yaml
# .github/workflows/settle.yml
name: Daily Settlement
on:
  schedule:
    - cron: '0 21 * * *'   # 21:00 UTC daily
  workflow_dispatch:

jobs:
  settle:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -e ".[pipeline]"

      - name: Write Betfair certs
        run: |
          mkdir -p ~/.betfair/certs
          echo "${{ secrets.BETFAIR_CERT }}" > ~/.betfair/certs/betfair.crt
          echo "${{ secrets.BETFAIR_KEY }}" > ~/.betfair/certs/betfair.key

      - name: Settle and report
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: eu-west-2
          BETFAIR_USERNAME: ${{ secrets.BETFAIR_USERNAME }}
          BETFAIR_PASSWORD: ${{ secrets.BETFAIR_PASSWORD }}
          BETFAIR_APP_KEY: ${{ secrets.BETFAIR_APP_KEY }}
          SMTP_HOST: ${{ secrets.SMTP_HOST }}
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASS: ${{ secrets.SMTP_PASS }}
          REPORT_EMAIL: ${{ secrets.REPORT_EMAIL }}
        run: python -m pipeline.settle

      # Outputs:
      #   s3://ultra-betting/settlements/{date}.csv
      #   s3://ultra-betting/daily_pnl/{date}.json
      #   s3://ultra-betting/cumulative_pnl.json  (appended)
      #   s3://ultra-betting/reports/{date}.html
      #   Email sent to REPORT_EMAIL
```

---

## 6. Pipeline Entry Points

### 6.1 predict.py

```python
"""
Morning prediction job.

1. Determine today's race meetings (from racing calendar or Betfair API)
2. Pull required feature data (form, ratings, etc.)
3. Run Ashcroft model
4. Write predictions CSV to S3

Predictions CSV schema:
  date, venue, race_time, market_id, selection_id, runner_name,
  <model output columns — you define these>,
  predicted_at
"""
```

### 6.2 execute.py

```python
"""
Pre-race execution job. Runs every 30 mins.

1. Load today's predictions from S3
2. Load today's executions from S3 (for idempotency)
3. Filter to predictions not yet executed
4. Filter to markets that are open and within timing window
5. Authenticate with Betfair
6. For each eligible prediction:
   a. Fetch live prices
   b. Run rules engine
   c. If bet instruction returned:
      - Apply guardrails
      - Place bet (or dry-run log)
      - Append to executions CSV
7. Write updated executions CSV to S3

Executions CSV schema:
  date, venue, race_time, market_id, selection_id, runner_name,
  side, stake, price_requested, price_matched, size_matched,
  bet_id, status, dry_run, reasoning, executed_at
"""
```

### 6.3 settle.py

```python
"""
Evening settlement job.

1. Load today's executions from S3
2. Authenticate with Betfair
3. Fetch settled bets from Betfair API
4. Match settlements to executions
5. Calculate per-bet P&L
6. Calculate daily totals (bets placed, winners, P&L, ROI)
7. Update cumulative P&L tracker
8. Generate HTML report with embedded chart
9. Email report
10. Write settlements + daily P&L to S3

Settlements CSV schema:
  date, venue, race_time, market_id, selection_id, runner_name,
  side, stake, price_matched, result, pnl, commission,
  net_pnl, bet_id, settled_at

Daily P&L JSON schema:
  {
    "date": "2026-03-16",
    "bets_placed": 12,
    "bets_won": 4,
    "bets_lost": 8,
    "gross_pnl": 45.60,
    "commission": -2.28,
    "net_pnl": 43.32,
    "roi_percent": 18.05,
    "bank_start": 1000.00,
    "bank_end": 1043.32
  }
"""
```

---

## 7. Daily Report (Email)

HTML email containing:

### Header
- Date
- Bets placed / won / lost
- Net P&L for the day
- ROI %
- Bank balance

### Bet Details Table
| Race | Runner | Side | Stake | Price | Result | P&L |
|------|--------|------|-------|-------|--------|-----|
| 2:30 Cheltenham | Constitution Hill | BACK | £10 | 2.50 | WON | +£15.00 |
| 3:00 Kempton | Some Horse | BACK | £5 | 6.00 | LOST | -£5.00 |

### Cumulative P&L Chart
- Line chart: daily net P&L cumulative from day 1
- X-axis: date
- Y-axis: cumulative net P&L (£)
- Embedded as inline base64 PNG in the email

### Skipped Bets (optional section)
- Predictions that didn't result in bets, with the reason (no edge, market closed, guardrail hit)
- Useful for model monitoring

---

## 8. MCP Server (Claude Code Interface)

Same tools as the v1 spec, plus pipeline-aware tools:

| Tool | Purpose |
|---|---|
| `betfair_login` | Authenticate (auto on first call) |
| `betfair_account_balance` | Check funds |
| `betfair_list_events` | Browse today's meetings |
| `betfair_list_markets` | Browse markets at a venue |
| `betfair_list_runners` | Live prices for a market |
| `betfair_place_bet` | Manual bet with confirmation flow |
| `betfair_confirm_bet` | Confirm a pending manual bet |
| `betfair_cancel_bet` | Cancel unmatched bet |
| `betfair_list_current_bets` | View open bets |
| `betfair_list_settled_bets` | View settled bets |
| **`pipeline_today_predictions`** | Pull today's predictions from S3 |
| **`pipeline_today_executions`** | Pull today's executed bets from S3 |
| **`pipeline_today_settlements`** | Pull today's settlements from S3 |
| **`pipeline_pnl_history`** | Pull cumulative P&L data |
| **`pipeline_explain_skip`** | Why didn't we bet on a specific runner? |
| **`pipeline_trigger_predict`** | Manually trigger prediction job |
| **`pipeline_trigger_execute`** | Manually trigger execution (with dry_run option) |

Registration in Claude Code:

```bash
claude mcp add betfair \
  --command "python" \
  --args "-m mcp_server"
```

---

## 9. S3 Data Layout

```
s3://ultra-betting/
├── predictions/
│   ├── 2026-03-16.csv
│   ├── 2026-03-17.csv
│   └── ...
├── executions/
│   ├── 2026-03-16.csv
│   └── ...
├── settlements/
│   ├── 2026-03-16.csv
│   └── ...
├── daily_pnl/
│   ├── 2026-03-16.json
│   └── ...
├── cumulative_pnl.json          # Single file, appended daily
├── reports/
│   ├── 2026-03-16.html
│   └── ...
└── model/
    ├── ashcroft_latest.pkl      # Current model artifact
    └── feature_config.json      # Feature definitions
```

---

## 10. GitHub Secrets Required

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | S3 access |
| `AWS_SECRET_ACCESS_KEY` | S3 access |
| `BETFAIR_USERNAME` | Betfair login |
| `BETFAIR_PASSWORD` | Betfair login |
| `BETFAIR_APP_KEY` | Betfair API key |
| `BETFAIR_CERT` | SSL certificate (PEM contents) |
| `BETFAIR_KEY` | SSL private key (PEM contents) |
| `SMTP_HOST` | Email sending |
| `SMTP_USER` | Email sending |
| `SMTP_PASS` | Email sending |
| `REPORT_EMAIL` | Where daily reports go |

---

## 11. Guardrails (guardrails.yaml)

```yaml
# Hard limits — enforced server-side, cannot be overridden
max_stake: 50.00               # Per bet, in GBP
max_liability: 100.00          # Per lay bet
max_daily_bets: 30             # Circuit breaker
max_daily_loss: -200.00        # Stop-loss: halt execution if daily P&L hits this
allowed_event_types: ["7"]     # Horse racing only
allowed_countries: ["GB", "IE", "AU", "NZ", "FR", "US", "AE"]

# Execution constraints
min_minutes_to_race: 5         # Don't bet within 5 mins of off
min_market_total_matched: 5000 # Skip illiquid markets

# Dry run
dry_run: false                 # Set true to simulate everything
```

---

## 12. Idempotency

The execute job runs every 30 minutes. It must be idempotent:

- Each prediction gets a deterministic ID: `{date}_{market_id}_{selection_id}`
- Before placing a bet, check if this ID exists in today's executions CSV
- If it does, skip (already handled)
- If the bet failed on a previous run, the execution log records the failure — a subsequent run will retry

---

## 13. Phasing

### Phase 1: Scaffold + Manual

- [ ] `ultra_betting` package structure
- [ ] Betfair client (auth, markets, pricing, execution, settlement)
- [ ] S3 data layer
- [ ] Guardrails
- [ ] Audit logging
- [ ] MCP server with all tools
- [ ] `rules.yaml` template (empty — you fill in)
- [ ] Rules engine skeleton (reads yaml, has clear extension points)

### Phase 2: Pipeline

- [ ] `predict.py` pipeline job (connected to Ashcroft)
- [ ] `execute.py` pipeline job
- [ ] `settle.py` pipeline job
- [ ] GitHub Actions workflows with cron
- [ ] Idempotency layer

### Phase 3: Reporting

- [ ] Daily P&L calculator
- [ ] Cumulative P&L tracker
- [ ] HTML email report with chart
- [ ] Email sending via SMTP
- [ ] Skipped-bet explanations

### Phase 4: Harden

- [ ] Retry logic for transient Betfair API failures
- [ ] Session token refresh mid-execution
- [ ] GitHub Actions failure notifications
- [ ] Guardrail: stop-loss circuit breaker
- [ ] Weekly model performance review report
