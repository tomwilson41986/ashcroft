# Pre-registration: the greyhound favourite back, closed at the SP

racing² - greyhound research. Registered 10 Nov 2026, before any of its test days exist. Paper only: nothing bets
until the owner decides.

**Why.** The owner, 10 Nov 2026: "I want to back. Either back and trade out at BFSP, or back just to win." The back
study (reports/greyhound_model.md section 39; fast-loop batch 22, run 42906178212) read 3,054 back rules: backing on
its own loses, no back-to-win rule holds, and one trade-out family survives a 10% false discovery rate on the new
model's prices: the favourite backed at the first traded book at the forecast SP plus 20%, closed at the SP (holdout
+10.50%, 90% 5.38 to 14.37, 157 trades in 7 days).

**The rule, fixed now.**
- **Price:** the live model's (greyhound/stack_model.py, `greyhound/live/predictions/<day>.csv`), through the limit
  study's forecast of the SP at the first traded book (research/greyhound/limit_study.py `forecast`: the conditional
  logit on the book and our price, fitted on every recorded day before the day).
- **Which dog:** the market's favourite at its race's first traded book (the shortest mid).
- **The order:** a back at the forecast SP x 1.20 on Betfair's ladder (rounded up), resting until the off, taken at
  once where the best back price already reaches it. Stake to win GBP100.
- **Filled:** 'strict', a later traded price beyond our limit (the queue ahead certainly cleared).
- **The exit:** closed at the SP (the back's winnings laid at the SP), settled by the result; 2% commission on each
  market's net.
- **Books:** the recorder's, re-timed, as the study read them (`limit_study.load(source='retimed')`); the live-key
  books (5 s) read beside them for the fill, not for the gate.

**The gate, fixed now.** Read on the first 20 test days from 11 Nov 2026 with at least 300 matched backs: PASS when
the return on the money at risk is above zero with its 90% interval (a bootstrap over the test days) above zero, and
the matched backs' mean CLV against the SP is above zero. FAIL otherwise. Nothing earlier counts.

**Beside it, not gated:** the model's favourite instead of the market's; the 10% margin; the same entries greened in
the last book and held to the result (the owner's 'back to win').
