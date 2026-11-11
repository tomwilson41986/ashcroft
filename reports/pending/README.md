# racing² - changes waiting for the owner's word

Changes to the live trader are not merged without the owner's go-ahead. Until then the change is kept here as a
patch, so the live code is untouched and the work is not lost.

| Patch | What it does | Evidence | To apply |
|---|---|---|---|
| `live_trader_lost_reply_1111.patch` | A lost Betfair reply on a top-up, or on a top-up's lay at the SP, is settled only by an order the trader has not seen before (never by the horse's earlier fill or earlier lay); a horse whose lay is past its tries is not laid again because another horse in the race is. | Edge investigation findings trading_code-00, -01, -03 (11 Nov 2026), each confirmed by two independent checks; 5 new tests in the patch, each failing on the code as it stands; 347 tests pass with it across the trading, greyhound, football, morning and desk suites. | `git apply reports/pending/live_trader_lost_reply_1111.patch`, run `pytest tests/test_live_trading.py`, then delete the patch. |
