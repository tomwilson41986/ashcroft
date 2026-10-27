"""The Professor: an automated tipster on the Ashcroft model (the owner's ask of 23 Oct 2026; the design in
reports/professor_design.md, the rule's parameters from reports/professor_rule.md).

The persona: a former horse-racing trader and betting-syndicate quant who publishes the model's selections at
10:00 UK once the morning's Betfair trades are in: two free tips a day on X and Instagram, every tip in the Telegram
members channel, a write-up a horse, nothing over 50/1, every advised price available with three or more major
bookmakers. The selection (config.json): among the horses the syndicate holds at 09:55 UK, each whose bookmaker price
clears the closing model's expected BSP by 10%, one a race, ranked by the expected win probability, at most 5 a day,
the top two free. The record is published to the advised price and to the Betfair SP (the claim the model
supports: the price move), and the syndicate's positions are disclosed. Proof first, then paid membership.

Phase 0 (this package's config and the backtest) is done; the modules of the design follow in Phase 1.
"""
