"""The one greyhound trial count (the owner, 20 Nov 2026: the discovery round, weakness 5; reports/
greyhound_discovery_1120_map.md). Every deflated Sharpe on a greyhound rule is deflated by every greyhound rule the
ledger holds (desk/trials.py ``count_rules``: a line's ``n_rules`` where it carries one, else one a trial line), never
by a count typed into the study. A study passes its own count as the floor, so the figure only ever rises.

    python -m research.greyhound.trial_count
"""

from __future__ import annotations


def greyhound_trials(floor: int = 0, ledger=None) -> int:
    from desk.trials import LEDGER, count_rules, parse
    lines, _ = parse((ledger or LEDGER).read_text())
    return max(int(floor), count_rules(lines, "greyhounds"))


if __name__ == "__main__":
    print(greyhound_trials())
