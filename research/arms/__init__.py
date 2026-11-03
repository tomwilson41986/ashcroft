"""The strategy arms (the owner, 3 Nov 2026): six ways to earn more from the model and the recorded books, each
replayed read-only through the live trader's own code on the days the record holds. Nothing in trading/ changes.

1. re-weighting the money the rule already places (a stake floor on long prices, a bar that rises with the clock,
   fewer top-ups);
2. the lay side: the rule's mirror, laying the horses the model has too short, closed by a back at the BSP;
3. the place markets, with the model's place prices (Harville) and a closing model fitted to the place books;
4. the evening before: the same rule on tomorrow's books from 17:00 UK;
5. whole-race books (model/race_book.py) on the recorded books;
6. exits before the off (take-profit, stop-loss) against the lay at the BSP, on the trader's own fills.

    python -m research.arms.run --days 2026-10-30,2026-10-31 --arms all --jobs 2 --out out/arms
"""
