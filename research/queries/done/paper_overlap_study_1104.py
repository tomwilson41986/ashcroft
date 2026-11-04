"""The overlap study (the owner, 4 Nov 2026: "We can back the horses we've already backed by the live trader ... we
can just back them again if the steamer model still thinks the price is relevant").

30 Sep - 2 Nov 2026, the recorded books: each variant alone on the book after the live trader's fills, win and place,
backed at 10:00 UK (never earlier) and held to the result (research/paper/morning_edges.py overlap_study). The
variants: the quick return and the steamer model on the horses the live trader did not back and on those it did; and
the steamer model re-checked at 10:00 (back only while the predicted shortening exceeds what has already happened
since 08:00). For the live trader's horses: its backs before 10:00 and after (the money a paper back competes with).
Prints only; writes nothing to S3.
"""
import sys

sys.path.insert(0, ".")
from research.paper import morning_edges as ME  # noqa: E402

ME.main(["--overlap-study", "2026-09-30..2026-11-02"])
