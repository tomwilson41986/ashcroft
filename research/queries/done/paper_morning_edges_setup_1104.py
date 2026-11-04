"""Set up the morning edges paper trader (the owner, 4 Nov 2026: "set this up as a paper trader model. Ensure it
doesn't bet before 10:00 so it doesnt take prices from the live trader").

Run through the research-query job because the trader's own workflow (paper-morning-edges.yml) can only be scheduled
once it is on the default branch. This run: (1) fits and freezes the CLEAN steamer model to S3 (training cut-off
31 Oct 2026, before any forward day); (2) fills the Timeform cache in S3 from the Railway backend; (3) replays every
racing day with recorded books from 30 Sep 2026 to two days ago -- the pre-registered fill check (spec section 9),
shown apart from the forward test, which starts on 5 Nov 2026. Writes only under s3://$PRED_BUCKET/paper/morning_edges/.
"""
import sys

sys.path.insert(0, ".")
from research.paper import morning_edges as ME  # noqa: E402

ME.main(["--train", "--cutoff", "2026-10-31"])
ME.main(["--days", "pending", "--backfill-from", "2026-09-30"])
