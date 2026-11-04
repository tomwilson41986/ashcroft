"""The traders' dashboard (the owner, 4 Nov 2026): every trader, live and paper, one page per sport.

``dashboard.traders`` reads each trader's own ledger in S3 (read-only) and reduces it to a table of markets a day
(staked, expected value, P&L once settled); ``dashboard.page`` renders the page. Nothing here trades, and nothing
here changes a trader: the ledgers are read the way each trader's own settle step reads them.

    python -m dashboard --out out/dashboard            # build traders.json + index.html locally
    python -m dashboard --out out/dashboard --upload   # ... and copy both to s3://$PRED_BUCKET/dashboard/
"""
