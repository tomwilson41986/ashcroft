"""
Backup horse_racing.db to S3 bucket 'horseracingresults'.

Usage:
    python backup_to_s3.py              # Upload current DB
    python backup_to_s3.py --versioned  # Upload with date-stamped copy

Requires AWS credentials configured via:
    - ~/.aws/credentials
    - or environment variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
"""

import logging
import os
import sys
from datetime import date

import boto3
from botocore.exceptions import ClientError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
BUCKET = "horseracingresults"
S3_KEY = "horse_racing.db"
REGION = "us-east-1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def upload_db(versioned: bool = False):
    """Upload horse_racing.db to S3."""
    if not os.path.exists(DB_PATH):
        log.error(f"Database not found: {DB_PATH}")
        sys.exit(1)

    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    log.info(f"Uploading {DB_PATH} ({size_mb:.1f} MB) to s3://{BUCKET}/{S3_KEY}")

    s3 = boto3.client("s3", region_name=REGION)

    try:
        s3.upload_file(DB_PATH, BUCKET, S3_KEY)
        log.info(f"Uploaded to s3://{BUCKET}/{S3_KEY}")

        if versioned:
            dated_key = f"backups/horse_racing_{date.today().isoformat()}.db"
            s3.upload_file(DB_PATH, BUCKET, dated_key)
            log.info(f"Versioned copy: s3://{BUCKET}/{dated_key}")

    except ClientError as e:
        log.error(f"S3 upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    versioned = "--versioned" in sys.argv
    upload_db(versioned=versioned)
