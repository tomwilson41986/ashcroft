"""
Backup horse_racing.db to S3.

Usage:
    python backup_to_s3.py              # Upload current DB
    python backup_to_s3.py --versioned  # Upload with date-stamped copy

Requires AWS credentials configured via:
    - Environment variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    - or ~/.aws/credentials

Environment variables:
    AWS_ACCESS_KEY_ID       - AWS access key
    AWS_SECRET_ACCESS_KEY   - AWS secret key
    AWS_DEFAULT_REGION      - AWS region (default: us-east-1)
    S3_BUCKET               - S3 bucket name (default: horseracingresults)
    S3_DB_KEY               - S3 object key (default: horse_racing.db)
"""

import logging
import os
import sys
from datetime import date

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
BUCKET = os.getenv("S3_BUCKET", "horseracingresults")
S3_KEY = os.getenv("S3_DB_KEY", "horse_racing.db")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def upload_db(
    db_path: str = DB_PATH,
    bucket: str = BUCKET,
    key: str = S3_KEY,
    region: str = REGION,
    versioned: bool = False,
):
    """Upload horse_racing.db to S3."""
    if not os.path.exists(db_path):
        log.error(f"Database not found: {db_path}")
        sys.exit(1)

    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    log.info(f"Uploading {db_path} ({size_mb:.1f} MB) to s3://{bucket}/{key}")

    s3 = boto3.client("s3", region_name=region)

    try:
        s3.upload_file(db_path, bucket, key)
        log.info(f"Uploaded to s3://{bucket}/{key}")

        if versioned:
            dated_key = f"backups/horse_racing_{date.today().isoformat()}.db"
            s3.upload_file(db_path, bucket, dated_key)
            log.info(f"Versioned copy: s3://{bucket}/{dated_key}")

    except ClientError as e:
        log.error(f"S3 upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    versioned = "--versioned" in sys.argv
    upload_db(versioned=versioned)
