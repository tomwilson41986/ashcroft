"""Common pipeline setup — logging, S3, auth."""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local")


def setup_logging(name: str) -> logging.Logger:
    """Configure structured logging for pipeline jobs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(name)


def ensure_database() -> Path:
    """Ensure the horse_racing.db is available, downloading from S3 if needed."""
    from ultra_betting.config import DB_PATH, S3_DB_BUCKET, S3_DB_KEY

    if DB_PATH.exists():
        return DB_PATH

    log = logging.getLogger(__name__)
    log.info("Database not found locally, downloading from S3...")

    import boto3
    s3 = boto3.client("s3")
    s3.download_file(S3_DB_BUCKET, S3_DB_KEY, str(DB_PATH))
    log.info(f"Downloaded {S3_DB_KEY} from s3://{S3_DB_BUCKET}")
    return DB_PATH
