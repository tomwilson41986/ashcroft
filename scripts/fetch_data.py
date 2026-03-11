"""
Fetch historic data from S3 and model artifacts from GitHub releases.

Downloads the SQLite database and trained model files needed to run
the daily predictions pipeline in CI/CD or on a fresh machine.

Usage:
    python scripts/fetch_data.py                          # Fetch both DB and model
    python scripts/fetch_data.py --db-only                # Fetch only the database
    python scripts/fetch_data.py --model-only             # Fetch only the model
    python scripts/fetch_data.py --s3-bucket my-bucket    # Override S3 bucket

Environment variables:
    AWS_ACCESS_KEY_ID       - AWS credentials (or use IAM role)
    AWS_SECRET_ACCESS_KEY   - AWS credentials
    AWS_DEFAULT_REGION      - AWS region (default: us-east-1)
    S3_BUCKET               - S3 bucket name (default: horseracingresults)
    S3_DB_KEY               - S3 object key for the database (default: horse_racing.db)
    GITHUB_TOKEN            - GitHub token for downloading release assets
    GITHUB_REPO             - GitHub repo (default: tomwilson41986/ashcroft)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(PROJECT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")

DEFAULT_BUCKET = "horseracingresults"
DEFAULT_DB_KEY = "horse_racing.db"
DEFAULT_REGION = "us-east-1"
DEFAULT_REPO = "tomwilson41986/ashcroft"


def fetch_db_from_s3(
    bucket: str = DEFAULT_BUCKET,
    key: str = DEFAULT_DB_KEY,
    dest: str = DB_PATH,
    region: str = DEFAULT_REGION,
) -> bool:
    """Download the SQLite database from S3."""
    try:
        import boto3
    except ImportError:
        log.error("boto3 not installed. Run: pip install boto3")
        return False

    log.info(f"Downloading s3://{bucket}/{key} -> {dest}")

    try:
        s3 = boto3.client("s3", region_name=region)
        s3.download_file(bucket, key, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        log.info(f"Downloaded database ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        log.error(f"Failed to download from S3: {e}")
        return False


def upload_db_to_s3(
    db_path: str = DB_PATH,
    bucket: str = DEFAULT_BUCKET,
    key: str = DEFAULT_DB_KEY,
    region: str = DEFAULT_REGION,
) -> bool:
    """Upload the SQLite database to S3."""
    try:
        import boto3
    except ImportError:
        log.error("boto3 not installed. Run: pip install boto3")
        return False

    if not os.path.exists(db_path):
        log.error(f"Database not found: {db_path}")
        return False

    size_mb = os.path.getsize(db_path) / (1024 * 1024)
    log.info(f"Uploading {db_path} ({size_mb:.1f} MB) -> s3://{bucket}/{key}")

    try:
        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(db_path, bucket, key)
        log.info("Upload complete")
        return True
    except Exception as e:
        log.error(f"Failed to upload to S3: {e}")
        return False


def fetch_model_from_github(
    repo: str = DEFAULT_REPO,
    dest_dir: str = MODEL_DIR,
    tag: str = "latest",
) -> bool:
    """Download trained model artifacts from the latest GitHub release.

    Expects the release to contain:
      - probability_model.lgb
      - probability_model_meta.json
      - blend_config.json
    """
    token = os.getenv("GITHUB_TOKEN", "")

    log.info(f"Fetching model from GitHub release ({repo}, tag={tag})...")

    try:
        # Use gh CLI if available (handles auth automatically)
        result = subprocess.run(
            ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return _download_via_gh(repo, tag, dest_dir, result.stdout)

        # Fallback to API
        return _download_via_api(repo, tag, dest_dir, token)

    except FileNotFoundError:
        # gh CLI not installed, use API
        return _download_via_api(repo, tag, dest_dir, token)


def _download_via_gh(
    repo: str, tag: str, dest_dir: str, assets_json: str
) -> bool:
    """Download release assets using the gh CLI."""
    os.makedirs(dest_dir, exist_ok=True)

    assets = json.loads(assets_json).get("assets", [])
    model_files = [
        "probability_model.lgb",
        "probability_model_meta.json",
        "blend_config.json",
        "training_summary.json",
    ]

    downloaded = 0
    for asset in assets:
        name = asset.get("name", "")
        if name in model_files:
            log.info(f"  Downloading {name}...")
            try:
                subprocess.run(
                    [
                        "gh", "release", "download", tag,
                        "--repo", repo,
                        "--pattern", name,
                        "--dir", dest_dir,
                        "--clobber",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                downloaded += 1
            except subprocess.CalledProcessError as e:
                log.error(f"  Failed to download {name}: {e.stderr}")
                return False

    if downloaded == 0:
        log.warning("No model files found in release assets")
        return False

    log.info(f"Downloaded {downloaded} model files to {dest_dir}")
    return True


def _download_via_api(
    repo: str, tag: str, dest_dir: str, token: str
) -> bool:
    """Download release assets via the GitHub API."""
    os.makedirs(dest_dir, exist_ok=True)

    if tag == "latest":
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    else:
        api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            release = json.loads(resp.read().decode())
    except Exception as e:
        log.error(f"Failed to fetch release info: {e}")
        return False

    model_files = [
        "probability_model.lgb",
        "probability_model_meta.json",
        "blend_config.json",
        "training_summary.json",
    ]

    downloaded = 0
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if name in model_files:
            download_url = asset["browser_download_url"]
            dest = os.path.join(dest_dir, name)

            log.info(f"  Downloading {name}...")
            dl_headers = {"Accept": "application/octet-stream"}
            if token:
                dl_headers["Authorization"] = f"Bearer {token}"

            dl_req = urllib.request.Request(download_url, headers=dl_headers)
            try:
                with urllib.request.urlopen(dl_req) as resp:
                    with open(dest, "wb") as f:
                        f.write(resp.read())
                downloaded += 1
            except Exception as e:
                log.error(f"  Failed to download {name}: {e}")
                return False

    if downloaded == 0:
        log.warning("No model files found in release assets")
        return False

    log.info(f"Downloaded {downloaded} model files to {dest_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historic data from S3 and model from GitHub releases"
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Only fetch the database from S3",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only fetch the model from GitHub releases",
    )
    parser.add_argument(
        "--upload-db",
        action="store_true",
        help="Upload local database to S3 (instead of downloading)",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.getenv("S3_BUCKET", DEFAULT_BUCKET),
        help=f"S3 bucket name (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--s3-key",
        type=str,
        default=os.getenv("S3_DB_KEY", DEFAULT_DB_KEY),
        help=f"S3 object key for the database (default: {DEFAULT_DB_KEY})",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_DEFAULT_REGION", DEFAULT_REGION),
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        default=os.getenv("GITHUB_REPO", DEFAULT_REPO),
        help=f"GitHub repo for model releases (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--release-tag",
        type=str,
        default="latest",
        help="GitHub release tag to download (default: latest)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DB_PATH,
        help=f"Local database path (default: {DB_PATH})",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help=f"Local model directory (default: {MODEL_DIR})",
    )
    args = parser.parse_args()

    if args.upload_db:
        success = upload_db_to_s3(
            db_path=args.db_path,
            bucket=args.s3_bucket,
            key=args.s3_key,
            region=args.region,
        )
        sys.exit(0 if success else 1)

    fetch_db = not args.model_only
    fetch_model = not args.db_only

    ok = True

    if fetch_db:
        if not fetch_db_from_s3(
            bucket=args.s3_bucket,
            key=args.s3_key,
            dest=args.db_path,
            region=args.region,
        ):
            ok = False

    if fetch_model:
        if not fetch_model_from_github(
            repo=args.github_repo,
            dest_dir=args.model_dir,
            tag=args.release_tag,
        ):
            ok = False

    if ok:
        log.info("All data fetched successfully")
    else:
        log.error("Some downloads failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
