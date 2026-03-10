"""
Upload trained model artifacts to a GitHub release.

After training a model locally, use this script to publish the
model files as a GitHub release so the daily predictions workflow
can download them.

Usage:
    python scripts/upload_model.py                        # Create release with latest model
    python scripts/upload_model.py --tag v2026.03.10      # Specific tag name
    python scripts/upload_model.py --model-dir data/models

Requires:
    gh CLI (https://cli.github.com/) authenticated with your account.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")

MODEL_FILES = [
    "probability_model.lgb",
    "probability_model_meta.json",
    "blend_config.json",
    "training_summary.json",
]


def create_release(
    model_dir: str = MODEL_DIR,
    tag: str | None = None,
    repo: str | None = None,
) -> bool:
    """Create a GitHub release with model artifacts."""
    # Validate model files exist
    existing = []
    for fname in MODEL_FILES:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            existing.append(path)
        else:
            log.warning(f"Model file not found: {path}")

    if not existing:
        log.error(f"No model files found in {model_dir}")
        return False

    # Generate tag name
    if tag is None:
        tag = f"model-{date.today().isoformat()}"

    # Load training summary for release notes
    summary_path = os.path.join(model_dir, "training_summary.json")
    notes = f"Model trained on {date.today().isoformat()}"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        metrics = summary.get("final_model_metrics", {})
        data_range = summary.get("data_range", {})
        notes = (
            f"Model trained on {date.today().isoformat()}\n\n"
            f"- Data range: {data_range.get('min_date', '?')} to "
            f"{data_range.get('max_date', '?')}\n"
            f"- Total rows: {data_range.get('total_rows', '?'):,}\n"
            f"- Log-loss: {metrics.get('val_logloss_normalised', '?')}\n"
            f"- Lambda: {summary.get('optimal_lambda', '?')}\n"
            f"- Folds: {summary.get('n_folds', '?')}\n"
        )

    # Build gh release command
    cmd = [
        "gh", "release", "create", tag,
        "--title", f"Model {tag}",
        "--notes", notes,
    ]

    if repo:
        cmd.extend(["--repo", repo])

    # Add model files
    cmd.extend(existing)

    log.info(f"Creating GitHub release '{tag}' with {len(existing)} files...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error(f"Failed to create release: {result.stderr}")
        return False

    log.info(f"Release created: {result.stdout.strip()}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained model to GitHub release"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help=f"Directory containing model artifacts (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Release tag name (default: model-YYYY-MM-DD)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="GitHub repository (default: current repo)",
    )
    args = parser.parse_args()

    success = create_release(
        model_dir=args.model_dir,
        tag=args.tag,
        repo=args.repo,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
