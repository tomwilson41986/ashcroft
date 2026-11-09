"""One writer of horse_racing.db at a time: every workflow that uploads the database shares one concurrency group.

The database is one SQLite file in S3; a job downloads it, changes it and uploads the whole file, so two that
overlap lose the first one's rows. The group makes a run wait for the one in front."""
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
UPLOADS = ("--upload-db", "backup_to_s3.py")        # how a workflow writes horse_racing.db back to S3
GROUP = "horse-racing-db"


def _writers():
    return sorted(p for p in WORKFLOWS.glob("*.yml") if any(u in p.read_text() for u in UPLOADS))


def test_the_known_writers_are_found():
    names = {p.name for p in _writers()}
    assert {"daily-results.yml", "backfill-results.yml", "blandford-sync.yml", "betfair-prices-import.yml",
            "blandford-sectionals.yml"} <= names


def test_every_workflow_that_uploads_the_database_waits_its_turn():
    for p in _writers():
        spec = yaml.safe_load(p.read_text())
        c = spec.get("concurrency") or {}
        assert c.get("group") == GROUP, f"{p.name} uploads horse_racing.db outside the {GROUP} group"
        assert c.get("cancel-in-progress") is False, f"{p.name} must wait, not cancel the writer in front"
