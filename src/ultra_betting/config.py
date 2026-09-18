"""Configuration loading for Ultra Betting."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Project root (where this repo is checked out)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"

# S3 bucket for ultra-betting pipeline data
S3_BUCKET = os.getenv("ULTRA_BETTING_S3_BUCKET", "ashcroft")
S3_DB_BUCKET = os.getenv("S3_BUCKET", "horseracingresults")
S3_DB_KEY = os.getenv("S3_DB_KEY", "horse_racing.db")

# Database
DB_PATH = PROJECT_ROOT / "horse_racing.db"

def _env(name: str, default: str = "") -> str:
    """An environment variable, treating empty as absent.

    A workflow that writes `VAR: ${{ secrets.MISSING }}` sets the variable to an
    empty string rather than leaving it unset, so `os.getenv(name, default)`
    returns "" and the default never applies. That is not a hypothetical: it
    turned an unset SMTP_PORT into `int("")` here, at module scope, where it
    would take down every importer of this module -- and it silently defeated
    the SMTP_USERNAME fallback below for six months.
    """
    value = os.getenv(name)
    return value if value not in (None, "") else default


# Email
REPORT_EMAIL = _env("REPORT_EMAIL", "racingsquared@gmail.com")
SMTP_HOST = _env("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_env("SMTP_PORT", "587"))
SMTP_USER = _env("SMTP_USER") or _env("SMTP_USERNAME")
SMTP_PASS = _env("SMTP_PASS") or _env("SMTP_PASSWORD")


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_guardrails() -> dict:
    """Load guardrails configuration."""
    return load_yaml(CONFIG_DIR / "guardrails.yaml")


def load_rules() -> dict:
    """Load betting rules configuration."""
    return load_yaml(CONFIG_DIR / "rules.yaml")
