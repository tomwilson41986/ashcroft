"""Structured JSON audit logging."""

import json
import logging
from datetime import datetime

log = logging.getLogger("ultra_betting.audit")


def audit_event(event_type: str, **kwargs) -> None:
    """Log a structured audit event."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        **kwargs,
    }
    log.info(json.dumps(entry, default=str))
