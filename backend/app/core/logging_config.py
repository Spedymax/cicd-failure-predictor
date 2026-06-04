"""structlog configuration with email masking (NFR-11, audit hygiene).

The mask processor scans every log event value (recursively) and replaces
email addresses with ``f***@host.tld`` so personally identifiable
information never lands in stdout / log aggregators in cleartext.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import structlog

_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def mask_email(text: str) -> str:
    return _EMAIL_RE.sub(r"\1***\2", text)


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        return mask_email(value)
    if isinstance(value, dict):
        return {k: _mask_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_mask_value(v) for v in value)
    return value


def email_masking_processor(_logger, _method_name, event_dict: dict) -> dict:
    return {k: _mask_value(v) for k, v in event_dict.items()}


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            email_masking_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
