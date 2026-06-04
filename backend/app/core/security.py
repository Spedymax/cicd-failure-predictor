"""HMAC-SHA256 verification for GitHub webhook signatures.

GitHub signs the raw request body with a shared secret and sends the
signature in the ``X-Hub-Signature-256`` header. We must reject any
request whose signature does not match — otherwise an attacker could
forge ``push`` events and pollute the prediction database.
"""

from __future__ import annotations

import hashlib
import hmac

SIGNATURE_HEADER = "X-Hub-Signature-256"
SIGNATURE_PREFIX = "sha256="


def compute_signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{SIGNATURE_PREFIX}{digest}"


def verify_signature(secret: str, body: bytes, header_value: str | None) -> bool:
    if not header_value:
        return False
    expected = compute_signature(secret, body)
    return hmac.compare_digest(expected, header_value)
