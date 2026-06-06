"""Thin Redis helper. Used for webhook idempotency and (future) feature cache."""

from __future__ import annotations

import logging
from functools import lru_cache

import redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True, socket_timeout=2.0)


def claim_delivery(delivery_id: str, ttl_seconds: int = 3600) -> bool:
    """Mark a webhook delivery as claimed. Returns True on first claim,
    False on duplicate. Never raises — Redis outage degrades to "not idempotent"
    rather than dropping webhooks.
    """
    if not delivery_id:
        return True
    try:
        return bool(
            get_redis().set(f"webhook:delivery:{delivery_id}", "1", nx=True, ex=ttl_seconds)
        )
    except redis.RedisError as exc:
        logger.warning("redis idempotency check failed (%s); allowing through", exc)
        return True
