"""Bounded retry for calls to external dependencies."""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


def call_with_retries(
    fn: Callable[[], T],
    *,
    retries: int,
    retry_on: tuple[type[BaseException], ...],
    backoff_s: float = 0.3,
) -> T:
    """Run fn, retrying up to `retries` extra times on the given transient errors."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == retries:
                raise
            log.warning(
                "transient error, retrying", extra={"attempt": attempt + 1, "err": str(exc)}
            )
            time.sleep(backoff_s * (2**attempt))
    raise AssertionError("unreachable")
