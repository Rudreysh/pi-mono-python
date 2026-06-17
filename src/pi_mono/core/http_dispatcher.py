import math
from typing import Any

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000


def parse_http_idle_timeout_ms(value: Any) -> int | None:
    """Parse HTTP idle timeout value into milliseconds, returning None if invalid."""
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() == "disabled":
            return 0
        if len(trimmed) == 0:
            return None
        try:
            return parse_http_idle_timeout_ms(float(trimmed))
        except ValueError:
            return None

    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float)):
        return None

    if not math.isfinite(value) or value < 0:
        return None

    return int(value)
