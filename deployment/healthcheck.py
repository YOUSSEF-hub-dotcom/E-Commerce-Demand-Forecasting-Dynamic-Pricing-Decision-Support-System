"""Container healthcheck probe: exit 0 when the target URL returns 2xx.

Usage:
    python deployment/healthcheck.py <url> [timeout_seconds]
"""

from __future__ import annotations

import sys
import urllib.request

DEFAULT_TIMEOUT = 5


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: healthcheck.py <url> [timeout_seconds]", file=sys.stderr)
        return 2
    url = argv[1]
    timeout = int(argv[2]) if len(argv) > 2 else DEFAULT_TIMEOUT
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 0 if 200 <= resp.status < 300 else 1
    except Exception as exc:  # noqa: BLE001 - health probes must never raise
        print(f"healthcheck {url}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))