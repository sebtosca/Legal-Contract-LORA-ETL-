"""Rate-limited HTTP client for SEC EDGAR, respecting the fair-access policy."""

import os
import threading
import time

import requests

EDGAR_MAX_REQUESTS_PER_SECOND = 10


class RateLimiter:
    """Simple token-bucket limiter: at most `rate` acquisitions per second."""

    def __init__(self, rate: float):
        self._interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self._interval
            wait = start - now
        if wait > 0:
            time.sleep(wait)


class EdgarClient:
    """Thin wrapper around `requests` enforcing SEC's User-Agent requirement
    and fair-access rate limit (~10 req/sec) on every call.
    """

    def __init__(self, user_agent: str | None = None, rate: float = EDGAR_MAX_REQUESTS_PER_SECOND):
        user_agent = user_agent or os.environ.get("SEC_EDGAR_USER_AGENT")
        if not user_agent:
            raise RuntimeError(
                "SEC_EDGAR_USER_AGENT is not set. SEC requires an identifying "
                "User-Agent (e.g. 'Your Name you@example.com') on every request."
            )
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._limiter = RateLimiter(rate)

    def get(self, url: str, **kwargs) -> requests.Response:
        self._limiter.acquire()
        response = self._session.get(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response

    def get_json(self, url: str, **kwargs) -> dict:
        return self.get(url, **kwargs).json()

    def get_text(self, url: str, **kwargs) -> str:
        return self.get(url, **kwargs).text
