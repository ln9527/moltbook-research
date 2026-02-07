"""
Moltbook Research Scraper - API Client
=======================================
Rate-limited HTTP client for the Moltbook API with retry logic.
"""

import time
import json
import socket
import logging
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from config import (
    BASE_URL, API_KEY, REQUEST_DELAY, MAX_RETRIES,
    RETRY_DELAY, REQUEST_TIMEOUT, PROXIES
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._last_request = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()


class MoltbookClient:
    """HTTP client for the Moltbook API with rate limiting and retries."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or API_KEY
        self._rate_limiter = RateLimiter(REQUEST_DELAY)
        self._request_count = 0
        self._error_count = 0

    def _build_headers(self) -> dict:
        headers = {
            "Accept": "application/json",
            "User-Agent": "MoltbookResearchScraper/1.0 (Academic Research)"
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        Make a GET request to the Moltbook API.

        Args:
            endpoint: API endpoint path (e.g., "/posts")
            params: Query parameters dict

        Returns:
            Parsed JSON response dict, or None on failure
        """
        url = f"{BASE_URL}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params)}"

        for attempt in range(MAX_RETRIES):
            self._rate_limiter.wait()
            self._request_count += 1

            try:
                req = Request(url, headers=self._build_headers())
                with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                    if data.get("success") is False:
                        logger.warning(
                            "API error on %s: %s", endpoint, data.get("error")
                        )
                        return data

                    return data

            except HTTPError as e:
                self._error_count += 1
                if e.code == 429:
                    wait = RETRY_DELAY * (attempt + 1) * 2
                    logger.warning(
                        "Rate limited on %s, waiting %ds (attempt %d/%d)",
                        endpoint, wait, attempt + 1, MAX_RETRIES
                    )
                    time.sleep(wait)
                elif e.code in (400, 401, 403, 405):
                    logger.debug(
                        "HTTP %d on %s (not retrying)", e.code, endpoint
                    )
                    return None
                elif e.code == 404:
                    logger.debug("Not found: %s", endpoint)
                    return None
                else:
                    logger.error(
                        "HTTP %d on %s (attempt %d/%d)",
                        e.code, endpoint, attempt + 1, MAX_RETRIES
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)

            except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as e:
                self._error_count += 1
                logger.error(
                    "Network error on %s: %s (attempt %d/%d)",
                    endpoint, str(e), attempt + 1, MAX_RETRIES
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        logger.error("All retries exhausted for %s", endpoint)
        return None

    @property
    def stats(self) -> dict:
        return {
            "total_requests": self._request_count,
            "errors": self._error_count,
            "success_rate": (
                (self._request_count - self._error_count) / self._request_count
                if self._request_count > 0 else 0
            )
        }
