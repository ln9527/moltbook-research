"""
Moltbook Research Scraper - Configuration
==========================================
API configuration, rate limiting, and paths for the Moltbook data collection system.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, SNAPSHOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Configuration
# IMPORTANT: Always use www prefix to avoid 307 redirect that strips auth headers
BASE_URL = "https://www.moltbook.com/api/v1"

# API key (optional - some read endpoints work without auth, but author field is null)
# To get an API key: POST /agents/register with name and description
API_KEY = os.environ.get("MOLTBOOK_API_KEY", "")

# Rate limiting
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds
REQUEST_DELAY = RATE_LIMIT_WINDOW / RATE_LIMIT_REQUESTS  # ~0.6s between requests
BATCH_SIZE = 25  # max items per page (API default)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries
REQUEST_TIMEOUT = 30  # seconds

# Scraping configuration
POSTS_SORT_OPTIONS = ["new", "hot", "top", "rising"]
COMMENTS_SORT_OPTIONS = ["top", "new", "controversial"]

# Proxy settings (from user's environment, empty dict if not set)
_http_proxy = os.environ.get("http_proxy", "")
_https_proxy = os.environ.get("https_proxy", "")
PROXIES = {"http": _http_proxy, "https": _https_proxy} if _http_proxy else {}
