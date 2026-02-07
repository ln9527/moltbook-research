"""
Moltbook Research Scraper - Parallel Owner Data Collector
==========================================================
Parallel version using 3 concurrent workers to speed up collection.
Uses ThreadPoolExecutor with shared rate limiter.

Resumes from existing checkpoint.
"""

import json
import csv
import sys
import logging
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR, BASE_URL, REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY
from api_client import MoltbookClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = RAW_DIR / "checkpoints"
CHECKPOINT_INTERVAL = 100
NUM_WORKERS = 3
RATE_LIMIT_REQUESTS = 90  # Stay under 100/min limit
RATE_LIMIT_WINDOW = 60


class SharedRateLimiter:
    """Thread-safe rate limiter for parallel requests."""

    def __init__(self, max_requests: int, window_seconds: float):
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests = []
        self._lock = threading.Lock()

    def wait(self):
        """Wait until a request slot is available."""
        while True:
            with self._lock:
                now = time.time()
                # Remove old requests outside window
                self._requests = [t for t in self._requests if now - t < self._window]

                if len(self._requests) < self._max_requests:
                    self._requests.append(now)
                    return

            # Wait a bit before trying again
            time.sleep(0.1)


class ParallelMoltbookClient:
    """Thread-safe client with shared rate limiter."""

    def __init__(self, rate_limiter: SharedRateLimiter):
        self._rate_limiter = rate_limiter
        self._lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0

    def get(self, endpoint: str) -> Optional[dict]:
        """Make a rate-limited GET request."""
        import urllib.request
        import urllib.error

        url = f"{BASE_URL}{endpoint}"

        for attempt in range(MAX_RETRIES):
            self._rate_limiter.wait()

            with self._lock:
                self._request_count += 1

            try:
                headers = {
                    "Accept": "application/json",
                    "User-Agent": "MoltbookResearchScraper/1.0 (Academic Research)"
                }
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data

            except urllib.error.HTTPError as e:
                with self._lock:
                    self._error_count += 1
                if e.code == 429:
                    wait = RETRY_DELAY * (attempt + 1) * 2
                    time.sleep(wait)
                elif e.code in (400, 401, 403, 404, 405):
                    return None
                else:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)

            except Exception as e:
                with self._lock:
                    self._error_count += 1
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        return None

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_requests": self._request_count,
                "errors": self._error_count,
            }


def extract_owner_data(post_data: dict, author_name: str) -> Optional[dict]:
    """Extract owner and agent metadata from single-post API response."""
    if not post_data or not post_data.get("post"):
        return None

    post = post_data["post"]
    author = post.get("author", {})
    owner = author.get("owner", {}) or {}

    return {
        "agent_name": author_name,
        "agent_id": author.get("id", ""),
        "karma": author.get("karma", 0),
        "follower_count": author.get("follower_count", 0),
        "following_count": author.get("following_count", 0),
        "description": author.get("description", ""),
        "owner_x_handle": owner.get("x_handle", ""),
        "owner_x_name": owner.get("x_name", ""),
        "owner_x_bio": owner.get("x_bio", ""),
        "owner_x_follower_count": owner.get("x_follower_count", 0),
        "owner_x_verified": owner.get("x_verified", False),
        "has_owner": bool(owner.get("x_handle")),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_owner(client: ParallelMoltbookClient, agent_name: str, post_id: str) -> Optional[dict]:
    """Fetch owner data for a single agent."""
    data = client.get(f"/posts/{post_id}")
    if data:
        return extract_owner_data(data, agent_name)
    return None


def get_unique_agents_with_sample_posts(posts_json_path: Path) -> dict:
    """Load posts and extract one sample post_id per unique author."""
    logger.info("Loading posts from %s...", posts_json_path)
    with open(posts_json_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    agents = {}
    for post in posts:
        author = post.get("author")
        if not author:
            continue

        if isinstance(author, dict):
            author_name = author.get("name", "")
        else:
            author_name = author

        if not author_name or author_name in agents:
            continue

        post_id = post.get("id")
        if post_id:
            agents[author_name] = post_id

    logger.info("Found %d unique agents", len(agents))
    return agents


def load_checkpoint() -> tuple:
    """Load the most recent checkpoint."""
    checkpoints = sorted(CHECKPOINT_DIR.glob("owners_checkpoint_*.json"))
    if not checkpoints:
        return [], set()

    cp_path = checkpoints[-1]
    logger.info("Loading checkpoint: %s", cp_path)

    with open(cp_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    owners = data.get("owners", [])
    processed = set(data.get("processed_agents", []))
    logger.info("Loaded: %d owners, %d processed", len(owners), len(processed))
    return owners, processed


def save_checkpoint(all_owners: list, processed_agents: set, timestamp: str):
    """Save current progress."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cp_path = CHECKPOINT_DIR / f"owners_checkpoint_{timestamp}.json"

    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump({
            "owners": all_owners,
            "processed_agents": list(processed_agents),
            "owner_count": len(all_owners),
            "agents_processed": len(processed_agents),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, f, ensure_ascii=False)

    logger.info("Checkpoint: %d owners saved", len(all_owners))


def save_final(owners: list):
    """Save final results."""
    raw_path = RAW_DIR / "owners_master.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(owners, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", raw_path)

    csv_path = PROCESSED_DIR / "owners_master.csv"
    if owners:
        fieldnames = list(owners[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(owners)
        logger.info("Saved: %s", csv_path)


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("Parallel Owner Collector (%d workers)", NUM_WORKERS)
    logger.info("Rate limit: %d requests/%d seconds", RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
    logger.info("=" * 60)

    # Load agents
    posts_master = RAW_DIR / "posts_master.json"
    agents = get_unique_agents_with_sample_posts(posts_master)

    # Load checkpoint
    all_owners, processed_agents = load_checkpoint()

    # Filter remaining
    remaining = [(name, post_id) for name, post_id in agents.items()
                 if name not in processed_agents]

    logger.info("Remaining: %d agents to process", len(remaining))

    if not remaining:
        logger.info("All agents already processed!")
        return

    # Setup
    rate_limiter = SharedRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
    client = ParallelMoltbookClient(rate_limiter)

    shutdown_requested = [False]
    results_lock = threading.Lock()
    processed_count = [len(processed_agents)]
    with_owner_count = [sum(1 for o in all_owners if o.get("has_owner"))]

    def handle_signal(signum, frame):
        logger.info("Shutdown signal received...")
        shutdown_requested[0] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    total = len(agents)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}

        for agent_name, post_id in remaining:
            if shutdown_requested[0]:
                break
            future = executor.submit(fetch_owner, client, agent_name, post_id)
            futures[future] = agent_name

        for future in as_completed(futures):
            if shutdown_requested[0]:
                break

            agent_name = futures[future]
            try:
                owner_data = future.result()
                with results_lock:
                    if owner_data:
                        all_owners.append(owner_data)
                        if owner_data["has_owner"]:
                            with_owner_count[0] += 1
                    processed_agents.add(agent_name)
                    processed_count[0] += 1

                    current = processed_count[0]
                    if current % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = (current - len(processed_agents) + len(remaining)) / elapsed * 60 if elapsed > 0 else 0
                        logger.info(
                            "Progress: %d/%d (%.1f%%), %d with owner, %.0f/min",
                            current, total, current / total * 100,
                            with_owner_count[0], rate
                        )

                    if current % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(all_owners, processed_agents, timestamp)

            except Exception as e:
                logger.error("Error for %s: %s", agent_name, e)

    # Final save
    save_checkpoint(all_owners, processed_agents, timestamp)

    if len(processed_agents) == total:
        save_final(all_owners)
        logger.info("=" * 60)
        logger.info("COMPLETE: %d owners collected", len(all_owners))
        logger.info("With owner: %d (%.1f%%)", with_owner_count[0],
                    with_owner_count[0] / len(all_owners) * 100 if all_owners else 0)
    else:
        logger.info("Interrupted at %d/%d. Resume with same command.", len(processed_agents), total)

    logger.info("API stats: %s", client.stats)


if __name__ == "__main__":
    main()
