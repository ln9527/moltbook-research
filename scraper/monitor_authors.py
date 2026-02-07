"""
Moltbook Author API Monitor
============================
Checks every N minutes whether the API returns author data.
When authors become available, automatically starts the backfill.

Usage:
    python3 scraper/monitor_authors.py              # Check every 20 min
    python3 scraper/monitor_authors.py --interval 10 # Check every 10 min
    python3 scraper/monitor_authors.py --check-only  # Single check, no loop
"""

import json
import sys
import time
import signal
import argparse
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

_stop = False


def _signal_handler(signum, frame):
    global _stop
    _stop = True
    logger.info("Shutdown requested")


def check_author_availability() -> dict:
    """
    Test multiple posts to see if the API returns author data.
    Returns dict with check results.
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_reachable": False,
        "posts_checked": 0,
        "posts_with_author": 0,
        "sample_author": None,
        "error": None,
    }

    try:
        # First check: list endpoint
        req = Request(
            f"{BASE_URL}/posts?sort=new&limit=5",
            headers={"Accept": "application/json", "User-Agent": "MoltbookResearchScraper/1.0"}
        )
        resp = urlopen(req, timeout=20)
        data = json.loads(resp.read())
        results["api_reachable"] = True

        posts = data.get("posts", [])
        for p in posts:
            results["posts_checked"] += 1
            author = p.get("author")
            if author is not None and author != "":
                results["posts_with_author"] += 1
                if results["sample_author"] is None:
                    results["sample_author"] = author

        # If list didn't have authors, try single-post endpoint
        if results["posts_with_author"] == 0 and posts:
            time.sleep(1)
            pid = posts[0]["id"]
            req2 = Request(
                f"{BASE_URL}/posts/{pid}",
                headers={"Accept": "application/json", "User-Agent": "MoltbookResearchScraper/1.0"}
            )
            resp2 = urlopen(req2, timeout=20)
            data2 = json.loads(resp2.read())
            results["posts_checked"] += 1
            author = data2.get("author")
            if author is not None and author != "":
                results["posts_with_author"] += 1
                results["sample_author"] = author

    except (HTTPError, URLError, TimeoutError, OSError) as e:
        results["error"] = str(e)

    return results


def run_backfill():
    """Launch the author backfill as a subprocess."""
    logger.info("Starting author backfill...")
    script = Path(__file__).parent / "update.py"
    proc = subprocess.Popen(
        [sys.executable, str(script), "--fetch-authors"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info("[backfill] %s", line)

    proc.wait()
    logger.info("Backfill exited with code %d", proc.returncode)
    return proc.returncode


def monitor(interval_minutes: int = 20, check_only: bool = False):
    """Main monitoring loop."""
    global _stop

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    check_count = 0
    log_path = Path(__file__).parent.parent / "data" / "state" / "author_monitor.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Load previous checks
    checks = []
    if log_path.exists():
        with open(log_path, "r") as f:
            checks = json.load(f)

    while not _stop:
        check_count += 1
        logger.info("=" * 50)
        logger.info("Author API check #%d", check_count)

        result = check_author_availability()

        if not result["api_reachable"]:
            logger.warning("API not reachable: %s", result.get("error", "unknown"))
        elif result["posts_with_author"] > 0:
            logger.info(
                "AUTHOR DATA AVAILABLE! %d/%d posts have authors. Sample: %s",
                result["posts_with_author"],
                result["posts_checked"],
                json.dumps(result["sample_author"]),
            )
        else:
            logger.info(
                "No author data. Checked %d posts - all null. Next check in %d min.",
                result["posts_checked"],
                interval_minutes,
            )

        # Save check log
        checks.append(result)
        with open(log_path, "w") as f:
            json.dump(checks, f, indent=2, ensure_ascii=False)

        # If authors available, start backfill
        if result["posts_with_author"] > 0:
            logger.info("Launching author backfill automatically...")
            run_backfill()
            logger.info("Backfill complete. Continuing to monitor.")

        if check_only:
            break

        # Wait interval (in 10-second increments so we can catch shutdown)
        wait_seconds = interval_minutes * 60
        waited = 0
        while waited < wait_seconds and not _stop:
            time.sleep(10)
            waited += 10

    logger.info("Monitor stopped after %d checks", check_count)


def main():
    parser = argparse.ArgumentParser(description="Monitor Moltbook API for author data availability")
    parser.add_argument("--interval", type=int, default=20,
                        help="Check interval in minutes (default: 20)")
    parser.add_argument("--check-only", action="store_true",
                        help="Single check, no loop")
    args = parser.parse_args()

    monitor(interval_minutes=args.interval, check_only=args.check_only)


if __name__ == "__main__":
    main()
