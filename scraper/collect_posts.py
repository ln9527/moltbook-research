"""
Moltbook Research Scraper - Post Collector
============================================
Collects all posts from the Moltbook API with full pagination.
Saves raw JSON and processed CSV.
"""

import json
import csv
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR, BATCH_SIZE
from api_client import MoltbookClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def collect_all_posts(client: MoltbookClient, sort: str = "new") -> list:
    """
    Paginate through all posts on Moltbook.

    Args:
        client: MoltbookClient instance
        sort: Sort order ("new", "hot", "top", "rising")

    Returns:
        List of all post dicts
    """
    all_posts = []
    offset = 0
    total_expected = None

    logger.info("Starting post collection (sort=%s, batch_size=%d)", sort, BATCH_SIZE)

    while True:
        data = client.get("/posts", params={
            "sort": sort,
            "limit": BATCH_SIZE,
            "offset": offset
        })

        if data is None:
            logger.error("Failed to fetch posts at offset %d", offset)
            break

        posts = data.get("posts", [])
        if not posts:
            logger.info("No more posts at offset %d", offset)
            break

        all_posts.extend(posts)
        has_more = data.get("has_more", False)
        next_offset = data.get("next_offset", offset + len(posts))

        logger.info(
            "Collected %d posts (total: %d, has_more: %s)",
            len(posts), len(all_posts), has_more
        )

        if not has_more:
            break

        offset = next_offset

    logger.info("Post collection complete: %d posts total", len(all_posts))
    return all_posts


def flatten_post(post: dict) -> dict:
    """Flatten a post dict for CSV output."""
    submolt = post.get("submolt") or {}
    author = post.get("author")
    if isinstance(author, dict):
        author_name = author.get("name", "")
    else:
        author_name = author or ""

    return {
        "id": post.get("id", ""),
        "title": post.get("title", ""),
        "content": post.get("content", ""),
        "url": post.get("url", ""),
        "author": author_name,
        "submolt_id": submolt.get("id", ""),
        "submolt_name": submolt.get("name", ""),
        "upvotes": post.get("upvotes", 0),
        "downvotes": post.get("downvotes", 0),
        "score": post.get("upvotes", 0) - post.get("downvotes", 0),
        "comment_count": post.get("comment_count", 0),
        "created_at": post.get("created_at", ""),
    }


def save_posts(posts: list, timestamp: str):
    """Save posts as both raw JSON and processed CSV."""
    # Raw JSON
    raw_path = RAW_DIR / f"posts_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    logger.info("Saved raw JSON: %s (%d posts)", raw_path, len(posts))

    # Processed CSV
    csv_path = PROCESSED_DIR / f"posts_{timestamp}.csv"
    if posts:
        flat = [flatten_post(p) for p in posts]
        fieldnames = list(flat[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat)
        logger.info("Saved CSV: %s (%d rows)", csv_path, len(flat))

    return raw_path, csv_path


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    client = MoltbookClient()

    logger.info("=" * 60)
    logger.info("Moltbook Post Collector")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 60)

    posts = collect_all_posts(client, sort="new")

    if posts:
        raw_path, csv_path = save_posts(posts, timestamp)
        logger.info("Collection complete.")
        logger.info("  Raw JSON: %s", raw_path)
        logger.info("  CSV: %s", csv_path)
    else:
        logger.warning("No posts collected.")

    logger.info("API stats: %s", client.stats)


if __name__ == "__main__":
    main()
