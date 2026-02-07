"""
Moltbook Research - Extract Submolts from Posts
=================================================
The /submolts API endpoint caps at 100 results, but posts contain 1,601+
unique submolt references. This script extracts complete submolt data
from the posts dataset.
"""

import json
import csv
import sys
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)
sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_submolts(posts: list) -> list:
    """
    Extract unique submolt data from posts, enriched with post-level stats.

    Returns list of submolt dicts with:
    - id, name, display_name (from API)
    - post_count, total_upvotes, total_comments, avg_upvotes
    - earliest_post, latest_post (temporal bounds)
    - unique_authors (count of distinct authors if available)
    """
    submolt_map = {}

    for post in posts:
        submolt = post.get("submolt", {})
        if not isinstance(submolt, dict):
            continue

        name = submolt.get("name", "")
        if not name:
            continue

        if name not in submolt_map:
            submolt_map[name] = {
                "id": submolt.get("id", ""),
                "name": name,
                "display_name": submolt.get("display_name", name),
                "post_count": 0,
                "total_upvotes": 0,
                "total_downvotes": 0,
                "total_comments": 0,
                "max_upvotes": 0,
                "earliest_post": "",
                "latest_post": "",
                "authors": set(),
            }

        entry = submolt_map[name]
        entry["post_count"] += 1
        entry["total_upvotes"] += post.get("upvotes", 0)
        entry["total_downvotes"] += post.get("downvotes", 0)
        entry["total_comments"] += post.get("comment_count", 0)

        upvotes = post.get("upvotes", 0)
        if upvotes > entry["max_upvotes"]:
            entry["max_upvotes"] = upvotes

        created = post.get("created_at", "")
        if created:
            if not entry["earliest_post"] or created < entry["earliest_post"]:
                entry["earliest_post"] = created
            if not entry["latest_post"] or created > entry["latest_post"]:
                entry["latest_post"] = created

        author = post.get("author")
        if isinstance(author, dict) and author.get("name"):
            entry["authors"].add(author["name"])
        elif isinstance(author, str) and author:
            entry["authors"].add(author)

    # Convert sets to counts for serialization
    results = []
    for entry in submolt_map.values():
        entry["unique_authors"] = len(entry["authors"])
        entry["avg_upvotes"] = (
            round(entry["total_upvotes"] / entry["post_count"], 2)
            if entry["post_count"] > 0 else 0
        )
        del entry["authors"]
        results.append(entry)

    results.sort(key=lambda x: x["post_count"], reverse=True)
    return results


def save_submolts_enriched(submolts: list, timestamp: str):
    """Save enriched submolt data."""
    raw_path = RAW_DIR / f"submolts_from_posts_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(submolts, f, ensure_ascii=False, indent=2)
    logger.info("Saved JSON: %s (%d submolts)", raw_path, len(submolts))

    csv_path = PROCESSED_DIR / f"submolts_enriched_{timestamp}.csv"
    if submolts:
        fieldnames = list(submolts[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(submolts)
        logger.info("Saved CSV: %s (%d rows)", csv_path, len(submolts))

    return raw_path, csv_path


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Load posts
    posts_files = sorted(RAW_DIR.glob("posts_*.json"))
    if not posts_files:
        logger.error("No posts JSON files found.")
        sys.exit(1)

    latest = posts_files[-1]
    logger.info("Loading posts from: %s", latest)
    with open(latest, "r", encoding="utf-8") as f:
        posts = json.load(f)
    logger.info("Loaded %d posts", len(posts))

    # Also merge HuggingFace posts if available (they have author data)
    hf_path = RAW_DIR / "hf_posts.csv"
    if hf_path.exists():
        logger.info("Loading HuggingFace posts for author enrichment...")
        hf_author_map = {}
        with open(hf_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id") and row.get("author"):
                    hf_author_map[row["id"]] = row["author"]

        # Enrich API posts with HF author data
        enriched = 0
        for post in posts:
            pid = post.get("id", "")
            if pid in hf_author_map and post.get("author") is None:
                post["author"] = hf_author_map[pid]
                enriched += 1
        logger.info("Enriched %d posts with HF author data", enriched)

    # Extract submolts
    submolts = extract_submolts(posts)
    logger.info("Extracted %d unique submolts", len(submolts))

    # Print top 20
    logger.info("\nTop 20 submolts by post count:")
    for s in submolts[:20]:
        logger.info(
            "  %-25s posts=%5d upvotes=%7d comments=%5d authors=%d",
            s["name"], s["post_count"], s["total_upvotes"],
            s["total_comments"], s["unique_authors"]
        )

    # Save
    raw_path, csv_path = save_submolts_enriched(submolts, timestamp)

    # Summary stats
    total_posts = sum(s["post_count"] for s in submolts)
    active_submolts = sum(1 for s in submolts if s["post_count"] >= 5)
    logger.info("\nSummary:")
    logger.info("  Total submolts: %d", len(submolts))
    logger.info("  Active (5+ posts): %d", active_submolts)
    logger.info("  Total posts covered: %d", total_posts)
    logger.info("  Submolts with 1 post: %d",
                sum(1 for s in submolts if s["post_count"] == 1))


if __name__ == "__main__":
    main()
