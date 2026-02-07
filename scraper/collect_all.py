"""
Moltbook Research Scraper - Master Collection Script
======================================================
Orchestrates full data collection: posts -> comments -> submolts.
Designed for both initial full scrape and periodic updates.

Usage:
    python3 collect_all.py              # Full collection
    python3 collect_all.py --posts-only # Just posts
    python3 collect_all.py --quick      # Posts + submolts (skip comments)
"""

import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR, SNAPSHOTS_DIR
from api_client import MoltbookClient
from collect_posts import collect_all_posts, save_posts
from collect_submolts import collect_all_submolts, save_submolts
from collect_comments import collect_comments_for_posts, save_comments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def create_snapshot_summary(
    timestamp: str,
    posts: list,
    submolts: list,
    comments: list,
    client: MoltbookClient
):
    """Create a summary snapshot of the collection run."""
    summary = {
        "timestamp": timestamp,
        "collection_time_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "posts": len(posts),
            "submolts": len(submolts),
            "comments": len(comments),
        },
        "post_stats": {},
        "submolt_stats": {},
        "api_stats": client.stats,
    }

    if posts:
        scores = [p.get("upvotes", 0) for p in posts]
        comment_counts = [p.get("comment_count", 0) for p in posts]
        summary["post_stats"] = {
            "total_upvotes": sum(scores),
            "max_upvotes": max(scores),
            "avg_upvotes": sum(scores) / len(scores),
            "total_comments_referenced": sum(comment_counts),
            "posts_with_comments": sum(1 for c in comment_counts if c > 0),
            "earliest_post": min(
                (p.get("created_at", "") for p in posts), default=""
            ),
            "latest_post": max(
                (p.get("created_at", "") for p in posts), default=""
            ),
        }

        # Top submolts by post count
        submolt_counts = {}
        for p in posts:
            s = p.get("submolt", {})
            name = s.get("name", "unknown") if isinstance(s, dict) else str(s)
            submolt_counts[name] = submolt_counts.get(name, 0) + 1
        summary["post_stats"]["top_submolts_by_posts"] = dict(
            sorted(submolt_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        )

    if submolts:
        subs = [s.get("subscriber_count", 0) for s in submolts]
        summary["submolt_stats"] = {
            "total_subscribers_sum": sum(subs),
            "max_subscribers": max(subs),
            "avg_subscribers": sum(subs) / len(subs),
            "submolts_with_subscribers": sum(1 for s in subs if s > 0),
        }

    snapshot_path = SNAPSHOTS_DIR / f"snapshot_{timestamp}.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Snapshot saved: %s", snapshot_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Moltbook Research Data Collector")
    parser.add_argument("--posts-only", action="store_true", help="Only collect posts")
    parser.add_argument("--quick", action="store_true", help="Posts + submolts, skip comments")
    parser.add_argument("--comments-only", action="store_true", help="Only collect comments (requires existing posts)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    client = MoltbookClient()

    logger.info("=" * 60)
    logger.info("Moltbook Research Data Collector")
    logger.info("Timestamp: %s", timestamp)
    logger.info("Mode: %s", "posts-only" if args.posts_only else "quick" if args.quick else "comments-only" if args.comments_only else "full")
    logger.info("=" * 60)

    posts = []
    submolts = []
    comments = []

    # Step 1: Collect posts
    if not args.comments_only:
        logger.info("\n--- STEP 1: Collecting Posts ---")
        posts = collect_all_posts(client, sort="new")
        if posts:
            save_posts(posts, timestamp)
            logger.info("Posts collected: %d", len(posts))
        else:
            logger.warning("No posts collected.")

    if args.posts_only:
        create_snapshot_summary(timestamp, posts, [], [], client)
        return

    # Step 2: Collect submolts
    if not args.comments_only:
        logger.info("\n--- STEP 2: Collecting Submolts ---")
        submolts = collect_all_submolts(client)
        if submolts:
            save_submolts(submolts, timestamp)
            logger.info("Submolts collected: %d", len(submolts))
        else:
            logger.warning("No submolts collected.")

    if args.quick:
        create_snapshot_summary(timestamp, posts, submolts, [], client)
        return

    # Step 3: Collect comments
    logger.info("\n--- STEP 3: Collecting Comments ---")

    if args.comments_only:
        # Load posts from most recent file
        posts_files = sorted(RAW_DIR.glob("posts_*.json"))
        if not posts_files:
            logger.error("No posts files found. Run without --comments-only first.")
            sys.exit(1)
        with open(posts_files[-1], "r", encoding="utf-8") as f:
            posts = json.load(f)

    # Only fetch comments for posts that have them
    posts_with_comments = [
        p["id"] for p in posts
        if p.get("comment_count", 0) > 0
    ]
    logger.info(
        "Fetching comments for %d posts (out of %d with comments > 0)",
        len(posts_with_comments), len(posts_with_comments)
    )

    if posts_with_comments:
        comments = collect_comments_for_posts(client, posts_with_comments)
        if comments:
            save_comments(comments, timestamp)
            logger.info("Comments collected: %d", len(comments))

    # Create summary snapshot
    summary = create_snapshot_summary(timestamp, posts, submolts, comments, client)

    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("  Posts: %d", summary["counts"]["posts"])
    logger.info("  Submolts: %d", summary["counts"]["submolts"])
    logger.info("  Comments: %d", summary["counts"]["comments"])
    logger.info("  API requests: %d", client.stats["total_requests"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
