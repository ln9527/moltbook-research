"""
Moltbook Research Scraper - Comment Collector
===============================================
Collects all comments for each post by fetching individual post pages.
The single-post endpoint (/posts/:id) returns comments inline.

Supports checkpoint/resume for long-running collection.
"""

import json
import csv
import sys
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR
from api_client import MoltbookClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = RAW_DIR / "checkpoints"
CHECKPOINT_INTERVAL = 500  # Save checkpoint every N posts


def flatten_comments(comments: list, post_id: str, depth: int = 0) -> list:
    """
    Recursively flatten nested comment tree into a flat list.

    Each comment gains: post_id, depth, has_replies fields.
    """
    flat = []
    for comment in comments:
        author = comment.get("author")
        if isinstance(author, dict):
            author_name = author.get("name", "")
        else:
            author_name = author or ""

        flat_comment = {
            "comment_id": comment.get("id", ""),
            "post_id": post_id,
            "parent_id": comment.get("parent_id", ""),
            "author": author_name,
            "content": comment.get("content", ""),
            "upvotes": comment.get("upvotes", 0),
            "downvotes": comment.get("downvotes", 0),
            "created_at": comment.get("created_at", ""),
            "depth": depth,
            "has_replies": len(comment.get("replies", [])) > 0,
            "reply_count": len(comment.get("replies", [])),
        }
        flat.append(flat_comment)

        replies = comment.get("replies", [])
        if replies:
            flat.extend(flatten_comments(replies, post_id, depth + 1))

    return flat


def save_checkpoint(all_comments: list, processed_ids: set, timestamp: str):
    """Save current progress to checkpoint files."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cp_path = CHECKPOINT_DIR / f"comments_checkpoint_{timestamp}.json"
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump({
            "comments": all_comments,
            "processed_post_ids": list(processed_ids),
            "comment_count": len(all_comments),
            "posts_processed": len(processed_ids),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, f, ensure_ascii=False)
    logger.info(
        "Checkpoint saved: %d comments, %d posts processed -> %s",
        len(all_comments), len(processed_ids), cp_path
    )
    return cp_path


def load_checkpoint(timestamp: str) -> tuple:
    """Load from checkpoint if available. Returns (comments, processed_ids)."""
    cp_path = CHECKPOINT_DIR / f"comments_checkpoint_{timestamp}.json"
    if not cp_path.exists():
        # Try to find any recent checkpoint
        checkpoints = sorted(CHECKPOINT_DIR.glob("comments_checkpoint_*.json"))
        if checkpoints:
            cp_path = checkpoints[-1]
            logger.info("Found checkpoint: %s", cp_path)
        else:
            return [], set()

    with open(cp_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    comments = data.get("comments", [])
    processed = set(data.get("processed_post_ids", []))
    logger.info(
        "Loaded checkpoint: %d comments, %d posts already processed",
        len(comments), len(processed)
    )
    return comments, processed


def collect_comments_for_posts(
    client: MoltbookClient,
    post_ids: list,
    timestamp: str,
    resume: bool = True,
) -> list:
    """
    Collect comments for a list of post IDs by fetching each post individually.

    Supports checkpoint/resume for fault tolerance.
    """
    all_comments = []
    processed_ids = set()

    # Try to resume from checkpoint
    if resume:
        all_comments, processed_ids = load_checkpoint(timestamp)

    # Filter out already-processed posts
    remaining = [pid for pid in post_ids if pid not in processed_ids]
    total = len(post_ids)
    start_from = len(processed_ids)

    if processed_ids:
        logger.info(
            "Resuming: %d/%d posts already done, %d remaining",
            start_from, total, len(remaining)
        )

    # Handle graceful shutdown
    shutdown_requested = [False]

    def handle_signal(signum, frame):
        logger.info("Shutdown signal received, saving checkpoint...")
        shutdown_requested[0] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    errors = 0
    for i, post_id in enumerate(remaining):
        if shutdown_requested[0]:
            logger.info("Graceful shutdown - saving progress")
            break

        data = client.get(f"/posts/{post_id}")

        if data is None:
            errors += 1
            if errors > 50:
                logger.error("Too many errors (%d), saving checkpoint", errors)
                break
            continue

        comments = data.get("comments", [])
        processed_ids.add(post_id)

        if comments:
            flat = flatten_comments(comments, post_id)
            all_comments.extend(flat)

        current = start_from + i + 1
        if current % 100 == 0:
            logger.info(
                "Progress: %d/%d posts (%d%%), %d comments collected, %d errors",
                current, total, int(current / total * 100),
                len(all_comments), errors
            )

        # Periodic checkpoint
        if current % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(all_comments, processed_ids, timestamp)

    # Final checkpoint
    save_checkpoint(all_comments, processed_ids, timestamp)

    logger.info(
        "Comment collection %s: %d comments from %d posts (%d errors)",
        "interrupted" if shutdown_requested[0] else "complete",
        len(all_comments), len(processed_ids), errors
    )
    return all_comments


def save_comments(comments: list, timestamp: str):
    """Save comments as both raw JSON and processed CSV."""
    raw_path = RAW_DIR / f"comments_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    logger.info("Saved raw JSON: %s (%d comments)", raw_path, len(comments))

    csv_path = PROCESSED_DIR / f"comments_{timestamp}.csv"
    if comments:
        fieldnames = list(comments[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comments)
        logger.info("Saved CSV: %s (%d rows)", csv_path, len(comments))

    return raw_path, csv_path


def load_post_ids(posts_json_path: Path) -> list:
    """Load post IDs from a previously collected posts JSON file."""
    with open(posts_json_path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    return [p["id"] for p in posts if p.get("id")]


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    client = MoltbookClient()

    logger.info("=" * 60)
    logger.info("Moltbook Comment Collector (with checkpoint/resume)")
    logger.info("Timestamp: %s", timestamp)
    logger.info("Press Ctrl+C to gracefully stop and save progress")
    logger.info("=" * 60)

    # Find the most recent posts JSON file
    posts_files = sorted(RAW_DIR.glob("posts_2*.json"))
    if not posts_files:
        logger.error("No posts JSON files found. Run collect_posts.py first.")
        sys.exit(1)

    latest_posts = posts_files[-1]
    logger.info("Using posts file: %s", latest_posts)

    # Load and filter posts
    with open(latest_posts, "r", encoding="utf-8") as f:
        posts_data = json.load(f)

    posts_with_comments = [
        p["id"] for p in posts_data
        if p.get("comment_count", 0) > 0
    ]
    logger.info(
        "Filtering to %d posts with comments (out of %d total)",
        len(posts_with_comments), len(posts_data)
    )

    if not posts_with_comments:
        logger.info("No posts with comments found.")
        return

    # Sort by comment count descending (get highest-value posts first)
    comment_counts = {p["id"]: p.get("comment_count", 0) for p in posts_data}
    posts_with_comments.sort(key=lambda pid: comment_counts.get(pid, 0), reverse=True)

    comments = collect_comments_for_posts(
        client, posts_with_comments, timestamp, resume=True
    )

    if comments:
        raw_path, csv_path = save_comments(comments, timestamp)
        logger.info("Collection complete.")
        logger.info("  Raw JSON: %s", raw_path)
        logger.info("  CSV: %s", csv_path)
    else:
        logger.warning("No comments collected.")

    logger.info("API stats: %s", client.stats)


if __name__ == "__main__":
    main()
