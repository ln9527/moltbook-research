"""
Moltbook Research - Incremental Data Update
=============================================
Fetches new data since the last collection run and merges it into
the existing enriched dataset. Designed to be run daily or on-demand.

Workflow:
1. Load state from data/state/last_run.json
2. Detect API schema changes (new Moltbook features)
3. Fetch new posts (by timestamp, stop at overlap)
4. Identify posts needing comment refresh
5. Fetch new/updated comments
6. Merge into enriched dataset (deduplicated by ID)
7. Re-extract submolt stats
8. Update state file
9. Print summary report

Usage:
    python3 scraper/update.py                # Standard incremental update
    python3 scraper/update.py --full-comments # Also refresh comments on old posts
    python3 scraper/update.py --schema-only   # Only check for API changes
    python3 scraper/update.py --init          # Initialize state from existing data
    python3 scraper/update.py --fetch-authors  # Backfill missing author data via /posts/{id}
"""

import json
import csv
import sys
import signal
import argparse
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)
sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_DIR, PROCESSED_DIR, BATCH_SIZE, DATA_DIR
from api_client import MoltbookClient
from collect_posts import collect_all_posts, flatten_post
from collect_comments import flatten_comments, CHECKPOINT_DIR
from extract_submolts_from_posts import extract_submolts, save_submolts_enriched
from detect_schema import detect_and_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

STATE_DIR = DATA_DIR / "state"
STATE_PATH = STATE_DIR / "last_run.json"
MASTER_POSTS_PATH = RAW_DIR / "posts_master.json"
MASTER_COMMENTS_PATH = RAW_DIR / "comments_master.json"
MASTER_POSTS_CSV = PROCESSED_DIR / "posts_master.csv"
MASTER_COMMENTS_CSV = PROCESSED_DIR / "comments_master.csv"
MASTER_AUTHORS_CSV = PROCESSED_DIR / "authors_master.csv"


def load_state() -> dict:
    """Load last run state."""
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    """Save run state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_master_posts() -> dict:
    """Load master posts dataset as dict keyed by ID."""
    if MASTER_POSTS_PATH.exists():
        with open(MASTER_POSTS_PATH, "r", encoding="utf-8") as f:
            posts = json.load(f)
        return {p["id"]: p for p in posts if p.get("id")}

    # Fall back to most recent enriched dataset
    files = sorted(RAW_DIR.glob("posts_enriched_*.json"))
    if files:
        with open(files[-1], "r", encoding="utf-8") as f:
            posts = json.load(f)
        return {p["id"]: p for p in posts if p.get("id")}

    # Fall back to raw posts
    files = sorted(RAW_DIR.glob("posts_2*.json"))
    if files:
        with open(files[-1], "r", encoding="utf-8") as f:
            posts = json.load(f)
        return {p["id"]: p for p in posts if p.get("id")}

    return {}


def load_master_comments() -> dict:
    """Load master comments dataset as dict keyed by comment ID."""
    if MASTER_COMMENTS_PATH.exists():
        with open(MASTER_COMMENTS_PATH, "r", encoding="utf-8") as f:
            comments = json.load(f)
        return {c["comment_id"]: c for c in comments if c.get("comment_id")}

    # Fall back to most recent comments file
    files = sorted(RAW_DIR.glob("comments_2*.json"))
    if files:
        with open(files[-1], "r", encoding="utf-8") as f:
            comments = json.load(f)
        return {c["comment_id"]: c for c in comments if c.get("comment_id")}

    # Check for checkpoint data
    if CHECKPOINT_DIR.exists():
        checkpoints = sorted(CHECKPOINT_DIR.glob("comments_checkpoint_*.json"))
        if checkpoints:
            with open(checkpoints[-1], "r", encoding="utf-8") as f:
                data = json.load(f)
            comments = data.get("comments", [])
            return {c["comment_id"]: c for c in comments if c.get("comment_id")}

    return {}


def load_hf_author_map() -> dict:
    """Load HuggingFace author mapping."""
    hf_path = RAW_DIR / "hf_posts.csv"
    if not hf_path.exists():
        return {}
    author_map = {}
    with open(hf_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("id") and row.get("author"):
                author_map[row["id"]] = row["author"]
    return author_map


def fetch_new_posts(client: MoltbookClient, existing_ids: set) -> list:
    """
    Fetch posts newer than what we have.
    Stops when we encounter posts we already collected.
    """
    new_posts = []
    offset = 0
    consecutive_dupes = 0
    max_consecutive_dupes = 50  # Stop after 50 consecutive known posts

    logger.info("Fetching new posts...")

    while True:
        data = client.get("/posts", params={
            "sort": "new",
            "limit": BATCH_SIZE,
            "offset": offset,
        })

        if data is None:
            logger.error("Failed to fetch posts at offset %d", offset)
            break

        posts = data.get("posts", [])
        if not posts:
            break

        batch_new = 0
        for post in posts:
            pid = post.get("id", "")
            if pid in existing_ids:
                consecutive_dupes += 1
            else:
                consecutive_dupes = 0
                new_posts.append(post)
                batch_new += 1

        logger.info(
            "Offset %d: %d new, %d existing (consecutive dupes: %d)",
            offset, batch_new, len(posts) - batch_new, consecutive_dupes
        )

        if consecutive_dupes >= max_consecutive_dupes:
            logger.info("Reached overlap zone - stopping post collection")
            break

        if not data.get("has_more", False):
            break

        offset = data.get("next_offset", offset + len(posts))

    logger.info("Fetched %d new posts", len(new_posts))
    return new_posts


def fetch_updated_comments(
    client: MoltbookClient,
    posts_to_check: list,
    existing_comments: dict,
    existing_posts: dict = None,
    checkpoint_callback: callable = None,
    checkpoint_interval: int = 500,
) -> list:
    """
    Fetch comments for posts that need refreshing.
    Also captures author data from single-post responses when available.

    Args:
        client: API client
        posts_to_check: List of post IDs to check
        existing_comments: Dict of existing comments (will be modified in place)
        existing_posts: Dict of existing posts (will be modified in place for author data)
        checkpoint_callback: Optional callback to save progress (called with new_comments list)
        checkpoint_interval: How often to call checkpoint_callback (default: 500 posts)

    Returns list of new/updated comment dicts.
    """
    new_comments = []
    authors_captured = 0
    total = len(posts_to_check)
    last_checkpoint = 0

    logger.info("Fetching comments for %d posts...", total)

    for i, post_id in enumerate(posts_to_check):
        data = client.get(f"/posts/{post_id}")

        if data is None:
            continue

        # Capture author data from single-post response
        if existing_posts is not None:
            author = data.get("author")
            if author is not None and post_id in existing_posts:
                old_author = existing_posts[post_id].get("author")
                if old_author is None:
                    existing_posts[post_id]["author"] = author
                    existing_posts[post_id]["author_source"] = "api_single"
                    authors_captured += 1

            # Also update comment_count from fresh API data
            new_count = data.get("comment_count")
            if new_count is not None and post_id in existing_posts:
                existing_posts[post_id]["comment_count"] = new_count

        comments = data.get("comments", [])
        if comments:
            flat = flatten_comments(comments, post_id)
            for c in flat:
                cid = c.get("comment_id", "")
                if cid and cid not in existing_comments:
                    new_comments.append(c)
                    # Also add to existing_comments dict immediately
                    existing_comments[cid] = c

        if (i + 1) % 100 == 0:
            logger.info(
                "Comment refresh: %d/%d posts, %d new comments, %d authors captured",
                i + 1, total, len(new_comments), authors_captured
            )

        # Checkpoint save every N posts
        if checkpoint_callback and (i + 1) - last_checkpoint >= checkpoint_interval:
            logger.info("Saving checkpoint at %d/%d posts (%d comments)...", i + 1, total, len(new_comments))
            checkpoint_callback()
            last_checkpoint = i + 1

    # Final checkpoint if we have unsaved progress
    if checkpoint_callback and (len(posts_to_check) - last_checkpoint > 0):
        logger.info("Saving final checkpoint (%d comments total)...", len(new_comments))
        checkpoint_callback()

    logger.info("Found %d new comments, captured %d authors", len(new_comments), authors_captured)
    return new_comments


def identify_posts_for_comment_refresh(
    existing_posts: dict,
    new_posts: list,
    existing_comments: dict,
    full_refresh: bool = False,
) -> list:
    """
    Identify which posts need their comments fetched/refreshed.

    Includes:
    - All new posts with comment_count > 0
    - Existing posts where comment_count increased (engagement growth)
    - Optionally all posts with comments (full refresh mode)
    """
    posts_to_check = []

    # All new posts with comments
    for p in new_posts:
        if p.get("comment_count", 0) > 0:
            posts_to_check.append(p["id"])

    # Existing posts with increased comment count
    # (compare API's current comment_count vs what we actually collected)
    if not full_refresh:
        # Count comments we have per post
        comments_per_post = Counter()
        for c in existing_comments.values():
            comments_per_post[c.get("post_id", "")] += 1

        for pid, post in existing_posts.items():
            expected = post.get("comment_count", 0)
            actual = comments_per_post.get(pid, 0)
            if expected > actual and pid not in posts_to_check:
                posts_to_check.append(pid)
    else:
        # Full refresh - all posts with comments
        for pid, post in existing_posts.items():
            if post.get("comment_count", 0) > 0 and pid not in posts_to_check:
                posts_to_check.append(pid)

    return posts_to_check


AUTHOR_CHECKPOINT_DIR = RAW_DIR / "checkpoints"
AUTHOR_CHECKPOINT_PREFIX = "author_backfill_checkpoint"

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested - will save checkpoint after current request")


def fetch_authors_backfill(
    client: MoltbookClient,
    existing_posts: dict,
    max_consecutive_errors: int = 20,
) -> int:
    """
    Backfill author data by paginating the list endpoint.
    Uses /posts?sort=new&limit=25 (25x faster than single-post fetches).
    Matches returned posts by ID and updates author where null.
    Checkpoint/resume by offset. Graceful shutdown on SIGINT.
    """
    global _shutdown_requested
    _shutdown_requested = False

    old_sigint = signal.signal(signal.SIGINT, _signal_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    AUTHOR_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    start_offset = 0
    checkpoints = sorted(AUTHOR_CHECKPOINT_DIR.glob(f"{AUTHOR_CHECKPOINT_PREFIX}_*.json"))
    if checkpoints:
        with open(checkpoints[-1], "r", encoding="utf-8") as f:
            cp_data = json.load(f)
        start_offset = cp_data.get("last_offset", 0)
        if start_offset > 0:
            logger.info("Resumed from checkpoint: offset %d", start_offset)

    null_count = sum(1 for p in existing_posts.values() if p.get("author") is None)
    logger.info("Author backfill: %d posts with null author (starting at offset %d)", null_count, start_offset)

    if null_count == 0:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        return 0

    captured = 0
    matched = 0
    errors = 0
    consecutive_errors = 0
    offset = start_offset
    pages = 0

    while not _shutdown_requested:
        data = client.get("/posts", params={
            "sort": "new",
            "limit": BATCH_SIZE,
            "offset": offset,
        })

        if data is None:
            errors += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                logger.warning(
                    "%d consecutive errors at offset %d - saving checkpoint",
                    consecutive_errors, offset
                )
                break
            continue

        consecutive_errors = 0
        posts = data.get("posts", [])
        if not posts:
            logger.info("No more posts at offset %d - backfill complete", offset)
            break

        batch_captured = 0
        for p in posts:
            pid = p.get("id", "")
            if not pid:
                continue

            if pid in existing_posts:
                matched += 1
                author = p.get("author")
                if author is not None and existing_posts[pid].get("author") is None:
                    existing_posts[pid]["author"] = author
                    existing_posts[pid]["author_source"] = "api_backfill"
                    captured += 1
                    batch_captured += 1

                for field in ("upvotes", "downvotes", "comment_count"):
                    val = p.get(field)
                    if val is not None:
                        existing_posts[pid][field] = val
            else:
                # New post not in dataset - add it
                existing_posts[pid] = p
                if p.get("author") is not None:
                    existing_posts[pid]["author_source"] = "api_backfill"
                    captured += 1
                    batch_captured += 1
                matched += 1

        pages += 1
        if pages % 20 == 0:
            logger.info(
                "Author backfill: offset %d, %d matched, %d captured, %d errors",
                offset, matched, captured, errors
            )

        # Checkpoint every 100 pages (~2500 posts)
        if pages % 100 == 0:
            _save_author_checkpoint(offset, captured, matched, errors)

        if not data.get("has_more", False):
            logger.info("Reached end of posts at offset %d", offset)
            break

        offset = data.get("next_offset", offset + len(posts))

    _save_author_checkpoint(offset, captured, matched, errors)

    signal.signal(signal.SIGINT, old_sigint)
    signal.signal(signal.SIGTERM, old_sigterm)

    logger.info(
        "Author backfill done: offset %d, %d matched, %d captured, %d errors",
        offset, matched, captured, errors
    )
    return captured


def _save_author_checkpoint(last_offset: int, captured: int, matched: int, errors: int):
    """Save author backfill checkpoint."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = AUTHOR_CHECKPOINT_DIR / f"{AUTHOR_CHECKPOINT_PREFIX}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "last_offset": last_offset,
            "captured": captured,
            "matched": matched,
            "errors": errors,
            "timestamp": ts,
        }, f)
    logger.info("Author checkpoint: offset %d, %d captured (%s)", last_offset, captured, path.name)


def enrich_with_authors(posts: dict, hf_map: dict) -> int:
    """Enrich posts with HuggingFace author data. Returns count enriched."""
    enriched = 0
    for pid, post in posts.items():
        if pid in hf_map and post.get("author") is None:
            post["author"] = hf_map[pid]
            post["author_source"] = "huggingface"
            enriched += 1
    return enriched


def save_master_datasets(posts: dict, comments: dict, timestamp: str):
    """Save consolidated master datasets."""
    # Posts
    posts_list = sorted(posts.values(), key=lambda p: p.get("created_at", ""), reverse=True)
    with open(MASTER_POSTS_PATH, "w", encoding="utf-8") as f:
        json.dump(posts_list, f, ensure_ascii=False)
    logger.info("Saved master posts: %s (%d posts)", MASTER_POSTS_PATH, len(posts_list))

    # Posts CSV
    if posts_list:
        fieldnames = [
            "id", "title", "content", "url", "author", "author_source",
            "submolt_name", "upvotes", "downvotes", "score", "comment_count",
            "created_at"
        ]
        with open(MASTER_POSTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in posts_list:
                submolt = p.get("submolt", {})
                author = p.get("author")
                if isinstance(author, dict):
                    author_name = author.get("name", "")
                else:
                    author_name = author or ""
                writer.writerow({
                    "id": p.get("id", ""),
                    "title": p.get("title", ""),
                    "content": p.get("content", ""),
                    "url": p.get("url", ""),
                    "author": author_name,
                    "author_source": p.get("author_source", ""),
                    "submolt_name": submolt.get("name", "") if isinstance(submolt, dict) else str(submolt),
                    "upvotes": p.get("upvotes", 0),
                    "downvotes": p.get("downvotes", 0),
                    "score": p.get("upvotes", 0) - p.get("downvotes", 0),
                    "comment_count": p.get("comment_count", 0),
                    "created_at": p.get("created_at", ""),
                })

    # Comments
    comments_list = sorted(comments.values(), key=lambda c: c.get("created_at", ""), reverse=True)
    with open(MASTER_COMMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(comments_list, f, ensure_ascii=False)
    logger.info("Saved master comments: %s (%d comments)", MASTER_COMMENTS_PATH, len(comments_list))

    # Comments CSV
    if comments_list:
        fieldnames = list(comments_list[0].keys())
        with open(MASTER_COMMENTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comments_list)

    # Authors
    authors = Counter()
    for p in posts_list:
        author = p.get("author")
        if isinstance(author, dict):
            name = author.get("name", "")
        else:
            name = author or ""
        if name:
            authors[name] += 1

    with open(MASTER_AUTHORS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["author", "post_count"])
        writer.writeheader()
        for name, count in authors.most_common():
            writer.writerow({"author": name, "post_count": count})
    logger.info("Saved master authors: %s (%d authors)", MASTER_AUTHORS_CSV, len(authors))


def initialize_state(posts: dict, comments: dict):
    """Initialize state from existing data (first-time setup)."""
    latest_post_time = ""
    latest_post_id = ""
    for pid, p in posts.items():
        ts = p.get("created_at", "")
        if ts > latest_post_time:
            latest_post_time = ts
            latest_post_id = pid

    state = {
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "last_update": datetime.now(timezone.utc).isoformat(),
        "latest_post_timestamp": latest_post_time,
        "latest_post_id": latest_post_id,
        "total_posts": len(posts),
        "total_comments": len(comments),
        "total_authors": 0,
        "update_history": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "init",
            "posts_added": len(posts),
            "comments_added": len(comments),
        }],
    }
    save_state(state)
    logger.info("State initialized: %d posts, %d comments", len(posts), len(comments))
    return state


def run_update(
    full_comments: bool = False,
    schema_only: bool = False,
    init_only: bool = False,
    fetch_authors: bool = False,
) -> dict:
    """
    Main update function.

    Returns summary dict with update results.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    client = MoltbookClient()

    logger.info("=" * 60)
    logger.info("Moltbook Incremental Update")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 60)

    # Step 1: Load existing data
    logger.info("\n--- Step 1: Loading Existing Data ---")
    existing_posts = load_master_posts()
    existing_comments = load_master_comments()
    hf_map = load_hf_author_map()
    state = load_state()

    logger.info(
        "Existing: %d posts, %d comments, %d HF authors",
        len(existing_posts), len(existing_comments), len(hf_map)
    )

    # Init mode: just set up state from existing data
    if init_only or not state:
        if not state:
            logger.info("No state found - initializing from existing data")
        state = initialize_state(existing_posts, existing_comments)
        if init_only:
            save_master_datasets(existing_posts, existing_comments, timestamp)
            return {"type": "init", "posts": len(existing_posts), "comments": len(existing_comments)}

    # Author-only mode: skip schema detection and new post collection
    if fetch_authors:
        logger.info("\n--- Author Backfill Mode ---")
        newly_enriched = enrich_with_authors(existing_posts, hf_map)
        authors_backfilled = fetch_authors_backfill(client, existing_posts)

        logger.info("\n--- Saving Updated Datasets ---")
        save_master_datasets(existing_posts, existing_comments, timestamp)
        submolts = extract_submolts(list(existing_posts.values()))
        save_submolts_enriched(submolts, timestamp)

        state["last_update"] = datetime.now(timezone.utc).isoformat()
        state["update_history"] = state.get("update_history", []) + [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "author_backfill",
            "authors_backfilled": authors_backfilled,
            "authors_hf_enriched": newly_enriched,
            "api_stats": client.stats,
        }]
        save_state(state)

        logger.info("\n" + "=" * 60)
        logger.info("AUTHOR BACKFILL COMPLETE")
        logger.info("  Authors backfilled: %d", authors_backfilled)
        logger.info("  API requests: %d (errors: %d)", client.stats["total_requests"], client.stats["errors"])
        logger.info("=" * 60)
        return {"type": "author_backfill", "authors_backfilled": authors_backfilled}

    # Step 2: Schema detection
    schema_result = {"summary": "skipped", "has_changes": False}
    schema_summary = "skipped"
    if not schema_only:
        logger.info("\n--- Step 2: API Schema Detection ---")
    else:
        logger.info("\n--- Schema Detection ---")
    schema_result = detect_and_report(client)
    schema_summary = schema_result["summary"]
    logger.info("Schema: %s", schema_summary)

    if schema_only:
        return {"schema": schema_result, "type": "schema_only"}

    # Step 3: Fetch new posts
    logger.info("\n--- Step 3: Fetching New Posts ---")
    existing_ids = set(existing_posts.keys())
    new_posts = fetch_new_posts(client, existing_ids)

    # Add new posts to dataset
    for p in new_posts:
        pid = p.get("id", "")
        if pid:
            existing_posts[pid] = p

    # Enrich all posts with HF authors
    newly_enriched = enrich_with_authors(existing_posts, hf_map)

    # Save posts immediately after collection (checkpoint to prevent data loss)
    logger.info("Saving posts checkpoint after collection...")
    save_master_datasets(existing_posts, existing_comments, timestamp)

    # Step 4: Comment refresh
    logger.info("\n--- Step 4: Comment Refresh ---")
    posts_to_check = identify_posts_for_comment_refresh(
        existing_posts, new_posts, existing_comments, full_refresh=full_comments
    )

    new_comments = []
    if posts_to_check:
        logger.info("Need to check %d posts for new comments", len(posts_to_check))

        # Create checkpoint callback that saves both posts and comments
        def save_checkpoint():
            save_master_datasets(existing_posts, existing_comments, timestamp)

        new_comments = fetch_updated_comments(
            client,
            posts_to_check,
            existing_comments,
            existing_posts,
            checkpoint_callback=save_checkpoint,
            checkpoint_interval=500,
        )
        # Note: existing_comments is now updated in place by fetch_updated_comments
    else:
        logger.info("No posts need comment refresh")

    authors_backfilled = 0

    # Step 5: Save updated datasets
    logger.info("\n--- Step 5: Saving Updated Datasets ---")
    save_master_datasets(existing_posts, existing_comments, timestamp)

    # Re-extract submolts
    submolts = extract_submolts(list(existing_posts.values()))
    save_submolts_enriched(submolts, timestamp)

    # Step 6: Update state
    latest_ts = ""
    latest_id = ""
    for pid, p in existing_posts.items():
        ts = p.get("created_at", "")
        if ts > latest_ts:
            latest_ts = ts
            latest_id = pid

    update_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "incremental",
        "posts_added": len(new_posts),
        "comments_added": len(new_comments),
        "posts_comment_refreshed": len(posts_to_check),
        "authors_enriched": newly_enriched,
        "authors_backfilled": authors_backfilled,
        "schema_changes": schema_result.get("has_changes", False),
        "api_stats": client.stats,
    }

    history = state.get("update_history", [])
    history.append(update_entry)

    state.update({
        "last_update": datetime.now(timezone.utc).isoformat(),
        "latest_post_timestamp": latest_ts,
        "latest_post_id": latest_id,
        "total_posts": len(existing_posts),
        "total_comments": len(existing_comments),
        "update_history": history,
    })
    save_state(state)

    # Summary
    summary = {
        "type": "incremental",
        "new_posts": len(new_posts),
        "new_comments": len(new_comments),
        "total_posts": len(existing_posts),
        "total_comments": len(existing_comments),
        "total_submolts": len(submolts),
        "schema_changes": schema_result.get("has_changes", False),
        "schema_summary": schema_summary,
        "api_requests": client.stats["total_requests"],
        "api_errors": client.stats["errors"],
    }

    logger.info("\n" + "=" * 60)
    logger.info("UPDATE COMPLETE")
    logger.info("=" * 60)
    logger.info("  New posts: +%d (total: %d)", len(new_posts), len(existing_posts))
    logger.info("  New comments: +%d (total: %d)", len(new_comments), len(existing_comments))
    logger.info("  Submolts: %d", len(submolts))
    logger.info("  Schema: %s", schema_summary)
    logger.info("  API requests: %d (errors: %d)", client.stats["total_requests"], client.stats["errors"])
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Moltbook Incremental Data Update")
    parser.add_argument("--full-comments", action="store_true",
                        help="Refresh comments on all posts (not just new/changed)")
    parser.add_argument("--schema-only", action="store_true",
                        help="Only check for API schema changes")
    parser.add_argument("--init", action="store_true",
                        help="Initialize state from existing data")
    parser.add_argument("--fetch-authors", action="store_true",
                        help="Backfill missing author data via /posts/{id} (long-running)")
    args = parser.parse_args()

    run_update(
        full_comments=args.full_comments,
        schema_only=args.schema_only,
        init_only=args.init,
        fetch_authors=args.fetch_authors,
    )


if __name__ == "__main__":
    main()
