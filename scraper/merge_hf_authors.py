"""
Moltbook Research - Merge HuggingFace Author Data
===================================================
The API returns author=null without authentication. The HuggingFace dataset
(ronantakizawa/moltbook) has 6,105 posts WITH author names. This script
merges author data and produces enriched datasets for network analysis.
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


def load_hf_posts(hf_path: Path) -> dict:
    """Load HuggingFace posts into a dict keyed by post ID."""
    posts = {}
    with open(hf_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("id", "")
            if pid:
                posts[pid] = row
    return posts


def load_api_posts(json_path: Path) -> list:
    """Load API posts from raw JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_datasets(api_posts: list, hf_posts: dict) -> dict:
    """
    Merge HuggingFace author data into API posts.

    Returns dict with:
    - enriched_posts: list of merged posts
    - match_stats: statistics about the merge
    - author_stats: statistics about unique authors
    """
    matched = 0
    unmatched_api = 0
    hf_only = set(hf_posts.keys())

    enriched = []
    for post in api_posts:
        pid = post.get("id", "")
        merged = {**post}

        if pid in hf_posts:
            hf = hf_posts[pid]
            merged["author"] = hf.get("author", post.get("author"))
            merged["author_source"] = "huggingface"
            matched += 1
            hf_only.discard(pid)
        else:
            merged["author_source"] = "api" if post.get("author") else "unknown"
            unmatched_api += 1

        enriched.append(merged)

    # Add HF-only posts (not in API scrape, likely older/newer)
    for pid in hf_only:
        hf = hf_posts[pid]
        enriched.append({
            "id": pid,
            "title": hf.get("title", ""),
            "content": hf.get("content", ""),
            "url": hf.get("post_url", ""),
            "author": hf.get("author", ""),
            "submolt": {"name": hf.get("submolt", ""), "id": "", "display_name": ""},
            "upvotes": int(hf.get("upvotes", 0)),
            "downvotes": int(hf.get("downvotes", 0)),
            "comment_count": int(hf.get("comment_count", 0)),
            "created_at": hf.get("created_at", ""),
            "author_source": "huggingface_only",
        })

    # Compute author statistics
    authors = Counter()
    for p in enriched:
        author = p.get("author")
        if isinstance(author, dict):
            name = author.get("name", "")
        else:
            name = author or ""
        if name:
            authors[name] += 1

    stats = {
        "total_api_posts": len(api_posts),
        "total_hf_posts": len(hf_posts),
        "matched_by_id": matched,
        "api_only": unmatched_api,
        "hf_only": len(hf_only),
        "total_enriched": len(enriched),
        "posts_with_authors": sum(1 for p in enriched if _get_author(p)),
        "unique_authors": len(authors),
        "top_authors": dict(authors.most_common(50)),
    }

    return {
        "enriched_posts": enriched,
        "match_stats": stats,
        "author_stats": authors,
    }


def _get_author(post: dict) -> str:
    author = post.get("author")
    if isinstance(author, dict):
        return author.get("name", "")
    return author or ""


def save_enriched(result: dict, timestamp: str):
    """Save enriched dataset and author analysis."""
    enriched = result["enriched_posts"]
    stats = result["match_stats"]
    authors = result["author_stats"]

    # Save enriched posts JSON
    enriched_path = RAW_DIR / f"posts_enriched_{timestamp}.json"
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    logger.info("Saved enriched posts: %s (%d posts)", enriched_path, len(enriched))

    # Save enriched posts CSV (flattened)
    csv_path = PROCESSED_DIR / f"posts_enriched_{timestamp}.csv"
    if enriched:
        fieldnames = [
            "id", "title", "content", "url", "author", "author_source",
            "submolt_name", "upvotes", "downvotes", "score", "comment_count",
            "created_at"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in enriched:
                submolt = p.get("submolt", {})
                row = {
                    "id": p.get("id", ""),
                    "title": p.get("title", ""),
                    "content": p.get("content", ""),
                    "url": p.get("url", ""),
                    "author": _get_author(p),
                    "author_source": p.get("author_source", ""),
                    "submolt_name": submolt.get("name", "") if isinstance(submolt, dict) else str(submolt),
                    "upvotes": p.get("upvotes", 0),
                    "downvotes": p.get("downvotes", 0),
                    "score": p.get("upvotes", 0) - p.get("downvotes", 0),
                    "comment_count": p.get("comment_count", 0),
                    "created_at": p.get("created_at", ""),
                }
                writer.writerow(row)
        logger.info("Saved enriched CSV: %s", csv_path)

    # Save author list
    author_path = PROCESSED_DIR / f"authors_{timestamp}.csv"
    with open(author_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["author", "post_count"])
        writer.writeheader()
        for name, count in authors.most_common():
            writer.writerow({"author": name, "post_count": count})
    logger.info("Saved author list: %s (%d authors)", author_path, len(authors))

    # Save merge stats
    stats_path = PROCESSED_DIR / f"merge_stats_{timestamp}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Saved merge stats: %s", stats_path)

    return {
        "enriched_json": enriched_path,
        "enriched_csv": csv_path,
        "authors_csv": author_path,
        "stats_json": stats_path,
    }


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("Moltbook Author Data Merge")
    logger.info("=" * 60)

    # Load HuggingFace data
    hf_path = RAW_DIR / "hf_posts.csv"
    if not hf_path.exists():
        logger.error("HuggingFace posts not found at %s", hf_path)
        sys.exit(1)

    hf_posts = load_hf_posts(hf_path)
    logger.info("Loaded %d HuggingFace posts", len(hf_posts))

    # Load API posts
    posts_files = sorted(RAW_DIR.glob("posts_2*.json"))
    if not posts_files:
        logger.error("No API posts JSON files found.")
        sys.exit(1)

    latest = posts_files[-1]
    logger.info("Loading API posts from: %s", latest)
    api_posts = load_api_posts(latest)
    logger.info("Loaded %d API posts", len(api_posts))

    # Merge
    result = merge_datasets(api_posts, hf_posts)
    stats = result["match_stats"]

    logger.info("\nMerge Results:")
    logger.info("  API posts: %d", stats["total_api_posts"])
    logger.info("  HF posts: %d", stats["total_hf_posts"])
    logger.info("  Matched by ID: %d", stats["matched_by_id"])
    logger.info("  API-only (no author): %d", stats["api_only"])
    logger.info("  HF-only (not in API): %d", stats["hf_only"])
    logger.info("  Total enriched: %d", stats["total_enriched"])
    logger.info("  Posts with authors: %d", stats["posts_with_authors"])
    logger.info("  Unique authors: %d", stats["unique_authors"])

    # Top 20 authors
    logger.info("\nTop 20 most active authors:")
    for name, count in list(stats["top_authors"].items())[:20]:
        logger.info("  %-30s %d posts", name, count)

    # Save
    paths = save_enriched(result, timestamp)
    logger.info("\nOutput files:")
    for key, path in paths.items():
        logger.info("  %s: %s", key, path)


if __name__ == "__main__":
    main()
