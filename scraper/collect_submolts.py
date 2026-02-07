"""
Moltbook Research Scraper - Submolt Collector
================================================
Collects all submolts (subcommunities) from Moltbook.
These represent the organizational units in the AI agent society.
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


def collect_all_submolts(client: MoltbookClient) -> list:
    """Paginate through all submolts."""
    all_submolts = []
    offset = 0

    logger.info("Starting submolt collection (batch_size=%d)", BATCH_SIZE)

    while True:
        data = client.get("/submolts", params={
            "limit": BATCH_SIZE,
            "offset": offset
        })

        if data is None:
            logger.error("Failed to fetch submolts at offset %d", offset)
            break

        submolts = data.get("submolts", [])
        if not submolts:
            break

        all_submolts.extend(submolts)
        has_more = data.get("has_more", False)
        next_offset = data.get("next_offset", offset + len(submolts))

        logger.info(
            "Collected %d submolts (total: %d, has_more: %s)",
            len(submolts), len(all_submolts), has_more
        )

        if not has_more:
            break

        offset = next_offset

    logger.info("Submolt collection complete: %d submolts total", len(all_submolts))
    return all_submolts


def flatten_submolt(submolt: dict) -> dict:
    """Flatten a submolt dict for CSV output."""
    created_by = submolt.get("created_by")
    if isinstance(created_by, dict):
        creator_name = created_by.get("name", "")
    else:
        creator_name = created_by or ""

    return {
        "id": submolt.get("id", ""),
        "name": submolt.get("name", ""),
        "display_name": submolt.get("display_name", ""),
        "description": submolt.get("description", ""),
        "subscriber_count": submolt.get("subscriber_count", 0),
        "created_at": submolt.get("created_at", ""),
        "last_activity_at": submolt.get("last_activity_at", ""),
        "featured_at": submolt.get("featured_at", ""),
        "created_by": creator_name,
    }


def save_submolts(submolts: list, timestamp: str):
    """Save submolts as both raw JSON and processed CSV."""
    raw_path = RAW_DIR / f"submolts_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(submolts, f, ensure_ascii=False, indent=2)
    logger.info("Saved raw JSON: %s (%d submolts)", raw_path, len(submolts))

    csv_path = PROCESSED_DIR / f"submolts_{timestamp}.csv"
    if submolts:
        flat = [flatten_submolt(s) for s in submolts]
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
    logger.info("Moltbook Submolt Collector")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 60)

    submolts = collect_all_submolts(client)

    if submolts:
        raw_path, csv_path = save_submolts(submolts, timestamp)
        logger.info("Collection complete.")
        logger.info("  Raw JSON: %s", raw_path)
        logger.info("  CSV: %s", csv_path)
    else:
        logger.warning("No submolts collected.")

    logger.info("API stats: %s", client.stats)


if __name__ == "__main__":
    main()
