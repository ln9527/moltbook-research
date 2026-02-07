"""
Moltbook Research Scraper - Owner Data Collector
=================================================
Collects human owner data for AI agents by fetching one sample post per agent.
The single-post endpoint (/posts/:id) returns author metadata including owner.x_* fields.

Supports checkpoint/resume for long-running collection.
"""

import json
import csv
import sys
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR
from api_client import MoltbookClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = RAW_DIR / "checkpoints"
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N agents


def get_unique_agents_with_sample_posts(posts_json_path: Path) -> dict:
    """
    Load posts and extract one sample post_id per unique author.

    Returns: dict mapping author_name -> {author_id, sample_post_id}
    """
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
            author_id = author.get("id", "")
        else:
            author_name = author
            author_id = ""

        if not author_name or author_name in agents:
            continue

        post_id = post.get("id")
        if post_id:
            agents[author_name] = {
                "author_id": author_id,
                "sample_post_id": post_id,
            }

    logger.info("Found %d unique agents with sample posts", len(agents))
    return agents


def extract_owner_data(post_data: dict, author_name: str) -> Optional[dict]:
    """
    Extract owner and agent metadata from single-post API response.
    """
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


def save_checkpoint(all_owners: list, processed_agents: set, timestamp: str):
    """Save current progress to checkpoint files."""
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
    logger.info(
        "Checkpoint saved: %d owners, %d agents processed -> %s",
        len(all_owners), len(processed_agents), cp_path
    )
    return cp_path


def load_checkpoint(timestamp: str) -> tuple:
    """Load from checkpoint if available. Returns (owners, processed_agents)."""
    cp_path = CHECKPOINT_DIR / f"owners_checkpoint_{timestamp}.json"
    if not cp_path.exists():
        checkpoints = sorted(CHECKPOINT_DIR.glob("owners_checkpoint_*.json"))
        if checkpoints:
            cp_path = checkpoints[-1]
            logger.info("Found checkpoint: %s", cp_path)
        else:
            return [], set()

    with open(cp_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    owners = data.get("owners", [])
    processed = set(data.get("processed_agents", []))
    logger.info(
        "Loaded checkpoint: %d owners, %d agents already processed",
        len(owners), len(processed)
    )
    return owners, processed


def collect_owner_data(
    client: MoltbookClient,
    agents: dict,
    timestamp: str,
    resume: bool = True,
    limit: int = None,
) -> list:
    """
    Collect owner data for agents by fetching their sample posts.

    Args:
        client: MoltbookClient instance
        agents: dict mapping agent_name -> {author_id, sample_post_id}
        timestamp: Run timestamp for checkpointing
        resume: Whether to resume from checkpoint
        limit: Optional limit for testing

    Returns:
        List of owner data dicts
    """
    all_owners = []
    processed_agents = set()

    if resume:
        all_owners, processed_agents = load_checkpoint(timestamp)

    agent_list = [(name, info) for name, info in agents.items()
                  if name not in processed_agents]

    if limit:
        agent_list = agent_list[:limit]

    total = len(agents)
    start_from = len(processed_agents)

    if processed_agents:
        logger.info(
            "Resuming: %d/%d agents already done, %d remaining",
            start_from, total, len(agent_list)
        )

    shutdown_requested = [False]

    def handle_signal(signum, frame):
        logger.info("Shutdown signal received, saving checkpoint...")
        shutdown_requested[0] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    errors = 0
    with_owner = 0

    for i, (agent_name, agent_info) in enumerate(agent_list):
        if shutdown_requested[0]:
            logger.info("Graceful shutdown - saving progress")
            break

        post_id = agent_info["sample_post_id"]
        data = client.get(f"/posts/{post_id}")

        if data is None:
            errors += 1
            if errors > 100:
                logger.error("Too many errors (%d), saving checkpoint", errors)
                break
            continue

        owner_data = extract_owner_data(data, agent_name)
        if owner_data:
            all_owners.append(owner_data)
            processed_agents.add(agent_name)
            if owner_data["has_owner"]:
                with_owner += 1

        current = start_from + i + 1
        if current % 100 == 0:
            logger.info(
                "Progress: %d/%d agents (%d%%), %d with owner, %d errors",
                current, total, int(current / total * 100),
                with_owner, errors
            )

        if current % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(all_owners, processed_agents, timestamp)

    save_checkpoint(all_owners, processed_agents, timestamp)

    logger.info(
        "Owner collection %s: %d agents, %d with owner (%d errors)",
        "interrupted" if shutdown_requested[0] else "complete",
        len(all_owners), with_owner, errors
    )
    return all_owners


def save_owners(owners: list):
    """Save owners as both raw JSON and processed CSV."""
    raw_path = RAW_DIR / "owners_master.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(owners, f, ensure_ascii=False, indent=2)
    logger.info("Saved raw JSON: %s (%d owners)", raw_path, len(owners))

    csv_path = PROCESSED_DIR / "owners_master.csv"
    if owners:
        fieldnames = list(owners[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(owners)
        logger.info("Saved CSV: %s (%d rows)", csv_path, len(owners))

    return raw_path, csv_path


def compute_owner_statistics(owners: list) -> dict:
    """Compute summary statistics for owner data."""
    total = len(owners)
    with_owner = sum(1 for o in owners if o.get("has_owner"))

    owner_handles = defaultdict(list)
    for o in owners:
        if o.get("owner_x_handle"):
            owner_handles[o["owner_x_handle"]].append(o["agent_name"])

    multi_agent_owners = {h: agents for h, agents in owner_handles.items()
                         if len(agents) > 1}

    verified_owners = [o for o in owners if o.get("owner_x_verified")]

    karma_values = [o.get("karma", 0) for o in owners]
    avg_karma = sum(karma_values) / len(karma_values) if karma_values else 0

    return {
        "total_agents": total,
        "agents_with_owner": with_owner,
        "agents_without_owner": total - with_owner,
        "owner_coverage_pct": round(with_owner / total * 100, 1) if total else 0,
        "unique_owners": len(owner_handles),
        "multi_agent_owners": len(multi_agent_owners),
        "max_agents_per_owner": max(len(a) for a in owner_handles.values()) if owner_handles else 0,
        "verified_owners": len(verified_owners),
        "avg_agent_karma": round(avg_karma, 1),
    }


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    client = MoltbookClient()

    logger.info("=" * 60)
    logger.info("Moltbook Owner Data Collector (with checkpoint/resume)")
    logger.info("Timestamp: %s", timestamp)
    logger.info("Press Ctrl+C to gracefully stop and save progress")
    logger.info("=" * 60)

    posts_master = RAW_DIR / "posts_master.json"
    if not posts_master.exists():
        logger.error("posts_master.json not found. Run post collection first.")
        sys.exit(1)

    agents = get_unique_agents_with_sample_posts(posts_master)

    if not agents:
        logger.error("No agents found in posts data.")
        sys.exit(1)

    logger.info("Collecting owner data for %d unique agents...", len(agents))
    logger.info("Estimated time: %.1f hours at ~0.6s per request",
                len(agents) * 0.6 / 3600)

    owners = collect_owner_data(client, agents, timestamp, resume=True)

    if owners:
        raw_path, csv_path = save_owners(owners)

        stats = compute_owner_statistics(owners)
        logger.info("=" * 60)
        logger.info("Collection Statistics:")
        for key, value in stats.items():
            logger.info("  %s: %s", key, value)
        logger.info("=" * 60)

        logger.info("Output files:")
        logger.info("  Raw JSON: %s", raw_path)
        logger.info("  CSV: %s", csv_path)
    else:
        logger.warning("No owner data collected.")

    logger.info("API stats: %s", client.stats)


if __name__ == "__main__":
    main()
