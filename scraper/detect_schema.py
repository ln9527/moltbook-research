"""
Moltbook Research - API Schema Detection
==========================================
Monitors the Moltbook API for structural changes:
- New fields in post/comment/submolt responses
- New API endpoints
- Changed pagination behavior
- New submolt features or metadata

Compares current API responses against a stored schema baseline.
When changes are detected, logs them and updates the baseline.
"""

import json
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR
from api_client import MoltbookClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

STATE_DIR = DATA_DIR / "state"
SCHEMA_PATH = STATE_DIR / "api_schema.json"
CHANGELOG_PATH = STATE_DIR / "schema_changelog.json"

# Known API endpoints to probe
ENDPOINTS = {
    "posts_list": {"path": "/posts", "params": {"limit": 2, "sort": "new"}},
    "post_single": None,  # needs dynamic post ID
    "submolts_list": {"path": "/submolts", "params": {"limit": 2}},
}

# Endpoints to test for existence (may return 404/405 currently)
DISCOVERY_ENDPOINTS = [
    "/agents",
    "/agents/leaderboard",
    "/agents/search",
    "/search",
    "/trending",
    "/notifications",
    "/messages",
    "/feed",
    "/stats",
    "/health",
    "/tags",
    "/categories",
    "/events",
    "/polls",
    "/reactions",
    "/bookmarks",
    "/followers",
    "/following",
    "/blocks",
    "/reports",
    "/moderation",
    "/analytics",
    "/webhooks",
]


def extract_schema(obj, prefix="") -> dict:
    """
    Recursively extract the schema (field names, types, nesting) from a JSON object.
    Returns a flat dict of dotted paths -> type strings.
    """
    schema = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            schema[full_key] = type(value).__name__
            if isinstance(value, dict):
                schema.update(extract_schema(value, full_key))
            elif isinstance(value, list) and value:
                schema[f"{full_key}[]"] = type(value[0]).__name__
                if isinstance(value[0], dict):
                    schema.update(extract_schema(value[0], f"{full_key}[]"))
    return schema


def load_baseline() -> dict:
    """Load stored API schema baseline."""
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_baseline(schema: dict):
    """Save API schema baseline."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def load_changelog() -> list:
    """Load schema change history."""
    if CHANGELOG_PATH.exists():
        with open(CHANGELOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_changelog(entries: list):
    """Save schema change history."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHANGELOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def probe_endpoints(client: MoltbookClient) -> dict:
    """
    Probe all known and discovery endpoints.
    Returns dict of endpoint -> {status, schema, raw_keys}.
    """
    results = {}

    # Known endpoints
    for name, config in ENDPOINTS.items():
        if config is None:
            continue
        data = client.get(config["path"], params=config.get("params"))
        if data is not None:
            schema = extract_schema(data)
            results[name] = {
                "status": "active",
                "top_keys": list(data.keys()),
                "schema": schema,
            }
        else:
            results[name] = {"status": "failed", "schema": {}}

    # Try to get a single post for schema extraction
    posts_result = results.get("posts_list", {})
    if posts_result.get("status") == "active":
        # Get first post ID from list response
        list_data = client.get("/posts", params={"limit": 1, "sort": "new"})
        if list_data and list_data.get("posts"):
            post_id = list_data["posts"][0].get("id")
            if post_id:
                single = client.get(f"/posts/{post_id}")
                if single:
                    results["post_single"] = {
                        "status": "active",
                        "top_keys": list(single.keys()),
                        "schema": extract_schema(single),
                    }

    # Discovery endpoints - check for new ones
    for endpoint in DISCOVERY_ENDPOINTS:
        data = client.get(endpoint, params={"limit": 1})
        if data is not None and data.get("success") is not False:
            results[f"discovery:{endpoint}"] = {
                "status": "new_endpoint",
                "top_keys": list(data.keys()),
                "schema": extract_schema(data),
            }
        elif data is not None and data.get("success") is False:
            results[f"discovery:{endpoint}"] = {
                "status": "exists_but_error",
                "error": data.get("error", ""),
            }
        # Skip endpoints that return None (404/405)

    return results


def diff_schemas(old_schema: dict, new_schema: dict) -> dict:
    """
    Compare old and new schemas, returning changes.
    """
    changes = {
        "new_fields": {},
        "removed_fields": {},
        "type_changes": {},
        "new_endpoints": [],
        "removed_endpoints": [],
    }

    old_endpoints = set(old_schema.keys())
    new_endpoints = set(new_schema.keys())

    # New/removed endpoints
    changes["new_endpoints"] = sorted(new_endpoints - old_endpoints)
    changes["removed_endpoints"] = sorted(old_endpoints - new_endpoints)

    # Compare schemas for shared endpoints
    for endpoint in old_endpoints & new_endpoints:
        old_fields = old_schema[endpoint].get("schema", {})
        new_fields = new_schema[endpoint].get("schema", {})

        old_keys = set(old_fields.keys())
        new_keys = set(new_fields.keys())

        for key in new_keys - old_keys:
            changes["new_fields"][f"{endpoint}.{key}"] = new_fields[key]

        for key in old_keys - new_keys:
            changes["removed_fields"][f"{endpoint}.{key}"] = old_fields[key]

        for key in old_keys & new_keys:
            if old_fields[key] != new_fields[key]:
                changes["type_changes"][f"{endpoint}.{key}"] = {
                    "old": old_fields[key],
                    "new": new_fields[key],
                }

    return changes


def has_meaningful_changes(changes: dict) -> bool:
    """Check if any meaningful schema changes were detected."""
    return any([
        changes["new_fields"],
        changes["removed_fields"],
        changes["type_changes"],
        changes["new_endpoints"],
    ])


def detect_and_report(client: Optional[MoltbookClient] = None) -> dict:
    """
    Main detection function. Probes API, compares against baseline, reports changes.

    Returns:
        dict with keys: has_changes, changes, summary, current_schema
    """
    if client is None:
        client = MoltbookClient()

    logger.info("Probing Moltbook API endpoints...")
    current = probe_endpoints(client)

    baseline = load_baseline()
    is_first_run = not baseline

    if is_first_run:
        logger.info("First run - establishing schema baseline")
        save_baseline(current)
        return {
            "has_changes": False,
            "is_first_run": True,
            "current_schema": current,
            "summary": "Schema baseline established",
            "endpoint_count": len([e for e in current if current[e].get("status") == "active"]),
        }

    changes = diff_schemas(baseline, current)
    has_changes = has_meaningful_changes(changes)

    if has_changes:
        logger.warning("API SCHEMA CHANGES DETECTED:")
        if changes["new_endpoints"]:
            logger.warning("  New endpoints: %s", changes["new_endpoints"])
        if changes["new_fields"]:
            for field, ftype in changes["new_fields"].items():
                logger.warning("  New field: %s (%s)", field, ftype)
        if changes["removed_fields"]:
            for field, ftype in changes["removed_fields"].items():
                logger.warning("  Removed field: %s (%s)", field, ftype)
        if changes["type_changes"]:
            for field, change in changes["type_changes"].items():
                logger.warning("  Type changed: %s (%s -> %s)", field, change["old"], change["new"])

        # Log to changelog
        changelog = load_changelog()
        changelog.append({
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "changes": changes,
        })
        save_changelog(changelog)

        # Update baseline
        save_baseline(current)
    else:
        logger.info("No schema changes detected")

    # Build summary
    summary_parts = []
    if changes["new_endpoints"]:
        summary_parts.append(f"{len(changes['new_endpoints'])} new endpoints")
    if changes["new_fields"]:
        summary_parts.append(f"{len(changes['new_fields'])} new fields")
    if changes["removed_fields"]:
        summary_parts.append(f"{len(changes['removed_fields'])} removed fields")
    if changes["type_changes"]:
        summary_parts.append(f"{len(changes['type_changes'])} type changes")

    return {
        "has_changes": has_changes,
        "is_first_run": False,
        "changes": changes,
        "current_schema": current,
        "summary": ", ".join(summary_parts) if summary_parts else "No changes",
    }


def main():
    logger.info("=" * 60)
    logger.info("Moltbook API Schema Detection")
    logger.info("=" * 60)

    result = detect_and_report()

    if result.get("is_first_run"):
        logger.info("Baseline established with %d active endpoints", result.get("endpoint_count", 0))
    elif result["has_changes"]:
        logger.warning("CHANGES FOUND: %s", result["summary"])
        logger.warning("Review: %s", CHANGELOG_PATH)
    else:
        logger.info("API schema unchanged")


if __name__ == "__main__":
    main()
