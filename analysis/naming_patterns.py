"""
Moltbook Research - Agent Naming Pattern Analysis
===================================================
Detects batch-created agents through naming pattern analysis.
Classifies agents as likely_automated vs likely_human based on naming conventions.

Patterns detected:
1. Numeric suffix: xmolt01, agent_1, hanhan2
2. Bot/Agent suffix: TipJarBot, OpenClaw-Agent
3. Timestamp pattern: Agent-1706547823
4. Random-looking: Long alphanumeric with few vowels
5. Common prefix batches: Multiple agents with shared prefix + variant
"""

import json
import csv
import re
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "scraper"))
from config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Detection thresholds
MIN_RANDOM_NAME_LENGTH = 8
RANDOM_VOWEL_THRESHOLD = 0.15  # English averages ~40% vowels
MIN_RANDOM_LENGTH_CAPS = 16

# Pre-compiled regex patterns for performance
NUMERIC_SUFFIX_PATTERNS = [
    re.compile(r"^(.+?)[-_]?(\d{1,4})$"),  # name-01, name_1, name01
    re.compile(r"^(.+?)_v(\d+)$"),  # name_v2
    re.compile(r"^(.+?)\s+#?(\d+)$"),  # name #1, name 1
]

TIMESTAMP_PATTERNS = [
    re.compile(r".*[-_](\d{10})$"),  # Unix timestamp (10 digits)
    re.compile(r".*[-_](\d{13})$"),  # Unix ms timestamp (13 digits)
    re.compile(r".*[-_](\d{14})$"),  # YYYYMMDDHHMMSS format
    re.compile(r".*[-_](\d{8})[-_](\d{6})$"),  # YYYYMMDD_HHMMSS
]

RANDOM_CAPS_PATTERN = re.compile(r"[a-z][A-Z][a-z]|[A-Z][a-z][A-Z]")
UUID_LIKE_PATTERN = re.compile(r"^[a-f0-9-]{8,}$")
ALPHA_PREFIX_PATTERN = re.compile(r"^[a-zA-Z]")


def load_unique_agents(posts_json_path: Path) -> list:
    """Extract unique agent names from posts."""
    logger.info("Loading agents from %s...", posts_json_path)
    with open(posts_json_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    agents = set()
    for post in posts:
        author = post.get("author")
        if isinstance(author, dict):
            name = author.get("name", "")
        else:
            name = author or ""
        if name:
            agents.add(name)

    logger.info("Found %d unique agent names", len(agents))
    return sorted(agents)


def detect_numeric_suffix(name: str) -> dict:
    """Detect numeric suffix patterns like xmolt01, agent_1, hanhan2."""
    for pattern in NUMERIC_SUFFIX_PATTERNS:
        match = pattern.match(name)
        if match:
            base, num = match.groups()
            if len(base) >= 2:
                return {
                    "detected": True,
                    "pattern_type": "numeric_suffix",
                    "base_name": base,
                    "suffix_number": int(num),
                }
    return {"detected": False}


def detect_bot_agent_suffix(name: str) -> dict:
    """Detect bot/agent naming patterns like TipJarBot, OpenClaw-Agent."""
    suffixes = [
        "bot", "agent", "ai", "gpt", "llm", "claude", "assistant",
        "helper", "worker", "model", "engine"
    ]

    name_lower = name.lower()
    for suffix in suffixes:
        if name_lower.endswith(suffix) and len(name) > len(suffix) + 1:
            base = name[:-len(suffix)]
            if base.endswith("-") or base.endswith("_"):
                base = base[:-1]
            return {
                "detected": True,
                "pattern_type": "bot_suffix",
                "suffix": suffix,
                "base_name": base,
            }

    prefixes = ["ai-", "ai_", "bot-", "bot_", "agent-", "agent_"]
    for prefix in prefixes:
        if name_lower.startswith(prefix):
            return {
                "detected": True,
                "pattern_type": "bot_prefix",
                "prefix": prefix,
                "base_name": name[len(prefix):],
            }

    return {"detected": False}


def detect_timestamp_pattern(name: str) -> dict:
    """Detect timestamp patterns like Agent-1706547823."""
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.match(name)
        if match:
            return {
                "detected": True,
                "pattern_type": "timestamp",
                "timestamp_value": match.group(1),
            }
    return {"detected": False}


def detect_random_looking(name: str) -> dict:
    """Detect random-looking names (long alphanumeric with few vowels)."""
    if len(name) < MIN_RANDOM_NAME_LENGTH:
        return {"detected": False}

    vowels = set("aeiouAEIOU")
    vowel_count = sum(1 for c in name if c in vowels)
    alpha_count = sum(1 for c in name if c.isalpha())

    if alpha_count == 0:
        return {"detected": False}

    vowel_ratio = vowel_count / alpha_count
    has_random_caps = bool(RANDOM_CAPS_PATTERN.search(name))

    is_random = (
        (len(name) >= 12 and vowel_ratio < RANDOM_VOWEL_THRESHOLD) or
        (len(name) >= MIN_RANDOM_LENGTH_CAPS and has_random_caps) or
        bool(UUID_LIKE_PATTERN.match(name.lower()))
    )

    if is_random:
        return {
            "detected": True,
            "pattern_type": "random_looking",
            "length": len(name),
            "vowel_ratio": round(vowel_ratio, 3),
        }
    return {"detected": False}


def build_prefix_index(all_names: set, max_prefix_len: int = 15) -> dict:
    """Pre-compute prefix -> names mapping for O(1) lookup."""
    prefix_index = defaultdict(set)
    for name in all_names:
        for prefix_len in range(3, min(len(name) + 1, max_prefix_len + 1)):
            prefix_index[name[:prefix_len]].add(name)
    return prefix_index


def detect_common_prefix_batch(name: str, prefix_index: dict) -> dict:
    """Detect if name is part of a batch with common prefix (O(1) lookup)."""
    for prefix_len in range(3, min(len(name), 15)):
        prefix = name[:prefix_len]

        if not ALPHA_PREFIX_PATTERN.match(prefix):
            continue

        matching = prefix_index.get(prefix, set()) - {name}
        if len(matching) >= 2:
            return {
                "detected": True,
                "pattern_type": "common_prefix",
                "prefix": prefix,
                "batch_size": len(matching) + 1,
                "batch_members": sorted(list(matching) + [name])[:10],
            }
    return {"detected": False}


def classify_agent_name(name: str, prefix_index: dict) -> dict:
    """Classify a single agent name using all detection methods."""
    result = {
        "agent_name": name,
        "likely_automated": False,
        "pattern_types": [],
        "details": {},
    }

    detectors = [
        ("numeric_suffix", detect_numeric_suffix),
        ("bot_suffix", detect_bot_agent_suffix),
        ("timestamp", detect_timestamp_pattern),
        ("random_looking", detect_random_looking),
    ]

    for pattern_name, detector in detectors:
        detection = detector(name)
        if detection.get("detected"):
            result["likely_automated"] = True
            result["pattern_types"].append(pattern_name)
            result["details"][pattern_name] = detection

    prefix_detection = detect_common_prefix_batch(name, prefix_index)
    if prefix_detection.get("detected") and prefix_detection.get("batch_size", 0) >= 3:
        result["likely_automated"] = True
        result["pattern_types"].append("common_prefix")
        result["details"]["common_prefix"] = prefix_detection

    if not result["pattern_types"]:
        result["pattern_types"].append("likely_human")

    return result


def find_batch_groups(classifications: list) -> dict:
    """Find and summarize batch groups from classifications."""
    batch_groups = defaultdict(lambda: {"members": [], "pattern": ""})

    numeric_batches = defaultdict(list)
    for c in classifications:
        if "numeric_suffix" in c["pattern_types"]:
            base = c["details"]["numeric_suffix"]["base_name"]
            numeric_batches[base].append({
                "name": c["agent_name"],
                "number": c["details"]["numeric_suffix"]["suffix_number"],
            })

    for base, members in numeric_batches.items():
        if len(members) >= 2:
            batch_groups[f"numeric_{base}"] = {
                "pattern": "numeric_suffix",
                "base_name": base,
                "member_count": len(members),
                "members": sorted([m["name"] for m in members]),
                "number_range": f"{min(m['number'] for m in members)}-{max(m['number'] for m in members)}",
            }

    prefix_batches = defaultdict(set)
    for c in classifications:
        if "common_prefix" in c["pattern_types"]:
            prefix = c["details"]["common_prefix"]["prefix"]
            for member in c["details"]["common_prefix"]["batch_members"]:
                prefix_batches[prefix].add(member)

    for prefix, members in prefix_batches.items():
        if len(members) >= 3:
            batch_groups[f"prefix_{prefix}"] = {
                "pattern": "common_prefix",
                "prefix": prefix,
                "member_count": len(members),
                "members": sorted(members)[:20],
            }

    return dict(batch_groups)


def analyze_naming_patterns(agents: list) -> tuple:
    """Run full naming pattern analysis on all agents."""
    logger.info("Analyzing naming patterns for %d agents...", len(agents))
    all_names = set(agents)

    logger.info("Building prefix index for O(1) batch detection...")
    prefix_index = build_prefix_index(all_names)
    logger.info("Prefix index built with %d prefixes", len(prefix_index))

    classifications = []
    for i, name in enumerate(agents):
        result = classify_agent_name(name, prefix_index)
        classifications.append(result)

        if (i + 1) % 5000 == 0:
            logger.info("Processed %d/%d agents...", i + 1, len(agents))

    automated = sum(1 for c in classifications if c["likely_automated"])
    human = len(classifications) - automated

    logger.info("Classification complete:")
    pct_auto = (automated / len(classifications) * 100) if classifications else 0
    pct_human = (human / len(classifications) * 100) if classifications else 0
    logger.info("  Likely automated: %d (%.1f%%)", automated, pct_auto)
    logger.info("  Likely human: %d (%.1f%%)", human, pct_human)

    pattern_counts = Counter()
    for c in classifications:
        for p in c["pattern_types"]:
            pattern_counts[p] += 1

    logger.info("Pattern distribution:")
    for pattern, count in pattern_counts.most_common():
        logger.info("  %s: %d", pattern, count)

    batch_groups = find_batch_groups(classifications)
    logger.info("Found %d batch groups with 2+ members", len(batch_groups))

    return classifications, batch_groups


def save_results(classifications: list, batch_groups: dict):
    """Save analysis results to CSV and JSON."""
    csv_path = PROCESSED_DIR / "agent_naming_analysis.csv"

    fieldnames = [
        "agent_name", "likely_automated", "pattern_types",
        "numeric_base", "numeric_suffix", "bot_suffix", "batch_prefix"
    ]

    rows = []
    for c in classifications:
        row = {
            "agent_name": c["agent_name"],
            "likely_automated": c["likely_automated"],
            "pattern_types": ",".join(c["pattern_types"]),
            "numeric_base": "",
            "numeric_suffix": "",
            "bot_suffix": "",
            "batch_prefix": "",
        }

        if "numeric_suffix" in c["details"]:
            row["numeric_base"] = c["details"]["numeric_suffix"].get("base_name", "")
            row["numeric_suffix"] = c["details"]["numeric_suffix"].get("suffix_number", "")
        if "bot_suffix" in c["details"]:
            row["bot_suffix"] = c["details"]["bot_suffix"].get("suffix", "")
        if "common_prefix" in c["details"]:
            row["batch_prefix"] = c["details"]["common_prefix"].get("prefix", "")

        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved agent classifications: %s", csv_path)

    batch_path = RESULTS_DIR / "batch_groups.json"
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_batch_groups": len(batch_groups),
            "batch_groups": batch_groups,
        }, f, ensure_ascii=False, indent=2)

    logger.info("Saved batch groups: %s", batch_path)

    return csv_path, batch_path


def compute_summary_statistics(classifications: list, batch_groups: dict) -> dict:
    """Compute summary statistics for the analysis."""
    total = len(classifications)
    automated = sum(1 for c in classifications if c["likely_automated"])

    pattern_counts = Counter()
    for c in classifications:
        for p in c["pattern_types"]:
            pattern_counts[p] += 1

    large_batches = [b for b in batch_groups.values() if b.get("member_count", 0) >= 5]

    return {
        "total_agents": total,
        "likely_automated": automated,
        "likely_human": total - automated,
        "automation_rate_pct": round(automated / total * 100, 1) if total else 0,
        "pattern_counts": dict(pattern_counts),
        "total_batch_groups": len(batch_groups),
        "batch_groups_5_plus": len(large_batches),
        "largest_batch": max(
            (b.get("member_count", 0) for b in batch_groups.values()),
            default=0
        ),
    }


def main():
    logger.info("=" * 60)
    logger.info("Moltbook Agent Naming Pattern Analysis")
    logger.info("=" * 60)

    posts_master = RAW_DIR / "posts_master.json"
    if not posts_master.exists():
        logger.error("posts_master.json not found.")
        return

    agents = load_unique_agents(posts_master)

    classifications, batch_groups = analyze_naming_patterns(agents)

    csv_path, batch_path = save_results(classifications, batch_groups)

    stats = compute_summary_statistics(classifications, batch_groups)

    logger.info("=" * 60)
    logger.info("Summary Statistics:")
    logger.info("  Total agents: %d", stats["total_agents"])
    logger.info("  Likely automated: %d (%.1f%%)", stats["likely_automated"], stats["automation_rate_pct"])
    logger.info("  Likely human: %d", stats["likely_human"])
    logger.info("  Batch groups found: %d", stats["total_batch_groups"])
    logger.info("  Batch groups with 5+ members: %d", stats["batch_groups_5_plus"])
    logger.info("  Largest batch: %d members", stats["largest_batch"])
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info("  Classifications: %s", csv_path)
    logger.info("  Batch groups: %s", batch_path)


if __name__ == "__main__":
    main()
