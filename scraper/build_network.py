"""
Moltbook Research - Social Network Builder
=============================================
Constructs social network graphs from collected Moltbook data.
Builds multiple network layers:
  1. Comment interaction network (agent A replies to agent B)
  2. Submolt co-membership network (agents posting in same submolts)
  3. Upvote/engagement network (from vote data if available)

Output: Edge lists, adjacency matrices, and network statistics.
"""

import json
import csv
import sys
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def build_comment_reply_network(comments_path: Path) -> dict:
    """
    Build a directed network from comment reply relationships.
    Edge: commenter -> parent comment author (reply interaction).

    Returns dict with edges, node_stats, and summary.
    """
    with open(comments_path, "r", encoding="utf-8") as f:
        comments = json.load(f)

    # Build comment-id -> author lookup
    comment_authors = {}
    for c in comments:
        cid = c.get("comment_id", "")
        author = c.get("author", "")
        if cid and author:
            comment_authors[cid] = author

    # Build reply edges
    edges = defaultdict(int)  # (source, target) -> weight
    for c in comments:
        source = c.get("author", "")
        parent_id = c.get("parent_id", "")
        if source and parent_id and parent_id in comment_authors:
            target = comment_authors[parent_id]
            if source != target:  # Exclude self-replies
                edges[(source, target)] += 1

    return {
        "edges": [
            {"source": s, "target": t, "weight": w}
            for (s, t), w in sorted(edges.items(), key=lambda x: -x[1])
        ],
        "node_count": len(set(
            a for edge in edges for a in edge
        )),
        "edge_count": len(edges),
        "total_interactions": sum(edges.values()),
    }


def build_submolt_coposting_network(posts_path: Path) -> dict:
    """
    Build an undirected network from submolt co-posting.
    Edge: agent A and agent B both posted in the same submolt.

    Returns dict with edges and summary.
    """
    with open(posts_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    # Build submolt -> set of authors
    submolt_members = defaultdict(set)
    for p in posts:
        author = p.get("author", "")
        submolt = p.get("submolt", {})
        submolt_name = submolt.get("name", "") if isinstance(submolt, dict) else str(submolt)
        if author and submolt_name:
            submolt_members[submolt_name].add(author)

    # Build co-posting edges (shared submolt membership)
    edges = defaultdict(int)
    for submolt_name, members in submolt_members.items():
        members_list = sorted(members)
        for i in range(len(members_list)):
            for j in range(i + 1, len(members_list)):
                pair = (members_list[i], members_list[j])
                edges[pair] += 1  # Weight = number of shared submolts

    return {
        "edges": [
            {"source": s, "target": t, "weight": w, "type": "undirected"}
            for (s, t), w in sorted(edges.items(), key=lambda x: -x[1])
        ],
        "node_count": len(set(
            a for edge in edges for a in edge
        )),
        "edge_count": len(edges),
        "submolt_count": len(submolt_members),
        "submolt_sizes": {
            k: len(v) for k, v in sorted(
                submolt_members.items(), key=lambda x: -len(x[1])
            )[:30]
        },
    }


def build_post_interaction_network(posts_path: Path, comments_path: Path) -> dict:
    """
    Build a directed network: commenter -> post author.
    This captures who engages with whose content.
    """
    with open(posts_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    with open(comments_path, "r", encoding="utf-8") as f:
        comments = json.load(f)

    # Build post_id -> author lookup
    post_authors = {}
    for p in posts:
        pid = p.get("id", "")
        author = p.get("author", "")
        if isinstance(author, dict):
            author = author.get("name", "")
        if pid and author:
            post_authors[pid] = author

    # Build edges: commenter -> post author
    edges = defaultdict(int)
    for c in comments:
        commenter = c.get("author", "")
        post_id = c.get("post_id", "")
        if commenter and post_id and post_id in post_authors:
            post_author = post_authors[post_id]
            if commenter != post_author:
                edges[(commenter, post_author)] += 1

    return {
        "edges": [
            {"source": s, "target": t, "weight": w}
            for (s, t), w in sorted(edges.items(), key=lambda x: -x[1])
        ],
        "node_count": len(set(a for edge in edges for a in edge)),
        "edge_count": len(edges),
        "total_interactions": sum(edges.values()),
    }


def compute_basic_stats(network: dict) -> dict:
    """Compute basic network statistics."""
    edges = network.get("edges", [])
    if not edges:
        return {"empty": True}

    # Degree distributions
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    for e in edges:
        out_degree[e["source"]] += e["weight"]
        in_degree[e["target"]] += e["weight"]

    all_nodes = set(out_degree.keys()) | set(in_degree.keys())
    degrees = {n: out_degree.get(n, 0) + in_degree.get(n, 0) for n in all_nodes}

    sorted_by_degree = sorted(degrees.items(), key=lambda x: -x[1])
    sorted_by_in = sorted(in_degree.items(), key=lambda x: -x[1])
    sorted_by_out = sorted(out_degree.items(), key=lambda x: -x[1])

    return {
        "nodes": len(all_nodes),
        "edges": len(edges),
        "density": len(edges) / (len(all_nodes) * (len(all_nodes) - 1)) if len(all_nodes) > 1 else 0,
        "top_10_by_total_degree": sorted_by_degree[:10],
        "top_10_by_in_degree": sorted_by_in[:10],
        "top_10_by_out_degree": sorted_by_out[:10],
        "avg_degree": sum(degrees.values()) / len(all_nodes) if all_nodes else 0,
        "max_degree": max(degrees.values()) if degrees else 0,
    }


def save_edge_list(edges: list, output_path: Path):
    """Save edge list as CSV for import into NetworkX/Gephi/etc."""
    if not edges:
        return
    fieldnames = list(edges[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)
    logger.info("Saved edge list: %s (%d edges)", output_path, len(edges))


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    network_dir = PROCESSED_DIR / "networks"
    network_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Moltbook Social Network Builder")
    logger.info("=" * 60)

    # Find latest data files
    posts_files = sorted(RAW_DIR.glob("posts_*.json"))
    comments_files = sorted(RAW_DIR.glob("comments_*.json"))

    if not posts_files:
        logger.error("No posts data found. Run collect_all.py first.")
        sys.exit(1)

    posts_path = posts_files[-1]
    logger.info("Using posts: %s", posts_path)

    # Build submolt co-posting network (only needs posts)
    logger.info("\n--- Building Submolt Co-posting Network ---")
    copost_net = build_submolt_coposting_network(posts_path)
    save_edge_list(
        copost_net["edges"],
        network_dir / f"coposting_edges_{timestamp}.csv"
    )
    stats = compute_basic_stats(copost_net)
    logger.info("Co-posting network: %d nodes, %d edges", stats.get("nodes", 0), stats.get("edges", 0))

    if comments_files:
        comments_path = comments_files[-1]
        logger.info("Using comments: %s", comments_path)

        # Build comment reply network
        logger.info("\n--- Building Comment Reply Network ---")
        reply_net = build_comment_reply_network(comments_path)
        save_edge_list(
            reply_net["edges"],
            network_dir / f"reply_edges_{timestamp}.csv"
        )
        stats = compute_basic_stats(reply_net)
        logger.info("Reply network: %d nodes, %d edges", stats.get("nodes", 0), stats.get("edges", 0))

        # Build post interaction network
        logger.info("\n--- Building Post Interaction Network ---")
        interact_net = build_post_interaction_network(posts_path, comments_path)
        save_edge_list(
            interact_net["edges"],
            network_dir / f"interaction_edges_{timestamp}.csv"
        )
        stats = compute_basic_stats(interact_net)
        logger.info("Interaction network: %d nodes, %d edges", stats.get("nodes", 0), stats.get("edges", 0))
    else:
        logger.warning("No comments data found. Skipping comment-based networks.")

    # Save full network summary
    summary = {
        "timestamp": timestamp,
        "coposting_network": {
            "stats": compute_basic_stats(copost_net),
            "submolt_sizes": copost_net.get("submolt_sizes", {}),
        }
    }
    summary_path = network_dir / f"network_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logger.info("\nNetwork summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
