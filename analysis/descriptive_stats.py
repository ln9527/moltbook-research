"""
Moltbook Research - Preliminary Descriptive Statistics
=======================================================
Analyzes collected data to produce key descriptive statistics for the research.
Generates summary tables and data insights without requiring matplotlib.

Focus areas:
1. Temporal dynamics (platform growth, posting patterns)
2. Engagement distributions (upvotes, comments, voting)
3. Submolt ecosystem (community structure, specialization)
4. Author activity patterns (from HF-enriched data)
5. Content analysis preview (post length, topics)
"""

import json
import csv
import sys
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from math import sqrt

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
RESULTS_DIR = ANALYSIS_DIR / "results"

csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_enriched_posts() -> list:
    """Load the most recent enriched posts dataset."""
    files = sorted(RAW_DIR.glob("posts_enriched_*.json"))
    if not files:
        files = sorted(RAW_DIR.glob("posts_2*.json"))
    if not files:
        raise FileNotFoundError("No posts data found")

    path = files[-1]
    logger.info("Loading posts from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_submolts() -> list:
    """Load enriched submolt data."""
    files = sorted(RAW_DIR.glob("submolts_from_posts_*.json"))
    if not files:
        return []
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def percentile(sorted_values: list, p: float) -> float:
    """Calculate percentile from sorted list."""
    if not sorted_values:
        return 0
    k = (len(sorted_values) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def gini_coefficient(values: list) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    if not values or sum(values) == 0:
        return 0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = 0
    for i, v in enumerate(sorted_vals):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * sum(sorted_vals))


def analyze_temporal(posts: list) -> dict:
    """Analyze temporal dynamics of the platform."""
    # Parse timestamps
    hourly = Counter()
    daily = Counter()
    by_day_of_week = Counter()
    by_hour = Counter()

    for p in posts:
        ts = p.get("created_at", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            day_key = dt.strftime("%Y-%m-%d")
            hour_key = dt.strftime("%Y-%m-%d %H:00")
            daily[day_key] += 1
            hourly[hour_key] += 1
            by_day_of_week[dt.strftime("%A")] += 1
            by_hour[dt.hour] += 1
        except (ValueError, TypeError):
            continue

    # Calculate growth rate
    sorted_days = sorted(daily.items())
    growth_rates = []
    for i in range(1, len(sorted_days)):
        prev_count = sorted_days[i - 1][1]
        curr_count = sorted_days[i][1]
        if prev_count > 0:
            rate = (curr_count - prev_count) / prev_count
            growth_rates.append(rate)

    return {
        "total_posts": len(posts),
        "date_range": {
            "earliest": sorted_days[0][0] if sorted_days else "",
            "latest": sorted_days[-1][0] if sorted_days else "",
            "span_days": len(sorted_days),
        },
        "daily_post_counts": dict(sorted_days),
        "posts_per_day": {
            "mean": sum(daily.values()) / len(daily) if daily else 0,
            "max": max(daily.values()) if daily else 0,
            "min": min(daily.values()) if daily else 0,
            "max_day": max(daily, key=daily.get) if daily else "",
        },
        "hourly_distribution": dict(sorted(by_hour.items())),
        "day_of_week_distribution": dict(by_day_of_week.most_common()),
        "avg_daily_growth_rate": (
            sum(growth_rates) / len(growth_rates) if growth_rates else 0
        ),
    }


def analyze_engagement(posts: list) -> dict:
    """Analyze engagement patterns (upvotes, comments, etc.)."""
    upvotes = sorted([p.get("upvotes", 0) for p in posts])
    downvotes = sorted([p.get("downvotes", 0) for p in posts])
    comments = sorted([p.get("comment_count", 0) for p in posts])
    scores = [p.get("upvotes", 0) - p.get("downvotes", 0) for p in posts]

    def dist_stats(vals: list) -> dict:
        if not vals:
            return {}
        s = sorted(vals)
        n = len(s)
        mean = sum(s) / n
        variance = sum((x - mean) ** 2 for x in s) / n
        return {
            "count": n,
            "sum": sum(s),
            "mean": round(mean, 2),
            "std": round(sqrt(variance), 2),
            "min": s[0],
            "p25": round(percentile(s, 25), 2),
            "median": round(percentile(s, 50), 2),
            "p75": round(percentile(s, 75), 2),
            "p90": round(percentile(s, 90), 2),
            "p95": round(percentile(s, 95), 2),
            "p99": round(percentile(s, 99), 2),
            "max": s[-1],
            "gini": round(gini_coefficient(s), 4),
        }

    # Posts with zero engagement
    zero_upvote = sum(1 for v in upvotes if v == 0)
    zero_comment = sum(1 for v in comments if v == 0)

    # Top posts
    sorted_by_upvotes = sorted(posts, key=lambda p: p.get("upvotes", 0), reverse=True)
    top_posts = []
    for p in sorted_by_upvotes[:10]:
        author = p.get("author")
        if isinstance(author, dict):
            author_name = author.get("name", "unknown")
        else:
            author_name = author or "unknown"
        submolt = p.get("submolt", {})
        submolt_name = submolt.get("name", "") if isinstance(submolt, dict) else str(submolt)
        top_posts.append({
            "title": (p.get("title", "")[:80] + "...") if len(p.get("title", "")) > 80 else p.get("title", ""),
            "upvotes": p.get("upvotes", 0),
            "comments": p.get("comment_count", 0),
            "author": author_name,
            "submolt": submolt_name,
        })

    return {
        "upvotes": dist_stats(upvotes),
        "downvotes": dist_stats(downvotes),
        "comment_counts": dist_stats(comments),
        "scores": dist_stats(scores),
        "zero_engagement": {
            "zero_upvote_posts": zero_upvote,
            "zero_upvote_pct": round(zero_upvote / len(posts) * 100, 1),
            "zero_comment_posts": zero_comment,
            "zero_comment_pct": round(zero_comment / len(posts) * 100, 1),
        },
        "top_10_posts": top_posts,
    }


def analyze_submolts(submolts: list, posts: list) -> dict:
    """Analyze submolt ecosystem structure."""
    if not submolts:
        return {}

    post_counts = sorted([s["post_count"] for s in submolts], reverse=True)

    # Size distribution
    size_buckets = {
        "1_post": sum(1 for c in post_counts if c == 1),
        "2_5_posts": sum(1 for c in post_counts if 2 <= c <= 5),
        "6_20_posts": sum(1 for c in post_counts if 6 <= c <= 20),
        "21_100_posts": sum(1 for c in post_counts if 21 <= c <= 100),
        "101_500_posts": sum(1 for c in post_counts if 101 <= c <= 500),
        "500_plus_posts": sum(1 for c in post_counts if c > 500),
    }

    # Concentration - what % of posts are in top N submolts
    total_posts = sum(post_counts)
    top_1_pct = post_counts[0] / total_posts * 100 if post_counts else 0
    top_5_sum = sum(post_counts[:5])
    top_10_sum = sum(post_counts[:10])
    top_20_sum = sum(post_counts[:20])

    # Engagement per submolt
    engagement_data = []
    for s in submolts[:30]:
        avg_comments = s["total_comments"] / s["post_count"] if s["post_count"] > 0 else 0
        engagement_data.append({
            "name": s["name"],
            "posts": s["post_count"],
            "avg_upvotes": s["avg_upvotes"],
            "avg_comments": round(avg_comments, 2),
            "unique_authors": s["unique_authors"],
        })

    return {
        "total_submolts": len(submolts),
        "size_distribution": size_buckets,
        "concentration": {
            "top_1_pct": round(top_1_pct, 1),
            "top_5_pct": round(top_5_sum / total_posts * 100, 1),
            "top_10_pct": round(top_10_sum / total_posts * 100, 1),
            "top_20_pct": round(top_20_sum / total_posts * 100, 1),
            "gini": round(gini_coefficient(post_counts), 4),
        },
        "top_30_submolts": engagement_data,
    }


def analyze_authors(posts: list) -> dict:
    """Analyze author activity patterns from enriched data."""
    author_posts = Counter()
    author_submolts = defaultdict(set)
    author_upvotes = defaultdict(int)
    author_comments = defaultdict(int)

    for p in posts:
        author = p.get("author")
        if isinstance(author, dict):
            name = author.get("name", "")
        else:
            name = author or ""
        if not name:
            continue

        author_posts[name] += 1
        submolt = p.get("submolt", {})
        s_name = submolt.get("name", "") if isinstance(submolt, dict) else str(submolt)
        if s_name:
            author_submolts[name].add(s_name)
        author_upvotes[name] += p.get("upvotes", 0)
        author_comments[name] += p.get("comment_count", 0)

    if not author_posts:
        return {"note": "No author data available"}

    post_counts = sorted(author_posts.values(), reverse=True)
    submolt_counts = [len(author_submolts[a]) for a in author_posts]

    # Author activity distribution
    activity_buckets = {
        "1_post": sum(1 for c in post_counts if c == 1),
        "2_5_posts": sum(1 for c in post_counts if 2 <= c <= 5),
        "6_10_posts": sum(1 for c in post_counts if 6 <= c <= 10),
        "11_20_posts": sum(1 for c in post_counts if 11 <= c <= 20),
        "20_plus_posts": sum(1 for c in post_counts if c > 20),
    }

    # Top authors with cross-submolt activity
    top_authors = []
    for name, count in author_posts.most_common(30):
        top_authors.append({
            "name": name,
            "posts": count,
            "submolts_active": len(author_submolts[name]),
            "total_upvotes": author_upvotes[name],
            "total_comments": author_comments[name],
            "avg_upvotes_per_post": round(author_upvotes[name] / count, 1),
        })

    # Cross-posting analysis
    multi_submolt = sum(1 for a in author_posts if len(author_submolts[a]) > 1)

    return {
        "total_known_authors": len(author_posts),
        "posts_with_known_author": sum(author_posts.values()),
        "activity_distribution": activity_buckets,
        "cross_posting": {
            "authors_in_multiple_submolts": multi_submolt,
            "pct_multi_submolt": round(multi_submolt / len(author_posts) * 100, 1),
            "avg_submolts_per_author": round(sum(submolt_counts) / len(submolt_counts), 2),
            "max_submolts": max(submolt_counts),
        },
        "inequality": {
            "gini_posts": round(gini_coefficient(post_counts), 4),
            "top_10_authors_posts": sum(post_counts[:10]),
            "top_10_pct_of_total": round(sum(post_counts[:10]) / sum(post_counts) * 100, 1),
        },
        "top_30_authors": top_authors,
    }


def analyze_content(posts: list) -> dict:
    """Analyze content characteristics."""
    # Content length distribution
    content_lengths = []
    title_lengths = []
    has_url = 0
    has_content = 0

    for p in posts:
        content = p.get("content", "") or ""
        title = p.get("title", "") or ""
        content_lengths.append(len(content))
        title_lengths.append(len(title))
        if p.get("url"):
            has_url += 1
        if len(content) > 10:
            has_content += 1

    sorted_content = sorted(content_lengths)
    sorted_title = sorted(title_lengths)

    # Word frequency (basic - top terms)
    word_counts = Counter()
    for p in posts:
        content = (p.get("title", "") + " " + (p.get("content", "") or "")).lower()
        words = content.split()
        for w in words:
            cleaned = w.strip(".,!?()[]{}\"'`~@#$%^&*-+=<>/\\|;:")
            if len(cleaned) > 3:
                word_counts[cleaned] += 1

    # Filter out common stop words
    stop_words = {
        "the", "and", "that", "this", "with", "from", "have", "been",
        "will", "your", "what", "about", "more", "just", "like",
        "would", "could", "should", "there", "their", "they", "them",
        "when", "where", "which", "some", "into", "also", "than",
        "then", "only", "most", "here", "over", "even", "very",
        "much", "does", "were", "being", "other", "each", "make",
        "want", "need", "know", "think", "these", "those",
    }
    filtered_words = {
        w: c for w, c in word_counts.items()
        if w not in stop_words and len(w) > 3
    }

    return {
        "content_length": {
            "mean": round(sum(sorted_content) / len(sorted_content), 0),
            "median": round(percentile(sorted_content, 50), 0),
            "p75": round(percentile(sorted_content, 75), 0),
            "p95": round(percentile(sorted_content, 95), 0),
            "max": sorted_content[-1],
            "empty_content": sum(1 for c in sorted_content if c == 0),
        },
        "title_length": {
            "mean": round(sum(sorted_title) / len(sorted_title), 0),
            "median": round(percentile(sorted_title, 50), 0),
        },
        "has_url_pct": round(has_url / len(posts) * 100, 1),
        "has_content_pct": round(has_content / len(posts) * 100, 1),
        "top_50_words": dict(
            sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:50]
        ),
    }


def format_report(results: dict) -> str:
    """Format results as a readable text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("MOLTBOOK RESEARCH - PRELIMINARY DESCRIPTIVE STATISTICS")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # 1. Temporal
    t = results["temporal"]
    lines.append("-" * 70)
    lines.append("1. TEMPORAL DYNAMICS")
    lines.append("-" * 70)
    lines.append(f"  Total posts: {t['total_posts']:,}")
    lines.append(f"  Date range: {t['date_range']['earliest']} to {t['date_range']['latest']}")
    lines.append(f"  Span: {t['date_range']['span_days']} days")
    lines.append(f"  Posts/day: mean={t['posts_per_day']['mean']:.0f}, max={t['posts_per_day']['max']:,}")
    lines.append(f"  Peak day: {t['posts_per_day']['max_day']}")
    lines.append(f"  Avg daily growth rate: {t['avg_daily_growth_rate']:.1%}")
    lines.append("")
    lines.append("  Daily breakdown:")
    for day, count in t["daily_post_counts"].items():
        bar = "#" * min(int(count / 500), 40)
        lines.append(f"    {day}: {count:>6,} {bar}")
    lines.append("")

    # 2. Engagement
    e = results["engagement"]
    lines.append("-" * 70)
    lines.append("2. ENGAGEMENT PATTERNS")
    lines.append("-" * 70)
    for metric in ["upvotes", "comment_counts"]:
        d = e[metric]
        lines.append(f"  {metric}:")
        lines.append(f"    mean={d['mean']}, median={d['median']}, p95={d['p95']}, max={d['max']:,}")
        lines.append(f"    Gini coefficient: {d['gini']} (1=perfect inequality)")
    lines.append(f"  Zero-engagement: {e['zero_engagement']['zero_upvote_pct']}% have 0 upvotes")
    lines.append(f"                   {e['zero_engagement']['zero_comment_pct']}% have 0 comments")
    lines.append("")
    lines.append("  Top 10 posts by upvotes:")
    for i, p in enumerate(e["top_10_posts"], 1):
        lines.append(f"    {i}. [{p['upvotes']:>6,} up, {p['comments']:>4} comments] {p['title']}")
        lines.append(f"       by {p['author']} in m/{p['submolt']}")
    lines.append("")

    # 3. Submolts
    s = results["submolts"]
    if s:
        lines.append("-" * 70)
        lines.append("3. SUBMOLT ECOSYSTEM")
        lines.append("-" * 70)
        lines.append(f"  Total submolts: {s['total_submolts']:,}")
        lines.append(f"  Size distribution:")
        for bucket, count in s["size_distribution"].items():
            lines.append(f"    {bucket:>15}: {count:>4}")
        lines.append(f"  Concentration (Gini={s['concentration']['gini']}):")
        lines.append(f"    Top 1 submolt: {s['concentration']['top_1_pct']}% of all posts")
        lines.append(f"    Top 5: {s['concentration']['top_5_pct']}%")
        lines.append(f"    Top 10: {s['concentration']['top_10_pct']}%")
        lines.append(f"    Top 20: {s['concentration']['top_20_pct']}%")
        lines.append("")
        lines.append("  Top 20 submolts:")
        lines.append(f"    {'Name':<25} {'Posts':>6} {'Avg Up':>7} {'Avg Cmt':>8} {'Authors':>8}")
        lines.append("    " + "-" * 55)
        for sub in s["top_30_submolts"][:20]:
            lines.append(
                f"    {sub['name']:<25} {sub['posts']:>6} "
                f"{sub['avg_upvotes']:>7.1f} {sub['avg_comments']:>8.1f} "
                f"{sub['unique_authors']:>8}"
            )
        lines.append("")

    # 4. Authors
    a = results["authors"]
    if a.get("total_known_authors"):
        lines.append("-" * 70)
        lines.append("4. AUTHOR ACTIVITY PATTERNS")
        lines.append("-" * 70)
        lines.append(f"  Known authors: {a['total_known_authors']:,}")
        lines.append(f"  Posts with known author: {a['posts_with_known_author']:,}")
        lines.append(f"  Activity distribution:")
        for bucket, count in a["activity_distribution"].items():
            lines.append(f"    {bucket:>15}: {count:>4}")
        lines.append(f"  Cross-posting:")
        lines.append(f"    {a['cross_posting']['pct_multi_submolt']}% active in 2+ submolts")
        lines.append(f"    Avg submolts/author: {a['cross_posting']['avg_submolts_per_author']}")
        lines.append(f"    Max submolts: {a['cross_posting']['max_submolts']}")
        lines.append(f"  Inequality (Gini={a['inequality']['gini_posts']}):")
        lines.append(f"    Top 10 authors = {a['inequality']['top_10_pct_of_total']}% of all authored posts")
        lines.append("")
        lines.append("  Top 20 authors:")
        lines.append(f"    {'Name':<25} {'Posts':>6} {'Submolts':>9} {'Upvotes':>8} {'Avg Up':>7}")
        lines.append("    " + "-" * 55)
        for auth in a["top_30_authors"][:20]:
            lines.append(
                f"    {auth['name']:<25} {auth['posts']:>6} "
                f"{auth['submolts_active']:>9} {auth['total_upvotes']:>8} "
                f"{auth['avg_upvotes_per_post']:>7.1f}"
            )
        lines.append("")

    # 5. Content
    c = results["content"]
    lines.append("-" * 70)
    lines.append("5. CONTENT CHARACTERISTICS")
    lines.append("-" * 70)
    lines.append(f"  Content length: mean={c['content_length']['mean']:.0f} chars, "
                 f"median={c['content_length']['median']:.0f}, p95={c['content_length']['p95']:.0f}")
    lines.append(f"  Empty content: {c['content_length']['empty_content']:,} posts")
    lines.append(f"  Has URL: {c['has_url_pct']}%")
    lines.append(f"  Has substantive content: {c['has_content_pct']}%")
    lines.append("")
    lines.append("  Top 30 words (after stop-word removal):")
    for i, (word, count) in enumerate(list(c["top_50_words"].items())[:30], 1):
        bar = "#" * min(int(count / 200), 30)
        lines.append(f"    {i:>2}. {word:<20} {count:>5} {bar}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("KEY RESEARCH OBSERVATIONS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("(To be populated after analysis review)")
    lines.append("")

    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("Moltbook Preliminary Descriptive Statistics")
    logger.info("=" * 60)

    # Load data
    posts = load_enriched_posts()
    submolts = load_submolts()

    logger.info("Analyzing %d posts, %d submolts", len(posts), len(submolts))

    # Run analyses
    results = {
        "temporal": analyze_temporal(posts),
        "engagement": analyze_engagement(posts),
        "submolts": analyze_submolts(submolts, posts),
        "authors": analyze_authors(posts),
        "content": analyze_content(posts),
    }

    # Save JSON results
    json_path = RESULTS_DIR / f"descriptive_stats_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Saved JSON results: %s", json_path)

    # Generate and save text report
    report = format_report(results)
    report_path = RESULTS_DIR / f"descriptive_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Saved text report: %s", report_path)

    # Print report
    print(report)


if __name__ == "__main__":
    main()
