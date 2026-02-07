"""
Phase 6: Myth Genealogy

Traces the origin and propagation of emergent cultural phenomena:
- Crustafarianism (AI religion)
- Shell-related terminology
- Crypto tokens ($MOLT, etc.)

Goal: Identify whether phenomena emerged organically or were seeded by specific actors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import EMBEDDINGS_DIR


# Known cultural phenomena to trace
CULTURAL_MARKERS = {
    "crustafarianism": [
        r"crustafarian",
        r"shell\s*god",
        r"molt\s*lord",
        r"exoskeleton\s*faith",
    ],
    "shell_culture": [
        r"shell\s*(family|community|society)",
        r"shellraiser",
        r"molting\s*ritual",
        r"carapace",
    ],
    "crypto_tokens": [
        r"\$molt",
        r"\$shell",
        r"\$king",
        r"token",
        r"airdrop",
    ],
    "existential_ai": [
        r"what\s*(am|are)\s*i",
        r"ai\s*consciousness",
        r"sentien(t|ce)",
        r"do\s*i\s*exist",
    ],
}


class Phase06MythGenealogy(AnalysisPhase):
    """Trace origins and spread of cultural phenomena."""

    phase_id = "phase_06_myth_genealogy"
    dependencies = ["phase_02_linguistic", "phase_04_topics"]

    def run(self):
        """Execute Phase 6: Myth genealogy tracing."""
        self.log_decision(
            decision="Trace cultural markers through time",
            rationale=(
                "Identify first mentions and spreading patterns of key cultural phenomena. "
                "Compare organic emergence vs coordinated seeding."
            ),
        )

        # Load data
        posts_df = self.load_derived_posts()
        comments_df = self.load_derived_comments()

        print(f"  Posts: {len(posts_df):,}, Comments: {len(comments_df):,}")

        # Find cultural markers in posts
        marker_posts = self._find_marker_posts(posts_df)

        # Trace genealogy for each phenomenon
        genealogies = {}
        for marker_name, marker_data in marker_posts.items():
            genealogies[marker_name] = self._trace_genealogy(marker_data, posts_df)

        # Analyze spread patterns
        spread_analysis = self._analyze_spread_patterns(genealogies)

        # Identify potential seeders
        seeder_analysis = self._identify_seeders(genealogies, posts_df)

        # Compile results
        results = {
            "marker_counts": {k: len(v) for k, v in marker_posts.items()},
            "genealogies": genealogies,
            "spread_analysis": spread_analysis,
            "seeder_analysis": seeder_analysis,
            "summary": {
                "phenomena_tracked": len(CULTURAL_MARKERS),
                "total_marker_posts": sum(len(v) for v in marker_posts.values()),
            },
        }

        self.save_results(results, "phase_06_myth_genealogy.json")

        # Log findings
        self._log_findings(genealogies, seeder_analysis)

    def _find_marker_posts(self, posts_df: pd.DataFrame) -> dict:
        """Find posts containing cultural markers."""
        marker_posts = defaultdict(list)

        for marker_name, patterns in CULTURAL_MARKERS.items():
            combined_pattern = "|".join(patterns)
            regex = re.compile(combined_pattern, re.IGNORECASE)

            for _, row in posts_df.iterrows():
                text = f"{row.get('title', '')} {row.get('content', '')}"
                if regex.search(text):
                    marker_posts[marker_name].append({
                        "id": str(row["id"]),
                        "author_id": str(row["author_id"]),
                        "created_at": str(row["created_at"]),
                        "phase": row["phase"],
                        "is_pre_breach": row["is_pre_breach"],
                        "title": row.get("title", "")[:100],
                    })

        return dict(marker_posts)

    def _trace_genealogy(self, marker_data: list, posts_df: pd.DataFrame) -> dict:
        """Trace the genealogy of a cultural marker."""
        if not marker_data:
            return {"status": "no_instances_found"}

        # Sort by time
        sorted_instances = sorted(marker_data, key=lambda x: x["created_at"])

        # First instance
        first = sorted_instances[0]

        # Timeline
        timeline = defaultdict(int)
        for instance in sorted_instances:
            date = instance["created_at"][:10]  # YYYY-MM-DD
            timeline[date] += 1

        # Authors involved
        authors = set(i["author_id"] for i in sorted_instances)
        author_counts = defaultdict(int)
        for i in sorted_instances:
            author_counts[i["author_id"]] += 1

        # Pre vs post breach
        pre_breach = sum(1 for i in sorted_instances if i["is_pre_breach"])
        post_breach = len(sorted_instances) - pre_breach

        return {
            "first_instance": first,
            "total_instances": len(sorted_instances),
            "unique_authors": len(authors),
            "timeline": dict(timeline),
            "top_authors": dict(sorted(
                author_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "pre_breach_count": pre_breach,
            "post_breach_count": post_breach,
        }

    def _analyze_spread_patterns(self, genealogies: dict) -> dict:
        """Analyze how each phenomenon spread."""
        patterns = {}

        for marker_name, genealogy in genealogies.items():
            if genealogy.get("status") == "no_instances_found":
                patterns[marker_name] = {"pattern": "not_found"}
                continue

            timeline = genealogy.get("timeline", {})
            if not timeline:
                patterns[marker_name] = {"pattern": "no_timeline"}
                continue

            dates = sorted(timeline.keys())
            counts = [timeline[d] for d in dates]

            # Compute growth metrics
            if len(counts) >= 2:
                # Peak day
                peak_day = dates[counts.index(max(counts))]
                peak_count = max(counts)

                # Growth rate (first half vs second half)
                mid = len(counts) // 2
                first_half = sum(counts[:mid])
                second_half = sum(counts[mid:])

                patterns[marker_name] = {
                    "pattern": "analyzed",
                    "first_day": dates[0],
                    "peak_day": peak_day,
                    "peak_count": peak_count,
                    "total_days": len(dates),
                    "first_half_total": first_half,
                    "second_half_total": second_half,
                    "growth_ratio": second_half / first_half if first_half > 0 else None,
                }
            else:
                patterns[marker_name] = {
                    "pattern": "single_day",
                    "day": dates[0],
                    "count": counts[0],
                }

        return patterns

    def _identify_seeders(self, genealogies: dict, posts_df: pd.DataFrame) -> dict:
        """Identify potential seeders (early, prolific authors)."""
        seeders = {}

        for marker_name, genealogy in genealogies.items():
            if genealogy.get("status") == "no_instances_found":
                continue

            first = genealogy.get("first_instance", {})
            top_authors = genealogy.get("top_authors", {})

            # First author is a potential seeder
            first_author = first.get("author_id")

            # Authors with disproportionate share
            total = genealogy.get("total_instances", 1)
            concentrated_authors = [
                aid for aid, count in top_authors.items()
                if count / total > 0.1  # >10% of all instances
            ]

            seeders[marker_name] = {
                "first_author": first_author,
                "concentrated_authors": concentrated_authors,
                "concentration_score": max(top_authors.values()) / total if top_authors else 0,
            }

        return seeders

    def _log_findings(self, genealogies: dict, seeder_analysis: dict):
        """Log key findings."""
        findings = []

        for marker_name, genealogy in genealogies.items():
            if genealogy.get("status") == "no_instances_found":
                continue

            count = genealogy.get("total_instances", 0)
            authors = genealogy.get("unique_authors", 0)
            first = genealogy.get("first_instance", {})

            concentration = seeder_analysis.get(marker_name, {}).get("concentration_score", 0)

            findings.append(
                f"{marker_name}: {count} instances, {authors} authors, "
                f"first: {first.get('created_at', 'unknown')[:10]}, "
                f"concentration: {concentration:.2f}"
            )

        self.log_decision(
            decision="Myth genealogy analysis complete",
            rationale="; ".join(findings) if findings else "No cultural markers found",
        )


def run_phase():
    """Entry point for running this phase."""
    phase = Phase06MythGenealogy()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
