"""
##############################################################################
#                                                                            #
#  WARNING: INVERTED LABEL LOGIC - DO NOT USE WITHOUT CORRECTION             #
#                                                                            #
#  The classification labels in this module are INVERTED:                    #
#    - "HIGH_AUTONOMY" actually indicates HUMAN-PROMPTED behavior            #
#      (high CoV = irregular = human typing/prompting)                       #
#    - "HIGH_HUMAN_INFLUENCE" actually indicates AUTONOMOUS behavior         #
#      (low CoV = regular = automated scheduling/heartbeat)                  #
#                                                                            #
#  The underlying CoV values are CORRECT, only the labels are wrong.         #
#                                                                            #
#  Results file renamed to: phase_01_temporal_analysis_DEPRECATED_INVERTED   #
#  Statistical analysis file: statistical_analysis_DEPRECATED_INVERTED       #
#                                                                            #
#  To fix: Swap the label names in _classify_authors() method                #
#                                                                            #
##############################################################################

Phase 1: Temporal Analysis - Heartbeat Detection

Analyzes inter-event time patterns to classify agent autonomy:
- HIGH_AUTONOMY: Irregular, human-like timing patterns
- MIXED: Some regularity mixed with variation
- HIGH_HUMAN_INFLUENCE: Very regular "heartbeat" patterns suggesting scheduling

The hypothesis: Truly autonomous agents should exhibit irregular posting patterns,
while human-operated or scheduled agents show metronomic regularity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import (
    HEARTBEAT_BIN_SIZE_SECONDS,
    AUTONOMY_REGULARITY_THRESHOLD,
)


class Phase01Temporal(AnalysisPhase):
    """Detect heartbeat patterns in agent posting behavior."""

    phase_id = "phase_01_temporal"
    dependencies = ["phase_00_data_audit"]

    def run(self):
        """Execute Phase 1: Temporal heartbeat analysis."""
        self.log_decision(
            decision="Use coefficient of variation (CoV) for regularity detection",
            rationale=(
                "CoV = std/mean of inter-event times. Low CoV indicates regular scheduling; "
                "high CoV suggests organic/autonomous behavior. Threshold at 0.8 distinguishes "
                "HIGH_HUMAN_INFLUENCE (CoV < 0.8) from MIXED/HIGH_AUTONOMY."
            ),
            alternatives=[
                "Autocorrelation analysis (more complex)",
                "Fourier analysis for periodicity (overengineered)",
                "Simple median gap comparison (less informative)",
            ],
        )

        # Load derived data
        posts_df = self.load_derived_posts()
        comments_df = self.load_derived_comments()

        print(f"  Posts: {len(posts_df):,}, Comments: {len(comments_df):,}")

        # Combine post and comment activity per author
        all_activity = self._combine_activity(posts_df, comments_df)

        # Analyze each author's temporal patterns
        author_stats = self._analyze_authors(all_activity)

        # Classify authors
        classifications = self._classify_authors(author_stats)

        # Generate summary results
        results = self._generate_results(author_stats, classifications)

        self.save_results(results, "phase_01_temporal_analysis.json")

        # Log key findings
        self._log_findings(classifications)

    def _combine_activity(
        self,
        posts_df: pd.DataFrame,
        comments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine posts and comments into unified activity timeline."""
        # Posts
        posts_activity = posts_df[["author_id", "created_at"]].copy()
        posts_activity["activity_type"] = "post"

        # Comments
        comments_activity = comments_df[["author_id", "created_at"]].copy()
        comments_activity["activity_type"] = "comment"

        # Combine and sort
        all_activity = pd.concat([posts_activity, comments_activity], ignore_index=True)
        all_activity = all_activity.sort_values(["author_id", "created_at"])

        return all_activity

    def _analyze_authors(self, all_activity: pd.DataFrame) -> dict:
        """Analyze temporal patterns for each author."""
        author_stats = {}
        min_events = 5  # Need at least 5 events for meaningful analysis

        for author_id, group in all_activity.groupby("author_id"):
            if len(group) < min_events:
                continue

            # Compute inter-event times
            times = group["created_at"].sort_values()
            gaps = times.diff().dropna().dt.total_seconds()

            if len(gaps) < 4:
                continue

            # Filter out very short gaps (< 1 second, likely API artifacts)
            gaps = gaps[gaps >= 1]

            if len(gaps) < 4:
                continue

            # Compute statistics
            mean_gap = gaps.mean()
            std_gap = gaps.std()
            median_gap = gaps.median()
            cov = std_gap / mean_gap if mean_gap > 0 else 0  # Coefficient of variation

            # Check for periodicity (look for mode in binned gaps)
            bins = np.arange(0, gaps.max() + HEARTBEAT_BIN_SIZE_SECONDS, HEARTBEAT_BIN_SIZE_SECONDS)
            hist, _ = np.histogram(gaps, bins=bins)
            if len(hist) > 0 and hist.max() > 0:
                mode_bin = np.argmax(hist)
                mode_gap = (bins[mode_bin] + bins[mode_bin + 1]) / 2
                mode_frequency = hist.max() / len(gaps)
            else:
                mode_gap = median_gap
                mode_frequency = 0

            author_stats[author_id] = {
                "event_count": len(group),
                "post_count": len(group[group["activity_type"] == "post"]),
                "comment_count": len(group[group["activity_type"] == "comment"]),
                "mean_gap_seconds": mean_gap,
                "std_gap_seconds": std_gap,
                "median_gap_seconds": median_gap,
                "cov": cov,
                "mode_gap_seconds": mode_gap,
                "mode_frequency": mode_frequency,
                "min_gap": gaps.min(),
                "max_gap": gaps.max(),
                "gap_range": gaps.max() - gaps.min(),
            }

        self.log_filtering(
            description="Authors with >= 5 events for temporal analysis",
            before=all_activity["author_id"].nunique(),
            after=len(author_stats),
            rationale="Need sufficient events for meaningful inter-event time analysis",
        )

        return author_stats

    def _classify_authors(self, author_stats: dict) -> dict:
        """Classify authors by autonomy level based on temporal patterns."""
        classifications = {}

        for author_id, stats in author_stats.items():
            cov = stats["cov"]
            mode_freq = stats["mode_frequency"]

            # Classification logic
            if cov < AUTONOMY_REGULARITY_THRESHOLD:
                # Low variation = regular scheduling
                if mode_freq > 0.5:
                    # Strong mode = very regular heartbeat
                    classification = "HIGH_HUMAN_INFLUENCE"
                    confidence = min(1.0, (AUTONOMY_REGULARITY_THRESHOLD - cov) * 2 + mode_freq * 0.5)
                else:
                    classification = "HIGH_HUMAN_INFLUENCE"
                    confidence = min(1.0, (AUTONOMY_REGULARITY_THRESHOLD - cov) * 2)
            elif cov > 1.5:
                # Very high variation = likely autonomous
                classification = "HIGH_AUTONOMY"
                confidence = min(1.0, (cov - 1.0) * 0.5)
            else:
                # Middle ground
                classification = "MIXED"
                confidence = 1.0 - abs(cov - 1.0) * 0.5

            classifications[author_id] = {
                "classification": classification,
                "confidence": round(confidence, 3),
                "cov": round(cov, 3),
                "mode_frequency": round(mode_freq, 3),
            }

        return classifications

    def _generate_results(self, author_stats: dict, classifications: dict) -> dict:
        """Generate summary results."""
        # Count classifications
        class_counts = defaultdict(int)
        for data in classifications.values():
            class_counts[data["classification"]] += 1

        # Get CoV distribution
        covs = [s["cov"] for s in author_stats.values()]

        results = {
            "summary": {
                "total_authors_analyzed": len(author_stats),
                "classification_counts": dict(class_counts),
                "cov_stats": {
                    "mean": np.mean(covs),
                    "median": np.median(covs),
                    "std": np.std(covs),
                    "min": np.min(covs),
                    "max": np.max(covs),
                },
            },
            "threshold_used": AUTONOMY_REGULARITY_THRESHOLD,
            "author_classifications": classifications,
            "author_stats": {
                k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in author_stats.items()
            },
        }

        return results

    def _log_findings(self, classifications: dict):
        """Log key findings."""
        counts = defaultdict(int)
        for data in classifications.values():
            counts[data["classification"]] += 1

        total = len(classifications)
        self.log_decision(
            decision="Temporal classification complete",
            rationale=(
                f"Of {total} authors analyzed: "
                f"HIGH_AUTONOMY={counts['HIGH_AUTONOMY']} ({counts['HIGH_AUTONOMY']/total*100:.1f}%), "
                f"MIXED={counts['MIXED']} ({counts['MIXED']/total*100:.1f}%), "
                f"HIGH_HUMAN_INFLUENCE={counts['HIGH_HUMAN_INFLUENCE']} ({counts['HIGH_HUMAN_INFLUENCE']/total*100:.1f}%)"
            ),
        )


def run_phase():
    """Entry point for running this phase."""
    phase = Phase01Temporal()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
