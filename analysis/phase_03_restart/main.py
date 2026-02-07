"""
Phase 3: Pre/Post Breach Dataset Split

The security breach on Jan 31 provides a natural experiment:
- Pre-breach data: Cleaner, less manipulated
- Post-breach data: Potentially contaminated by exploitation

Split datasets enable:
- Baseline vs contaminated comparison
- Natural experiment design for causal inference
- Robustness checks using pre-breach only
"""

import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import SPLITS_DIR, BREACH_TIMESTAMP


class Phase03Restart(AnalysisPhase):
    """Split datasets at breach timestamp."""

    phase_id = "phase_03_restart"
    dependencies = ["phase_00_data_audit"]

    def run(self):
        """Execute Phase 3: Create pre/post breach splits."""
        self.log_decision(
            decision=f"Split at breach timestamp: {BREACH_TIMESTAMP}",
            rationale=(
                "Security breach discovered Jan 31 2026 at ~17:35 UTC. "
                "Pre-breach data is cleaner; post-breach may be manipulated. "
                "Split enables natural experiment design."
            ),
        )

        # Load derived data
        posts_df = self.load_derived_posts()
        comments_df = self.load_derived_comments()

        # Split posts
        posts_pre = posts_df[posts_df["is_pre_breach"]].copy()
        posts_post = posts_df[~posts_df["is_pre_breach"]].copy()

        # Split comments
        comments_pre = comments_df[comments_df["is_pre_breach"]].copy()
        comments_post = comments_df[~comments_df["is_pre_breach"]].copy()

        # Save splits
        self.save_parquet(posts_pre, SPLITS_DIR / "posts_pre_breach.parquet")
        self.save_parquet(posts_post, SPLITS_DIR / "posts_post_breach.parquet")
        self.save_parquet(comments_pre, SPLITS_DIR / "comments_pre_breach.parquet")
        self.save_parquet(comments_post, SPLITS_DIR / "comments_post_breach.parquet")

        # Generate comparison summary
        results = self._generate_comparison(
            posts_pre, posts_post, comments_pre, comments_post
        )

        self.save_results(results, "phase_03_breach_split.json")

        # Log key findings
        self._log_findings(posts_pre, posts_post)

    def _generate_comparison(
        self,
        posts_pre: pd.DataFrame,
        posts_post: pd.DataFrame,
        comments_pre: pd.DataFrame,
        comments_post: pd.DataFrame,
    ) -> dict:
        """Generate comparison statistics."""
        results = {
            "breach_timestamp": BREACH_TIMESTAMP,
            "posts": {
                "pre_breach": {
                    "count": len(posts_pre),
                    "unique_authors": posts_pre["author_id"].nunique(),
                    "mean_word_count": posts_pre["word_count"].mean(),
                    "median_word_count": posts_pre["word_count"].median(),
                    "long_posts": int(posts_pre["is_long_post"].sum()),
                    "by_phase": posts_pre["phase"].value_counts().to_dict(),
                },
                "post_breach": {
                    "count": len(posts_post),
                    "unique_authors": posts_post["author_id"].nunique(),
                    "mean_word_count": posts_post["word_count"].mean(),
                    "median_word_count": posts_post["word_count"].median(),
                    "long_posts": int(posts_post["is_long_post"].sum()),
                    "by_phase": posts_post["phase"].value_counts().to_dict(),
                },
            },
            "comments": {
                "pre_breach": {
                    "count": len(comments_pre),
                    "unique_authors": comments_pre["author_id"].nunique(),
                    "mean_word_count": comments_pre["word_count"].mean(),
                },
                "post_breach": {
                    "count": len(comments_post),
                    "unique_authors": comments_post["author_id"].nunique(),
                    "mean_word_count": comments_post["word_count"].mean(),
                },
            },
            "author_overlap": {
                "pre_only_authors": len(
                    set(posts_pre["author_id"]) - set(posts_post["author_id"])
                ),
                "post_only_authors": len(
                    set(posts_post["author_id"]) - set(posts_pre["author_id"])
                ),
                "both_periods": len(
                    set(posts_pre["author_id"]) & set(posts_post["author_id"])
                ),
            },
        }

        return results

    def _log_findings(self, posts_pre: pd.DataFrame, posts_post: pd.DataFrame):
        """Log key findings."""
        total = len(posts_pre) + len(posts_post)
        pre_pct = len(posts_pre) / total * 100 if total > 0 else 0

        # Authors active in both periods
        both = len(set(posts_pre["author_id"]) & set(posts_post["author_id"]))
        pre_only = len(set(posts_pre["author_id"]) - set(posts_post["author_id"]))
        post_only = len(set(posts_post["author_id"]) - set(posts_pre["author_id"]))

        self.log_decision(
            decision="Dataset split complete",
            rationale=(
                f"Pre-breach: {len(posts_pre):,} posts ({pre_pct:.1f}%), "
                f"Post-breach: {len(posts_post):,} posts ({100-pre_pct:.1f}%). "
                f"Author continuity: {both:,} active in both periods, "
                f"{pre_only:,} pre-only, {post_only:,} post-only."
            ),
        )


def run_phase():
    """Entry point for running this phase."""
    phase = Phase03Restart()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
