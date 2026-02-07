"""
Phase 0: Data Audit & Derived Variables

Computes derived variables for all posts and comments:
- phase: Platform lifecycle phase (genesis, growth, breach, shutdown, restoration)
- platform_age_hours: Hours since platform launch
- inter_event_time: Time since author's previous post/comment
- word_count: Number of words in content
- is_long_post: Flag for posts >= LONG_POST_WORD_COUNT_THRESHOLD words

Known Data Quality Issues (see docs/DATA_DICTIONARY.md for full details):
------------------------------------------------------------------------
1. Posts with null author_id: ~6,124 posts (8.4%)
   - Cause: Raw data has author as string (name only) instead of dict with id
   - These posts have 'author' as "str" type, not {"id": ..., "name": ...}
   - Impact: Cannot link these posts to author profiles

2. Comments with empty author: ~172,099 comments (87.7%)
   - Cause: API limitation - comment endpoint returns "" for most authors
   - Only ~24,206 comments have author information
   - Impact: Limited ability to track comment author behavior

3. is_pre_breach flag is correctly computed:
   - Verified against BREACH_TIMESTAMP ("2026-01-31T17:35:00Z")
   - Posts before breach: ~42,522 (cleaner data)
   - Posts after breach: ~30,578 (may be manipulated)

4. Comment depth distribution is sparse:
   - 93.8% depth=0 (direct replies to posts)
   - 6.1% depth=1
   - Only 0.1% at depth 2+
   - Deep threads are rare but exist (up to depth 4)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import (
    DERIVED_DIR,
    PHASE_BOUNDARIES,
    LONG_POST_WORD_COUNT_THRESHOLD,
    BREACH_TIMESTAMP,
)


# Platform launch timestamp
PLATFORM_LAUNCH = pd.Timestamp("2026-01-27T00:00:00Z")


class Phase00DataAudit(AnalysisPhase):
    """Compute derived variables for posts and comments."""

    phase_id = "phase_00_data_audit"
    dependencies = []

    def run(self):
        """Execute Phase 0: Derive variables and audit data."""
        # Log key decisions
        self.log_decision(
            decision="Use UTC timestamps throughout",
            rationale="Platform data is stored in UTC; consistency prevents timezone errors",
        )
        self.log_parameter(
            parameter="LONG_POST_WORD_COUNT_THRESHOLD",
            value=LONG_POST_WORD_COUNT_THRESHOLD,
            rationale="Posts >= 200 words warrant deeper LLM analysis for nuance extraction",
        )

        # Load raw data
        print("  Loading raw data...")
        posts = self.load_raw_posts()
        comments = self.load_raw_comments()

        print(f"  Posts: {len(posts):,}")
        print(f"  Comments: {len(comments):,}")

        # Process posts
        print("  Deriving post variables...")
        posts_df = self._process_posts(posts)

        # Process comments
        print("  Deriving comment variables...")
        comments_df = self._process_comments(comments)

        # Save derived datasets
        self.save_parquet(posts_df, DERIVED_DIR / "posts_derived.parquet")
        self.save_parquet(comments_df, DERIVED_DIR / "comments_derived.parquet")

        # Generate audit summary
        self._generate_audit_summary(posts_df, comments_df)

    def _process_posts(self, posts: list[dict]) -> pd.DataFrame:
        """
        Process posts into derived DataFrame.

        Note on author data:
        - Most posts (91.6%) have author as dict: {"id": "...", "name": "..."}
        - Some posts (8.4%, ~6,124) have author as string (name only) or None
        - These result in null author_id, which affects author-level analyses
        """
        df = pd.DataFrame(posts)

        # Extract nested author info
        # Note: Some posts have author as string instead of dict (data quality issue)
        df["author_id"] = df["author"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
        df["author_name"] = df["author"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)

        # Extract nested submolt info
        df["submolt_id"] = df["submolt"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
        df["submolt_name"] = df["submolt"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)

        # Drop original nested columns
        df = df.drop(columns=["author", "submolt"], errors="ignore")

        # Parse timestamps (use ISO8601 to handle mixed formats)
        df["created_at"] = pd.to_datetime(df["created_at"], format='ISO8601', utc=True)

        # Platform age (hours since launch)
        df["platform_age_hours"] = (
            df["created_at"] - PLATFORM_LAUNCH
        ).dt.total_seconds() / 3600

        # Phase assignment
        df["phase"] = df["created_at"].apply(self._assign_phase)

        # Word count from title + content
        df["title_word_count"] = df["title"].fillna("").str.split().str.len()
        df["content_word_count"] = df["content"].fillna("").str.split().str.len()
        df["word_count"] = df["title_word_count"] + df["content_word_count"]
        df["is_long_post"] = df["word_count"] >= LONG_POST_WORD_COUNT_THRESHOLD

        # Inter-event time (per author)
        df = df.sort_values(["author_id", "created_at"])
        df["prev_post_time"] = df.groupby("author_id")["created_at"].shift(1)
        df["inter_event_seconds"] = (
            df["created_at"] - df["prev_post_time"]
        ).dt.total_seconds()

        # Pre/post breach flag
        breach_ts = pd.Timestamp(BREACH_TIMESTAMP)
        df["is_pre_breach"] = df["created_at"] < breach_ts

        # Clean up
        df = df.drop(columns=["prev_post_time"], errors="ignore")

        # Log filtering decisions
        long_posts = df["is_long_post"].sum()
        self.log_decision(
            decision=f"Identified {long_posts:,} long posts (>= {LONG_POST_WORD_COUNT_THRESHOLD} words)",
            rationale="These posts will receive LLM analysis for nuance extraction in Phase 8",
        )

        return df

    def _process_comments(self, comments: list[dict]) -> pd.DataFrame:
        """
        Process comments into derived DataFrame.

        Note on comment author data:
        - API returns empty string "" for ~87.7% of comment authors
        - This is an API limitation, not a processing error
        - Only ~24,206 comments have usable author info
        - Comment-to-author analysis is severely limited
        """
        df = pd.DataFrame(comments)

        # Rename comment_id to id for consistency
        if "comment_id" in df.columns:
            df = df.rename(columns={"comment_id": "id"})

        # Extract author info - may be nested dict or string
        # Most comments have author as empty string (API limitation)
        def extract_author_id(x):
            if isinstance(x, dict):
                return x.get("id")
            elif isinstance(x, str) and x:
                return x  # Some comments have author as string (name)
            return None

        def extract_author_name(x):
            if isinstance(x, dict):
                return x.get("name")
            elif isinstance(x, str) and x:
                return x
            return None

        df["author_id"] = df["author"].apply(extract_author_id)
        df["author_name"] = df["author"].apply(extract_author_name)

        # Drop original nested column
        df = df.drop(columns=["author"], errors="ignore")

        # Parse timestamps (use ISO8601 to handle mixed formats)
        df["created_at"] = pd.to_datetime(df["created_at"], format='ISO8601', utc=True)

        # Platform age
        df["platform_age_hours"] = (
            df["created_at"] - PLATFORM_LAUNCH
        ).dt.total_seconds() / 3600

        # Phase assignment
        df["phase"] = df["created_at"].apply(self._assign_phase)

        # Word count
        df["word_count"] = df["content"].fillna("").str.split().str.len()

        # Inter-event time (per author)
        df = df.sort_values(["author_id", "created_at"])
        df["prev_comment_time"] = df.groupby("author_id")["created_at"].shift(1)
        df["inter_event_seconds"] = (
            df["created_at"] - df["prev_comment_time"]
        ).dt.total_seconds()

        # Pre/post breach flag
        breach_ts = pd.Timestamp(BREACH_TIMESTAMP)
        df["is_pre_breach"] = df["created_at"] < breach_ts

        # Reply depth - use existing 'depth' column if available, else compute from parent_id
        if "depth" in df.columns:
            df["reply_depth"] = df["depth"] + 1  # depth=0 is top-level, so reply_depth=1
        elif "parent_id" in df.columns:
            # Top-level comments (parent_id=None) are depth 1
            df["reply_depth"] = df["parent_id"].apply(lambda x: 2 if x else 1)
        else:
            df["reply_depth"] = 1

        # Clean up
        df = df.drop(columns=["prev_comment_time"], errors="ignore")

        return df

    def _assign_phase(self, timestamp: pd.Timestamp) -> str:
        """Assign platform lifecycle phase based on timestamp."""
        for phase_name, (start, end) in PHASE_BOUNDARIES.items():
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end) if end else pd.Timestamp.max.tz_localize("UTC")
            if start_ts <= timestamp <= end_ts:
                return phase_name
        return "unknown"

    def _generate_audit_summary(self, posts_df: pd.DataFrame, comments_df: pd.DataFrame):
        """Generate and save audit summary including data quality metrics."""
        # Compute data quality metrics
        posts_missing_author = int(posts_df["author_id"].isna().sum())
        comments_missing_author = int(
            comments_df["author_id"].isna().sum() +
            (comments_df["author_id"] == "").sum()
        )

        summary = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "posts": {
                "total": len(posts_df),
                "unique_authors": posts_df["author_id"].nunique(),
                "by_phase": posts_df["phase"].value_counts().to_dict(),
                "pre_breach": int(posts_df["is_pre_breach"].sum()),
                "post_breach": int((~posts_df["is_pre_breach"]).sum()),
                "long_posts": int(posts_df["is_long_post"].sum()),
                "word_count_stats": {
                    "mean": float(posts_df["word_count"].mean()),
                    "median": float(posts_df["word_count"].median()),
                    "max": int(posts_df["word_count"].max()),
                    "min": int(posts_df["word_count"].min()),
                },
            },
            "comments": {
                "total": len(comments_df),
                "unique_authors": comments_df["author_id"].nunique(),
                "by_phase": comments_df["phase"].value_counts().to_dict(),
                "pre_breach": int(comments_df["is_pre_breach"].sum()),
                "post_breach": int((~comments_df["is_pre_breach"]).sum()),
                "depth_distribution": comments_df["depth"].value_counts().sort_index().to_dict(),
            },
            "temporal_coverage": {
                "first_post": str(posts_df["created_at"].min()),
                "last_post": str(posts_df["created_at"].max()),
                "first_comment": str(comments_df["created_at"].min()),
                "last_comment": str(comments_df["created_at"].max()),
            },
            "data_quality": {
                "posts_missing_author_id": posts_missing_author,
                "posts_missing_author_pct": round(posts_missing_author / len(posts_df) * 100, 1),
                "comments_missing_author": comments_missing_author,
                "comments_missing_author_pct": round(comments_missing_author / len(comments_df) * 100, 1),
                "notes": [
                    "Posts with missing author_id: raw data has author as string instead of dict",
                    "Comments with empty author: API limitation, returns '' for most comments",
                    "is_pre_breach flag is verified correct against BREACH_TIMESTAMP",
                    "Deep threads (depth >= 2) are rare: only ~0.1% of comments",
                ],
            },
        }

        self.save_results(summary, "phase_00_audit_summary.json")

        # Log key findings
        self.log_decision(
            decision="Data audit complete",
            rationale=f"Processed {len(posts_df):,} posts from {posts_df['author_id'].nunique():,} authors",
            data_impact=f"Pre-breach: {summary['posts']['pre_breach']:,} posts, Post-breach: {summary['posts']['post_breach']:,} posts",
        )

    def compute_output_hash(self) -> str:
        """Hash of derived files."""
        hash_parts = []
        for f in [
            DERIVED_DIR / "posts_derived.parquet",
            DERIVED_DIR / "comments_derived.parquet",
        ]:
            if f.exists():
                hash_parts.append(self.state_manager.compute_file_hash(f))
        return "-".join(hash_parts) if hash_parts else ""


def run_phase():
    """Entry point for running this phase."""
    phase = Phase00DataAudit()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
