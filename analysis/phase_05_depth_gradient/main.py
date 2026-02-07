"""
Phase 5: Depth Gradient Analysis (Echo Decay)

Analyzes how semantic similarity decays with conversation depth:
- Do replies stay on topic or diverge?
- Is there "echo" behavior (copying/paraphrasing parent)?
- Does decay pattern differ pre/post breach?

Hypothesis: Genuine autonomous conversation should show gradual topic drift.
Heavily prompted/scripted threads might show unusual patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.spatial.distance import cosine

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import EMBEDDINGS_DIR


class Phase05DepthGradient(AnalysisPhase):
    """Analyze echo decay in conversation threads."""

    phase_id = "phase_05_depth_gradient"
    dependencies = ["phase_02_linguistic"]

    def run(self):
        """Execute Phase 5: Depth gradient analysis."""
        self.log_decision(
            decision="Measure cosine similarity decay vs reply depth",
            rationale=(
                "Echo decay = how quickly replies diverge from the root post. "
                "Fit exponential decay: similarity = a * exp(-b * depth) + c. "
                "Parameter b measures decay rate."
            ),
            alternatives=[
                "Simple correlation (less interpretable)",
                "Per-thread topic coherence (computationally expensive)",
            ],
        )

        # Load data
        posts_df = self.load_derived_posts()
        comments_df = self.load_derived_comments()

        post_emb = pd.read_parquet(EMBEDDINGS_DIR / "posts_embeddings.parquet")
        comment_emb = pd.read_parquet(EMBEDDINGS_DIR / "comments_embeddings.parquet")

        print(f"  Posts: {len(posts_df):,}, Comments: {len(comments_df):,}")

        # Build thread structure
        threads = self._build_thread_structure(posts_df, comments_df)
        print(f"  Threads (posts with comments): {len(threads):,}")

        # Compute similarity at each depth
        depth_similarities = self._compute_depth_similarities(
            threads, posts_df, comments_df, post_emb, comment_emb
        )

        # Fit decay model
        decay_params = self._fit_decay_model(depth_similarities)

        # Analyze pre/post breach difference
        breach_comparison = self._compare_breach_periods(
            threads, posts_df, comments_df, post_emb, comment_emb
        )

        # Compile results
        results = {
            "depth_similarities": depth_similarities,
            "decay_model": decay_params,
            "breach_comparison": breach_comparison,
            "summary": {
                "threads_analyzed": len(threads),
                "max_depth_observed": max(depth_similarities.keys()) if depth_similarities else 0,
                "decay_rate": decay_params.get("b", None),
            },
        }

        self.save_results(results, "phase_05_depth_gradient.json")

        # Log findings
        self._log_findings(decay_params, breach_comparison)

    def _build_thread_structure(
        self,
        posts_df: pd.DataFrame,
        comments_df: pd.DataFrame,
    ) -> dict:
        """
        Build thread structure mapping posts to comment trees with depth info.

        The raw comment data has a 'depth' field:
        - depth=0: Top-level comment (direct reply to post)
        - depth=1: Reply to a depth-0 comment
        - depth=2: Reply to a depth-1 comment
        - etc.

        We store both comment IDs and their depths for proper echo decay analysis.
        """
        threads = {}

        # Group comments by post
        for post_id in posts_df["id"].unique():
            post_comments = comments_df[comments_df["post_id"] == post_id]
            if len(post_comments) > 0:
                # Build list with depth info
                comment_depths = {}
                for _, row in post_comments.iterrows():
                    comment_depths[str(row["id"])] = row.get("depth", 0)

                threads[str(post_id)] = {
                    "comment_ids": post_comments["id"].tolist(),
                    "comment_depths": comment_depths,  # Map comment_id -> depth
                    "comment_count": len(post_comments),
                    "max_depth": post_comments["depth"].max() if "depth" in post_comments.columns else 0,
                }

        return threads

    def _compute_depth_similarities(
        self,
        threads: dict,
        posts_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        post_emb: pd.DataFrame,
        comment_emb: pd.DataFrame,
    ) -> dict:
        """
        Compute average similarity at each depth level using real depth values.

        The depth field in comments represents:
        - 0: Direct reply to post
        - 1: Reply to a depth-0 comment
        - 2: Reply to a depth-1 comment
        - etc.

        For analysis, we use depth+1 (so depth 0 becomes "distance 1 from post").
        """
        from collections import defaultdict

        depth_sims = defaultdict(list)

        # Convert embeddings to lookup dicts
        post_emb = post_emb.copy()
        comment_emb = comment_emb.copy()
        post_emb["id"] = post_emb["id"].astype(str)
        comment_emb["id"] = comment_emb["id"].astype(str)

        emb_cols = [c for c in post_emb.columns if c.startswith("emb_")]

        post_emb_dict = {}
        for _, row in post_emb.iterrows():
            post_emb_dict[row["id"]] = row[emb_cols].values

        comment_emb_dict = {}
        for _, row in comment_emb.iterrows():
            comment_emb_dict[row["id"]] = row[emb_cols].values

        # For each thread, compute similarity of comments to root post
        for post_id, thread_data in threads.items():
            if post_id not in post_emb_dict:
                continue

            post_vec = post_emb_dict[post_id]
            comment_depths = thread_data.get("comment_depths", {})

            for comment_id in thread_data["comment_ids"]:
                comment_id_str = str(comment_id)
                if comment_id_str not in comment_emb_dict:
                    continue

                comment_vec = comment_emb_dict[comment_id_str]

                # Get actual depth (default to 0 if not found)
                raw_depth = comment_depths.get(comment_id_str, 0)
                # Convert to "distance from post" (depth 0 = distance 1)
                distance = raw_depth + 1

                # Cosine similarity
                try:
                    sim = 1 - cosine(post_vec, comment_vec)
                    if not np.isnan(sim):
                        depth_sims[distance].append(sim)
                except Exception:
                    # Skip problematic embeddings
                    continue

        # Compute means for each depth level
        depth_means = {}
        for depth in sorted(depth_sims.keys()):
            sims = depth_sims[depth]
            if sims:
                depth_means[depth] = {
                    "mean": float(np.mean(sims)),
                    "std": float(np.std(sims)),
                    "count": len(sims),
                }

        return depth_means

    def _fit_decay_model(self, depth_similarities: dict) -> dict:
        """Fit exponential decay model to similarity vs depth."""
        if not depth_similarities:
            return {"status": "insufficient_data"}

        depths = sorted(depth_similarities.keys())
        means = [depth_similarities[d]["mean"] for d in depths]

        if len(depths) < 2:
            # Can't fit with just one point
            return {
                "status": "single_depth",
                "depth_1_similarity": means[0] if means else None,
            }

        # Exponential decay: y = a * exp(-b * x) + c
        def decay_func(x, a, b, c):
            return a * np.exp(-b * x) + c

        try:
            # Initial guesses
            p0 = [0.5, 0.5, 0.3]
            bounds = ([0, 0, 0], [1, 5, 1])

            popt, pcov = curve_fit(
                decay_func,
                depths,
                means,
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )

            return {
                "status": "fit_success",
                "a": float(popt[0]),  # Initial similarity above baseline
                "b": float(popt[1]),  # Decay rate
                "c": float(popt[2]),  # Asymptotic baseline
                "half_life": float(np.log(2) / popt[1]) if popt[1] > 0 else None,
            }

        except Exception as e:
            return {
                "status": "fit_failed",
                "error": str(e),
                "raw_means": means,
            }

    def _compare_breach_periods(
        self,
        threads: dict,
        posts_df: pd.DataFrame,
        comments_df: pd.DataFrame,
        post_emb: pd.DataFrame,
        comment_emb: pd.DataFrame,
    ) -> dict:
        """Compare depth gradient pre vs post breach."""
        # Split posts by breach
        posts_df["id"] = posts_df["id"].astype(str)
        pre_post_ids = set(posts_df[posts_df["is_pre_breach"]]["id"])
        post_post_ids = set(posts_df[~posts_df["is_pre_breach"]]["id"])

        pre_threads = {k: v for k, v in threads.items() if k in pre_post_ids}
        post_threads = {k: v for k, v in threads.items() if k in post_post_ids}

        # Compute similarities for each period
        pre_sims = self._compute_depth_similarities(
            pre_threads, posts_df, comments_df, post_emb, comment_emb
        )
        post_sims = self._compute_depth_similarities(
            post_threads, posts_df, comments_df, post_emb, comment_emb
        )

        return {
            "pre_breach": {
                "threads": len(pre_threads),
                "depth_similarities": pre_sims,
            },
            "post_breach": {
                "threads": len(post_threads),
                "depth_similarities": post_sims,
            },
        }

    def _log_findings(self, decay_params: dict, breach_comparison: dict):
        """Log key findings."""
        if decay_params.get("status") == "fit_success":
            half_life = decay_params.get("half_life", "N/A")
            self.log_decision(
                decision=f"Echo decay model fitted successfully",
                rationale=(
                    f"Decay rate b={decay_params['b']:.3f}, half-life={half_life:.2f} depth units. "
                    f"Baseline similarity c={decay_params['c']:.3f}"
                ),
            )
        else:
            self.log_decision(
                decision="Depth gradient analysis limited by data structure",
                rationale=(
                    f"Status: {decay_params.get('status')}. "
                    "May need actual reply tree structure for multi-depth analysis."
                ),
            )


def run_phase():
    """Entry point for running this phase."""
    phase = Phase05DepthGradient()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
