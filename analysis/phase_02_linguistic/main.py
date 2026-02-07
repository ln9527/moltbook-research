"""
Phase 2: Linguistic Features & Embeddings

Two-pronged text analysis approach:
1. EMBEDDINGS (all content): Compute vector representations for distance/consistency analysis
   - Uses OpenRouter qwen/qwen3-embedding-8b (4096 dim → 768 via PCA)
   - Enables: similarity, clustering, topic modeling, depth gradient analysis

2. LLM ANALYSIS (long posts only): Deep nuance extraction
   - Uses Claude/GPT for structured analysis
   - Targets posts >= 200 words (more substantive content)
   - Extracts: tone, intent, autonomy signals, structure

Both approaches are cached and incrementally updated.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import (
    EMBEDDINGS_DIR,
    LLM_ANALYSES_DIR,
    EMBEDDING_RAW_DIM,
    EMBEDDING_REDUCED_DIM,
    EMBEDDING_CHECKPOINT_INTERVAL,
    LONG_POST_WORD_COUNT_THRESHOLD,
    LONG_THREAD_DEPTH_THRESHOLD,
    MAX_LLM_ANALYSIS_POSTS,
    MAX_LLM_ANALYSIS_THREADS,
)
from .embedding_client import EmbeddingClient
from .llm_analyzer import LLMAnalyzer


class Phase02Linguistic(AnalysisPhase):
    """Compute embeddings and LLM analyses for posts."""

    phase_id = "phase_02_linguistic"
    dependencies = ["phase_00_data_audit"]

    def run(self):
        """Execute Phase 2: Embeddings + LLM analysis."""
        # Log methodology decisions
        self.log_analysis_approach()

        # Load derived posts
        posts_df = self.load_derived_posts()
        print(f"  Loaded {len(posts_df):,} posts")

        # Load comments for thread analysis
        comments_df = self.load_derived_comments()
        print(f"  Loaded {len(comments_df):,} comments")

        # STEP 1: Compute embeddings for all posts
        self._compute_embeddings(posts_df)

        # STEP 2: LLM analysis for long threads (depth >= 3) and long posts
        self._compute_llm_analyses(posts_df, comments_df)

        # STEP 3: Also embed comments (for depth gradient analysis)
        self._compute_comment_embeddings(comments_df)

    def log_analysis_approach(self):
        """Log methodological decisions for text analysis."""
        self.log_decision(
            decision="Two-pronged text analysis: embeddings + LLM",
            rationale=(
                "Embeddings provide scalable distance/consistency metrics for all content. "
                "LLM analysis extracts nuances (tone, intent, autonomy signals) that embeddings miss, "
                "but is expensive so limited to long posts (>= 200 words)."
            ),
            alternatives=[
                "Embeddings only (miss nuance)",
                "LLM only (prohibitively expensive at scale)",
                "Traditional NLP features (less semantic richness)",
            ],
        )

        self.log_model_choice(
            model_type="Embedding model",
            model_name="qwen/qwen3-embedding-8b via OpenRouter",
            rationale=(
                "4096-dim embeddings capture semantic content well. "
                "PCA to 768 dims reduces storage/computation while preserving >95% variance."
            ),
            alternatives=[
                "OpenAI text-embedding-3-large (expensive)",
                "sentence-transformers (local but less capable)",
                "BGE embeddings (alternative quality)",
            ],
        )

        self.log_parameter(
            parameter="PCA_TARGET_DIM",
            value=EMBEDDING_REDUCED_DIM,
            rationale="768 dims balances information preservation with computational efficiency",
        )

    def _compute_embeddings(self, posts_df: pd.DataFrame):
        """Compute embeddings for all posts."""
        print("\n  === Computing Post Embeddings ===")

        # Prepare texts (combine title + content)
        texts = []
        ids = []
        for _, row in posts_df.iterrows():
            title = row.get("title", "") or ""
            content = row.get("content", "") or ""
            text = f"{title}\n\n{content}".strip()
            texts.append(text if text else "(empty post)")
            ids.append(str(row["id"]))

        # Check for existing embeddings
        raw_checkpoint = EMBEDDINGS_DIR / "posts_raw_checkpoint.npz"
        pca_model_path = EMBEDDINGS_DIR / "pca_model.joblib"
        output_path = EMBEDDINGS_DIR / "posts_embeddings.parquet"

        # Compute raw embeddings with checkpointing
        client = EmbeddingClient()
        raw_embeddings = client.embed_with_checkpoint(
            texts=texts,
            ids=ids,
            checkpoint_file=raw_checkpoint,
            checkpoint_interval=EMBEDDING_CHECKPOINT_INTERVAL,
        )

        print(f"  Raw embeddings shape: {raw_embeddings.shape}")

        # Fit/apply PCA
        if pca_model_path.exists():
            print("  Loading existing PCA model...")
            pca = joblib.load(pca_model_path)
            reduced_embeddings = pca.transform(raw_embeddings)
        else:
            print(f"  Fitting PCA to reduce {EMBEDDING_RAW_DIM} -> {EMBEDDING_REDUCED_DIM} dims...")
            pca = PCA(n_components=EMBEDDING_REDUCED_DIM, random_state=42)
            reduced_embeddings = pca.fit_transform(raw_embeddings)

            # Save PCA model
            joblib.dump(pca, pca_model_path)

            # Log variance explained
            variance_explained = sum(pca.explained_variance_ratio_)
            self.log_decision(
                decision=f"PCA fitted: {variance_explained:.1%} variance retained",
                rationale=f"Reduced {EMBEDDING_RAW_DIM} -> {EMBEDDING_REDUCED_DIM} dims",
            )

        # Save as parquet
        emb_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f"emb_{i}" for i in range(EMBEDDING_REDUCED_DIM)],
        )
        emb_df["id"] = ids
        emb_df = emb_df[["id"] + [f"emb_{i}" for i in range(EMBEDDING_REDUCED_DIM)]]

        self.save_parquet(emb_df, output_path)
        print(f"  Saved {len(emb_df):,} post embeddings")

    def _compute_comment_embeddings(self, comments_df: pd.DataFrame):
        """Compute embeddings for comments (for depth analysis)."""
        print("\n  === Computing Comment Embeddings ===")

        texts = []
        ids = []
        for _, row in comments_df.iterrows():
            content = row.get("content", "") or ""
            texts.append(content if content else "(empty comment)")
            ids.append(str(row["id"]))

        # Checkpoint file
        raw_checkpoint = EMBEDDINGS_DIR / "comments_raw_checkpoint.npz"
        output_path = EMBEDDINGS_DIR / "comments_embeddings.parquet"

        # Compute raw embeddings
        client = EmbeddingClient()
        raw_embeddings = client.embed_with_checkpoint(
            texts=texts,
            ids=ids,
            checkpoint_file=raw_checkpoint,
            checkpoint_interval=EMBEDDING_CHECKPOINT_INTERVAL,
        )

        print(f"  Raw comment embeddings shape: {raw_embeddings.shape}")

        # Apply existing PCA model
        pca_model_path = EMBEDDINGS_DIR / "pca_model.joblib"
        if pca_model_path.exists():
            pca = joblib.load(pca_model_path)
            reduced_embeddings = pca.transform(raw_embeddings)
        else:
            print("  WARNING: No PCA model found, saving raw embeddings")
            reduced_embeddings = raw_embeddings

        # Save
        emb_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f"emb_{i}" for i in range(reduced_embeddings.shape[1])],
        )
        emb_df["id"] = ids
        cols = ["id"] + [f"emb_{i}" for i in range(reduced_embeddings.shape[1])]
        emb_df = emb_df[cols]

        self.save_parquet(emb_df, output_path)
        print(f"  Saved {len(emb_df):,} comment embeddings")

    def _compute_llm_analyses(self, posts_df: pd.DataFrame, comments_df: pd.DataFrame):
        """Compute LLM analyses for long threads and long posts."""
        print("\n  === Computing LLM Analyses ===")

        analyzer = LLMAnalyzer()

        # PART 1: Analyze long threads (depth >= 3)
        print(f"\n  --- Long Thread Analysis (depth >= {LONG_THREAD_DEPTH_THRESHOLD}) ---")
        long_threads = self._identify_long_threads(posts_df, comments_df)
        print(f"  Found {len(long_threads):,} threads with depth >= {LONG_THREAD_DEPTH_THRESHOLD}")

        if long_threads:
            # Cap if too many
            if len(long_threads) > MAX_LLM_ANALYSIS_THREADS:
                self.log_decision(
                    decision=f"Sampling {MAX_LLM_ANALYSIS_THREADS:,} from {len(long_threads):,} long threads",
                    rationale=f"Thread analysis is expensive; capping at {MAX_LLM_ANALYSIS_THREADS:,} threads",
                )
                # Sample by post_id
                sampled_ids = np.random.RandomState(42).choice(
                    list(long_threads.keys()),
                    size=min(MAX_LLM_ANALYSIS_THREADS, len(long_threads)),
                    replace=False,
                )
                long_threads = {k: long_threads[k] for k in sampled_ids}

            # Run thread analysis
            thread_checkpoint = LLM_ANALYSES_DIR / "thread_analysis.jsonl"
            thread_analyses = analyzer.analyze_threads_batch(
                threads=long_threads,
                checkpoint_file=thread_checkpoint,
                checkpoint_interval=25,
            )
            print(f"  Completed {len(thread_analyses)} thread analyses")

            if thread_analyses:
                conv_scores = [a.get("conversation_naturalness", 3) for a in thread_analyses if isinstance(a, dict)]
                if conv_scores:
                    self.log_decision(
                        decision="Thread conversation analysis complete",
                        rationale=(
                            f"Analyzed {len(thread_analyses)} long threads. "
                            f"Conversation naturalness: mean={np.mean(conv_scores):.2f}"
                        ),
                    )

        # PART 2: Analyze long posts (>= word threshold) - uses cheaper model
        print(f"\n  --- Long Post Analysis (>= {LONG_POST_WORD_COUNT_THRESHOLD} words) ---")
        long_posts = posts_df[posts_df["is_long_post"]].copy()
        print(f"  Long posts: {len(long_posts):,}")

        if len(long_posts) > 0:
            # Cap if too many
            if len(long_posts) > MAX_LLM_ANALYSIS_POSTS:
                self.log_decision(
                    decision=f"Sampling {MAX_LLM_ANALYSIS_POSTS:,} from {len(long_posts):,} long posts",
                    rationale=f"LLM analysis is expensive; capping at {MAX_LLM_ANALYSIS_POSTS:,} posts",
                )
                long_posts = long_posts.sample(n=MAX_LLM_ANALYSIS_POSTS, random_state=42)

            # Prepare for analysis
            posts_to_analyze = [
                {
                    "id": str(row["id"]),
                    "title": row.get("title", ""),
                    "content": row.get("content", ""),
                }
                for _, row in long_posts.iterrows()
            ]

            # Run LLM analysis (uses fallback model for cost-effectiveness)
            checkpoint_file = LLM_ANALYSES_DIR / "long_posts_analysis.jsonl"
            analyses = analyzer.analyze_posts_batch(
                posts=posts_to_analyze,
                checkpoint_file=checkpoint_file,
                checkpoint_interval=50,
                use_fallback=True,  # Use cheaper model for individual posts
            )

            print(f"  Completed {len(analyses)} post analyses")

            # Summarize autonomy scores
            if analyses:
                autonomy_scores = [a.autonomy_score for a in analyses]
                self.log_decision(
                    decision="LLM autonomy scoring complete",
                    rationale=(
                        f"Analyzed {len(analyses)} long posts. "
                        f"Autonomy scores: mean={np.mean(autonomy_scores):.2f}, "
                        f"std={np.std(autonomy_scores):.2f}"
                    ),
                )

    def _identify_long_threads(
        self, posts_df: pd.DataFrame, comments_df: pd.DataFrame
    ) -> dict:
        """
        Identify threads (posts) with comment chains reaching depth >= threshold.

        Returns:
            dict mapping post_id -> thread data (post + comment chain)
        """
        long_threads = {}

        # Group comments by post
        comments_by_post = comments_df.groupby("post_id")

        for post_id, post_comments in comments_by_post:
            # Compute max depth in this thread
            max_depth = post_comments["depth"].max() if "depth" in post_comments.columns else 0

            if max_depth >= LONG_THREAD_DEPTH_THRESHOLD:
                # Get the post
                post_row = posts_df[posts_df["id"] == post_id]
                if post_row.empty:
                    continue

                post_data = post_row.iloc[0].to_dict()

                # Build comment chain (sorted by depth, then by created_at)
                sorted_comments = post_comments.sort_values(
                    by=["depth", "created_at"] if "depth" in post_comments.columns else ["created_at"]
                )

                comment_chain = [
                    {
                        "id": str(row["id"]),
                        "author_id": str(row.get("author_id", "")),
                        "content": row.get("content", ""),
                        "depth": row.get("depth", 0),
                        "parent_id": str(row.get("parent_id", "")) if row.get("parent_id") else None,
                    }
                    for _, row in sorted_comments.iterrows()
                ]

                long_threads[str(post_id)] = {
                    "post_id": str(post_id),
                    "post_title": post_data.get("title", ""),
                    "post_content": post_data.get("content", ""),
                    "post_author_id": str(post_data.get("author_id", "")),
                    "max_depth": int(max_depth),
                    "comment_count": len(comment_chain),
                    "comments": comment_chain,
                }

        return long_threads

    def compute_output_hash(self) -> str:
        """Hash of output files."""
        hash_parts = []
        for f in [
            EMBEDDINGS_DIR / "posts_embeddings.parquet",
            EMBEDDINGS_DIR / "comments_embeddings.parquet",
            LLM_ANALYSES_DIR / "long_posts_analysis.jsonl",
        ]:
            if f.exists():
                hash_parts.append(self.state_manager.compute_file_hash(f))
        return "-".join(hash_parts) if hash_parts else ""


def run_phase():
    """Entry point for running this phase."""
    phase = Phase02Linguistic()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
