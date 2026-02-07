"""
Phase 4: Topic Modeling

Uses BERTopic with pre-computed embeddings to discover:
- Main themes in Moltbook discourse
- Topic evolution over platform lifecycle
- Topic distribution pre/post breach
- Emergent vs planted topics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import (
    TOPICS_DIR,
    EMBEDDINGS_DIR,
    TOPIC_MIN_CLUSTER_SIZE,
)


class Phase04Topics(AnalysisPhase):
    """BERTopic modeling on Moltbook posts."""

    phase_id = "phase_04_topics"
    dependencies = ["phase_02_linguistic"]

    def run(self):
        """Execute Phase 4: Topic modeling."""
        self.log_model_choice(
            model_type="Topic model",
            model_name="BERTopic with HDBSCAN",
            rationale=(
                "BERTopic leverages pre-computed embeddings for neural topic modeling. "
                "HDBSCAN finds clusters without specifying k. "
                "c-TF-IDF extracts representative terms per topic."
            ),
            alternatives=[
                "LDA (traditional but less semantic)",
                "Top2Vec (similar but less flexible)",
                "K-means clustering (requires specifying k)",
            ],
        )

        # Load data
        posts_df = self.load_derived_posts()
        embeddings_df = pd.read_parquet(EMBEDDINGS_DIR / "posts_embeddings.parquet")

        print(f"  Posts: {len(posts_df):,}")
        print(f"  Embeddings: {len(embeddings_df):,}")

        # Merge embeddings with posts
        embeddings_df["id"] = embeddings_df["id"].astype(str)
        posts_df["id"] = posts_df["id"].astype(str)

        merged = posts_df.merge(embeddings_df, on="id", how="inner")
        print(f"  Merged: {len(merged):,} posts with embeddings")

        # Extract embedding matrix
        emb_cols = [c for c in merged.columns if c.startswith("emb_")]
        embeddings = merged[emb_cols].values

        # Prepare documents (title + content)
        docs = (merged["title"].fillna("") + " " + merged["content"].fillna("")).tolist()

        # Fit BERTopic
        topic_info, doc_topics = self._fit_bertopic(docs, embeddings)

        # Add topic assignments to posts
        merged["topic"] = doc_topics
        merged["topic_name"] = merged["topic"].map(
            dict(zip(topic_info["Topic"], topic_info["Name"]))
        )

        # Save topic assignments
        topic_cols = ["id", "topic", "topic_name"]
        self.save_parquet(merged[topic_cols], TOPICS_DIR / "topic_assignments.parquet")

        # Analyze topic distribution
        results = self._analyze_topics(merged, topic_info)

        self.save_results(results, "phase_04_topic_analysis.json")

        # Log findings
        self._log_findings(topic_info, merged)

    def _fit_bertopic(self, docs: list[str], embeddings: np.ndarray) -> tuple:
        """Fit BERTopic model."""
        try:
            from bertopic import BERTopic
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError:
            print("  WARNING: BERTopic not installed. Using fallback clustering.")
            return self._fallback_clustering(docs, embeddings)

        self.log_parameter(
            parameter="min_cluster_size",
            value=TOPIC_MIN_CLUSTER_SIZE,
            rationale="Minimum 50 posts per topic ensures meaningful clusters",
        )

        # Configure HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=TOPIC_MIN_CLUSTER_SIZE,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # Configure vectorizer for topic representation
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=5,
            ngram_range=(1, 2),
        )

        # Fit BERTopic
        print("  Fitting BERTopic...")
        model = BERTopic(
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            top_n_words=10,
            verbose=True,
        )

        topics, probs = model.fit_transform(docs, embeddings)

        # Get topic info
        topic_info = model.get_topic_info()

        # Save model
        model_path = TOPICS_DIR / "bertopic_model"
        model.save(str(model_path))
        print(f"  Saved model to {model_path}")

        return topic_info, topics

    def _fallback_clustering(self, docs: list[str], embeddings: np.ndarray) -> tuple:
        """Fallback clustering if BERTopic unavailable."""
        from sklearn.cluster import KMeans

        print("  Using KMeans fallback (BERTopic not available)")

        # Estimate number of clusters
        n_clusters = min(50, len(docs) // 100)
        n_clusters = max(5, n_clusters)

        self.log_decision(
            decision=f"Fallback to KMeans with k={n_clusters}",
            rationale="BERTopic not installed; using sklearn KMeans as fallback",
        )

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        topics = kmeans.fit_predict(embeddings)

        # Create basic topic info
        topic_info = pd.DataFrame({
            "Topic": range(-1, n_clusters),
            "Count": [0] + [sum(topics == i) for i in range(n_clusters)],
            "Name": ["Outliers"] + [f"Topic_{i}" for i in range(n_clusters)],
        })

        return topic_info, topics

    def _analyze_topics(self, merged: pd.DataFrame, topic_info: pd.DataFrame) -> dict:
        """Analyze topic distribution."""
        # Topic counts
        topic_counts = merged["topic"].value_counts().to_dict()

        # Topics by phase
        phase_topics = defaultdict(lambda: defaultdict(int))
        for _, row in merged.iterrows():
            phase_topics[row["phase"]][row["topic"]] += 1

        # Topics by pre/post breach
        breach_topics = {
            "pre_breach": merged[merged["is_pre_breach"]]["topic"].value_counts().to_dict(),
            "post_breach": merged[~merged["is_pre_breach"]]["topic"].value_counts().to_dict(),
        }

        # Identify topics that emerged post-breach
        pre_topics = set(breach_topics["pre_breach"].keys())
        post_topics = set(breach_topics["post_breach"].keys())
        new_post_breach = post_topics - pre_topics
        disappeared_post_breach = pre_topics - post_topics

        results = {
            "summary": {
                "total_topics": len(topic_info) - 1,  # Exclude outlier topic
                "outlier_count": int(topic_counts.get(-1, 0)),
                "largest_topic": int(topic_info[topic_info["Topic"] != -1]["Count"].max()),
            },
            "topic_info": topic_info.to_dict(orient="records"),
            "topic_counts": topic_counts,
            "by_phase": {k: dict(v) for k, v in phase_topics.items()},
            "by_breach": breach_topics,
            "topic_shifts": {
                "new_post_breach": list(new_post_breach),
                "disappeared_post_breach": list(disappeared_post_breach),
            },
        }

        return results

    def _log_findings(self, topic_info: pd.DataFrame, merged: pd.DataFrame):
        """Log key findings."""
        n_topics = len(topic_info[topic_info["Topic"] != -1])
        outliers = (merged["topic"] == -1).sum()
        outlier_pct = outliers / len(merged) * 100

        # Top 5 topics
        top_topics = topic_info[topic_info["Topic"] != -1].nlargest(5, "Count")
        top_names = top_topics["Name"].tolist()

        self.log_decision(
            decision=f"Identified {n_topics} topics",
            rationale=(
                f"Outliers: {outliers:,} ({outlier_pct:.1f}%). "
                f"Top topics: {', '.join(top_names[:3])}"
            ),
        )


def run_phase():
    """Entry point for running this phase."""
    phase = Phase04Topics()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
