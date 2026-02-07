"""
Compute embeddings for comments.

Uses the same embedding model as posts (qwen/qwen3-embedding-8b).
Applies the existing PCA model (fitted on posts) to reduce to 768 dims.
Supports incremental processing with checkpointing.

Output: data/intermediate/embeddings/comments_embeddings.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import (
    EMBEDDINGS_DIR,
    DERIVED_DIR,
    EMBEDDING_RAW_DIM,
    EMBEDDING_REDUCED_DIM,
    EMBEDDING_CHECKPOINT_INTERVAL,
)
from analysis.phase_02_linguistic.embedding_client import EmbeddingClient


def compute_comment_embeddings():
    """Compute embeddings for all comments."""
    print("=" * 60)
    print("Computing Comment Embeddings")
    print("=" * 60)

    # Load comments
    print("\nLoading comments...")
    comments_df = pd.read_parquet(DERIVED_DIR / "comments_derived.parquet")
    print(f"  Total comments: {len(comments_df):,}")

    # Prepare texts
    texts = []
    ids = []
    for _, row in comments_df.iterrows():
        content = row.get("content", "") or ""
        texts.append(content if content else "(empty comment)")
        ids.append(str(row["id"]))

    print(f"  Prepared {len(texts):,} texts for embedding")

    # File paths
    raw_checkpoint = EMBEDDINGS_DIR / "comments_raw_checkpoint.npz"
    pca_model_path = EMBEDDINGS_DIR / "pca_model.joblib"
    output_path = EMBEDDINGS_DIR / "comments_embeddings.parquet"

    # Check for existing PCA model
    if not pca_model_path.exists():
        print("\nERROR: PCA model not found. Run posts embeddings first.")
        return

    print(f"\n  PCA model: {pca_model_path}")
    print(f"  Checkpoint: {raw_checkpoint}")
    print(f"  Output: {output_path}")

    # Compute raw embeddings with checkpointing
    print(f"\nComputing embeddings (checkpoint every {EMBEDDING_CHECKPOINT_INTERVAL})...")
    client = EmbeddingClient()
    raw_embeddings = client.embed_with_checkpoint(
        texts=texts,
        ids=ids,
        checkpoint_file=raw_checkpoint,
        checkpoint_interval=EMBEDDING_CHECKPOINT_INTERVAL,
    )

    print(f"\n  Raw embeddings shape: {raw_embeddings.shape}")

    # Apply PCA
    print(f"\nApplying PCA ({EMBEDDING_RAW_DIM} -> {EMBEDDING_REDUCED_DIM} dims)...")
    pca = joblib.load(pca_model_path)
    reduced_embeddings = pca.transform(raw_embeddings)
    print(f"  Reduced embeddings shape: {reduced_embeddings.shape}")

    # Save as parquet
    print("\nSaving to parquet...")
    emb_df = pd.DataFrame(
        reduced_embeddings,
        columns=[f"emb_{i}" for i in range(EMBEDDING_REDUCED_DIM)],
    )
    emb_df["id"] = ids
    emb_df = emb_df[["id"] + [f"emb_{i}" for i in range(EMBEDDING_REDUCED_DIM)]]

    emb_df.to_parquet(output_path, index=False)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n  Saved {len(emb_df):,} comment embeddings")
    print(f"  File size: {file_size:.1f} MB")

    # Verify
    print("\nVerification:")
    print(f"  NaN values: {emb_df.isna().sum().sum()}")
    print(f"  Embedding range: [{reduced_embeddings.min():.3f}, {reduced_embeddings.max():.3f}]")

    print("\n" + "=" * 60)
    print("Comment Embeddings Complete!")
    print("=" * 60)


if __name__ == "__main__":
    compute_comment_embeddings()
