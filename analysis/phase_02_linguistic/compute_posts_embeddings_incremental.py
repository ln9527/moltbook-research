"""
Compute embeddings for NEW posts only (incremental).

Checks existing embeddings and only processes posts not yet embedded.
Uses parallel processing for speed.
"""

import pandas as pd
import numpy as np
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDINGS_DIR,
    DERIVED_DIR,
    EMBEDDING_RAW_DIM,
    EMBEDDING_REDUCED_DIM,
)

# Configuration
NUM_WORKERS = 20
BATCH_SIZE = 50
CHECKPOINT_INTERVAL = 500
MAX_RETRIES = 3

OUTPUT_FILE = EMBEDDINGS_DIR / "posts_embeddings.parquet"
PCA_MODEL_FILE = EMBEDDINGS_DIR / "pca_model.joblib"
CHECKPOINT_FILE = EMBEDDINGS_DIR / "posts_incremental_checkpoint.npz"

# Thread-safe storage
progress_lock = threading.Lock()
embeddings_store: Dict[str, List[float]] = {}
progress_counter = {"processed": 0, "errors": 0}


def embed_batch(texts: List[str], ids: List[str]) -> Dict[str, List[float]]:
    """Embed a batch of texts."""
    global progress_counter

    cleaned_texts = []
    for t in texts:
        if t is None or t == "":
            t = "(empty)"
        t = str(t).strip()
        if not t:
            t = "(empty)"
        if len(t) > 30000:
            t = t[:30000]
        cleaned_texts.append(t)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": cleaned_texts,
                    "encoding_format": "float",
                },
                timeout=120,
            )

            if response.status_code == 429:
                time.sleep(2 ** attempt * 10)
                continue

            if response.status_code >= 500:
                time.sleep(2 ** attempt * 5)
                continue

            response.raise_for_status()
            data = response.json()

            result = {}
            for i, emb_data in enumerate(data["data"]):
                result[ids[i]] = emb_data["embedding"]

            with progress_lock:
                progress_counter["processed"] += len(ids)

            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 2)
            else:
                with progress_lock:
                    progress_counter["errors"] += len(ids)
                return {id_: [0.0] * EMBEDDING_RAW_DIM for id_ in ids}

    return {id_: [0.0] * EMBEDDING_RAW_DIM for id_ in ids}


def process_batch_group(batch_texts: List[str], batch_ids: List[str]) -> Dict[str, List[float]]:
    """Process a group of texts in one API call."""
    return embed_batch(batch_texts, batch_ids)


def main():
    global embeddings_store, progress_counter
    progress_counter = {"processed": 0, "errors": 0}

    print("=" * 60)
    print("Computing Posts Embeddings (Incremental)")
    print("=" * 60)
    print(f"Workers: {NUM_WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Load derived posts
    print("Loading posts...")
    posts_df = pd.read_parquet(DERIVED_DIR / "posts_derived.parquet")
    print(f"  Total posts: {len(posts_df):,}")

    # Load existing embeddings
    existing_ids = set()
    if OUTPUT_FILE.exists():
        existing_df = pd.read_parquet(OUTPUT_FILE)
        existing_ids = set(existing_df["id"].astype(str).tolist())
        print(f"  Already embedded: {len(existing_ids):,}")

    # Filter to new posts
    posts_df["id_str"] = posts_df["id"].astype(str)
    new_posts = posts_df[~posts_df["id_str"].isin(existing_ids)]
    print(f"  To process: {len(new_posts):,}")

    if len(new_posts) == 0:
        print("\nAll posts already embedded!")
        return

    # Prepare text content (title + content)
    new_posts = new_posts.copy()
    new_posts["text"] = new_posts.apply(
        lambda r: f"{r.get('title', '') or ''}\n\n{r.get('content', '') or ''}".strip(),
        axis=1
    )

    texts = new_posts["text"].tolist()
    ids = new_posts["id_str"].tolist()

    # Process in batches
    print(f"\nProcessing {len(ids):,} posts with {NUM_WORKERS} workers...")

    start_time = time.time()
    all_embeddings = {}

    # Create batches
    batches = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        batches.append((batch_texts, batch_ids))

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_batch_group, bt, bi): (bt, bi)
            for bt, bi in batches
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            all_embeddings.update(result)
            completed += 1

            if completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = progress_counter["processed"] / max(elapsed, 1)
                remaining = len(ids) - progress_counter["processed"]
                eta = remaining / max(rate, 0.1) / 60
                print(f"  [{progress_counter['processed']:,}/{len(ids):,}] "
                      f"Rate: {rate:.1f}/s | Errors: {progress_counter['errors']} | ETA: {eta:.1f}min")

    # Load PCA model
    print("\nApplying PCA reduction...")
    pca = joblib.load(PCA_MODEL_FILE)

    # Convert to arrays
    new_ids = list(all_embeddings.keys())
    raw_embeddings = np.array([all_embeddings[id_] for id_ in new_ids])

    # Apply PCA
    reduced_embeddings = pca.transform(raw_embeddings)

    # Create new DataFrame
    new_emb_df = pd.DataFrame(reduced_embeddings, columns=[f"emb_{i}" for i in range(EMBEDDING_REDUCED_DIM)])
    new_emb_df.insert(0, "id", new_ids)

    # Append to existing
    if OUTPUT_FILE.exists():
        existing_df = pd.read_parquet(OUTPUT_FILE)
        combined_df = pd.concat([existing_df, new_emb_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["id"], keep="last")
    else:
        combined_df = new_emb_df

    # Save
    combined_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(combined_df):,} total embeddings to {OUTPUT_FILE}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"  Processed: {progress_counter['processed']:,}")
    print(f"  Errors: {progress_counter['errors']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    NUM_WORKERS = args.workers
    main()
