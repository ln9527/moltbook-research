"""
OpenRouter Embedding Client

Handles batch embedding generation with:
- Rate limiting
- Checkpointing
- Error handling and retries
- Logging for debugging
"""

import requests
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_RAW_DIM,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_RATE_LIMIT_RPM,
    EMBEDDINGS_DIR,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)


@dataclass
class EmbeddingResult:
    """Result from embedding API call."""
    id: str
    embedding: list[float]
    model: str
    tokens_used: int


class EmbeddingClient:
    """Client for OpenRouter embedding API."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        rate_limit_rpm: int = EMBEDDING_RATE_LIMIT_RPM,
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.rate_limit_rpm = rate_limit_rpm
        self.min_request_interval = 60 / rate_limit_rpm
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def embed_texts(self, texts: list[str], ids: list[str]) -> list[EmbeddingResult]:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed
            ids: Corresponding IDs for each text

        Returns:
            List of EmbeddingResult objects
        """
        if len(texts) != len(ids):
            raise ValueError("texts and ids must have same length")

        results = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]

            self._rate_limit()
            batch_results = self._embed_batch(batch_texts, batch_ids)
            results.extend(batch_results)

            # Progress
            if (batch_end % 100 == 0) or (batch_end == len(texts)):
                print(f"    Embedded {batch_end}/{len(texts)} texts")

        return results

    def _embed_batch(
        self,
        texts: list[str],
        ids: list[str],
        max_retries: int = 3,
    ) -> list[EmbeddingResult]:
        """
        Embed a single batch with retries.

        Handles edge cases:
        - None/empty texts are converted to "(empty)"
        - Very long texts are truncated
        - API errors trigger retries with exponential backoff
        """
        if not texts:
            logger.warning("Empty batch received, skipping")
            return []

        # Clean texts (remove nulls, limit length)
        cleaned_texts = []
        for i, t in enumerate(texts):
            if t is None or t == "":
                t = "(empty)"
                logger.debug(f"Empty text at index {i} (id={ids[i]}), using placeholder")
            # Strip whitespace
            t = t.strip()
            if not t:
                t = "(empty)"
            # Truncate very long texts (API limit ~8k tokens)
            if len(t) > 30000:
                logger.debug(f"Truncating text at index {i} (id={ids[i]}) from {len(t)} to 30000 chars")
                t = t[:30000]
            cleaned_texts.append(t)

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": cleaned_texts,
                        "encoding_format": "float",
                    },
                    timeout=120,
                )

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt * 10
                    logger.warning(f"Rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue

                if response.status_code >= 500:
                    # Server error - retry
                    wait_time = 2 ** attempt * 5
                    logger.warning(f"Server error ({response.status_code}), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                results = []
                embeddings_data = data.get("data", [])

                if len(embeddings_data) != len(texts):
                    logger.warning(f"Mismatch: sent {len(texts)} texts, received {len(embeddings_data)} embeddings")

                for i, emb_data in enumerate(embeddings_data):
                    if i >= len(ids):
                        logger.warning(f"Extra embedding at index {i}, skipping")
                        break
                    embedding = emb_data.get("embedding", [])
                    if len(embedding) != EMBEDDING_RAW_DIM:
                        logger.warning(f"Unexpected embedding dim {len(embedding)} for id={ids[i]}, expected {EMBEDDING_RAW_DIM}")
                    results.append(EmbeddingResult(
                        id=ids[i],
                        embedding=embedding,
                        model=data.get("model", self.model),
                        tokens_used=data.get("usage", {}).get("total_tokens", 0) // max(len(texts), 1),
                    ))

                return results

            except requests.Timeout:
                wait_time = 2 ** attempt * 5
                logger.warning(f"Request timeout, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Request error: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        logger.error(f"Failed to embed batch after {max_retries} attempts")
        return []

    def embed_with_checkpoint(
        self,
        texts: list[str],
        ids: list[str],
        checkpoint_file: Path,
        checkpoint_interval: int = 500,
    ) -> np.ndarray:
        """
        Embed texts with periodic checkpointing.

        Args:
            texts: Texts to embed
            ids: IDs for each text
            checkpoint_file: Path to save/load checkpoints
            checkpoint_interval: Save checkpoint every N embeddings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if len(texts) != len(ids):
            raise ValueError(f"texts ({len(texts)}) and ids ({len(ids)}) must have same length")

        if not texts:
            logger.warning("No texts provided for embedding")
            return np.array([]).reshape(0, EMBEDDING_RAW_DIM)

        # Load existing checkpoint
        processed_ids = set()
        embeddings_dict = {}

        if checkpoint_file.exists():
            try:
                checkpoint = np.load(checkpoint_file, allow_pickle=True)
                processed_ids = set(checkpoint["ids"])
                for i, id_ in enumerate(checkpoint["ids"]):
                    embeddings_dict[id_] = checkpoint["embeddings"][i]
                logger.info(f"Loaded checkpoint with {len(processed_ids)} embeddings")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
                logger.info("Starting fresh embedding process")

        # Find unprocessed texts
        unprocessed = [
            (text, id_)
            for text, id_ in zip(texts, ids)
            if id_ not in processed_ids
        ]

        if not unprocessed:
            logger.info("All texts already embedded")
        else:
            logger.info(f"Embedding {len(unprocessed)} new texts...")

            unprocessed_texts, unprocessed_ids = zip(*unprocessed)
            unprocessed_texts = list(unprocessed_texts)
            unprocessed_ids = list(unprocessed_ids)

            batch_count = 0
            failed_batches = 0
            for batch_start in range(0, len(unprocessed_texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(unprocessed_texts))
                batch_texts = unprocessed_texts[batch_start:batch_end]
                batch_ids = unprocessed_ids[batch_start:batch_end]

                self._rate_limit()

                try:
                    results = self._embed_batch(batch_texts, batch_ids)

                    for result in results:
                        embeddings_dict[result.id] = result.embedding
                        processed_ids.add(result.id)

                    batch_count += len(batch_ids)
                except Exception as e:
                    failed_batches += 1
                    logger.error(f"Failed to embed batch {batch_start}-{batch_end}: {e}")
                    # Fill with zeros for failed batch
                    for bid in batch_ids:
                        if bid not in embeddings_dict:
                            embeddings_dict[bid] = [0.0] * EMBEDDING_RAW_DIM
                            processed_ids.add(bid)
                    continue

                # Checkpoint
                if batch_count % checkpoint_interval == 0 or batch_end == len(unprocessed_texts):
                    self._save_checkpoint(checkpoint_file, embeddings_dict)
                    logger.info(f"Checkpointed at {len(processed_ids)}/{len(ids)} embeddings")

            if failed_batches > 0:
                logger.warning(f"Completed with {failed_batches} failed batches (filled with zeros)")

        # Reconstruct in original order
        embeddings = []
        missing_count = 0
        for id_ in ids:
            if id_ in embeddings_dict:
                embeddings.append(embeddings_dict[id_])
            else:
                # Should not happen if all processed, but handle gracefully
                missing_count += 1
                embeddings.append([0.0] * EMBEDDING_RAW_DIM)

        if missing_count > 0:
            logger.warning(f"{missing_count} IDs missing from embeddings, filled with zeros")

        return np.array(embeddings)

    def _save_checkpoint(self, checkpoint_file: Path, embeddings_dict: dict):
        """Save checkpoint to disk."""
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[id_] for id_ in ids])
        np.savez_compressed(
            checkpoint_file,
            ids=np.array(ids),
            embeddings=embeddings,
        )
