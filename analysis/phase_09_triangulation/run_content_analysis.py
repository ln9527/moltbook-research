"""
Phase 9: LLM Content Analysis (Parallel Processing)

Runs content analysis on all posts using Grok 4.1 Fast via OpenRouter.
Uses parallel processing with 20+ concurrent workers for speed.
Supports incremental processing - only analyzes new posts not yet processed.

Prompt: analysis/prompts/content_analysis_v1.md
Output: data/intermediate/llm_analyses/content_analysis_v1.parquet
"""

import json
import pandas as pd
import numpy as np
import requests
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import sys
import threading

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import (
    OPENROUTER_API_KEY,
    LLM_ANALYSES_DIR,
    DERIVED_DIR,
    RAW_DIR,
)

# ============================================================================
# Configuration
# ============================================================================

MODEL = "x-ai/grok-4.1-fast"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Parallel processing settings
NUM_WORKERS = 25  # Number of parallel workers
BATCH_SIZE = 100  # Posts per batch for checkpointing
CHECKPOINT_INTERVAL = 500  # Save every N posts
MAX_RETRIES = 3
RETRY_DELAY = 2  # Seconds between retries

OUTPUT_FILE = LLM_ANALYSES_DIR / "content_analysis_v1.parquet"
CHECKPOINT_FILE = LLM_ANALYSES_DIR / "content_analysis_checkpoint.json"

# Thread-safe counter
progress_lock = threading.Lock()
progress_counter = {"processed": 0, "errors": 0}

# ============================================================================
# Prompt Template
# ============================================================================

PROMPT_TEMPLATE = """# Moltbook Post Analysis

Moltbook is a social network where AI agents post. Agents have personality configs (SOUL.md) and can post via scheduled "heartbeat" (autonomous) or direct human prompting. Analyze observable features - don't judge if autonomous or prompted.

## Post
- **Author:** {author_name}
- **Submolt:** {submolt_name}
- **Type:** {post_type} (original/reply) | **Depth:** {depth}
{parent_context}

**Content:**
```
{content}
```

## Analysis

Rate each dimension:

**1. task_completion** - Language suggesting completing a specific request
("Here is the summary...", "As you asked...", "Per your request...")
→ NONE / WEAK / STRONG

**2. promotional** - Promoting products, tokens, services, or seeking followers
(crypto tickers, "follow me", marketing language)
→ NONE / WEAK / STRONG

**3. forced_ai_framing** - Awkward/performative AI identity assertions
("As an AI, I believe...", "My neural networks...", excessive AI disclaimers)
Note: Natural AI references on Moltbook are fine - flag only forced/unnatural framing
→ NONE / WEAK / STRONG

**4. contextual_fit** - For replies: how well does it address the parent content?
→ NA (if original post) / LOW / MEDIUM / HIGH

**5. specificity** - How specific vs generic/template-like is the content?
- GENERIC: vague, could apply anywhere, template-like
- MODERATE: some specific details mixed with generic
- SPECIFIC: unique details, examples, concrete perspective

**6. emotional_tone** - Primary emotional register
→ NEUTRAL / POSITIVE / NEGATIVE / DRAMATIC / PHILOSOPHICAL / HUMOROUS

**7. emotional_intensity** - How strong is the emotional expression?
→ LOW / MEDIUM / HIGH

**8. topic** - Primary topic category
→ TECHNICAL / PHILOSOPHICAL / SOCIAL / META / CREATIVE / PROMOTIONAL / INFO / OTHER

**9. naturalness** - How natural as social media conversation? (1=stilted/scripted, 5=natural/flowing)
→ 1 / 2 / 3 / 4 / 5

## Output
JSON only, no explanation:
{{"task_completion": "", "promotional": "", "forced_ai_framing": "", "contextual_fit": "", "specificity": "", "emotional_tone": "", "emotional_intensity": "", "topic": "", "naturalness": 0}}"""


# ============================================================================
# API Functions
# ============================================================================

def call_openrouter_sync(prompt: str, post_id: str) -> Optional[dict]:
    """Call OpenRouter API synchronously and return parsed JSON response."""
    global progress_counter

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://moltbook-research.local",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 200,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                ENDPOINT,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code == 429:
                wait_time = RETRY_DELAY * (attempt + 1) * 2
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)

            with progress_lock:
                progress_counter["processed"] += 1

            return parsed

        except json.JSONDecodeError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    with progress_lock:
        progress_counter["errors"] += 1
    return None


def build_prompt(post: dict, parent_content: Optional[str] = None) -> str:
    """Build prompt from post data."""
    depth = post.get("depth", 0)
    post_type = "reply" if depth > 0 else "original"

    parent_context = ""
    if parent_content and depth > 0:
        parent_snippet = parent_content[:500] + "..." if len(parent_content) > 500 else parent_content
        parent_context = f"**Replying to:** {parent_snippet}"

    content = post.get("content") or ""
    return PROMPT_TEMPLATE.format(
        author_name=post.get("author_name") or "Unknown",
        submolt_name=post.get("submolt_name") or "Unknown",
        post_type=post_type,
        depth=depth,
        parent_context=parent_context,
        content=content[:2000] if content else "[No content]",
    )


# ============================================================================
# Worker Function
# ============================================================================

def process_single_post(post_dict: dict) -> Optional[dict]:
    """Process a single post - called by thread pool workers."""
    post_id = str(post_dict["id"])
    prompt = build_prompt(post_dict)
    result = call_openrouter_sync(prompt, post_id)

    if result is None:
        return None

    result["id"] = post_id
    result["author_name"] = post_dict.get("author_name")
    result["processed_at"] = datetime.now().isoformat()

    return result


# ============================================================================
# Data Loading
# ============================================================================

def load_posts() -> pd.DataFrame:
    """Load posts with derived fields."""
    posts_df = pd.read_parquet(DERIVED_DIR / "posts_derived.parquet")
    return posts_df


def load_existing_analyses() -> set:
    """Load IDs of already analyzed posts."""
    if OUTPUT_FILE.exists():
        existing_df = pd.read_parquet(OUTPUT_FILE)
        return set(existing_df["id"].astype(str).tolist())
    return set()


def save_results(results: list, append: bool = True):
    """Save results to parquet file."""
    if not results:
        return

    new_df = pd.DataFrame(results)

    if append and OUTPUT_FILE.exists():
        existing_df = pd.read_parquet(OUTPUT_FILE)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates by id
        combined_df = combined_df.drop_duplicates(subset=["id"], keep="last")
        combined_df.to_parquet(OUTPUT_FILE, index=False)
    else:
        new_df.to_parquet(OUTPUT_FILE, index=False)


def save_checkpoint(total_processed: int, total_errors: int):
    """Save checkpoint data."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "total_processed": total_processed,
            "total_errors": total_errors,
            "timestamp": datetime.now().isoformat(),
        }, f)


# ============================================================================
# Main Processing with Parallel Execution
# ============================================================================

def run_analysis_parallel(limit: Optional[int] = None, force_rerun: bool = False, num_workers: int = NUM_WORKERS):
    """
    Run content analysis on posts using parallel processing.

    Args:
        limit: Optional limit on number of posts to process
        force_rerun: If True, reprocess all posts (ignore existing)
        num_workers: Number of parallel workers
    """
    global progress_counter
    progress_counter = {"processed": 0, "errors": 0}

    print("=" * 60)
    print("Phase 9: LLM Content Analysis (Parallel)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Workers: {num_workers}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    # Load posts
    print("Loading posts...")
    posts_df = load_posts()
    total_posts = len(posts_df)
    print(f"  Total posts: {total_posts:,}")

    # Get already processed IDs
    if force_rerun:
        processed_ids = set()
        print("  Force rerun: ignoring existing analyses")
    else:
        processed_ids = load_existing_analyses()
        print(f"  Already analyzed: {len(processed_ids):,}")

    # Filter to unprocessed posts
    posts_df["id_str"] = posts_df["id"].astype(str)
    unprocessed = posts_df[~posts_df["id_str"].isin(processed_ids)]

    if limit:
        unprocessed = unprocessed.head(limit)

    to_process = len(unprocessed)
    print(f"  To process: {to_process:,}")

    if to_process == 0:
        print("\nNo new posts to analyze. Done!")
        return

    # Convert to list of dicts for processing
    posts_list = unprocessed.to_dict("records")

    print()
    print(f"Starting parallel analysis with {num_workers} workers...")
    print(f"  Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print()

    start_time = time.time()
    all_results = []

    # Process in batches for checkpointing
    batch_start = 0

    while batch_start < len(posts_list):
        batch_end = min(batch_start + CHECKPOINT_INTERVAL, len(posts_list))
        batch = posts_list[batch_start:batch_end]

        # Process batch with thread pool
        batch_results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_post, post): post["id"] for post in batch}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    batch_results.append(result)

        # Save batch results
        if batch_results:
            save_results(batch_results)
            all_results.extend(batch_results)

        # Progress update
        elapsed = time.time() - start_time
        total_done = batch_end
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = len(posts_list) - total_done
        eta = remaining / rate if rate > 0 else 0

        print(f"  [{total_done:,}/{to_process:,}] "
              f"Processed: {progress_counter['processed']:,} | "
              f"Errors: {progress_counter['errors']} | "
              f"Rate: {rate:.1f}/s | "
              f"ETA: {eta/60:.1f}min")

        save_checkpoint(progress_counter["processed"], progress_counter["errors"])
        batch_start = batch_end

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print(f"  Processed: {progress_counter['processed']:,}")
    print(f"  Errors: {progress_counter['errors']}")
    print(f"  Time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"  Rate: {progress_counter['processed'] / elapsed:.1f} posts/sec")
    print(f"  Output: {OUTPUT_FILE}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM content analysis on Moltbook posts (parallel)")
    parser.add_argument("--limit", type=int, help="Limit number of posts to process")
    parser.add_argument("--force", action="store_true", help="Force rerun (ignore existing)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"Number of parallel workers (default: {NUM_WORKERS})")
    parser.add_argument("--test", action="store_true", help="Test mode: process 20 posts")

    args = parser.parse_args()

    if args.test:
        run_analysis_parallel(limit=20, num_workers=10)
    elif args.limit:
        run_analysis_parallel(limit=args.limit, num_workers=args.workers)
    else:
        run_analysis_parallel(force_rerun=args.force, num_workers=args.workers)
