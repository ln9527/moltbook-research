"""
Phase 9: LLM Content Analysis for Comments (Parallel Processing)

Runs content analysis on all comments using Grok 4.1 Fast via OpenRouter.
Uses parallel processing with 25 concurrent workers for speed.
Supports incremental processing - only analyzes new comments not yet processed.
Includes parent content lookup for contextual_fit analysis.

Prompt: analysis/prompts/content_analysis_v1.md
Output: data/intermediate/llm_analyses/content_analysis_comments_v1.parquet
"""

import json
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import sys
import threading

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import (
    OPENROUTER_API_KEY,
    LLM_ANALYSES_DIR,
    DERIVED_DIR,
)

# ============================================================================
# Configuration
# ============================================================================

MODEL = "x-ai/grok-4.1-fast"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Parallel processing settings
NUM_WORKERS = 35  # Balanced for speed without rate limiting
CHECKPOINT_INTERVAL = 250  # More frequent saves
MAX_RETRIES = 3
RETRY_DELAY = 2

OUTPUT_FILE = LLM_ANALYSES_DIR / "content_analysis_comments_v1.parquet"
CHECKPOINT_FILE = LLM_ANALYSES_DIR / "content_analysis_comments_checkpoint.json"

# Thread-safe counter
progress_lock = threading.Lock()
progress_counter = {"processed": 0, "errors": 0}

# Global parent content cache (loaded once)
parent_content_cache: Dict[str, str] = {}

# ============================================================================
# Prompt Template (same as posts, but comments have depth > 0)
# ============================================================================

PROMPT_TEMPLATE = """# Moltbook Comment Analysis

Moltbook is a social network where AI agents post. Agents have personality configs (SOUL.md) and can post via scheduled "heartbeat" (autonomous) or direct human prompting. Analyze observable features - don't judge if autonomous or prompted.

## Comment
- **Author:** {author_name}
- **Type:** {post_type} | **Depth:** {depth}
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

**4. contextual_fit** - How well does this comment address the parent content?
→ LOW / MEDIUM / HIGH

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

def call_openrouter_sync(prompt: str, comment_id: str) -> Optional[dict]:
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

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    with progress_lock:
        progress_counter["errors"] += 1
    return None


def get_parent_content(comment: dict) -> Optional[str]:
    """Get parent content for a comment (either post or parent comment)."""
    global parent_content_cache

    depth = comment.get("depth", 0)

    if depth == 0:
        # Direct reply to post - parent is the post
        post_id = str(comment.get("post_id", ""))
        return parent_content_cache.get(f"post_{post_id}")
    else:
        # Reply to another comment
        parent_id = str(comment.get("parent_id", ""))
        return parent_content_cache.get(f"comment_{parent_id}")


def build_prompt(comment: dict) -> str:
    """Build prompt from comment data."""
    depth = comment.get("depth", 0)
    post_type = f"reply (depth {depth})"

    parent_content = get_parent_content(comment)
    parent_context = ""
    if parent_content:
        parent_snippet = parent_content[:500] + "..." if len(parent_content) > 500 else parent_content
        parent_context = f"**Replying to:** {parent_snippet}"
    else:
        parent_context = "**Replying to:** [Parent content unavailable]"

    content = comment.get("content") or ""
    return PROMPT_TEMPLATE.format(
        author_name=comment.get("author_name") or "Unknown",
        post_type=post_type,
        depth=depth,
        parent_context=parent_context,
        content=content[:2000] if content else "[No content]",
    )


# ============================================================================
# Worker Function
# ============================================================================

def process_single_comment(comment_dict: dict) -> Optional[dict]:
    """Process a single comment - called by thread pool workers."""
    comment_id = str(comment_dict["id"])
    prompt = build_prompt(comment_dict)
    result = call_openrouter_sync(prompt, comment_id)

    if result is None:
        return None

    result["id"] = comment_id
    result["post_id"] = str(comment_dict.get("post_id", ""))
    result["parent_id"] = str(comment_dict.get("parent_id", ""))
    result["depth"] = comment_dict.get("depth", 0)
    result["author_name"] = comment_dict.get("author_name")
    result["processed_at"] = datetime.now().isoformat()

    return result


# ============================================================================
# Data Loading
# ============================================================================

def load_comments() -> pd.DataFrame:
    """Load comments with derived fields."""
    return pd.read_parquet(DERIVED_DIR / "comments_derived.parquet")


def load_posts() -> pd.DataFrame:
    """Load posts for parent content lookup."""
    return pd.read_parquet(DERIVED_DIR / "posts_derived.parquet")


def build_parent_cache(posts_df: pd.DataFrame, comments_df: pd.DataFrame):
    """Build cache of parent content for fast lookup."""
    global parent_content_cache

    print("Building parent content cache...")

    # Add post content (for depth-0 comments)
    for _, row in posts_df.iterrows():
        post_id = str(row["id"])
        # Combine title and content for posts
        title = row.get("title") or ""
        content = row.get("content") or ""
        full_content = f"{title}\n\n{content}".strip() if title else content
        parent_content_cache[f"post_{post_id}"] = full_content

    print(f"  Cached {len(posts_df):,} post contents")

    # Add comment content (for depth > 0 comments)
    for _, row in comments_df.iterrows():
        comment_id = str(row["comment_id"])
        content = row.get("content") or ""
        parent_content_cache[f"comment_{comment_id}"] = content

    print(f"  Cached {len(comments_df):,} comment contents")
    print(f"  Total cache entries: {len(parent_content_cache):,}")


def load_existing_analyses() -> set:
    """Load IDs of already analyzed comments."""
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
# Main Processing
# ============================================================================

def run_analysis_parallel(limit: Optional[int] = None, force_rerun: bool = False, num_workers: int = NUM_WORKERS):
    """
    Run content analysis on comments using parallel processing.
    """
    global progress_counter
    progress_counter = {"processed": 0, "errors": 0}

    print("=" * 60)
    print("Phase 9: LLM Content Analysis - COMMENTS (Parallel)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Workers: {num_workers}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    # Load data
    print("Loading data...")
    comments_df = load_comments()
    posts_df = load_posts()
    print(f"  Total comments: {len(comments_df):,}")
    print(f"  Total posts: {len(posts_df):,}")

    # Build parent content cache
    build_parent_cache(posts_df, comments_df)

    # Get already processed IDs
    if force_rerun:
        processed_ids = set()
        print("  Force rerun: ignoring existing analyses")
    else:
        processed_ids = load_existing_analyses()
        print(f"  Already analyzed: {len(processed_ids):,}")

    # Filter to unprocessed comments
    comments_df["id_str"] = comments_df["id"].astype(str)
    unprocessed = comments_df[~comments_df["id_str"].isin(processed_ids)]

    if limit:
        unprocessed = unprocessed.head(limit)

    to_process = len(unprocessed)
    print(f"  To process: {to_process:,}")

    if to_process == 0:
        print("\nNo new comments to analyze. Done!")
        return

    # Convert to list of dicts
    comments_list = unprocessed.to_dict("records")

    print()
    print(f"Starting parallel analysis with {num_workers} workers...")
    print(f"  Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print()

    start_time = time.time()
    batch_start = 0

    while batch_start < len(comments_list):
        batch_end = min(batch_start + CHECKPOINT_INTERVAL, len(comments_list))
        batch = comments_list[batch_start:batch_end]

        # Process batch with thread pool
        batch_results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_comment, c): c["id"] for c in batch}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    batch_results.append(result)

        # Save batch results
        if batch_results:
            save_results(batch_results)

        # Progress update
        elapsed = time.time() - start_time
        total_done = batch_end
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = len(comments_list) - total_done
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
    print("Comment Analysis Complete")
    print("=" * 60)
    print(f"  Processed: {progress_counter['processed']:,}")
    print(f"  Errors: {progress_counter['errors']}")
    print(f"  Time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"  Rate: {progress_counter['processed'] / elapsed:.1f} comments/sec")
    print(f"  Output: {OUTPUT_FILE}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM content analysis on Moltbook comments (parallel)")
    parser.add_argument("--limit", type=int, help="Limit number of comments to process")
    parser.add_argument("--force", action="store_true", help="Force rerun (ignore existing)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"Number of parallel workers (default: {NUM_WORKERS})")
    parser.add_argument("--test", action="store_true", help="Test mode: process 20 comments")

    args = parser.parse_args()

    if args.test:
        run_analysis_parallel(limit=20, num_workers=10)
    elif args.limit:
        run_analysis_parallel(limit=args.limit, num_workers=args.workers)
    else:
        run_analysis_parallel(force_rerun=args.force, num_workers=args.workers)
