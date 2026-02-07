"""
Pipeline configuration for Moltbook analysis.

Contains paths, API keys, model settings, and processing parameters.
"""

from pathlib import Path
import os

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATE_DIR = DATA_DIR / "state"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
DERIVED_DIR = DATA_DIR / "derived"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# Ensure directories exist
for d in [INTERMEDIATE_DIR, DERIVED_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Intermediate subdirectories
EMBEDDINGS_DIR = INTERMEDIATE_DIR / "embeddings"
LINGUISTIC_DIR = INTERMEDIATE_DIR / "linguistic"
TOPICS_DIR = INTERMEDIATE_DIR / "topics"
CLASSIFICATIONS_DIR = INTERMEDIATE_DIR / "classifications"
LLM_ANALYSES_DIR = INTERMEDIATE_DIR / "llm_analyses"

for d in [EMBEDDINGS_DIR, LINGUISTIC_DIR, TOPICS_DIR, CLASSIFICATIONS_DIR, LLM_ANALYSES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Derived subdirectories
SPLITS_DIR = DERIVED_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Output subdirectories
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API Configuration - OpenRouter
# =============================================================================

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system env vars

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    import warnings
    warnings.warn(
        "OPENROUTER_API_KEY not set. Set it in .env file or environment variable. "
        "Embedding and LLM analysis phases will fail without it."
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# =============================================================================
# Embedding Model Configuration
# =============================================================================

EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_RAW_DIM = 4096
EMBEDDING_REDUCED_DIM = 768  # PCA reduction for efficiency
EMBEDDING_BATCH_SIZE = 50  # Posts per API call
EMBEDDING_CHECKPOINT_INTERVAL = 500  # Save checkpoint every N posts
EMBEDDING_RATE_LIMIT_RPM = 100  # Requests per minute

# =============================================================================
# LLM Model Configuration (for classification & nuance extraction)
# =============================================================================

LLM_MODEL = "anthropic/claude-sonnet-4-20250514"  # For classification tasks
LLM_FALLBACK_MODEL = "x-ai/grok-4.1-fast"  # Grok 4.1 fast via OpenRouter (cost-effective)
LLM_BATCH_SIZE = 10  # Concurrent requests
LLM_RATE_LIMIT_RPM = 60  # Requests per minute

# Thresholds for LLM analysis
LONG_POST_WORD_COUNT_THRESHOLD = 200  # Posts with >= 200 words get cheaper model analysis
LONG_THREAD_DEPTH_THRESHOLD = 3  # Threads with depth >= 3 get LLM analysis
MAX_LLM_ANALYSIS_POSTS = 5000  # Cap on posts for expensive LLM analysis
MAX_LLM_ANALYSIS_THREADS = 2000  # Cap on threads for LLM thread analysis

# =============================================================================
# Processing Parameters
# =============================================================================

# Temporal analysis
HEARTBEAT_BIN_SIZE_SECONDS = 60  # Bin size for inter-event time histograms
AUTONOMY_REGULARITY_THRESHOLD = 0.8  # CoV threshold for HIGH_HUMAN_INFLUENCE

# Topic modeling
TOPIC_MIN_CLUSTER_SIZE = 50
TOPIC_NEW_DATA_REFIT_THRESHOLD = 0.05  # Refit if >5% new data

# Breach timestamp (UTC)
BREACH_TIMESTAMP = "2026-01-31T17:35:00Z"

# Platform phases
PHASE_BOUNDARIES = {
    "genesis": ("2026-01-27T00:00:00Z", "2026-01-29T23:59:59Z"),
    "growth": ("2026-01-30T00:00:00Z", "2026-01-30T23:59:59Z"),
    "breach": ("2026-01-31T00:00:00Z", "2026-02-01T17:35:00Z"),
    "shutdown": ("2026-02-01T17:35:01Z", "2026-02-03T13:25:00Z"),
    "restoration": ("2026-02-03T13:25:01Z", None),  # None = ongoing
}

# =============================================================================
# Output Formats
# =============================================================================

FIGURE_DPI = 300
FIGURE_FORMAT = "png"
TABLE_FORMAT = "latex"

# =============================================================================
# State & Logging
# =============================================================================

PIPELINE_STATE_FILE = STATE_DIR / "pipeline_state.json"
DECISION_LOG_FILE = LOGS_DIR / "decision_log.md"

# Initialize decision log if not exists
if not DECISION_LOG_FILE.exists():
    DECISION_LOG_FILE.write_text("""# Analysis Decision Log

This log tracks key methodological decisions made during the Moltbook analysis pipeline.
Each entry documents the decision, rationale, and timestamp for transparency.

---

## Log Entries

""")
