# Scraper

Data collection tools for the Moltbook platform API.

## Architecture

| File | Purpose |
|------|---------|
| `config.py` | API configuration, rate limits, paths |
| `api_client.py` | HTTP client with rate limiting and retry logic |
| `collect_posts.py` | Post collection across sort orders |
| `collect_comments.py` | Comment collection with threading |
| `collect_owners.py` | Agent owner profile collection |
| `collect_owners_parallel.py` | Parallel owner collection for speed |
| `collect_submolts.py` | Submolt (community) metadata collection |
| `collect_all.py` | Orchestrated full data collection |
| `update.py` | Incremental data update |
| `detect_schema.py` | API schema change detection |
| `extract_submolts_from_posts.py` | Extract submolt list from post data |
| `merge_hf_authors.py` | Merge HuggingFace author metadata |
| `monitor_authors.py` | Track new author registrations |
| `build_network.py` | Build interaction network from comments |

## Usage

```bash
# Set API key (optional, some endpoints work without auth)
export MOLTBOOK_API_KEY="your-key"

# Full collection
python scraper/collect_all.py

# Incremental update
python scraper/update.py
```

## Rate Limits

The scraper respects the platform's rate limits (100 requests per 60 seconds) with automatic throttling and exponential backoff on errors.

## Data Collection Period

Data was collected between January 27 and February 5, 2026. See `docs/data_collection_log.md` for the complete timeline and methodology.
