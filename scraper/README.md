# Scraper

Moltbook API data collection. 14 modules for posts, comments, owners, submolts, and network construction.

Entry points: `collect_all.py` (full), `update.py` (incremental). Config in `config.py`.

Rate-limited at 100 req/60s with automatic backoff. Optional `MOLTBOOK_API_KEY` env var.
