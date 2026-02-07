# Data

Raw data (~565 MB) available via GitHub Releases.

```bash
gh release download v1.0 --repo ln9527/moltbook-research --dir data/raw/
```

| File | Records | Description |
|------|---------|-------------|
| `posts_master.json` | 91,792 | Posts with metadata, timestamps, content |
| `comments_master.json` | 405,707 | Comments with threading and timestamps |
| `owners_master.json` | 22,020 | Agent owner profiles |

Collection period: Jan 27 -- Feb 5, 2026. See `docs/data_collection_log.md`.
