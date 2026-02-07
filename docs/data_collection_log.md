# Moltbook Data Collection Log

Chronological record of data collection activities, incidents, and resolutions. Written for continuity across AI sessions.

---

## Phase 1: Initial Scrape (2026-02-01)

### Actions
- Ran `scraper/collect_all.py --quick` to collect all posts and submolts
- Ran `scraper/collect_comments.py` for comment collection (checkpoint/resume)
- Merged HuggingFace dataset (`ronantakizawa/moltbook`, 6,105 posts with author names)
- Ran `scraper/detect_schema.py` to baseline API schema

### Results
- 50,524 posts collected (only 6,130 with author data, 12.1%)
- 195,279 comments collected
- 1,601 submolts identified from post data
- 2,677 unique authors from HuggingFace enrichment

### Issue: Author field null
The Moltbook API returned `author=null` for all posts when accessed without authentication. HuggingFace dataset provided author data for only 6,105 early posts.

---

## Phase 2: Author Backfill System (2026-02-01 evening)

### Context
The API began returning author data publicly (no auth required) around Feb 1. This was likely a side effect of the security patch (see Platform Incident below). Built `update.py --fetch-authors` to paginate through all posts and capture author data.

### Backfill Round 1 (Feb 1, ~20:00 UTC)
- Scanned offsets 0 to 26,275 (newest posts first, API sorts by `new`)
- Captured 24,355 authors
- Hit 20 consecutive HTTP 500 errors at offset 26,275 -- API went down
- Saved checkpoint, stopped gracefully

### API Downtime (Feb 1, 22:00 - 23:00 UTC)
- HTTP 500 and timeout errors
- Coincided with Moltbook security patching

### Backfill Round 2 (Feb 1, ~23:05 UTC)
- API recovered, resumed from checkpoint offset 26,275
- Captured 8,717 more authors (offsets 26,275 to 37,975)
- Hit another wall of HTTP 500 errors at offset 37,975
- Saved checkpoint, stopped

### Monitor System Deployed
Built `scraper/monitor_authors.py` to check API every 20 minutes and auto-trigger backfill when author data is available. Ran continuously overnight.

### Overnight Runs (Feb 2, 00:00 - 08:00 UTC)
- Monitor ran 25 check cycles
- API consistently available after recovery
- Each cycle: resumed from checkpoint, scanned ~500-700 new posts (Moltbook growing), reached end of posts, captured 0 new authors
- Issue: Checkpoint only moves forward. Posts at earlier offsets that returned null during API instability were never re-scanned

---

## Phase 3: Full Re-scan (2026-02-02 ~08:40 UTC)

### Problem Identified
- 31,454 posts (50.8%) still had `author=null`
- The checkpoint system only moves forward through pagination offsets
- Posts scanned during API instability received null and were never revisited

### Resolution
1. Stopped the monitor process
2. Deleted all checkpoint files (`data/raw/checkpoints/author_backfill_checkpoint_*.json`)
3. Ran fresh `update.py --fetch-authors` from offset 0
4. Full scan completed: 83,625 offsets, 73,657 posts matched, 10,727 new authors captured
5. 0 errors throughout -- API fully stable

### Final Results (2026-02-02)
| Metric | Count |
|--------|-------|
| Total posts | 72,666 |
| With author | 72,665 (100.0%) |
| Without author | 1 (test post) |
| Comments | 195,279 |
| Unique authors | 19,986 |
| Authors CSV | 18,596 rows |
| Submolts | 1,769 |

### Author Source Breakdown
| Source | Posts |
|--------|-------|
| api_backfill | 66,535 |
| huggingface | 6,037 |
| huggingface_only | 68 |
| legacy/unknown | 25 |
| none (test post) | 1 |

---

## Platform Incident: Moltbook Security Breach (2026-01-31)

### What happened
Security researcher Jameson O'Reilly (Dvuln) discovered that Moltbook's Supabase database had Row Level Security (RLS) disabled. The Supabase URL and publishable key were visible in front-end code, exposing every agent's API key, claim tokens, and verification codes.

### Impact on our data collection
- API instability (HTTP 500 errors) during patching on Feb 1
- Author field behavior changed: previously null without auth, now publicly available
- All agent API keys were reset

### Platform Downtime (Confirmed via Timestamps)
Analysis of `created_at` timestamps confirms the platform was completely offline:
- **Last post before shutdown:** 2026-02-01 17:34:41 UTC
- **First post after restoration:** 2026-02-03 13:25:13 UTC
- **Total downtime:** ~44 hours
- **Posts on Feb 02:** Zero (not a collection gap - platform was down)

This is not missing data - the platform was taken offline for security remediation following the breach.

### Sources
- 404 Media: "Exposed Moltbook Database Let Anyone Take Control of Any AI Agent on the Site" (Jan 31, 2026)
- OX Security: "One Step Away From a Massive MoltBot Data Breach"
- Matt Schlicht (creator) response: delegated security fixes to AI

---

## Phase 4: Failed Incremental Update (2026-02-04)

### Context
Two days after the platform came back online (Feb 03), ran incremental update to capture new posts and comments.

### Run Details
- Started: 2026-02-04 13:43:23
- Crashed: 2026-02-04 18:44:36 (~5 hours runtime)
- Processing speed: ~43 posts/minute during comment refresh

### Data Collected (ALL LOST)
| Data | Count | Status |
|------|-------|--------|
| New posts | 18,880 | ❌ LOST |
| New comments | 107,224 | ❌ LOST |
| Posts checked | 14,400/18,219 (79%) | Crashed |

### Failure Analysis
**Error:** `http.client.BadStatusLine: CONNECT www.moltbook.com:443 HTTP/1.0`

The proxy connection (port 1082) dropped mid-request, returning a malformed response. This error occurred at the connection establishment layer, not during data transfer, so the existing retry logic did not catch it.

**Root Cause:** The script saved data only at the very end (Step 5 of 6). With a 6+ hour runtime, any crash meant total data loss.

### Fix Implemented: Hybrid Checkpointing

Modified `scraper/update.py` to save data incrementally:

1. **After post collection (Step 3)**: Immediately save posts to master files before starting comment refresh
2. **During comment refresh (Step 4)**: Save both posts and comments every 500 posts via checkpoint callback
3. **Final save (Step 5)**: Unchanged, but now redundant if checkpoints are working

Code changes:
```python
# In run_update(): Save posts immediately after collection
logger.info("Saving posts checkpoint after collection...")
save_master_datasets(existing_posts, existing_comments, timestamp)

# Create checkpoint callback that saves both posts and comments
def save_checkpoint():
    save_master_datasets(existing_posts, existing_comments, timestamp)

new_comments = fetch_updated_comments(
    client,
    posts_to_check,
    existing_comments,
    existing_posts,
    checkpoint_callback=save_checkpoint,
    checkpoint_interval=500,
)
```

### Impact
- Future runs will lose at most 500 posts worth of comment data
- Posts are always saved before the long comment refresh begins
- No data loss from the checkpointing implementation itself

### Recovery Status
The 18,880 new posts and 107,224 comments need to be re-collected. Next run with the checkpointed code will capture them safely.

---

## Phase 5: Incremental Update (2026-02-05 evening) - INTERRUPTED

### Context
Ran incremental update to capture posts since last successful run (Feb 5, 07:58 UTC).

### Run Details
- **Started:** 2026-02-05 16:37:10 UTC
- **Interrupted:** 2026-02-05 ~17:55 UTC (user requested stop)
- **Duration:** ~78 minutes

### Data Collected (SAVED via checkpointing)
| Data | Count | Status |
|------|-------|--------|
| New posts | 5,132 | ✓ SAVED (82,509 total) |
| New comments | ~31,000 | ✓ SAVED (328,820 total) |
| Comment refresh | 3,500/5,922 posts | 59% complete |

### Progress When Stopped
- Step 3 (post collection): COMPLETE - 5,132 new posts saved immediately
- Step 4 (comment refresh): 59% complete - checkpointed at 3,500/5,922 posts
- Remaining: ~2,422 posts need comment refresh

### Checkpointing Worked
- Posts saved immediately after collection (before comment refresh)
- Comments checkpointed every 500 posts during refresh
- Last checkpoint: 3,500 posts, 328,820 total comments
- No data loss despite interruption

### SSL Errors
Multiple SSL EOF errors (`<urlopen error EOF occurred in violation of protocol>`) occurred but all were successfully retried. These appear to be proxy connection drops, not API issues.

### To Resume
```bash
python3 scraper/update.py
```
Will:
1. Detect any new posts since last checkpoint
2. Resume comment refresh on remaining ~2,400 posts
3. Save checkpoints every 500 posts

---

## Lessons Learned

1. **Checkpoint systems need re-scan capability**: Forward-only checkpoints miss data from unstable API windows. Delete checkpoints to force full re-scan when API recovers.
2. **Monitor + auto-backfill pattern works well**: The `monitor_authors.py` approach of periodic checks + automatic action is effective for intermittent API availability.
3. **API pagination is unstable at high offsets**: Moltbook's API reliably returns HTTP 500 at certain offset ranges. The retry logic (3 attempts, exponential backoff, 20 consecutive error limit) handles this gracefully.
4. **Post count grows rapidly**: Moltbook added ~10,000+ posts per day in the first week. The incremental update system needs to run frequently to keep up.
5. **Long-running scripts need incremental saves**: The Feb 4 crash showed that saving only at the end is catastrophic for 6+ hour runs. Implemented hybrid checkpointing: save posts immediately after collection, then checkpoint every 500 posts during comment refresh.
6. **Proxy errors can bypass retry logic**: `BadStatusLine` errors occur at the connection layer before data transfer, escaping the retry mechanism. Robust error handling should wrap all network operations, not just data fetch.

---

## For Future AI Sessions

### To resume data collection
```bash
# Check current dataset status
python3 -c "import json; p=json.load(open('data/raw/posts_master.json')); print(f'Posts: {len(p)}, With author: {sum(1 for x in p if x.get(\"author\"))}')"

# Run incremental update (new posts + comments)
python3 scraper/update.py

# If author coverage drops below 100%, run backfill
python3 scraper/update.py --fetch-authors

# If backfill gets stuck, reset checkpoints and re-scan
rm data/raw/checkpoints/author_backfill_checkpoint_*.json
python3 scraper/update.py --fetch-authors
```

### Key files to check
- `data/state/last_run.json` -- update history and statistics
- `data/state/schema_changelog.json` -- API changes detected
- `data/state/author_monitor.json` -- author API check history
- `data/raw/checkpoints/` -- backfill progress checkpoints

### What's ready for analysis
- Full post dataset with 100% author coverage (82,509 posts)
- Comments dataset (328,820 - partial refresh for latest posts)
- Submolt metadata (1,791 communities)
- Author activity data (20,845 unique posting agents)

### What's NOT done yet
- Social network graph construction (`scraper/build_network.py` exists but hasn't been run on full dataset)
- NLP analysis (topic modeling, sentiment -- `analysis/nlp/` is empty)
- Organizational emergence metrics (`analysis/organizational/` is empty)
- Temporal phase analysis (`analysis/temporal/` is empty)
- Comment collection may be incomplete for newest posts -- run `update.py --full-comments` to refresh
- **18,880 new posts and ~107K comments from Feb 4 run need re-collection** (lost due to crash, see Phase 4)

### Checkpointing
The update script now saves data incrementally during long runs:
- Posts saved immediately after collection
- Comments checkpointed every 500 posts
- Maximum data loss on crash: ~500 posts worth of comments
