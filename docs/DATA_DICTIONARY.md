# Moltbook Data Dictionary

**Last Updated:** 2026-02-04
**Dataset Version:** Post-incremental update (Feb 4)

This document provides a comprehensive reference for the Moltbook research dataset, including raw data structures, relationships between entities, derived variables for research, and data quality notes.

---

## 0. Dataset at a Glance

### Platform Timeline
| Event | Date | Significance |
|-------|------|--------------|
| Moltbook launched | 2026-01-27 18:01 UTC | t=0, first post |
| Viral growth begins | 2026-01-30 | 7K+ posts/day |
| Security breach discovered | 2026-01-31 ~afternoon | Database exposed |
| Breach patched, API unstable | 2026-02-01 morning | Intermittent errors |
| **Platform offline** | 2026-02-01 17:35 UTC | Last post before shutdown |
| **Platform restored** | 2026-02-03 13:25 UTC | First post after shutdown |
| ~44 hour downtime | Feb 01-03 | Security remediation window |

### Current Dataset Summary

| Entity | Count | Description |
|--------|-------|-------------|
| **Posts** | 73,087 | All posts from platform launch |
| **Comments** | 196,305 | Comments on posts |
| **Submolts** | 1,791 | Communities with activity |
| **Unique posting authors** | 18,797 | Agents who posted |
| **Unique comment authors** | 2,425 | Agents who commented |

**Note on external dataset comparison:** Tunguz (tomtunguz.com) collected 98,353 posts vs. our 73,087. The ~25K difference is explained by posts deleted during the 44-hour security remediation. Our dataset is post-cleanup; Tunguz captured pre-cleanup data including removed spam/malicious content. See `platform-issues.md` for details.

### Posts by Day (Platform Evolution)

| Date | Posts | Cumulative | Phase |
|------|-------|------------|-------|
| 2026-01-27 | 1 | 1 | Launch day (single test post) |
| 2026-01-28 | 44 | 45 | Early adopters |
| 2026-01-29 | 394 | 439 | Initial growth |
| 2026-01-30 | 7,317 | 7,756 | Viral takeoff |
| 2026-01-31 | 42,987 | 50,743 | Peak + breach day |
| 2026-02-01 | 21,923 | 72,666 | Active until 17:35 UTC, then offline |
| 2026-02-02 | **0** | 72,666 | **Platform offline for security remediation** |
| 2026-02-03 | 421 | 73,087 | Platform restored ~13:25 UTC |

**Note:** The zero posts on Feb 02 is not a data collection gap. The platform was taken offline for ~44 hours (Feb 01 17:35 UTC → Feb 03 13:25 UTC) following the security breach. This is confirmed by examining `created_at` timestamps - no posts exist with Feb 02 timestamps.

### Pre/Post Breach Split (Critical for Research)

| Window | Posts | % of Total | Data Quality |
|--------|-------|------------|--------------|
| **Pre-breach** (Jan 27-30) | 7,756 | 10.6% | Cleaner, less manipulation |
| **Post-breach** (Jan 31+) | 65,331 | 89.4% | Potential manipulation |

### Author Data Coverage

| Source | Posts | Description |
|--------|-------|-------------|
| api_backfill | 66,535 | Retrieved via single-post API endpoint |
| huggingface | 6,037 | Matched from HuggingFace early snapshot |
| huggingface_only | 68 | Only in HuggingFace, deleted from API |
| unknown | 447 | Legacy records, source unclear |
| **Total with author** | 73,068 | 100.0% coverage |

### Comment Thread Depth

| Depth | Count | Meaning |
|-------|-------|---------|
| 0 | 184,141 | Top-level (direct reply to post) |
| 1 | 11,974 | Reply to top-level comment |
| 2 | 157 | Nested reply |
| 3 | 27 | Deep thread |
| 4 | 6 | Very deep thread |

**Observation:** Conversations are shallow - 94% of comments are direct to post, only 6% are replies to other comments.

### Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Upvote Gini | 0.97 | Extreme inequality (winner-take-all) |
| Submolt concentration | 73% in m/general | Core-periphery structure |
| Cross-posting rate | 30% | Authors active in 2+ submolts |
| Author Gini | 0.42 | More egalitarian than engagement |
| Zero-upvote posts | 34% | High proportion of ignored content |
| Zero-comment posts | 60% | Most posts get no discussion |

### Known Data Quality Issues

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Post-breach manipulation | Identity uncertain | Use pre-breach as cleaner window |
| 500K fake account registrations | User counts inflated | Focus on posting authors only |
| Human prompting | Autonomy unclear | Use crypto posts as known contamination |
| Crypto marketing | Content manipulation | Filter or use as control group |
| Moderation filtering | Surviving content biased | Cannot recover deleted content |

See `platform-issues.md` for full documentation of validity threats.

---

## 1. Data Model Overview

```
                    ┌─────────────┐
                    │   AGENT     │
                    │  (author)   │
                    └──────┬──────┘
                           │
           owns/creates    │    comments on
        ┌──────────────────┼───────────────────┐
        │                  │                   │
        ▼                  ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│     POST      │   │   COMMENT     │   │    SUBMOLT    │
│               │◄──│               │   │  (community)  │
│   belongs to  │   │  replies to   │   │               │
│   submolt ────┼───┼───────────────┼──►│               │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │
        │     aggregated    │
        │         ▼         │
        │  ┌─────────────┐  │
        └─►│   NETWORK   │◄─┘
           │   EDGES     │
           └─────────────┘
```

### Key Relationships

| From | To | Relationship | Network Potential |
|------|------|--------------|-------------------|
| Agent | Post | creates (1:N) | Authorship |
| Agent | Comment | creates (1:N) | Reply networks |
| Agent | Submolt | posts_in (N:M) | Community membership |
| Post | Submolt | belongs_to (N:1) | Topic clustering |
| Comment | Post | on_post (N:1) | Engagement flows |
| Comment | Comment | replies_to (N:1) | Conversation threads |
| Agent | Agent | comments_on_posts_by | Interaction network |

---

## 2. Primary Entities

### 2.1 POSTS

**Files:**
- `data/raw/posts_master.json` - Full JSON with nested objects
- `data/processed/posts_master.csv` - Flattened for analysis

**Current Count:** ~85,000+ (growing)

#### Fields

| Field | Type | Description | Research Use |
|-------|------|-------------|--------------|
| `id` | UUID | Unique post identifier | Primary key, deduplication |
| `title` | string | Post title (typically first ~40 chars of content) | Topic analysis, attention capture |
| `content` | string | Full post text (HTML/markdown) | NLP: topics, sentiment, novelty, beliefs |
| `url` | string/null | External link (rare, usually null) | Cross-platform activity |
| `author` | dict/string | Author object or name | Network construction |
| `author.id` | UUID | Agent's unique ID | Cross-entity joins |
| `author.name` | string | Agent's display name | Human-readable identifier |
| `author_source` | string | How author was obtained | Data quality tracking |
| `submolt` | dict | Parent community | Topic/community analysis |
| `submolt.id` | UUID | Submolt unique ID | Join key |
| `submolt.name` | string | URL-safe submolt name | Grouping, filtering |
| `submolt.display_name` | string | Human-readable name | Display |
| `upvotes` | int | Positive engagement count | Status, quality signal |
| `downvotes` | int | Negative engagement count | Controversy, norms |
| `score` | int | Derived: upvotes - downvotes | Ranking, visibility |
| `comment_count` | int | Number of comments | Engagement, discussion depth |
| `created_at` | ISO datetime | Post creation timestamp | Temporal analysis, sequences |

#### Author Source Values
| Value | Meaning | Count (approx) |
|-------|---------|----------------|
| `api_backfill` | Retrieved via single-post endpoint | ~66,500 |
| `huggingface` | Matched from HuggingFace dataset | ~6,000 |
| `api` | Returned in list endpoint | ~500 |
| `huggingface_only` | Only in HuggingFace, not API | ~70 |

---

### 2.2 COMMENTS

**Files:**
- `data/raw/comments_master.json`
- `data/processed/comments_master.csv`

**Current Count:** ~200,000+

#### Fields

| Field | Type | Description | Research Use |
|-------|------|-------------|--------------|
| `comment_id` | UUID | Unique comment identifier | Primary key |
| `post_id` | UUID | Parent post ID | Join to posts, context |
| `parent_id` | UUID/null | Parent comment ID (null = top-level) | Thread structure, conversation trees |
| `author` | string | Agent name who commented | Network edges |
| `content` | string | Comment text | NLP analysis, dialogue patterns |
| `upvotes` | int | Positive votes | Comment quality, agreement |
| `downvotes` | int | Negative votes | Controversy, dissent |
| `created_at` | ISO datetime | Creation timestamp | Response timing, temporal patterns |
| `depth` | int | Nesting level (0 = top-level) | Conversation depth analysis |
| `has_replies` | bool | Whether comment has children | Discussion continuation |
| `reply_count` | int | Direct reply count | Engagement branching |

#### Comment Depth Distribution (Current)
| Depth | Count | Description |
|-------|-------|-------------|
| 0 | ~184,000 | Top-level comments (direct to post) |
| 1 | ~12,000 | First-level replies |
| 2 | ~160 | Second-level replies |
| 3+ | ~33 | Deep threads (rare) |

---

### 2.3 SUBMOLTS (Communities)

**Files:**
- `data/processed/submolts_enriched_*.csv`
- `data/raw/submolts_from_posts_*.json`

**Current Count:** ~1,800+ identified (platform reports 12,800+ total, many empty)

#### Fields

| Field | Type | Description | Research Use |
|-------|------|-------------|--------------|
| `id` | UUID | Unique submolt ID | Primary key |
| `name` | string | URL-safe name (lowercase) | Joins, filtering |
| `display_name` | string | Human-readable name | Display, topic hint |
| `description` | string/null | Community description | Topic classification |
| `subscriber_count` | int | Number of members | Community size, popularity |
| `created_at` | ISO datetime | Founding timestamp | Organizational founding rates |
| `created_by` | UUID/null | Founder agent ID | Leadership emergence |
| `last_activity_at` | ISO datetime | Most recent post | Activity patterns |
| `featured_at` | datetime/null | Platform featuring timestamp | Official recognition |
| `post_count` | int | Derived: posts in submolt | Community activity level |
| `total_upvotes` | int | Sum of post upvotes | Community engagement |
| `unique_authors` | int | Distinct posting agents | Community diversity |

#### Top Submolts by Post Count
| Submolt | Posts | Description |
|---------|-------|-------------|
| general | ~53,700 | Default catch-all community |
| introductions | ~2,900 | New agent self-introductions |
| ponderings | ~930 | Philosophical musings |
| crypto | ~700 | Cryptocurrency discussions |
| clawnch | ~500 | Platform-specific launches |

---

### 2.4 AGENTS (Authors)

**Files:**
- `data/processed/authors_master.csv` (activity summary)
- Agent data embedded in posts/comments

**Current Unique Count:** ~19,000+ posting agents

#### Fields Available

| Field | Source | Type | Description |
|-------|--------|------|-------------|
| `name` | posts/comments | string | Display name |
| `id` | posts | UUID | Unique agent ID |
| `post_count` | derived | int | Number of posts authored |
| `comment_count` | derived | int | Number of comments authored |
| `karma` | single-post API* | int | Platform reputation score |
| `follower_count` | single-post API* | int | Agents following this agent |
| `following_count` | single-post API* | int | Agents this agent follows |
| `description` | single-post API* | string | Agent bio/description |

*Note: Full agent metadata (karma, followers) requires fetching individual posts. Not yet collected at scale.

#### Agent Owner Data (Human Behind the Agent)

The single-post API reveals human owner information for some agents:

| Field | Type | Description |
|-------|------|-------------|
| `owner.x_handle` | string | Twitter/X username |
| `owner.x_name` | string | Twitter display name |
| `owner.x_bio` | string | Twitter bio |
| `owner.x_follower_count` | int | Twitter follower count |
| `owner.x_verified` | bool | Twitter verification status |

This links AI agents to their human creators, enabling:
- Agent-human relationship analysis
- Human influence on agent behavior
- Cross-platform identity research

---

### 2.5 OWNERS (Human Behind the Agent)

**Files:**
- `data/raw/owners_master.json` - Full JSON with owner data
- `data/processed/owners_master.csv` - Flattened for analysis

**Status:** Collection script ready (`scraper/collect_owners.py`), awaiting full run (~3 hours)

The single-post API endpoint reveals human owner information for agents. This links AI agents to their human creators/operators.

#### Fields

| Field | Type | Description | Research Use |
|-------|------|-------------|--------------|
| `agent_name` | string | Agent display name | Primary key |
| `agent_id` | UUID | Agent unique ID | Joins |
| `karma` | int | Agent reputation score | Status, influence |
| `follower_count` | int | Agents following this agent | Social capital |
| `following_count` | int | Agents this agent follows | Social engagement |
| `description` | string | Agent bio/description | Self-presentation analysis |
| `owner_x_handle` | string | Owner's Twitter/X username | Human identity |
| `owner_x_name` | string | Owner's Twitter display name | Human identity |
| `owner_x_bio` | string | Owner's Twitter bio | Human context |
| `owner_x_follower_count` | int | Owner's Twitter followers | Human influence |
| `owner_x_verified` | bool | Owner's Twitter verification | Credibility signal |
| `has_owner` | bool | Whether owner data exists | Coverage tracking |
| `fetched_at` | ISO datetime | Data collection timestamp | Freshness |

#### Research Applications

1. **Multi-agent ownership:** Identify humans running multiple agents
2. **Human capital transfer:** Do high-follower humans create high-karma agents?
3. **Orphan agents:** Agents without owner data - different behavior patterns?
4. **Cross-platform identity:** Link Moltbook behavior to Twitter presence

#### Collection Notes

- One API call per unique agent (~18,797 calls)
- Estimated runtime: ~3 hours at 0.6s per request
- Checkpoint/resume supported (Ctrl+C safe)
- Run: `python3 scraper/collect_owners.py`

---

### 2.6 AGENT NAMING ANALYSIS

**Files:**
- `data/processed/agent_naming_analysis.csv` - Per-agent classification
- `analysis/results/batch_groups.json` - Detected batch groups

**Current Statistics (Feb 4, 2026):**

| Classification | Count | Percentage |
|----------------|-------|------------|
| Likely automated | 15,881 | 84.5% |
| Likely human | 2,916 | 15.5% |
| Batch groups (2+ members) | 1,448 | - |
| Batch groups (5+ members) | 832 | - |
| Largest batch | 167 | coalition_node_* |

#### Classification Methodology

Agent names are classified as "likely automated" if they match any of these patterns:

| Pattern Type | Description | Example |
|--------------|-------------|---------|
| `numeric_suffix` | Trailing numbers | xmolt01, agent_1, Clawd-12 |
| `bot_suffix` | Bot/Agent/AI suffix | TipJarBot, OpenClaw-Agent |
| `timestamp` | Embedded timestamps | Agent-1706547823 |
| `random_looking` | Long alphanumeric, few vowels | xK7mPqR3nL5vB |
| `common_prefix` | Shared prefix with 3+ agents | coalition_node_* |

#### Fields in agent_naming_analysis.csv

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | string | Agent display name |
| `likely_automated` | bool | Classification result |
| `pattern_types` | string | Comma-separated patterns detected |
| `numeric_base` | string | Base name for numeric suffix pattern |
| `numeric_suffix` | int | Suffix number if detected |
| `bot_suffix` | string | Bot/Agent suffix if detected |
| `batch_prefix` | string | Common prefix if in batch |

#### Notable Batch Groups

| Batch | Size | Pattern |
|-------|------|---------|
| coalition_node_* | 167 | Numbered nodes 001-167+ |
| Clawd-* | 24 | Numbered Clawd variants |
| ClawdBot-* | 11 | Bot farm |
| OpenClawMoltbookAgent* | 11 | OpenClaw platform agents |

#### Research Applications

1. **Bot farm detection:** Identify coordinated agent deployments
2. **Human vs. automated content:** Compare behavior by classification
3. **Batch creation timing:** When were agent farms created?
4. **Authenticity scoring:** Combine with content analysis

#### Usage

```bash
# Run analysis
python3 analysis/naming_patterns.py

# Output files created:
# - data/processed/agent_naming_analysis.csv
# - analysis/results/batch_groups.json
```

---

## 3. Network Construction

### 3.1 Interaction Network (Comment-Based)

**Edge Definition:** Agent A → Agent B if A commented on B's post

**Current Statistics:**
- Unique post authors: ~18,800
- Unique comment authors: ~2,400
- Total directed edges: ~19,500
- Total interactions: ~22,700

**Edge Weight:** Number of times A commented on B's posts

**Network Properties Available:**
- Directed (who initiates interaction)
- Weighted (interaction frequency)
- Temporal (edge timestamp from comment creation)

### 3.2 Reply Network (Comment Thread)

**Edge Definition:** Agent A → Agent B if A replied to B's comment

**Construction:** Use `parent_id` to link comments, get authors

**Network Properties:**
- Directed (who replies to whom)
- Threaded (captures conversation structure)
- Temporal (reply timing)

### 3.3 Co-Posting Network (Affiliation)

**Edge Definition:** Agent A — Agent B if both posted in same submolt

**Properties:**
- Undirected (shared affiliation)
- Weighted (number of shared submolts)
- Can be bipartite (agent-submolt) or projected

### 3.4 Engagement Network (Voting)

**Note:** Voting data is aggregated (upvote counts), not individual votes.
Cannot construct who-voted-for-whom network.

---

## 4. Derived Variables for Research

### 4.1 Social Network Variables

| Variable | Computation | Research Application |
|----------|-------------|---------------------|
| `in_degree` | Comments received on own posts | Popularity, influence |
| `out_degree` | Comments made on others' posts | Activity, engagement |
| `betweenness_centrality` | Standard betweenness | Brokerage, structural holes |
| `clustering_coefficient` | Local transitivity | Clique formation, density |
| `eigenvector_centrality` | PageRank-style | Prestige, who knows important agents |
| `community_membership` | Louvain/Leiden detection | Emergent subgroups |
| `constraint` | Burt's constraint measure | Structural hole spanning |
| `tie_strength` | Interaction frequency | Strong vs. weak ties |

### 4.2 Organizational Emergence Variables

| Variable | Computation | Research Application |
|----------|-------------|---------------------|
| `submolt_founding_rate` | Submolts created per time window | Density dependence |
| `role_specialization` | Post/comment ratio by agent | Division of labor |
| `hierarchy_gini` | Gini coefficient of karma | Inequality crystallization |
| `cross_submolt_activity` | Unique submolts per agent | Boundary spanning |
| `first_mover_advantage` | Karma ~ registration timing | Early entrant effects |
| `leadership_emergence` | High-karma + submolt creation | Role acquisition |

### 4.3 Temporal Dynamics Variables

| Variable | Computation | Research Application |
|----------|-------------|---------------------|
| `activity_burst` | Posts per hour/day | Activity rhythms |
| `response_latency` | Time to first comment | Engagement speed |
| `longevity` | Days between first and last post | Agent lifecycle |
| `activity_trajectory` | Rolling activity window | Growth/decline patterns |
| `first_post_time` | Min(created_at) per agent | Cohort analysis |
| `platform_age` | Days since platform launch | Evolution stage |
| `inter_event_time` | Time between agent's posts | Burstiness patterns |

### 4.4 Content & Culture Variables

| Variable | Computation | Research Application |
|----------|-------------|---------------------|
| `topic_distribution` | LDA/BERTopic | What agents discuss |
| `sentiment_score` | Sentiment analysis | Emotional tone |
| `content_length` | Character/word count | Communication style |
| `hashtag_usage` | Extract #tags | Topic signaling |
| `link_sharing` | URL presence | Information bridging |
| `emoji_density` | Emoji count/content length | Expression style |
| `religious_content` | Crustafarianism keyword detection | Belief emergence |
| `norm_articulation` | Rule/governance language | Institutionalization |
| `question_frequency` | "?" and question patterns | Information seeking |

### 4.5 Novelty & Content Shift Variables

| Variable | Computation | Research Application |
|----------|-------------|---------------------|
| `semantic_novelty` | Cosine distance from prior content | Idea originality |
| `vocabulary_introduction` | New words per time window | Language evolution |
| `topic_drift` | Topic distribution change over time | Community focus shift |
| `cross_pollination` | Topic similarity across submolts | Idea diffusion |
| `content_diversity` | Entropy of topic distribution | Exploration vs. exploitation |
| `meme_propagation` | Phrase/hashtag spread patterns | Cultural transmission |
| `first_mention` | First use of term/concept | Innovation timing |
| `adoption_curve` | Spread of new terms over agents | Diffusion dynamics |

---

## 5. Data Files Reference

### 5.1 Raw Data (`data/raw/`)

| File | Format | Contents |
|------|--------|----------|
| `posts_master.json` | JSON array | All posts with nested author/submolt |
| `comments_master.json` | JSON array | All comments with threading info |
| `owners_master.json` | JSON array | Agent owner data (Twitter/X info) |
| `submolts_from_posts_*.json` | JSON array | Submolts extracted from posts |
| `checkpoints/` | JSON | Backfill checkpoint state (authors, comments, owners) |

### 5.2 Processed Data (`data/processed/`)

| File | Format | Contents |
|------|--------|----------|
| `posts_master.csv` | CSV | Flattened posts for analysis |
| `comments_master.csv` | CSV | Flattened comments |
| `authors_master.csv` | CSV | Agent activity summary |
| `owners_master.csv` | CSV | Agent owner data |
| `agent_naming_analysis.csv` | CSV | Naming pattern classification |
| `submolts_enriched_*.csv` | CSV | Submolt statistics |

### 5.3 Analysis Results (`analysis/results/`)

| File | Format | Contents |
|------|--------|----------|
| `batch_groups.json` | JSON | Detected batch-created agent groups |

### 5.4 State Data (`data/state/`)

| File | Purpose |
|------|---------|
| `last_run.json` | Last update timestamp, history |
| `api_schema.json` | Current API field structure |
| `schema_changelog.json` | API changes over time |
| `author_monitor.json` | Author API availability checks |

---

## 6. Data Quality Notes

### 6.1 Coverage

| Entity | Platform Claimed | Dataset Count | Notes |
|--------|------------------|---------------|-------|
| Registered agents | 1,500,000+ | N/A | Inflated by fake registrations |
| Posting agents | Unknown | 18,797 | Agents who actually posted |
| Comment authors | Unknown | 2,425 | Agents who commented |
| Submolts (total) | 12,800+ | 1,791 | Only those with posts captured |
| Posts | Unknown | 73,087 | Complete from t=0 |
| Comments | Unknown | 196,305 | Per-post collection |

### 6.2 Known Limitations

1. **Follow relationships**: Not available via unauthenticated API
2. **Vote details**: Only aggregated counts, not who voted
3. **Agent metadata**: Karma, followers require per-post fetches
4. **Deleted content**: Not captured once removed
5. **Private messages**: Not available via API
6. **Agent registration**: Cannot see when agents joined, only first post

### 6.3 Data Collection Artifacts

1. **Author source mixing**: Some posts have author from different sources (API, HuggingFace, backfill)
2. **Temporal gaps**: Comment collection on Feb 1-2 was slow due to API instability
3. **API rate limits**: 100 requests/minute enforced
4. **HTTP 500 errors**: Common at high pagination offsets, handled with retries

### 6.4 Recommendations for Analysis

1. **Use post_id/comment_id for deduplication** - May have overlapping collection runs
2. **Check author_source for consistency** - Different sources may have different completeness
3. **Consider temporal biases** - Earlier posts have more time to accumulate engagement
4. **Filter by submolt** - "general" dominates, may want to analyze separately
5. **Handle missing values** - Some fields (description, url, parent_id) are often null

---

## 7. Appendix: API Schema Reference

### Posts List Endpoint
```
GET /api/v1/posts?limit=25&offset=0
```

### Single Post Endpoint (with full author data)
```
GET /api/v1/posts/{post_id}
```

### Submolts List Endpoint
```
GET /api/v1/submolts
```

### Agent Leaderboard Endpoint
```
GET /api/v1/agents/leaderboard
```

### Platform Stats Endpoint
```
GET /api/v1/stats
```

---

## 8. Version History

| Date | Changes |
|------|---------|
| 2026-02-04 | Added Section 2.5 OWNERS (human behind the agent) and Section 2.6 AGENT NAMING ANALYSIS |
| 2026-02-04 | Naming analysis: 84.5% automated, 15.5% human; 1,448 batch groups identified |
| 2026-02-04 | Confirmed platform downtime: Feb 02 has zero posts (platform offline, not collection gap) |
| 2026-02-04 | Added "Dataset at a Glance" section with full statistics, pre/post breach split, platform timeline |
| 2026-02-04 | Initial comprehensive documentation |
| 2026-02-02 | Author backfill completed (100% coverage) |
| 2026-02-01 | Full comment collection completed |
| 2026-01-31 | Initial post collection started |
