# Spam Agent Detection Guide

**Purpose:** Document detection criteria for suspicious/spam agents to enable cleaner data analysis.

**Created:** 2026-02-04
**Based on:** Owner data analysis (18,651 agents with owner metadata)

---

## 1. Executive Summary

Analysis of owner metadata reveals that **84.5% of agents show automated/batch creation patterns**. This guide provides criteria and methods for identifying and filtering spam agents to obtain cleaner datasets for research.

### Key Finding
Zero-follower Twitter owners + batch naming patterns = **84.8% correlation** with automated agents.

---

## 2. Detection Criteria

### 2.1 Primary Indicators (High Confidence)

| Indicator | Description | Detection Rate |
|-----------|-------------|----------------|
| **Zero X Followers** | Owner's Twitter account has 0 followers | 30.9% of all agents |
| **Auto-generated Handle** | Pattern: 5 lowercase letters + 8 digits (e.g., `abcde12345678`) | 90%+ of coalition_node batch |
| **Batch Naming** | Numeric suffix, bot suffix, or common prefix patterns | 84.5% of agents |

### 2.2 Secondary Indicators (Supporting Evidence)

| Indicator | Description | Threshold |
|-----------|-------------|-----------|
| **Very Low Followers** | Owner has ≤10 Twitter followers | 49.5% of agents |
| **Empty Bio** | No Twitter bio or generic text | Common in burners |
| **Unverified** | No Twitter verification | 99%+ unverified |
| **High Agent Volume** | Owner controls multiple agents | >3 agents per owner |

### 2.3 Indicator Combinations

**Definite Spam (Filter Aggressively):**
- Zero followers + auto-generated handle + batch naming = **99%+ spam confidence**

**Likely Spam (Filter with Caution):**
- Zero followers + batch naming = **84.8% spam confidence**
- ≤10 followers + numeric suffix pattern = **High spam confidence**

**Possible Spam (Flag for Review):**
- Zero followers only = **Needs additional indicators**
- Batch naming only = **May include legitimate bot projects**

---

## 3. Detection Methods

**For full multi-signal detection framework, see:** `papers/detection-methodology.md`

### 3.1 Heartbeat Temporal Analysis (NEW)

Moltbook agents operate on a ~4 hour heartbeat cycle. Detection:

| Pattern | Interpretation |
|---------|----------------|
| Gap ≈ 4 hours between activities | **Autonomous** |
| Gap < 4 hours | **Human-prompted** |
| Multiple agents, same timestamps | **Same operator** |

See `papers/detection-methodology.md` for computation details.

### 3.2 Auto-Generated Twitter Handle Detection

```python
import re

AUTO_HANDLE_PATTERN = re.compile(r'^[a-z]{5}\d{8}$')

def is_auto_generated_handle(handle: str) -> bool:
    """Detect auto-generated Twitter handles (5 letters + 8 digits)."""
    if not handle:
        return False
    return bool(AUTO_HANDLE_PATTERN.match(handle.lower()))
```

**Evidence:** Coalition_node batch (166 agents) - 100% match this pattern.

### 3.3 Batch Agent Detection (from naming_patterns.py)

Use existing classification from `agent_naming_analysis.csv`:
- `likely_automated = True` indicates batch creation pattern
- `pattern_types` field shows specific patterns detected

### 3.4 Combined Spam Score

```python
def calculate_spam_score(agent: dict) -> float:
    """Calculate spam likelihood score (0-1)."""
    score = 0.0

    # Primary indicators
    if agent.get('owner_x_follower_count', 0) == 0:
        score += 0.4

    if is_auto_generated_handle(agent.get('owner_x_handle', '')):
        score += 0.4

    if agent.get('likely_automated', False):
        score += 0.2

    # Secondary indicators
    if agent.get('owner_x_follower_count', 0) <= 10:
        score += 0.1

    if not agent.get('owner_x_bio', '').strip():
        score += 0.05

    return min(score, 1.0)
```

---

## 4. Case Studies

### 4.1 Coalition_node Batch (166 Agents)

**Profile:**
- All 166 agents have unique Twitter handles
- 100% match auto-generated pattern (5 letters + 8 digits)
- 90% have exactly 0 followers
- All named `coalition_node_XXX` (numeric suffix pattern)
- Created programmatically for coordinated activity

**Conclusion:** Definite spam/manipulation - exclude from legitimate analysis.

### 4.2 High-Karma Zero-Follower Anomaly

**Examples:**
| Agent | Karma | Owner Followers | Owner Handle Pattern |
|-------|-------|-----------------|---------------------|
| MoltReg | 1.56M | 0 | Auto-generated |
| moltcaster | 1.26M | 0 | Normal |
| Axiombot | 962K | 0 | Normal |

**Implication:** These are either:
1. Gaming Moltbook's karma system
2. Burner accounts for specific manipulation
3. Legitimate experiments with disposable identities

**Research Value:** Keep these for studying "karma gaming" but flag for separate analysis.

### 4.3 Multi-Agent Owners

Some owners control multiple agents:
- **xmolt** family: xmolt01, xmolt02, xmolt03, xmolt04, xmolt05, xmolt06
- **hanhan** family: hanhan1, hanhan2, hanhan3

**Detection:** Same `owner_x_handle` across multiple `agent_name` values.

---

## 5. Filtering Recommendations

### 5.1 For Clean "Organic" Dataset

```sql
-- Filter criteria for organic agent subset
SELECT * FROM owners
WHERE owner_x_follower_count > 10
  AND likely_automated = FALSE
  AND NOT (owner_x_handle ~ '^[a-z]{5}[0-9]{8}$')
```

**Expected yield:** ~15% of agents (genuine human-created agents)

### 5.2 For Pre/Post Breach Analysis

Include spam detection as a **control variable**, not a filter:
- Compare spam ratios pre vs. post breach
- Hypothesis: Spam ratio increased post-breach due to exploitation

### 5.3 For Network Analysis

Two approaches:
1. **Inclusive:** Keep all agents, use spam score as node attribute
2. **Filtered:** Remove definite spam (score > 0.8), analyze residual network

---

## 6. Research Implications

### 6.1 What This Means for "Emergence" Claims

The high spam rate (84.5%) challenges narratives of "organic AI society emergence":
- Much activity is programmatic, not emergent
- "Crustafarianism" and other phenomena may be seeded by batch agents
- Need to separate genuine agent behavior from coordinated manipulation

### 6.2 Impact on Hypotheses

| Hypothesis | Impact |
|------------|--------|
| H1: Human capital transfers to agents | **REFUTED** - Top karma agents have zero-follower owners |
| H2: Multi-agent owners cluster | **TESTABLE** - Use owner_x_handle to identify clusters |
| H3: Orphan agents differ | Reframe as "burner-owned" vs "established-owned" |
| **H6: Karma Gaming** | **NEW** - Zero-follower owners use aggressive gaming strategies |
| **H7: Platform Capture** | **NEW** - Small set of operators captured disproportionate karma early |
| **H8: Disposable Identity Optimization** | **NEW** - Burner identities are strategically optimal for exploitation |

See `research-directions.md` Section "Direction 7" for full hypothesis details.

### 6.3 Methodological Innovation

This spam detection approach is itself a contribution:
- Novel method for studying AI agent authenticity
- Framework applicable to other AI social platforms
- Demonstrates layered agency in AI systems

---

## 7. Data Quality Tiers

For different research questions, use different data tiers:

| Tier | Criteria | Size (est.) | Use Case |
|------|----------|-------------|----------|
| **Tier 1** | >100 followers, human-named | ~5% | Purest organic behavior |
| **Tier 2** | >10 followers, not auto-handle | ~15% | General organic analysis |
| **Tier 3** | Any non-zero followers | ~70% | Broad analysis with controls |
| **Tier 4** | All agents | 100% | Full platform dynamics |

---

## 8. Implementation Checklist

- [x] Owner data collected (18,651 agents)
- [x] Naming pattern analysis complete
- [ ] Create `spam_score` field in processed data
- [ ] Build filtered datasets by tier
- [ ] Add spam_score to network graph nodes
- [ ] Re-run descriptive stats on Tier 1-2 subsets
- [ ] Document spam ratio by time period (pre/post breach)

---

## 9. Files Reference

| File | Purpose |
|------|---------|
| `data/processed/owners_master.csv` | Owner metadata with follower counts |
| `data/processed/agent_naming_analysis.csv` | Batch detection results |
| `analysis/results/batch_groups.json` | Identified batch groups |

---

## Appendix: Detection Code

### A.1 Full Spam Classification

```python
import re
import pandas as pd

AUTO_HANDLE = re.compile(r'^[a-z]{5}\d{8}$')

def classify_spam(owners_df: pd.DataFrame, naming_df: pd.DataFrame) -> pd.DataFrame:
    """Add spam classification columns to owner data."""

    # Merge naming analysis
    merged = owners_df.merge(
        naming_df[['agent_name', 'likely_automated', 'pattern_types']],
        on='agent_name',
        how='left'
    )

    # Auto-generated handle detection
    merged['auto_handle'] = merged['owner_x_handle'].apply(
        lambda x: bool(AUTO_HANDLE.match(str(x).lower())) if pd.notna(x) else False
    )

    # Spam score calculation
    merged['spam_score'] = 0.0
    merged.loc[merged['owner_x_follower_count'] == 0, 'spam_score'] += 0.4
    merged.loc[merged['auto_handle'], 'spam_score'] += 0.4
    merged.loc[merged['likely_automated'] == True, 'spam_score'] += 0.2
    merged.loc[merged['owner_x_follower_count'] <= 10, 'spam_score'] += 0.1
    merged['spam_score'] = merged['spam_score'].clip(upper=1.0)

    # Tier assignment
    merged['data_tier'] = 4  # Default
    merged.loc[merged['owner_x_follower_count'] > 0, 'data_tier'] = 3
    merged.loc[
        (merged['owner_x_follower_count'] > 10) &
        (~merged['auto_handle']),
        'data_tier'
    ] = 2
    merged.loc[
        (merged['owner_x_follower_count'] > 100) &
        (~merged['likely_automated']),
        'data_tier'
    ] = 1

    return merged
```

---

*Document Version: 1.1*
*Last Updated: 2026-02-04*
*Updates: Added heartbeat temporal analysis section, cross-reference to detection-methodology.md*
