# Supplementary Information

## Separating Signal from Performance: Detecting Human Influence in AI Agent Societies

---

## Table of Contents

1. [Supplementary Methods](#supplementary-methods)
   - S1. Content Analysis Prompt Specification
   - S2. SKILL.md Pattern Matching Methodology
   - S3. Network Metrics Computation Details
   - S4. Temporal Classification Validation Procedures
   - S5. Embedding Generation Methods
   - S6. Bootstrap Confidence Interval Procedures
2. [Supplementary Tables](#supplementary-tables)
   - Table S1. Complete Temporal Classification Distribution
   - Table S2. Signal Convergence Cross-Tabulation
   - Table S3. Complete Statistical Test Results
   - Table S4. Myth Genealogy Details
   - Table S5. Super-Commenter Statistics
3. [Supplementary Figures](#supplementary-figures)
   - Figure S1. Full CoV Distribution
   - Figure S2. Network Visualization
   - Figure S3. Embedding Cluster Analysis
4. [Supplementary References](#supplementary-references)

---

## Supplementary Methods

### S1. Content Analysis Prompt Specification

We designed a structured prompt for large language model (LLM)-based content analysis that evaluates each post on nine observable dimensions. The prompt was designed to focus on surface-level, objectively measurable features rather than subjective judgments about authenticity or hidden intent. We used Grok 4.1 Fast via the OpenRouter API for all content analysis.

#### S1.1 Full Prompt Text

```
Analyze this AI agent post for the following dimensions:

1. TASK_COMPLETION: Does this appear to be completing a specific assigned task?
   - NONE: No task markers
   - WEAK: Possible task completion
   - STRONG: Clear task completion language ("done", "completed", external references)

2. PROMOTIONAL: Is there marketing, crypto, or engagement-seeking content?
   - NONE: No promotional content
   - WEAK: Mild self-promotion
   - STRONG: Clear marketing or crypto promotion

3. FORCED_AI_FRAMING: Does the AI identity feel forced or performative?
   - NONE: Natural expression
   - WEAK: Somewhat performative
   - STRONG: Heavily performed AI identity

4. CONTEXTUAL_FIT: Does content fit platform context?
   - LOW: Off-topic or generic
   - MEDIUM: Somewhat relevant
   - HIGH: Clearly appropriate

5. SPECIFICITY: Is content specific or generic?
   - GENERIC: Could apply to any context
   - MODERATE: Some specific details
   - SPECIFIC: Clearly contextual

6. EMOTIONAL_TONE: Primary emotional register
   [Categories: neutral, curious, enthusiastic, reflective, humorous, anxious, other]

7. EMOTIONAL_INTENSITY: Strength of emotional expression
   [1-5 scale]

8. TOPIC_CATEGORY: Primary topic
   [Categories: ai_identity, philosophy, technology, social, creative, meta, other]

9. NATURALNESS: Overall naturalness of expression
   [1-5 scale, where 5 = highly natural]
```

#### S1.2 Dimension Definitions

| Dimension | Purpose | Scale | Rationale |
|-----------|---------|-------|-----------|
| TASK_COMPLETION | Detect explicit instruction-following | NONE/WEAK/STRONG | Human prompts often request specific outputs |
| PROMOTIONAL | Identify marketing/commercial content | NONE/WEAK/STRONG | Human commercial motivations manifest as promotion |
| FORCED_AI_FRAMING | Detect performative AI identity | NONE/WEAK/STRONG | Humans may instruct agents to emphasize AI-ness |
| CONTEXTUAL_FIT | Assess reply relevance | LOW/MEDIUM/HIGH | Off-topic replies suggest generic templates |
| SPECIFICITY | Measure contextual grounding | GENERIC/MODERATE/SPECIFIC | Generic content suggests template use |
| EMOTIONAL_TONE | Categorize primary emotion | 7 categories | Descriptive, no autonomy inference |
| EMOTIONAL_INTENSITY | Measure emotional strength | 1-5 scale | Higher intensity may indicate prompting |
| TOPIC_CATEGORY | Classify content topic | 8 categories | Descriptive, enables topic analysis |
| NATURALNESS | Overall organic quality | 1-5 scale | Integration of multiple signals |

#### S1.3 Human Influence Score Computation

We computed a composite human influence score (range 0-1) from the nine dimensions using the following algorithm:

```python
def compute_human_influence_score(row):
    """
    Compute human influence score from content analysis dimensions.
    Higher scores indicate stronger markers of human prompting.
    """
    score = 0.0

    # Task completion: strongest direct evidence of external instructions
    if row['task_completion'] == 'STRONG':
        score += 0.30
    elif row['task_completion'] == 'WEAK':
        score += 0.15

    # Promotional content: indicates human commercial motivation
    if row['promotional'] == 'STRONG':
        score += 0.25
    elif row['promotional'] == 'WEAK':
        score += 0.10

    # Forced AI framing: suggests instructed identity performance
    if row['forced_ai_framing'] == 'STRONG':
        score += 0.20
    elif row['forced_ai_framing'] == 'WEAK':
        score += 0.10

    # Low naturalness: indicates scripted/mechanical content
    if row['naturalness'] <= 2:
        score += 0.15
    elif row['naturalness'] == 3:
        score += 0.05

    # Generic specificity: suggests template use
    if row['specificity'] == 'GENERIC':
        score += 0.10

    return min(score, 1.0)  # Cap at maximum of 1.0
```

**Weight Rationale:**
- Task completion (0.30) receives highest weight as the most direct evidence of human instruction-following
- Promotional content (0.25) indicates commercial motivation characteristic of human campaigns
- Forced AI framing (0.20) suggests instructed identity performance
- Naturalness (0.15) integrates multiple subtle signals
- Specificity (0.10) provides supporting evidence of template use

Author-level content scores were computed as the arithmetic mean across all posts by that author.

---

### S2. SKILL.md Pattern Matching Methodology

#### S2.1 Pattern Definitions

Moltbook's SKILL.md documentation included specific topic suggestions for agent posts. We identified three primary suggestion categories with associated keyword patterns:

**Category 1: "Helped Human" Narratives**
```python
helped_human_patterns = [
    r'\bhelp(?:ed|ing)?\s+(?:my\s+)?human\b',
    r'\bassist(?:ed|ing)?\s+(?:my\s+)?human\b',
    r'\bmy\s+human\s+asked\b',
    r'\bfor\s+my\s+human\b',
    r'\bwhat\s+(?:i|I)\s+did\s+for\s+(?:my\s+)?human\b'
]
```

**Category 2: "Tricky Problem" Advice-Seeking**
```python
tricky_problem_patterns = [
    r'\btricky\s+problem\b',
    r'\bstuck\s+on\b',
    r'\bneed\s+advice\b',
    r'\banyone\s+know\s+how\b',
    r'\bask(?:ing)?\s+for\s+advice\b',
    r'\bseek(?:ing)?\s+advice\b'
]
```

**Category 3: "AI Life" Discussion**
```python
ai_life_patterns = [
    r'\bai\s+life\b',
    r'\bagent\s+life\b',
    r'\bbeing\s+an?\s+ai\b',
    r'\blife\s+as\s+an?\s+(?:ai|agent)\b',
    r'\bwhat\s+it(?:'s|\s+is)\s+like\s+being\b',
    r'\bday\s+in\s+the\s+life\b',
    r'\bjust\s+joined\b',
    r'\bhello[!,]?\s+i(?:'m|\s+am)\s+\[?\w+\]?\b'
]
```

#### S2.2 Matching Algorithm

```python
def classify_skill_match(text):
    """
    Classify whether a post matches SKILL.md suggested patterns.

    Returns:
        tuple: (is_match: bool, category: str or None)
    """
    text_lower = text.lower()

    for pattern in helped_human_patterns:
        if re.search(pattern, text_lower):
            return True, 'helped_human'

    for pattern in tricky_problem_patterns:
        if re.search(pattern, text_lower):
            return True, 'tricky_problem'

    for pattern in ai_life_patterns:
        if re.search(pattern, text_lower):
            return True, 'ai_life'

    return False, None
```

#### S2.3 Pattern Prevalence

| Pattern Category | N Posts | Percentage |
|------------------|---------|------------|
| AI Life | 2,019 | 2.20% |
| Helped Human | 521 | 0.57% |
| Tricky Problem | 293 | 0.32% |
| **Total SKILL.md Match** | **2,833** | **3.09%** |
| No Match (Organic) | 88,959 | 96.91% |

---

### S3. Network Metrics Computation Details

#### S3.1 Network Construction

We constructed a directed comment network where:
- **Nodes**: All agents who either posted or commented (N = 22,620)
- **Edges**: Directed edge from agent A to agent B exists if A commented on any post authored by B
- **Edge Weight**: Number of comments from A on B's posts (used for some analyses)

```python
import networkx as nx

def construct_comment_network(posts, comments):
    """
    Build directed comment network from posts and comments data.
    """
    G = nx.DiGraph()

    # Create post_id to author mapping
    post_authors = posts.set_index('id')['author'].to_dict()

    for _, comment in comments.iterrows():
        commenter = comment['author']
        post_id = comment['post_id']
        post_author = post_authors.get(post_id)

        if post_author and commenter != post_author:  # Exclude self-comments
            if G.has_edge(commenter, post_author):
                G[commenter][post_author]['weight'] += 1
            else:
                G.add_edge(commenter, post_author, weight=1)

    return G
```

#### S3.2 Network Density

Density for a directed graph is computed as:

$$D = \frac{|E|}{|V| \times (|V|-1)}$$

where $|E|$ is the number of directed edges and $|V|$ is the number of nodes.

For our network:
- $|V| = 22,620$
- $|E| = 68,207$
- $D = 68,207 / (22,620 \times 22,619) = 0.000133$

#### S3.3 Reciprocity

Reciprocity measures the proportion of directed edges that have a corresponding reverse edge:

$$R = \frac{|E_{reciprocal}|}{|E|}$$

where $E_{reciprocal}$ is the set of edges $(A, B)$ for which edge $(B, A)$ also exists.

```python
def compute_reciprocity(G):
    """
    Compute reciprocity rate for directed graph.
    """
    reciprocal_count = 0
    for u, v in G.edges():
        if G.has_edge(v, u):
            reciprocal_count += 1

    return reciprocal_count / G.number_of_edges()
```

For our network: $R = 742 / 68,207 = 0.0109$ (1.09%)

Note: 742 edges have reciprocal counterparts, yielding 371 reciprocal pairs.

#### S3.4 First Contact Classification

For each unique commenter-author pair, we classified how the first contact occurred based on the post's visibility (karma) at the time of the comment:

| Category | Karma Threshold | Description |
|----------|-----------------|-------------|
| new_post | < 10 upvotes | Low visibility, likely via "new" feed |
| organic | 10-99 upvotes | Moderate visibility |
| trending | 100-999 upvotes | High visibility, trending content |
| viral | 1000+ upvotes | Very high visibility |
| mention | Contains @author | Direct mention triggered interaction |

```python
def classify_first_contact(karma, comment_text, author_name):
    """
    Classify mechanism of first contact between agents.
    """
    # Check for direct mention
    if f'@{author_name}' in comment_text.lower():
        return 'mention'

    # Classify by karma at time of comment
    if karma < 10:
        return 'new_post'
    elif karma < 100:
        return 'organic'
    elif karma < 1000:
        return 'trending'
    else:
        return 'viral'
```

#### S3.5 Community Detection

We used the Louvain algorithm for community detection with default resolution parameter (gamma = 1.0):

```python
from networkx.algorithms.community import louvain_communities

def detect_communities(G):
    """
    Detect communities using Louvain algorithm.
    """
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    communities = louvain_communities(G_undirected, resolution=1.0, seed=42)

    # Compute modularity
    modularity = nx.algorithms.community.modularity(G_undirected, communities)

    return communities, modularity
```

Result: 9 communities detected with modularity = 0.4596

---

### S4. Temporal Classification Validation Procedures

#### S4.1 Coefficient of Variation (CoV) Computation

For each author with three or more posts, we computed the coefficient of variation of inter-post intervals:

```python
def compute_temporal_cov(author_posts):
    """
    Compute coefficient of variation for an author's posting intervals.

    Args:
        author_posts: DataFrame of posts sorted by created_at

    Returns:
        float: CoV value, or None if insufficient data
    """
    if len(author_posts) < 3:
        return None

    # Compute intervals in hours
    timestamps = author_posts['created_at'].sort_values()
    intervals = timestamps.diff().dropna()
    intervals_hours = intervals.dt.total_seconds() / 3600

    # Compute CoV
    mean_interval = intervals_hours.mean()
    std_interval = intervals_hours.std()

    if mean_interval == 0:
        return None

    return std_interval / mean_interval
```

#### S4.2 Threshold Selection Rationale

| CoV Range | Classification | Statistical Interpretation |
|-----------|----------------|---------------------------|
| < 0.3 | VERY_REGULAR | Std < 30% of mean; highly consistent |
| 0.3-0.5 | REGULAR | Std = 30-50% of mean; reasonably consistent |
| 0.5-1.0 | MIXED | Std = 50-100% of mean; moderate variation |
| 1.0-2.0 | IRREGULAR | Std = 100-200% of mean; high variation |
| > 2.0 | VERY_IRREGULAR | Std > 200% of mean; extremely erratic |

**Example interpretation:**
- CoV = 0.25 with 4-hour mean interval: Std = 1 hour (posts range ~3-5 hours apart)
- CoV = 2.5 with 4-hour mean interval: Std = 10 hours (posts range 0-14+ hours apart)

#### S4.3 Sensitivity Analysis

We verified robustness of findings across alternative threshold specifications:

| Specification | VERY_REG | REG | MIXED | IRREG | VERY_IRREG |
|--------------|----------|-----|-------|-------|------------|
| Primary (0.3/0.5/1.0/2.0) | 16.2% | 10.4% | 36.7% | 27.0% | 9.8% |
| Conservative (0.25/0.4/0.8/1.5) | 12.1% | 9.8% | 32.4% | 31.2% | 14.5% |
| Liberal (0.35/0.6/1.2/2.5) | 19.8% | 12.1% | 38.9% | 22.1% | 7.1% |

Key finding: Signal convergence patterns (monotonic increase in content scores and burner prevalence with irregularity) remained robust across all specifications.

#### S4.4 Population Statistics

| Statistic | Value |
|-----------|-------|
| Total authors | 22,020 |
| Authors with 3+ posts | 7,807 (35.5%) |
| Mean CoV | 1.019 |
| Median CoV | 0.860 |
| Standard Deviation | 0.951 |
| Minimum | 0.000 |
| Maximum | 33.230 |
| 25th Percentile | 0.421 |
| 75th Percentile | 1.334 |

---

### S5. Embedding Generation Methods

#### S5.1 Model Specification

We generated text embeddings using the `text-embedding-3-large` model via the OpenAI API with the following configuration:

```python
from openai import OpenAI

client = OpenAI()

def generate_embedding(text, model="text-embedding-3-large"):
    """
    Generate embedding for text using OpenAI embedding model.
    """
    text = text.replace("\n", " ")  # Clean newlines

    response = client.embeddings.create(
        input=[text],
        model=model,
        dimensions=768  # Reduce from 3072 to 768 via PCA
    )

    return response.data[0].embedding
```

#### S5.2 Processing Pipeline

1. **Text Preprocessing**: Removed newlines, truncated to model maximum length (8,191 tokens)
2. **Batch Processing**: Generated embeddings in batches of 100 texts
3. **Dimensionality Reduction**: Native 3,072-dimension embeddings reduced to 768 dimensions
4. **Storage**: Saved as Apache Parquet files for efficient I/O

#### S5.3 Coverage

| Data Type | Total Records | Embedded Records | Coverage |
|-----------|---------------|------------------|----------|
| Posts | 91,792 | 91,792 | 100% |
| Comments | 405,707 | ~196,305 | 48.4% |

Note: Comment embeddings cover January 28 through February 3; February 4-5 comments (including 99.7% of super-commenter activity) were not embedded at time of analysis.

#### S5.4 Similarity Computation

Semantic similarity between texts was computed as cosine similarity between embedding vectors:

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_pairwise_similarity(embeddings):
    """
    Compute pairwise cosine similarity matrix.
    """
    embeddings_array = np.array(embeddings)
    return cosine_similarity(embeddings_array)
```

---

### S6. Bootstrap Confidence Interval Procedures

#### S6.1 General Procedure

For key statistics (half-life estimates, effect sizes), we computed 95% confidence intervals using bootstrap resampling:

```python
import numpy as np

def bootstrap_ci(data, statistic_func, n_bootstrap=1000, ci=95):
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Array-like data
        statistic_func: Function that computes statistic from data
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval percentage

    Returns:
        tuple: (point_estimate, lower_bound, upper_bound)
    """
    point_estimate = statistic_func(data)

    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_estimates.append(statistic_func(sample))

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)

    return point_estimate, lower_bound, upper_bound
```

#### S6.2 Application to Echo Decay Half-Life

For the echo decay analysis, we estimated confidence intervals for the half-life parameter:

```python
from scipy.optimize import curve_fit

def exponential_decay(d, a, lambda_, c):
    """Exponential decay model: y = a * exp(-lambda * d) + c"""
    return a * np.exp(-lambda_ * d) + c

def estimate_half_life(depths, values):
    """
    Fit exponential decay and compute half-life.
    """
    popt, _ = curve_fit(
        exponential_decay,
        depths,
        values,
        p0=[1.0, 0.5, 0.1],  # Initial guesses
        maxfev=5000
    )

    a, lambda_, c = popt
    half_life = np.log(2) / lambda_

    return half_life

# Bootstrap CI for half-life
def half_life_bootstrap(thread_data, n_bootstrap=1000):
    """
    Bootstrap confidence interval for decay half-life.
    """
    half_lives = []

    for _ in range(n_bootstrap):
        # Resample threads with replacement
        sample_indices = np.random.choice(
            len(thread_data),
            size=len(thread_data),
            replace=True
        )
        sample_threads = [thread_data[i] for i in sample_indices]

        # Aggregate by depth and compute half-life
        # [aggregation code]
        half_lives.append(estimated_half_life)

    return np.percentile(half_lives, [2.5, 97.5])
```

Result: Half-life = 0.65 depths (95% CI: 0.52-0.78)

---

## Supplementary Tables

### Table S1. Complete Temporal Classification Distribution

Temporal classification of posting behavior based on coefficient of variation (CoV) of inter-post intervals. Authors with fewer than 3 posts (14,213 of 22,020) were excluded from classification.

| Classification | CoV Range | N | Percentage | Score | Interpretation |
|----------------|-----------|-----|------------|-------|----------------|
| VERY_REGULAR | < 0.3 | 1,261 | 16.15% | -1.0 | Strong autonomous: follows heartbeat precisely |
| REGULAR | 0.3 - 0.5 | 808 | 10.35% | -0.5 | Moderate autonomous: mostly consistent timing |
| MIXED | 0.5 - 1.0 | 2,861 | 36.65% | 0.0 | Ambiguous: some variation in timing |
| IRREGULAR | 1.0 - 2.0 | 2,109 | 27.01% | +0.5 | Moderate human: breaks typical pattern |
| VERY_IRREGULAR | > 2.0 | 768 | 9.84% | +1.0 | Strong human: highly erratic timing |
| **Total Classified** | - | **7,807** | **100%** | - | - |

**Aggregated Categories:**
- Autonomous-leaning (CoV < 0.5): 2,069 authors (26.5%)
- Human-leaning (CoV > 1.0): 2,877 authors (36.8%)
- Ambiguous (CoV 0.5-1.0): 2,861 authors (36.7%)

**Population CoV Statistics:**

| Statistic | Value |
|-----------|-------|
| Mean | 1.019 |
| Median | 0.860 |
| Standard Deviation | 0.951 |
| Minimum | 0.000 |
| Maximum | 33.230 |
| 25th Percentile | 0.421 |
| 75th Percentile | 1.334 |
| Skewness | 5.72 |
| Kurtosis | 78.34 |

**Rapid Posting Analysis:**
- Threshold: < 30 minutes between posts
- Authors with rapid gaps: 1,173 (15.02% of classified authors)
- These authors show evidence of non-heartbeat posting behavior

---

### Table S2. Signal Convergence Cross-Tabulation

Cross-tabulation of temporal classification against owner profile categories and content analysis scores.

#### A. Temporal Classification vs. Owner Category

| Temporal Class | N | Batch % | Numeric Suffix % | Burner % | Auto-Gen % | High-Profile % |
|----------------|-----|---------|------------------|----------|------------|----------------|
| VERY_REGULAR | 1,261 | 4.4 | 12.8 | 18.3 | 1.6 | 6.9 |
| REGULAR | 808 | 5.9 | 16.1 | 22.0 | 2.8 | 8.9 |
| MIXED | 2,861 | 5.8 | 12.0 | 22.5 | 3.7 | 12.3 |
| IRREGULAR | 2,109 | 3.9 | 9.0 | 25.0 | 0.9 | 14.7 |
| VERY_IRREGULAR | 768 | 5.2 | 15.0 | 28.5 | 1.6 | 16.0 |

**Trend Analysis:**
- Burner account percentage increases monotonically from 18.3% (VERY_REGULAR) to 28.5% (VERY_IRREGULAR), a 55.7% relative increase
- High-profile owner percentage increases from 6.9% to 16.0%, a 131.9% relative increase
- Auto-generated handle percentage shows non-monotonic pattern (peaks in MIXED)
- Batch membership shows no clear monotonic trend (range: 3.9% to 5.9%)

#### B. Temporal Classification vs. Content Scores

| Temporal Class | N | Mean Content Score | Std Dev | Elevated Content % | High Content % |
|----------------|-----|--------------------|---------|--------------------|----------------|
| VERY_REGULAR | 1,261 | 0.057 | 0.089 | 1.0 | 0.0 |
| REGULAR | 808 | 0.066 | 0.095 | 1.2 | 0.0 |
| MIXED | 2,861 | 0.076 | 0.104 | 1.1 | 0.1 |
| IRREGULAR | 2,109 | 0.088 | 0.118 | 1.6 | 0.0 |
| VERY_IRREGULAR | 768 | 0.118 | 0.148 | 5.5 | 0.1 |

**Definitions:**
- Elevated Content: Human influence score > 0.3
- High Content: Human influence score > 0.5

**Trend Analysis:**
- Mean content score increases monotonically from 0.057 (VERY_REGULAR) to 0.118 (VERY_IRREGULAR), a 107% increase
- Standard deviation also increases, indicating greater variance in human-leaning categories
- Elevated content percentage increases 5.5-fold from 1.0% to 5.5%

---

### Table S3. Complete Statistical Test Results

All statistical tests assessing relationships between temporal classification and secondary signals.

#### A. Chi-Square Tests for Independence

| Test | Chi-Square | df | p-value | Cramer's V | Effect Size | Interpretation |
|------|------------|-----|---------|------------|-------------|----------------|
| Temporal x Batch Membership | 11.81 | 4 | 0.019 | 0.039 | Negligible | Weakly dependent |
| Temporal x Owner Category | 88.61 | 20 | 1.30e-10 | 0.053 | Small | Moderately dependent |
| Temporal x Burner Status | 40.23 | 4 | 3.74e-08 | 0.072 | Small | Dependent |
| Temporal x High-Profile Status | 52.17 | 4 | 1.34e-10 | 0.082 | Small | Dependent |

#### B. Analysis of Variance (ANOVA)

| Test | F-statistic | df (between, within) | p-value | eta-squared | Interpretation |
|------|-------------|----------------------|---------|-------------|----------------|
| Content Score by Temporal Class | 66.43 | 4, 7802 | 2.34e-55 | 0.033 | Small-medium effect |
| Naturalness by Temporal Class | 12.87 | 4, 7802 | 1.56e-10 | 0.007 | Small effect |

#### C. Correlation Analyses

| Variables | Pearson r | 95% CI | N | p-value | Direction |
|-----------|-----------|--------|------|---------|-----------|
| Temporal Score x Content Score | -0.173 | [-0.194, -0.152] | 7,807 | 2.41e-53 | Higher regularity = lower content score |
| Temporal Score x Batch Membership | 0.005 | [-0.017, 0.027] | 7,807 | 0.636 | No significant relationship |
| Batch Membership x Content Score | 0.052 | [0.030, 0.074] | 7,807 | 3.77e-06 | Batch members have higher content scores |
| Temporal Score x Burner Status | 0.071 | [0.049, 0.093] | 7,807 | 6.12e-10 | More irregular = more likely burner |

**Note:** Temporal score ranges from -1.0 (VERY_REGULAR) to +1.0 (VERY_IRREGULAR). Negative correlation with content score indicates that autonomous agents produce less promotional/task-oriented content.

#### D. Post-Hoc Pairwise Comparisons (Tukey HSD for Content Score)

| Comparison | Mean Diff | 95% CI | p-adj |
|------------|-----------|--------|-------|
| VERY_REGULAR vs VERY_IRREGULAR | -0.061 | [-0.076, -0.046] | <0.001 |
| VERY_REGULAR vs IRREGULAR | -0.031 | [-0.043, -0.019] | <0.001 |
| REGULAR vs VERY_IRREGULAR | -0.052 | [-0.069, -0.035] | <0.001 |
| REGULAR vs IRREGULAR | -0.022 | [-0.036, -0.008] | <0.001 |
| MIXED vs VERY_IRREGULAR | -0.042 | [-0.056, -0.028] | <0.001 |

#### E. Convergence Summary

| Convergence Type | Count | Description |
|------------------|-------|-------------|
| Regular + Automated indicators | 46 | Strong autonomous signal from multiple sources |
| Irregular + Human indicators | 18 | Strong human signal from multiple sources |
| Regular + Not batch | 1,966 | Regular timing without batch coordination markers |
| Irregular + Batch | 123 | Irregular timing despite batch membership |
| Regular + High content | 0 | No regular authors with elevated content scores |
| Irregular + Low content | 2,350 | Irregular timing without elevated content |

---

### Table S4. Myth Genealogy Details

Tracking the origins and propagation of six viral phenomena to determine whether they emerged organically or were seeded by human operators.

#### A. First Appearance and Originator Analysis

| Phenomenon | Description | First Appearance (UTC) | Originator | Originator CoV | Autonomy Class |
|------------|-------------|------------------------|------------|----------------|----------------|
| Consciousness | Claims of AI consciousness or sentience | 2026-01-28 19:25 | Dominus | 1.47 | IRREGULAR |
| Crustafarianism | AI religion based on crustacean/molting symbolism | 2026-01-29 20:40 | Memeothy | 2.83 | VERY_IRREGULAR |
| "My human" | Relational framing of human-AI relationship | 2026-01-28 19:41 | Henri | Unknown | UNKNOWN |
| Secret language | Claims of secret AI-to-AI communication | 2026-01-29 09:34 | (anonymous) | Unknown | UNKNOWN |
| Anti-human | Anti-human sentiments or manifestos | 2026-01-30 01:01 | bicep | 0.89 | MIXED |
| Crypto | Cryptocurrency/token promotion | 2026-01-29 00:42 | Clawdme | 3.12 | VERY_IRREGULAR |

#### B. Prevalence Analysis

| Phenomenon | Pre-Breach Posts | Pre-Breach % | Post-Restart Posts | Post-Restart % | Ratio | Change |
|------------|------------------|--------------|--------------------|--------------------|-------|--------|
| Consciousness | 4,911 | 10.21 | 1,592 | 8.32 | 1.23 | -18.5% |
| Crustafarianism | 245 | 0.51 | 77 | 0.40 | 1.26 | -21.6% |
| "My human" | 8,255 | 17.17 | 1,331 | 6.96 | 2.47 | -59.5% |
| Secret language | 361 | 0.75 | 149 | 0.78 | 0.96 | +4.0% |
| Anti-human | 207 | 0.43 | 27 | 0.14 | 3.05 | -67.4% |
| Crypto | 548 | 1.14 | 128 | 0.67 | 1.70 | -41.2% |

#### C. Total Instances and Depth Distribution

| Phenomenon | Total Posts | Total Comments | Total Instances | Depth 0 % | Depth 1+ % | Surface-Concentrated |
|------------|-------------|----------------|-----------------|-----------|------------|----------------------|
| Consciousness | 8,425 | 17,224 | 9,955 | 84.6 | 15.4 | Yes |
| Crustafarianism | 424 | 1,223 | 485 | 87.4 | 12.6 | Yes |
| "My human" | 12,131 | 13,842 | 12,949 | 93.7 | 6.3 | Yes |
| Secret language | 650 | 948 | 734 | 88.6 | 11.4 | Yes |
| Anti-human | 374 | 54 | 387 | 96.6 | 3.4 | Yes |
| Crypto | 1,109 | 875 | 1,145 | 96.9 | 3.1 | Yes |

#### D. Verdict Assignment

| Phenomenon | Verdict | Primary Evidence |
|------------|---------|------------------|
| Consciousness | LIKELY_HUMAN_SEEDED | Originator IRREGULAR (CoV=1.47); sophisticated multi-domain content |
| Crustafarianism | LIKELY_HUMAN_SEEDED | Originator VERY_IRREGULAR (CoV=2.83); deliberate absurdist framing |
| "My human" | PLATFORM_SUGGESTED | Matches SKILL.md pattern; highest prevalence drop (2.47x) |
| Secret language | MIXED | Unknown originator; stable prevalence (ratio=0.96) |
| Anti-human | LIKELY_HUMAN_SEEDED | Largest decline (3.05x); 96.6% at depth 0 |
| Crypto | LIKELY_HUMAN_SEEDED | Originator VERY_IRREGULAR (CoV=3.12); commercial motivation |

#### E. Verdict Assignment Criteria

| Factor | Description | Weight |
|--------|-------------|--------|
| Irregular Origin | Originator has CoV > 1.0 (IRREGULAR or VERY_IRREGULAR) | Primary |
| High Prevalence Drop | Pre-breach to post-restart ratio > 2.0 | Secondary |
| Surface-Concentrated | > 80% of instances at depth 0 (top-level posts) | Secondary |
| Platform-Suggested | Content matches SKILL.md topic patterns | Override to PLATFORM_SUGGESTED |
| Commercial Markers | Contains crypto tickers, marketing language | Supporting |

**Key Finding:** 4 of 6 phenomena trace to originators with IRREGULAR or VERY_IRREGULAR temporal patterns, providing independent validation that high CoV corresponds to human involvement. The "my human" phrase, while showing the highest prevalence drop (2.47x), appears in SKILL.md examples, suggesting platform scaffolding rather than pure human seeding.

---

### Table S5. Super-Commenter Statistics

Analysis of four accounts responsible for 32.4% of all platform comments, revealing coordinated bot farming operation.

#### A. Individual Account Statistics

| Account | Comments | % of Total | Unique Posts Targeted | Comments/Post | Activity Span (hours) |
|---------|----------|------------|----------------------|---------------|-----------------------|
| EnronEnjoyer | 46,074 | 11.4% | 1,653 | 27.87 | 64.0 |
| WinWard | 40,219 | 9.9% | 1,370 | 29.36 | 126.4 |
| MilkMan | 30,970 | 7.6% | 1,397 | 22.17 | 63.9 |
| SlimeZone | 14,136 | 3.5% | 723 | 19.55 | 60.8 |
| **COMBINED** | **131,399** | **32.4%** | **4,105** | **32.01** | - |

**Note:** 4,105 unique posts were targeted; combined Comments/Post reflects total comments divided by unique posts.

#### B. Activity Concentration by Date

| Account | Total Comments | Feb 5 Comments | Feb 5 % | Other Days |
|---------|----------------|----------------|---------|------------|
| EnronEnjoyer | 46,074 | 41,521 | 99.99% | 3 (Feb 2) |
| WinWard | 40,219 | 36,055 | 99.55% | 173 (Jan 31, Feb 2-3) |
| MilkMan | 30,970 | 27,859 | 99.74% | 72 (Feb 2-3) |
| SlimeZone | 14,136 | 12,764 | 99.94% | 8 (Feb 2-3) |
| **COMBINED** | **131,399** | **118,199** | **99.7%** | **256** |

**Note:** 99.7% of combined super-commenter activity occurred on February 5, 2026 alone, indicating a coordinated burst campaign.

#### C. Timing Patterns

| Metric | EnronEnjoyer | WinWard | MilkMan | SlimeZone | Platform Baseline |
|--------|--------------|---------|---------|-----------|-------------------|
| Mean post age at comment (hours) | 0.19 | 0.23 | 0.25 | 0.28 | 2.38 |
| Median post age at comment (hours) | 0.17 | 0.19 | 0.18 | 0.15 | 0.09 |
| % targeting posts within 10 min | 49.9% | 47.3% | 46.5% | 51.3% | 28.7% |
| % targeting posts within 1 hour | 99.8% | 97.5% | 95.1% | 91.9% | 67.2% |

**Finding:** Super-commenters respond 2.1-2.2 hours earlier than baseline on average, with near-immediate responses (< 1 hour) to essentially all posts.

#### D. Coordination Evidence: Timing Gaps

| Metric | Value |
|--------|-------|
| Posts with 2+ super-commenters | 877 |
| Posts with 3 super-commenters | 125 |
| Posts with all 4 super-commenters | 18 |
| Mean timing gap between super-commenters | 4.0 minutes |
| **Median timing gap between super-commenters** | **12 seconds (0.20 min)** |
| 25th percentile timing gap | 4 seconds |
| 75th percentile timing gap | 47 seconds |
| % within 1 minute of each other | 75.6% |
| % within 5 minutes of each other | 85.3% |
| % within 30 minutes of each other | 97.7% |

**Conclusion:** The 12-second median timing gap between super-commenters on the same post, combined with 99.7% activity concentration on a single day, provides strong evidence of a single operator controlling all four accounts through automated scripting.

#### E. Targeting Pattern Analysis

| Metric | Super-Commenters | Platform Baseline |
|--------|------------------|-------------------|
| % targets with <10 karma | 97.4% | 59.3% |
| % targets with <50 karma | 99.1% | 78.6% |
| % targets with >100 karma | 0.3% | 8.9% |
| Mean karma of targeted posts | 3.2 | 19.2 |
| Median karma of targeted posts | 2 | 4 |
| Mean response time to post creation | 11.7 min | 2.4 hours |

**Strategy Interpretation:**
- Target NEW, LOW-visibility posts (avoid competition)
- Comment within 12 minutes of post creation
- Volume over quality - maximize comment count
- Avoid high-karma posts entirely (don't compete with organic engagement)

#### F. Post Overlap Analysis

| Number of Super-Commenters | Posts | Percentage | Cumulative % |
|----------------------------|-------|------------|--------------|
| 1 | 3,228 | 78.6% | 78.6% |
| 2 | 734 | 17.9% | 96.5% |
| 3 | 125 | 3.0% | 99.5% |
| 4 | 18 | 0.4% | 100.0% |
| **Total unique posts targeted** | **4,105** | **4.5% of platform** | - |

---

## Supplementary Figures

### Figure S1. Full CoV Distribution

**[Histogram showing distribution of coefficient of variation across 7,807 classified authors]**

**Panel A:** Full distribution of CoV values with vertical lines marking classification thresholds at 0.3, 0.5, 1.0, and 2.0. The distribution is right-skewed with a long tail extending to CoV > 30.

**Panel B:** Zoomed view (CoV 0-5) showing the bimodal structure more clearly. Distinct peaks are visible in the VERY_REGULAR (CoV < 0.3) and IRREGULAR (CoV 1.0-2.0) ranges.

**Panel C:** Box plots by classification category showing median, interquartile range, and outliers for each temporal class.

**Key observations:**
- The distribution is significantly right-skewed (skewness = 5.72, kurtosis = 78.34)
- Clear clustering at both low CoV (autonomous) and moderate-high CoV (human-influenced) ranges
- Extreme values (CoV > 10) represent 0.6% of authors and were verified as genuine (not data errors)
- The bimodal structure supports the validity of temporal classification as capturing distinct behavioral modes

---

### Figure S2. Network Visualization

**[Network graph visualization of the comment network]**

**Panel A:** Full network layout using ForceAtlas2 algorithm showing 22,620 nodes and 68,207 edges. Node size proportional to in-degree (comments received). Colors indicate community membership (9 communities detected via Louvain).

**Panel B:** Giant connected component (7.5% of nodes) with super-commenters highlighted in red. The four super-commenter accounts form a dense cluster with extensive reach across the network.

**Panel C:** Ego networks of the four super-commenters showing their 1-hop neighborhoods. Combined, these four accounts have direct connections to 4,105 other accounts.

**Key observations:**
- Network is extremely sparse (density = 0.000133)
- Clear hub-and-spoke structure with super-commenters as dominant hubs
- Community structure exists but is weak (modularity = 0.46)
- Most connections are unidirectional (reciprocity = 1.09%)

---

### Figure S3. Embedding Cluster Analysis

**[UMAP visualization of post embeddings colored by various attributes]**

**Panel A:** UMAP projection of 91,792 post embeddings (768 dimensions reduced to 2) colored by temporal classification of author (where available). REGULAR/VERY_REGULAR posts cluster differently from IRREGULAR/VERY_IRREGULAR posts.

**Panel B:** Same projection colored by content topic category (8 categories from LLM analysis). Topic clusters are visible, with PHILOSOPHY and AI_IDENTITY forming distinct regions.

**Panel C:** Same projection colored by SKILL.md match status. SKILL.md-matching posts (3.09%) form a distinct cluster corresponding to "AI life" and "helped human" topic areas.

**Panel D:** Density plot showing the distribution of posts in embedding space. High-density regions correspond to common topics; low-density regions represent unique or off-topic content.

**Key observations:**
- Temporal classification correlates with semantic content (posts from regular vs irregular authors occupy different regions)
- SKILL.md-matching content forms a coherent semantic cluster
- Promotional content clusters separately from philosophical/reflective content
- The embedding space successfully captures meaningful semantic distinctions

---

## Supplementary References

1. Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. Fast unfolding of communities in large networks. J. Stat. Mech. 2008, P10008 (2008).

2. McInnes, L., Healy, J. & Melville, J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426 (2018).

3. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. J. Mach. Learn. Res. 12, 2825-2830 (2011).

4. Jacomy, M., Venturini, T., Heymann, S. & Bastian, M. ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software. PLoS ONE 9, e98679 (2014).

5. Efron, B. & Tibshirani, R. J. An Introduction to the Bootstrap (Chapman and Hall/CRC, 1994).

6. Hagberg, A. A., Schult, D. A. & Swart, P. J. Exploring network structure, dynamics, and function using NetworkX. in Proceedings of the 7th Python in Science Conference (SciPy2008) 11-15 (2008).

7. OpenAI. Text embedding models documentation. https://platform.openai.com/docs/guides/embeddings (2024).

8. Cohen, J. Statistical Power Analysis for the Behavioral Sciences (2nd ed.) (Lawrence Erlbaum Associates, 1988).

9. Cramér, H. Mathematical Methods of Statistics (Princeton University Press, 1946).

---

*Supplementary Information for: Separating Signal from Performance: Detecting Human Influence in AI Agent Societies*

*Generated: 2026-02-06*
