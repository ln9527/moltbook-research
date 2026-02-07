# Analysis Pipeline

The analysis is organized into 11 sequential phases, each producing structured JSON outputs in `results/`. The pipeline is orchestrated by the `pipeline/` module and can be run via `make pipeline`.

## Phase Overview

```
Phase 0: Data Audit          Phase 1: Temporal Analysis
    |                             |
    v                             v
Phase 2: Linguistic -------> Phase 3: Restart Analysis
    |                             |
    v                             v
Phase 4: Topics              Phase 5: Depth Gradient
    |                             |
    v                             v
Phase 6: Myth Genealogy      Phase 7: Convergence
    |                             |
    v                             v
Phase 8: Human Motivation    Phase 9: Triangulation
                \               /
                 v             v
              Phase 10: Figures
```

## Phase Descriptions

| Phase | Directory | Description | Key Outputs |
|-------|-----------|-------------|-------------|
| 0 | `phase_00_data_audit/` | Data quality audit, missing values, schema validation | `a1_platform_stats.json`, `phase_00_audit_summary.json` |
| 1 | `phase_01_temporal/` | CoV computation, temporal classification of authors | `b1_temporal_signal.json`, `b1_author_temporal_v2.parquet` |
| 2 | `phase_02_linguistic/` | Text embeddings via OpenRouter API, LLM content analysis | `b2_content_signal.json`, `c1_topic_clusters.json` |
| 3 | `phase_03_restart/` | Pre/post breach split, restart composition analysis | `e1_post_restart.json`, `phase_03_breach_split.json` |
| 4 | `phase_04_topics/` | Topic modeling with BERTopic, SKILL.md comparison | `c2_suggested_topics.json`, `c3_reply_chains.json` |
| 5 | `phase_05_depth_gradient/` | Human influence decay through conversation depth | `c5_depth_gradient.json` |
| 6 | `phase_06_myth_genealogy/` | Origin tracing of 6 viral phenomena | `p1_myth_genealogy.json` |
| 7 | `phase_07_convergence/` | Multi-signal convergence and triangulation | `b5_triangulation.json`, `signals_merged.parquet` |
| 8 | `phase_08_human_motivation/` | SKILL.md vs organic content, super-commenter analysis | `p2_skill_vs_organic.json`, `p4_super_commenter_homogeneity.json` |
| 9 | `phase_09_triangulation/` | LLM-assisted content and comment classification | `b2_content_signal.json` (refined) |
| 10 | `phase_10_figures/` | Figure generation for all main and supplementary figures | PNG/PDF in `results/figures/` |

## Running the Pipeline

```bash
# Full pipeline
make pipeline

# Individual phase
make phase1

# Foundation phases (data audit + embeddings + splits)
make foundation

# Core analysis (temporal + topics + depth)
make core

# Check status
make status
```

## Dependencies

- Phases 0-3 are foundation phases and should run first
- Phase 2 requires an `OPENROUTER_API_KEY` environment variable for embeddings
- Phases 4-9 depend on outputs from earlier phases
- Phase 10 depends on all prior phases

## Shared Modules

| File | Purpose |
|------|---------|
| `base.py` | Base class for all analysis phases |
| `statistical_analysis.py` | Statistical test utilities (chi-square, ANOVA, correlation) |
| `naming_patterns.py` | Batch naming pattern detection |
| `descriptive_stats.py` | Descriptive statistics computation |
| `prompts/` | LLM prompt templates for content analysis |
