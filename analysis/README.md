# Analysis Pipeline

11 sequential phases. Run via `make pipeline`. Each phase outputs JSON to `results/`.

| Phase | Directory | What it does | Key outputs |
|-------|-----------|--------------|-------------|
| 0 | `phase_00_data_audit/` | Data quality, schema validation | `a1_platform_stats.json` |
| 1 | `phase_01_temporal/` | CoV computation, author classification | `b1_temporal_signal.json` |
| 2 | `phase_02_linguistic/` | Embeddings + LLM content analysis | `b2_content_signal.json`, `c1_topic_clusters.json` |
| 3 | `phase_03_restart/` | Pre/post breach split | `e1_post_restart.json` |
| 4 | `phase_04_topics/` | BERTopic modeling, SKILL.md comparison | `c2_suggested_topics.json` |
| 5 | `phase_05_depth_gradient/` | Human influence decay by depth | `c5_depth_gradient.json` |
| 6 | `phase_06_myth_genealogy/` | Origin tracing of 6 viral phenomena | `p1_myth_genealogy.json` |
| 7 | `phase_07_convergence/` | Multi-signal triangulation | `b5_triangulation.json` |
| 8 | `phase_08_human_motivation/` | SKILL.md vs organic, super-commenters | `p2_skill_vs_organic.json` |
| 9 | `phase_09_triangulation/` | LLM-assisted content classification | refined `b2_content_signal.json` |
| 10 | `phase_10_figures/` | All figures | PNG/PDF in `results/figures/` |

Shared modules: `base.py` (phase base class), `statistical_analysis.py`, `naming_patterns.py`, `descriptive_stats.py`.

Phase 2 requires `OPENROUTER_API_KEY`. Phases 0-3 must run before 4-10.
