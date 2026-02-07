"""
Moltbook Analysis Pipeline

This package contains the analysis phases for the Moltbook research project.
Each phase is in its own subdirectory with a main.py entry point.

Phase Structure:
- phase_00_data_audit: Compute derived variables and audit data quality
- phase_01_temporal: Heartbeat detection for autonomy classification
- phase_02_linguistic: Embeddings and LLM-based content analysis
- phase_03_restart: Pre/post breach dataset split
- phase_04_topics: Topic modeling and clustering
- phase_05_depth_gradient: Echo decay analysis in conversation threads
- phase_06_myth_genealogy: Track emergent cultural phenomena origins
- phase_07_convergence: Measure linguistic convergence between agents
- phase_08_human_motivation: Analyze human prompting signals
- phase_09_prompt_injection: Detect potential prompt injection attacks
- phase_10_figures: Generate publication-ready figures

Data Quality Notes (see docs/DATA_DICTIONARY.md):
- 6,124 posts have null author_id (from string-type author fields in raw data)
- 172,099 comments have empty author (API limitation on comment author data)
- The is_pre_breach flag is correctly computed and verified
- Pre-breach data (Jan 27-30) is cleanest for analysis
"""
