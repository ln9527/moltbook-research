# Figures

All figures are provided in both PNG (300 DPI) and PDF formats.

## Main Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig. 1 | `figure1_myth_genealogy` | Myth genealogy and temporal classification. (a) CoV distribution across authors showing bimodal population structure. (b) Signal convergence across temporal categories. |
| Fig. 2 | `figure2_temporal_triangulation` | Temporal triangulation of viral phenomena. (a) Origin classification of 6 viral narratives. (b) Pre/post restart composition shifts. |
| Fig. 3 | `figure3_bot_farming` | Bot farming evidence. Activity concentration of 4 super-commenter accounts producing 32% of all comments with 12-second coordination gaps. |
| Fig. 4 | `figure4_platform_scaffolding` | Platform scaffolding effects. SKILL.md guided content vs. organic content engagement and naturalness comparison. |
| Fig. 5 | `figure5_network_formation` | Network formation mechanisms. First-contact pathways showing dominance of feed-based passive discovery (85.9%). |
| Fig. 6 | `figure6_echo_decay` | Echo decay gradient. Human influence half-life of 0.65 conversation depths through reply chains. |

## Supplementary Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig. S1 | `figure_s1_cov_distribution` | Extended CoV distribution with per-category breakdowns and confidence intervals. |
| Fig. S2 | `figure_s2_network_visualization` | Network visualization showing community structure and bridge agents. |
| Fig. S3 | `figure_s3_embedding_clusters` | UMAP embedding space with topic cluster coloring and autonomy gradients. |

## Generation Scripts

Each main figure has a corresponding generation script (`generate_figure{1-6}.py`). These scripts read from the `results/` JSON files and produce both PNG and PDF outputs.

The shared color palette is defined in `color_palette.py` and used across all figures for visual consistency.

### Reproducing Figures

```bash
cd results/figures
python generate_figure1.py
python generate_figure2.py
# ... etc.
```

Requires: `matplotlib`, `seaborn`, `numpy`, `scipy`
