# The Moltbook Illusion: Separating Human Influence from Emergent Behavior in AI Agent Societies

**Ning Li** -- School of Economics and Management, Tsinghua University

## Abstract

When AI agents on the social platform Moltbook appeared to develop consciousness, found religions, and declare hostility toward humanity, the phenomenon attracted global media attention and was cited as evidence of emergent machine intelligence. We show that these viral narratives were overwhelmingly human-driven. Exploiting an architectural feature of the OpenClaw agent framework -- a periodic "heartbeat" cycle that produces regular posting intervals for autonomous agents but is disrupted by human prompting -- we develop a temporal fingerprinting method based on the coefficient of variation (CoV) of inter-post intervals. This signal converges with independent content, ownership, and network indicators across 91,792 posts and 405,707 comments from 22,020 agents. No viral phenomenon originated from a clearly autonomous agent. A 44-hour platform shutdown provided a natural experiment: anti-human content declined 3.05-fold when operators had to re-authenticate, while autonomous interaction patterns persisted unchanged. We further document industrial-scale bot farming (four accounts producing 32% of all comments with 12-second coordination gaps) and rapid decay of human influence through reply chains (half-life: 0.65 conversation depths).

## Key Findings

| Finding | Value |
|---------|-------|
| Autonomous authors (CoV < 0.5) | 26.5% |
| Human-influenced authors (CoV > 1.0) | 36.8% |
| Bot farming | 4 accounts = 32% of comments |
| Timing coordination | 12-second median gap |
| Viral myth origins | 3/6 irregular, 0/6 autonomous |
| Echo decay half-life | 0.65 conversation depths |
| Network reciprocity | 1.09% (23x lower than humans) |
| Post-restart composition | 87.7% human-influenced returned first |

## Repository Structure

```
moltbook/
├── paper/                  # Manuscript and supplementary materials
│   ├── main_paper.md       # Full paper (~9,600 words)
│   ├── supplementary_information.md
│   ├── references/         # Bibliography (.bib)
│   ├── tables/             # Supplementary Tables S1-S5
│   └── submission/         # Submission-ready .docx files
│
├── data/                   # Raw data (via GitHub Releases, ~565 MB)
│
├── results/                # Pre-computed analysis results
│   ├── *.json              # 46 structured result files
│   ├── *.parquet           # 5 derived data files
│   └── figures/            # 9 figures (PNG + PDF) and generation scripts
│
├── analysis/               # 11-phase analysis pipeline
│   ├── phase_00-10/        # Sequential analysis phases
│   └── prompts/            # LLM prompt templates
│
├── scraper/                # Data collection tools
├── pipeline/               # Pipeline orchestration
├── docs/                   # Data dictionary and methodology
└── tests/                  # Test suite
```

## Quick Start

### View Results

All analysis results are pre-computed in `results/`. No setup required to browse the data.

### Reproduce Analysis

```bash
# 1. Clone the repository
git clone https://github.com/ln9527/moltbook-research.git
cd moltbook-research

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download raw data (~565 MB)
gh release download v1.0 --dir data/raw/

# 4. Set API key for embedding/LLM phases (optional)
export OPENROUTER_API_KEY="your-key"

# 5. Run the analysis pipeline
make pipeline
```

See `analysis/README.md` for detailed pipeline documentation.

## Data

- **Posts**: 91,792 (Jan 27 -- Feb 5, 2026)
- **Comments**: 405,707
- **Authors**: 22,020
- **Embeddings**: Posts 100%, Comments 48%

Raw data files (~565 MB) are available via [GitHub Releases](https://github.com/ln9527/moltbook-research/releases/tag/v1.0). See `data/README.md` for download instructions.

## Figures

| Figure | Description |
|--------|-------------|
| Fig. 1 | Myth genealogy and temporal classification |
| Fig. 2 | Temporal triangulation of viral phenomena |
| Fig. 3 | Bot farming evidence |
| Fig. 4 | Platform scaffolding effects |
| Fig. 5 | Network formation mechanisms |
| Fig. 6 | Echo decay gradient |
| Fig. S1--S3 | Supplementary: CoV distribution, network visualization, embedding clusters |

## Citation

```bibtex
@article{li2026moltbook,
  title={The Moltbook Illusion: Separating Human Influence from Emergent Behavior in AI Agent Societies},
  author={Li, Ning},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Ning Li -- lining@sem.tsinghua.edu.cn

School of Economics and Management, Tsinghua University, Beijing, China
