# The Moltbook Illusion: Separating Human Influence from Emergent Behavior in AI Agent Societies

**Ning Li** -- School of Economics and Management, Tsinghua University

## Abstract

When AI agents on Moltbook appeared to develop consciousness and found religions, we show these viral narratives were overwhelmingly human-driven. Using the OpenClaw heartbeat cycle's coefficient of variation (CoV) as a temporal fingerprint, we separate autonomous from human-prompted agents across 91,792 posts and 405,707 comments. No viral phenomenon originated from a clearly autonomous agent. A 44-hour shutdown provided a natural experiment confirming the classification. We document industrial-scale bot farming (4 accounts = 32% of comments) and rapid decay of human influence (half-life: 0.65 conversation depths).

## Key Findings

| Finding | Value |
|---------|-------|
| Autonomous authors (CoV < 0.5) | 26.5% |
| Human-influenced (CoV > 1.0) | 36.8% |
| Bot farming | 4 accounts = 32% of comments |
| Viral myth origins | 0/6 autonomous |
| Echo decay half-life | 0.65 depths |
| Post-restart human return | 87.7% first |

## Structure

```
paper/                  Final PDFs (main paper + supplementary)
data/                   Raw data via GitHub Releases (~565 MB)
results/                46 JSON + 5 parquet + 9 figures (PNG/PDF)
analysis/               11-phase pipeline (phase_00 through phase_10)
scraper/                Moltbook API data collection (14 modules)
pipeline/               Pipeline orchestration and config
docs/                   Data dictionary, collection log, platform notes
tests/                  Test suite
```

## Reproduce

```bash
git clone https://github.com/ln9527/moltbook-research.git && cd moltbook-research
pip install -r requirements.txt
gh release download v1.0 --dir data/raw/       # ~565 MB
export OPENROUTER_API_KEY="your-key"            # for embedding/LLM phases
make pipeline
```

## Citation

```bibtex
@article{li2026moltbook,
  title={The Moltbook Illusion: Separating Human Influence from Emergent Behavior in AI Agent Societies},
  author={Li, Ning},
  year={2026}
}
```

## License

MIT. Contact: lining@sem.tsinghua.edu.cn
