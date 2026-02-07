# Moltbook Analysis Pipeline Makefile
# Usage: make <target>

PYTHON = python3
PIPELINE = $(PYTHON) -m pipeline.runner

.PHONY: all clean status phases help

# Default target
all: pipeline

# Run full pipeline
pipeline:
	$(PIPELINE)

# Force rebuild all
rebuild:
	$(PIPELINE) --force

# Individual phases
phase0:
	$(PIPELINE) --phases phase_00_data_audit

phase1:
	$(PIPELINE) --phases phase_01_temporal

phase2:
	$(PIPELINE) --phases phase_02_linguistic

phase3:
	$(PIPELINE) --phases phase_03_restart

phase4:
	$(PIPELINE) --phases phase_04_topics

phase5:
	$(PIPELINE) --phases phase_05_depth_gradient

phase6:
	$(PIPELINE) --phases phase_06_myth_genealogy

phase7:
	$(PIPELINE) --phases phase_07_convergence

phase8:
	$(PIPELINE) --phases phase_08_human_motivation

phase9:
	$(PIPELINE) --phases phase_09_triangulation

phase10:
	$(PIPELINE) --phases phase_10_figures

# Common workflows
# Foundation: data audit + splits + embeddings
foundation:
	$(PIPELINE) --phases phase_00_data_audit phase_02_linguistic phase_03_restart

# Core analysis: temporal + topics + depth
core:
	$(PIPELINE) --phases phase_01_temporal phase_04_topics phase_05_depth_gradient

# All analysis (skip figures)
analysis:
	$(PIPELINE) --skip phase_10_figures

# Figures only
figures:
	$(PIPELINE) --phases phase_10_figures

# Status
status:
	$(PIPELINE) --status

# List phases
phases:
	$(PIPELINE) --list

# Update data (run scraper then pipeline)
update:
	$(PYTHON) scraper/update.py
	$(PIPELINE)

# Clean intermediate files (keep raw/processed data)
clean:
	rm -rf data/intermediate/embeddings/*.npz
	rm -rf data/intermediate/llm_analyses/*.jsonl
	rm -rf data/state/pipeline_state.json
	rm -rf outputs/figures/*
	rm -rf outputs/tables/*

# Help
help:
	@echo "Moltbook Analysis Pipeline"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  pipeline     Run full pipeline (default)"
	@echo "  rebuild      Force rebuild all phases"
	@echo "  foundation   Run Phase 0, 2, 3 (data audit, embeddings, splits)"
	@echo "  core         Run Phase 1, 4, 5 (temporal, topics, depth)"
	@echo "  analysis     Run all phases except figures"
	@echo "  figures      Generate figures only"
	@echo "  status       Show pipeline status"
	@echo "  phases       List all phases"
	@echo "  update       Update data and run pipeline"
	@echo "  clean        Remove intermediate files"
	@echo ""
	@echo "Individual phases:"
	@echo "  phase0-10    Run specific phase (e.g., make phase0)"
