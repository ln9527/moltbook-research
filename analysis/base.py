"""
Base class for all analysis phases.

Provides:
- Standard interface for running phases
- State management integration
- Decision logging
- Incremental processing helpers
"""

import json
import pandas as pd
import pyarrow.parquet as pq
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from pipeline.config import (
    RAW_DIR,
    PROCESSED_DIR,
    INTERMEDIATE_DIR,
    DERIVED_DIR,
    OUTPUTS_DIR,
)
from pipeline.state_manager import get_state_manager, StateManager
from pipeline.decision_logger import get_decision_logger, DecisionLogger


class AnalysisPhase(ABC):
    """
    Base class for all analysis phases.

    Subclasses should define:
    - phase_id: Unique identifier matching dependencies.py
    - dependencies: List of phase_ids this depends on
    - run(): Main execution logic

    Optionally override:
    - get_input_hash(): Custom input change detection
    - validate_outputs(): Custom output validation
    """

    phase_id: str = ""
    dependencies: list[str] = []

    def __init__(self):
        self.state_manager: StateManager = get_state_manager()
        self.decision_logger: DecisionLogger = get_decision_logger()
        self._start_time: Optional[datetime] = None

    # =========================================================================
    # Core Lifecycle
    # =========================================================================

    def execute(self, force_rebuild: bool = False) -> bool:
        """
        Execute the phase with state management.

        Args:
            force_rebuild: If True, ignore cached state and rebuild

        Returns:
            True if phase completed successfully
        """
        phase_state = self.state_manager.get_phase_state(self.phase_id)

        # Check if rebuild needed
        input_hash = self.get_input_hash()
        if not force_rebuild and not self.state_manager.needs_rebuild(self.phase_id, input_hash):
            print(f"[{self.phase_id}] Skipping - outputs up to date")
            return True

        self._start_time = datetime.utcnow()
        print(f"[{self.phase_id}] Starting at {self._start_time.isoformat()}")
        self.state_manager.start_phase(self.phase_id)
        self.state_manager.set_input_hash(self.phase_id, input_hash)

        try:
            self.run()
            output_hash = self.compute_output_hash()
            self.state_manager.complete_phase(self.phase_id, output_hash)
            duration = (datetime.utcnow() - self._start_time).total_seconds()
            print(f"[{self.phase_id}] Completed in {duration:.1f}s")
            return True

        except Exception as e:
            self.state_manager.fail_phase(self.phase_id, str(e))
            print(f"[{self.phase_id}] FAILED: {e}")
            raise

    @abstractmethod
    def run(self):
        """Main execution logic - implement in subclass."""
        raise NotImplementedError

    # =========================================================================
    # Data Loading Helpers
    # =========================================================================

    def load_raw_posts(self) -> list[dict]:
        """Load raw posts from JSON."""
        with open(RAW_DIR / "posts_master.json") as f:
            return json.load(f)

    def load_raw_comments(self) -> list[dict]:
        """Load raw comments from JSON."""
        with open(RAW_DIR / "comments_master.json") as f:
            return json.load(f)

    def load_posts_df(self) -> pd.DataFrame:
        """Load posts as DataFrame (CSV or derived parquet)."""
        derived = DERIVED_DIR / "posts_derived.parquet"
        if derived.exists():
            return pd.read_parquet(derived)
        return pd.read_csv(PROCESSED_DIR / "posts_master.csv")

    def load_comments_df(self) -> pd.DataFrame:
        """Load comments as DataFrame (CSV or derived parquet)."""
        derived = DERIVED_DIR / "comments_derived.parquet"
        if derived.exists():
            return pd.read_parquet(derived)
        return pd.read_csv(PROCESSED_DIR / "comments_master.csv")

    def load_derived_posts(self) -> pd.DataFrame:
        """Load derived posts parquet (requires Phase 0 complete)."""
        return pd.read_parquet(DERIVED_DIR / "posts_derived.parquet")

    def load_derived_comments(self) -> pd.DataFrame:
        """Load derived comments parquet (requires Phase 0 complete)."""
        return pd.read_parquet(DERIVED_DIR / "comments_derived.parquet")

    def load_embeddings(self, content_type: str = "posts") -> pd.DataFrame:
        """Load embeddings (requires Phase 2 complete)."""
        path = INTERMEDIATE_DIR / "embeddings" / f"{content_type}_embeddings.parquet"
        return pd.read_parquet(path)

    # =========================================================================
    # Output Helpers
    # =========================================================================

    def save_parquet(self, df: pd.DataFrame, path: Path):
        """Save DataFrame as parquet with compression."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy", index=False)
        print(f"  Saved {len(df):,} rows to {path.name}")

    def save_json(self, data: Any, path: Path, indent: int = 2):
        """Save data as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=indent, default=str)
        print(f"  Saved to {path.name}")

    def save_results(self, results: dict, filename: str):
        """Save results to analysis/results/."""
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        self.save_json(results, results_dir / filename)

    # =========================================================================
    # State & Hashing
    # =========================================================================

    def get_input_hash(self) -> str:
        """
        Compute hash of inputs for change detection.
        Override in subclass for custom logic.
        """
        # Default: hash raw data files
        hashes = []
        for f in [RAW_DIR / "posts_master.json", RAW_DIR / "comments_master.json"]:
            if f.exists():
                hashes.append(self.state_manager.compute_file_hash(f))
        return "-".join(hashes)

    def compute_output_hash(self) -> str:
        """
        Compute hash of outputs for downstream invalidation.
        Override in subclass for custom outputs.
        """
        return datetime.utcnow().isoformat()  # Default: timestamp

    def get_unprocessed_ids(self, all_ids: set) -> set:
        """Get IDs not yet processed by this phase."""
        return self.state_manager.get_unprocessed_ids(self.phase_id, all_ids)

    def mark_processed(self, item_ids: set):
        """Mark items as processed."""
        self.state_manager.mark_processed(self.phase_id, item_ids)

    # =========================================================================
    # Decision Logging
    # =========================================================================

    def log_decision(self, decision: str, rationale: str, **kwargs):
        """Log a methodological decision."""
        self.decision_logger.log_decision(
            phase=self.phase_id,
            decision=decision,
            rationale=rationale,
            **kwargs,
        )

    def log_parameter(self, parameter: str, value: Any, rationale: str):
        """Log a parameter choice."""
        self.decision_logger.log_parameter_choice(
            phase=self.phase_id,
            parameter=parameter,
            value=str(value),
            rationale=rationale,
        )

    def log_filtering(self, description: str, before: int, after: int, rationale: str):
        """Log data filtering."""
        self.decision_logger.log_data_filtering(
            phase=self.phase_id,
            filter_description=description,
            records_before=before,
            records_after=after,
            rationale=rationale,
        )

    def log_model_choice(self, model_type: str, model_name: str, rationale: str, alternatives: Optional[list[str]] = None):
        """Log model/algorithm choice."""
        self.decision_logger.log_model_choice(
            phase=self.phase_id,
            model_type=model_type,
            model_name=model_name,
            rationale=rationale,
            alternatives=alternatives,
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_outputs(self) -> bool:
        """
        Validate phase outputs exist and are valid.
        Override in subclass for custom validation.
        """
        return True
