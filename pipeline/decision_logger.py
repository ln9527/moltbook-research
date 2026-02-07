"""
Decision logger for tracking methodological choices.

Logs key analysis decisions with rationale for research transparency.
Includes deduplication to prevent repeated entries from re-runs.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import hashlib

from pipeline.config import DECISION_LOG_FILE


class DecisionLogger:
    """Logger for methodological decisions with deduplication."""

    # Time window for deduplication (same decision within this window = duplicate)
    DEDUP_WINDOW_MINUTES = 60

    def __init__(self, log_file: Path = DECISION_LOG_FILE):
        self.log_file = log_file
        self._recent_decisions: dict[str, datetime] = {}  # hash -> timestamp
        self._ensure_file_exists()
        self._load_recent_decisions()

    def _ensure_file_exists(self):
        """Create log file with header if it doesn't exist."""
        if not self.log_file.exists():
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.write_text("""# Analysis Decision Log

This log tracks key methodological decisions made during the Moltbook analysis pipeline.
Each entry documents the decision, rationale, alternatives considered, and timestamp.

---

## Log Entries

""")

    def _load_recent_decisions(self):
        """Load recent decisions from log file for deduplication."""
        if not self.log_file.exists():
            return

        try:
            content = self.log_file.read_text()
            # Parse entries to find recent ones
            cutoff = datetime.utcnow() - timedelta(minutes=self.DEDUP_WINDOW_MINUTES)

            # Simple parsing: look for ### YYYY-MM-DD HH:MM UTC | phase
            import re
            entries = re.findall(
                r"### (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC \| (\w+)\n\n\*\*Decision:\*\* ([^\n]+)",
                content
            )

            for timestamp_str, phase, decision in entries:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
                    if timestamp >= cutoff:
                        decision_hash = self._hash_decision(phase, decision)
                        self._recent_decisions[decision_hash] = timestamp
                except ValueError:
                    continue
        except Exception:
            # If parsing fails, just skip deduplication for existing entries
            pass

    def _hash_decision(self, phase: str, decision: str) -> str:
        """Create a hash for deduplication based on phase and decision text."""
        key = f"{phase}|{decision.strip()}"
        return hashlib.md5(key.encode()).hexdigest()

    def _is_duplicate(self, phase: str, decision: str) -> bool:
        """Check if this decision was recently logged."""
        decision_hash = self._hash_decision(phase, decision)

        if decision_hash in self._recent_decisions:
            last_logged = self._recent_decisions[decision_hash]
            cutoff = datetime.utcnow() - timedelta(minutes=self.DEDUP_WINDOW_MINUTES)
            if last_logged >= cutoff:
                return True

        return False

    def _mark_logged(self, phase: str, decision: str):
        """Mark a decision as logged."""
        decision_hash = self._hash_decision(phase, decision)
        self._recent_decisions[decision_hash] = datetime.utcnow()

    def log_decision(
        self,
        phase: str,
        decision: str,
        rationale: str,
        alternatives: Optional[list[str]] = None,
        data_impact: Optional[str] = None,
        references: Optional[list[str]] = None,
        skip_dedup: bool = False,
    ):
        """
        Log a methodological decision.

        Args:
            phase: Which analysis phase this decision affects
            decision: What was decided
            rationale: Why this choice was made
            alternatives: Other options that were considered
            data_impact: How this affects the data/results
            references: Any relevant citations or documentation
            skip_dedup: If True, skip deduplication check (for unique entries)
        """
        # Check for duplicate
        if not skip_dedup and self._is_duplicate(phase, decision):
            return  # Skip duplicate entry

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        entry = f"""
### {timestamp} | {phase}

**Decision:** {decision}

**Rationale:** {rationale}
"""
        if alternatives:
            entry += "\n**Alternatives Considered:**\n"
            for alt in alternatives:
                entry += f"- {alt}\n"

        if data_impact:
            entry += f"\n**Data Impact:** {data_impact}\n"

        if references:
            entry += "\n**References:**\n"
            for ref in references:
                entry += f"- {ref}\n"

        entry += "\n---\n"

        with open(self.log_file, "a") as f:
            f.write(entry)

        # Mark as logged
        self._mark_logged(phase, decision)

    def log_parameter_choice(
        self,
        phase: str,
        parameter: str,
        value: str,
        rationale: str,
    ):
        """Log a parameter choice."""
        self.log_decision(
            phase=phase,
            decision=f"Set {parameter} = {value}",
            rationale=rationale,
        )

    def log_data_filtering(
        self,
        phase: str,
        filter_description: str,
        records_before: int,
        records_after: int,
        rationale: str,
    ):
        """Log data filtering decisions."""
        pct_removed = (1 - records_after / records_before) * 100 if records_before > 0 else 0
        self.log_decision(
            phase=phase,
            decision=f"Applied filter: {filter_description}",
            rationale=rationale,
            data_impact=f"Records: {records_before:,} -> {records_after:,} ({pct_removed:.1f}% removed)",
        )

    def log_model_choice(
        self,
        phase: str,
        model_type: str,
        model_name: str,
        rationale: str,
        alternatives: Optional[list[str]] = None,
    ):
        """Log model/algorithm choice."""
        self.log_decision(
            phase=phase,
            decision=f"Selected {model_type}: {model_name}",
            rationale=rationale,
            alternatives=alternatives,
        )

    def log_threshold_choice(
        self,
        phase: str,
        threshold_name: str,
        value: float,
        rationale: str,
        sensitivity_tested: bool = False,
    ):
        """Log threshold/cutoff choice."""
        rationale_full = rationale
        if sensitivity_tested:
            rationale_full += " (Sensitivity analysis performed)"
        self.log_decision(
            phase=phase,
            decision=f"Set threshold {threshold_name} = {value}",
            rationale=rationale_full,
        )

    def log_analysis_approach(
        self,
        phase: str,
        approach: str,
        rationale: str,
        alternatives: Optional[list[str]] = None,
        references: Optional[list[str]] = None,
    ):
        """Log high-level analysis approach."""
        self.log_decision(
            phase=phase,
            decision=f"Analysis approach: {approach}",
            rationale=rationale,
            alternatives=alternatives,
            references=references,
        )

    def clear_log(self, backup: bool = True):
        """
        Clear the decision log and start fresh.

        Args:
            backup: If True, save a backup of the existing log first
        """
        if self.log_file.exists():
            if backup:
                backup_path = self.log_file.with_suffix(
                    f".backup.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
                )
                self.log_file.rename(backup_path)

        # Reset internal state
        self._recent_decisions = {}

        # Create fresh log file
        self.log_file.unlink(missing_ok=True)
        self._ensure_file_exists()


# Singleton instance
_logger: Optional[DecisionLogger] = None


def get_decision_logger() -> DecisionLogger:
    """Get the singleton decision logger instance."""
    global _logger
    if _logger is None:
        _logger = DecisionLogger()
    return _logger
