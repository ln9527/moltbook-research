"""
State management for incremental pipeline processing.

Tracks:
- Which items have been processed by each phase
- Output hashes for dependency invalidation
- Timestamps for audit trail
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from pipeline.config import PIPELINE_STATE_FILE


@dataclass
class PhaseState:
    """State for a single phase."""
    phase_id: str
    processed_ids: set = field(default_factory=set)
    last_processed_id: Optional[str] = None
    last_run_timestamp: Optional[str] = None
    output_hash: Optional[str] = None
    input_hash: Optional[str] = None  # Hash of inputs for invalidation
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "phase_id": self.phase_id,
            "processed_ids": list(self.processed_ids),
            "last_processed_id": self.last_processed_id,
            "last_run_timestamp": self.last_run_timestamp,
            "output_hash": self.output_hash,
            "input_hash": self.input_hash,
            "status": self.status,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhaseState":
        data = data.copy()
        data["processed_ids"] = set(data.get("processed_ids", []))
        return cls(**data)


class StateManager:
    """Manages pipeline state for incremental processing."""

    def __init__(self, state_file: Path = PIPELINE_STATE_FILE):
        self.state_file = state_file
        self.phases: dict[str, PhaseState] = {}
        self._load_state()

    def _load_state(self):
        """Load state from disk."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
            for phase_id, phase_data in data.get("phases", {}).items():
                self.phases[phase_id] = PhaseState.from_dict(phase_data)

    def _save_state(self):
        """Persist state to disk."""
        data = {
            "phases": {pid: ps.to_dict() for pid, ps in self.phases.items()},
            "last_updated": datetime.utcnow().isoformat() + "Z",
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def get_phase_state(self, phase_id: str) -> PhaseState:
        """Get or create state for a phase."""
        if phase_id not in self.phases:
            self.phases[phase_id] = PhaseState(phase_id=phase_id)
        return self.phases[phase_id]

    def get_unprocessed_ids(self, phase_id: str, all_ids: set) -> set:
        """Get IDs that haven't been processed by this phase."""
        state = self.get_phase_state(phase_id)
        return all_ids - state.processed_ids

    def mark_processed(self, phase_id: str, item_ids: set, save: bool = True):
        """Mark items as processed."""
        state = self.get_phase_state(phase_id)
        state.processed_ids.update(item_ids)
        if item_ids:
            state.last_processed_id = list(item_ids)[-1]
        if save:
            self._save_state()

    def start_phase(self, phase_id: str):
        """Mark phase as running."""
        state = self.get_phase_state(phase_id)
        state.status = "running"
        state.last_run_timestamp = datetime.utcnow().isoformat() + "Z"
        state.error_message = None
        self._save_state()

    def complete_phase(self, phase_id: str, output_hash: Optional[str] = None):
        """Mark phase as completed."""
        state = self.get_phase_state(phase_id)
        state.status = "completed"
        if output_hash:
            state.output_hash = output_hash
        self._save_state()

    def fail_phase(self, phase_id: str, error_message: str):
        """Mark phase as failed."""
        state = self.get_phase_state(phase_id)
        state.status = "failed"
        state.error_message = error_message
        self._save_state()

    def reset_phase(self, phase_id: str):
        """Reset a phase to initial state."""
        self.phases[phase_id] = PhaseState(phase_id=phase_id)
        self._save_state()

    def needs_rebuild(self, phase_id: str, input_hash: str) -> bool:
        """Check if phase needs rebuild due to input changes."""
        state = self.get_phase_state(phase_id)
        if state.input_hash != input_hash:
            return True
        if state.status != "completed":
            return True
        return False

    def set_input_hash(self, phase_id: str, input_hash: str):
        """Set the input hash for a phase."""
        state = self.get_phase_state(phase_id)
        state.input_hash = input_hash
        self._save_state()

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        if not file_path.exists():
            return ""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    @staticmethod
    def compute_data_hash(data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data).hexdigest()[:16]

    def get_summary(self) -> dict:
        """Get summary of all phase states."""
        return {
            pid: {
                "status": ps.status,
                "processed_count": len(ps.processed_ids),
                "last_run": ps.last_run_timestamp,
            }
            for pid, ps in self.phases.items()
        }


# Singleton instance
_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the singleton state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
