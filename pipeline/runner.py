"""
Pipeline Runner

Master orchestrator for the Moltbook analysis pipeline.
Runs phases in dependency order with:
- Selective rebuilds
- Parallel execution of independent phases
- State tracking and checkpointing
"""

import sys
import importlib
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.dependencies import (
    PHASES,
    get_execution_order,
    get_parallel_groups,
    get_downstream_phases,
)
from pipeline.state_manager import get_state_manager
from pipeline.decision_logger import get_decision_logger


def import_phase(phase_id: str):
    """Dynamically import a phase module."""
    phase_def = PHASES[phase_id]
    module_path = phase_def.module_path

    try:
        module = importlib.import_module(module_path)
        return module
    except ImportError as e:
        print(f"  WARNING: Could not import {phase_id}: {e}")
        return None


def run_phase(phase_id: str, force_rebuild: bool = False) -> bool:
    """Run a single phase."""
    phase_def = PHASES[phase_id]
    print(f"\n{'='*60}")
    print(f"Phase: {phase_def.name}")
    print(f"{'='*60}")

    module = import_phase(phase_id)
    if module is None:
        return False

    try:
        # Each phase module should have a run_phase() function
        if hasattr(module, "run_phase"):
            return module.run_phase()
        else:
            print(f"  WARNING: {phase_id} has no run_phase() function")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline(
    phases: list[str] = None,
    force_rebuild: bool = False,
    skip_phases: list[str] = None,
):
    """
    Run the full pipeline or specific phases.

    Args:
        phases: Specific phases to run (None = all)
        force_rebuild: Force rebuild even if outputs are up to date
        skip_phases: Phases to skip
    """
    print("="*60)
    print("MOLTBOOK ANALYSIS PIPELINE")
    print(f"Started at: {datetime.utcnow().isoformat()}Z")
    print("="*60)

    state_manager = get_state_manager()
    decision_logger = get_decision_logger()

    # Log pipeline run
    decision_logger.log_decision(
        phase="pipeline",
        decision="Pipeline run started",
        rationale=f"Phases: {phases if phases else 'all'}, Force: {force_rebuild}",
    )

    # Determine phases to run
    execution_order = get_execution_order()

    if phases:
        # Filter to requested phases (preserving dependency order)
        execution_order = [p for p in execution_order if p in phases]

    if skip_phases:
        execution_order = [p for p in execution_order if p not in skip_phases]

    print(f"\nExecution plan: {len(execution_order)} phases")
    for i, pid in enumerate(execution_order, 1):
        status = state_manager.get_phase_state(pid).status
        print(f"  {i}. {pid} [{status}]")

    # Run phases
    results = {}
    for phase_id in execution_order:
        success = run_phase(phase_id, force_rebuild=force_rebuild)
        results[phase_id] = "success" if success else "failed"

        if not success:
            print(f"\n*** Phase {phase_id} failed. Stopping pipeline. ***")
            break

    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)

    for phase_id, result in results.items():
        status_icon = "✓" if result == "success" else "✗"
        print(f"  {status_icon} {phase_id}: {result}")

    success_count = sum(1 for r in results.values() if r == "success")
    print(f"\nCompleted: {success_count}/{len(results)} phases")

    return all(r == "success" for r in results.values())


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Moltbook analysis pipeline")

    parser.add_argument(
        "--phases",
        nargs="+",
        help="Specific phases to run (default: all)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        help="Phases to skip",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if outputs are up to date",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all phases and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available phases:")
        for phase_id in get_execution_order():
            phase = PHASES[phase_id]
            deps = ", ".join(phase.dependencies) if phase.dependencies else "none"
            print(f"  {phase_id}")
            print(f"    {phase.description}")
            print(f"    Dependencies: {deps}")
            print()
        return

    if args.status:
        state_manager = get_state_manager()
        print("Pipeline status:")
        for phase_id in get_execution_order():
            state = state_manager.get_phase_state(phase_id)
            print(f"  {phase_id}: {state.status}")
            if state.last_run_timestamp:
                print(f"    Last run: {state.last_run_timestamp}")
            if state.processed_ids:
                print(f"    Processed: {len(state.processed_ids)} items")
        return

    success = run_pipeline(
        phases=args.phases,
        force_rebuild=args.force,
        skip_phases=args.skip,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
