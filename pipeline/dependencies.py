"""
Phase dependency graph for the Moltbook analysis pipeline.

Defines which phases depend on which others, enabling:
- Proper execution ordering
- Selective rebuilds when upstream changes
- Parallel execution of independent phases
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class PhaseStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseDefinition:
    """Definition of an analysis phase."""
    phase_id: str
    name: str
    description: str
    dependencies: list[str]
    priority: int  # Lower = higher priority (run first)
    fast_rebuild: bool = False  # True if phase is cheap to fully rebuild
    module_path: str = ""  # e.g., "analysis.phase_00_data_audit.main"


# =============================================================================
# Phase Definitions
# =============================================================================

PHASES = {
    "phase_00_data_audit": PhaseDefinition(
        phase_id="phase_00_data_audit",
        name="Data Audit & Derived Variables",
        description="Compute derived variables: phase labels, platform age, inter-event time, word counts",
        dependencies=[],  # No dependencies - runs first
        priority=0,
        fast_rebuild=True,
        module_path="analysis.phase_00_data_audit.main",
    ),
    "phase_01_temporal": PhaseDefinition(
        phase_id="phase_01_temporal",
        name="Temporal Analysis",
        description="Detect heartbeat patterns, classify agents by autonomy level",
        dependencies=["phase_00_data_audit"],
        priority=1,
        fast_rebuild=True,
        module_path="analysis.phase_01_temporal.main",
    ),
    "phase_02_linguistic": PhaseDefinition(
        phase_id="phase_02_linguistic",
        name="Linguistic Features & Embeddings",
        description="Compute embeddings via OpenRouter, PCA reduction, linguistic features",
        dependencies=["phase_00_data_audit"],
        priority=1,
        fast_rebuild=False,  # Expensive - embeddings are cached
        module_path="analysis.phase_02_linguistic.main",
    ),
    "phase_03_restart": PhaseDefinition(
        phase_id="phase_03_restart",
        name="Pre/Post Breach Split",
        description="Split dataset at breach timestamp for natural experiment",
        dependencies=["phase_00_data_audit"],
        priority=1,
        fast_rebuild=True,
        module_path="analysis.phase_03_restart.main",
    ),
    "phase_04_topics": PhaseDefinition(
        phase_id="phase_04_topics",
        name="Topic Modeling",
        description="BERTopic modeling, topic evolution tracking",
        dependencies=["phase_02_linguistic"],
        priority=2,
        fast_rebuild=False,  # Model fitting is expensive
        module_path="analysis.phase_04_topics.main",
    ),
    "phase_05_depth_gradient": PhaseDefinition(
        phase_id="phase_05_depth_gradient",
        name="Depth Gradient Analysis",
        description="Analyze echo decay in conversation threads",
        dependencies=["phase_02_linguistic"],
        priority=2,
        fast_rebuild=True,
        module_path="analysis.phase_05_depth_gradient.main",
    ),
    "phase_06_myth_genealogy": PhaseDefinition(
        phase_id="phase_06_myth_genealogy",
        name="Myth Genealogy",
        description="Trace viral claim origins and propagation",
        dependencies=["phase_02_linguistic", "phase_04_topics"],
        priority=3,
        fast_rebuild=True,
        module_path="analysis.phase_06_myth_genealogy.main",
    ),
    "phase_07_convergence": PhaseDefinition(
        phase_id="phase_07_convergence",
        name="Multi-Method Convergence",
        description="Validate findings across methods (temporal x content alignment)",
        dependencies=["phase_01_temporal", "phase_04_topics", "phase_05_depth_gradient"],
        priority=3,
        fast_rebuild=True,
        module_path="analysis.phase_07_convergence.main",
    ),
    "phase_08_human_motivation": PhaseDefinition(
        phase_id="phase_08_human_motivation",
        name="Human Motivation Detection",
        description="Classify prompted vs autonomous content using LLM",
        dependencies=["phase_00_data_audit", "phase_02_linguistic"],
        priority=2,
        fast_rebuild=False,  # LLM calls are expensive
        module_path="analysis.phase_08_human_motivation.main",
    ),
    "phase_09_prompt_injection": PhaseDefinition(
        phase_id="phase_09_prompt_injection",
        name="Prompt Injection Detection",
        description="Detect potential prompt injection artifacts",
        dependencies=["phase_02_linguistic"],
        priority=2,
        fast_rebuild=True,
        module_path="analysis.phase_09_prompt_injection.main",
    ),
    "phase_10_figures": PhaseDefinition(
        phase_id="phase_10_figures",
        name="Publication Outputs",
        description="Generate figures and tables for publication",
        dependencies=[
            "phase_01_temporal",
            "phase_03_restart",
            "phase_04_topics",
            "phase_05_depth_gradient",
            "phase_07_convergence",
        ],
        priority=4,
        fast_rebuild=True,
        module_path="analysis.phase_10_figures.main",
    ),
}


def get_execution_order() -> list[str]:
    """Get phases in dependency-respecting execution order."""
    # Topological sort based on dependencies and priority
    executed = set()
    order = []

    def can_execute(phase_id: str) -> bool:
        deps = PHASES[phase_id].dependencies
        return all(d in executed for d in deps)

    remaining = set(PHASES.keys())
    while remaining:
        # Find all phases that can execute
        ready = [pid for pid in remaining if can_execute(pid)]
        if not ready:
            # Circular dependency - should not happen
            raise ValueError(f"Circular dependency detected: {remaining}")

        # Sort by priority
        ready.sort(key=lambda pid: PHASES[pid].priority)

        # Execute first ready phase
        next_phase = ready[0]
        order.append(next_phase)
        executed.add(next_phase)
        remaining.remove(next_phase)

    return order


def get_downstream_phases(phase_id: str) -> set[str]:
    """Get all phases that depend on this phase (directly or indirectly)."""
    downstream = set()

    def find_dependents(pid: str):
        for other_id, phase in PHASES.items():
            if pid in phase.dependencies and other_id not in downstream:
                downstream.add(other_id)
                find_dependents(other_id)

    find_dependents(phase_id)
    return downstream


def get_upstream_phases(phase_id: str) -> set[str]:
    """Get all phases this phase depends on (directly or indirectly)."""
    upstream = set()

    def find_deps(pid: str):
        for dep in PHASES[pid].dependencies:
            if dep not in upstream:
                upstream.add(dep)
                find_deps(dep)

    find_deps(phase_id)
    return upstream


def get_parallel_groups() -> list[list[str]]:
    """Group phases that can run in parallel."""
    order = get_execution_order()
    groups = []
    executed = set()

    while len(executed) < len(order):
        # Find all phases whose deps are satisfied
        can_run = [
            pid
            for pid in order
            if pid not in executed
            and all(d in executed for d in PHASES[pid].dependencies)
        ]
        if not can_run:
            break
        groups.append(can_run)
        executed.update(can_run)

    return groups


# Print dependency info if run directly
if __name__ == "__main__":
    print("Execution order:")
    for i, phase_id in enumerate(get_execution_order(), 1):
        phase = PHASES[phase_id]
        deps = ", ".join(phase.dependencies) if phase.dependencies else "none"
        print(f"  {i}. {phase_id} (deps: {deps})")

    print("\nParallel groups:")
    for i, group in enumerate(get_parallel_groups(), 1):
        print(f"  Group {i}: {', '.join(group)}")
