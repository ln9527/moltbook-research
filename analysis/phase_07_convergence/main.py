"""
##############################################################################
#                                                                            #
#  WARNING: Uses INVERTED temporal classification labels from Phase 1        #
#                                                                            #
#  Lines ~95-97 use inverted labels:                                         #
#    - "HIGH_AUTONOMY" is actually HUMAN-PROMPTED                            #
#    - "HIGH_HUMAN_INFLUENCE" is actually AUTONOMOUS                         #
#                                                                            #
#  Convergence analysis is INVALID until Phase 1 labels are corrected.       #
#                                                                            #
##############################################################################

Phase 7: Multi-Method Convergence Validation

Validates findings by checking agreement across different analysis methods:
1. Temporal classification × Content analysis alignment
2. Pre/post breach baseline × Depth gradient consistency
3. Topic clustering × Myth genealogy overlap

This triangulation strengthens causal claims.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase


class Phase07Convergence(AnalysisPhase):
    """Multi-method convergence validation."""

    phase_id = "phase_07_convergence"
    dependencies = ["phase_01_temporal", "phase_04_topics", "phase_05_depth_gradient"]

    def run(self):
        """Execute Phase 7: Convergence validation."""
        self.log_decision(
            decision="Triangulate findings across methods",
            rationale=(
                "Cross-method agreement strengthens validity. "
                "Divergence indicates method-specific artifacts or complexity."
            ),
        )

        # Load results from previous phases
        results_dir = Path(__file__).parent.parent / "results"

        temporal_results = self._load_json(results_dir / "phase_01_temporal_analysis.json")
        topic_results = self._load_json(results_dir / "phase_04_topic_analysis.json")
        depth_results = self._load_json(results_dir / "phase_05_depth_gradient.json")
        breach_results = self._load_json(results_dir / "phase_03_breach_split.json")

        # Test 1: Temporal × Content alignment
        temporal_content = self._test_temporal_content_alignment(temporal_results, topic_results)

        # Test 2: Pre/post breach × Depth gradient
        breach_depth = self._test_breach_depth_consistency(breach_results, depth_results)

        # Test 3: Overall convergence score
        convergence_score = self._compute_convergence_score([temporal_content, breach_depth])

        # Compile results
        results = {
            "tests": {
                "temporal_content_alignment": temporal_content,
                "breach_depth_consistency": breach_depth,
            },
            "convergence_score": convergence_score,
            "interpretation": self._interpret_convergence(convergence_score),
        }

        self.save_results(results, "phase_07_convergence.json")

        # Log findings
        self._log_findings(results)

    def _load_json(self, path: Path) -> dict:
        """Load JSON results file."""
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def _test_temporal_content_alignment(
        self,
        temporal_results: dict,
        topic_results: dict,
    ) -> dict:
        """Test if temporal classifications align with content patterns."""
        if not temporal_results or not topic_results:
            return {"status": "missing_data", "agreement": None}

        # Get author classifications from temporal
        classifications = temporal_results.get("author_classifications", {})
        if not classifications:
            return {"status": "no_classifications", "agreement": None}

        # Group by classification
        high_autonomy = [aid for aid, data in classifications.items()
                        if data.get("classification") == "HIGH_AUTONOMY"]
        high_human = [aid for aid, data in classifications.items()
                     if data.get("classification") == "HIGH_HUMAN_INFLUENCE"]

        # If we had per-author topic distributions, we could test alignment
        # For now, compute basic statistics

        return {
            "status": "computed",
            "high_autonomy_count": len(high_autonomy),
            "high_human_influence_count": len(high_human),
            "hypothesis": (
                "HIGH_AUTONOMY authors should show more diverse topic engagement; "
                "HIGH_HUMAN_INFLUENCE should cluster on specific topics (e.g., crypto)"
            ),
            "agreement": None,  # Would need per-author topic data
            "note": "Full alignment test requires joining author classifications with topic assignments",
        }

    def _test_breach_depth_consistency(
        self,
        breach_results: dict,
        depth_results: dict,
    ) -> dict:
        """Test if breach split shows consistent patterns with depth gradient."""
        if not breach_results or not depth_results:
            return {"status": "missing_data", "agreement": None}

        # Get pre/post breach comparison from depth results
        breach_comparison = depth_results.get("breach_comparison", {})

        pre_sims = breach_comparison.get("pre_breach", {}).get("depth_similarities", {})
        post_sims = breach_comparison.get("post_breach", {}).get("depth_similarities", {})

        if not pre_sims or not post_sims:
            return {"status": "insufficient_depth_data", "agreement": None}

        # Compare depth-1 similarities
        pre_sim_1 = pre_sims.get(1, {}).get("mean", None)
        post_sim_1 = post_sims.get(1, {}).get("mean", None)

        if pre_sim_1 is None or post_sim_1 is None:
            return {"status": "missing_similarity_data", "agreement": None}

        # Hypothesis: Post-breach might show different echo patterns
        # (e.g., more scripted = higher similarity, or more chaos = lower)
        diff = post_sim_1 - pre_sim_1
        diff_pct = (diff / pre_sim_1) * 100 if pre_sim_1 != 0 else 0

        return {
            "status": "computed",
            "pre_breach_depth1_sim": round(pre_sim_1, 4),
            "post_breach_depth1_sim": round(post_sim_1, 4),
            "difference": round(diff, 4),
            "difference_pct": round(diff_pct, 2),
            "interpretation": (
                "Higher post-breach similarity could indicate more scripted/echo behavior. "
                "Lower could indicate chaotic/manipulated content."
            ),
            "agreement": abs(diff_pct) < 10,  # <10% change = consistent
        }

    def _compute_convergence_score(self, test_results: list) -> dict:
        """Compute overall convergence score."""
        agreements = []
        for test in test_results:
            if test.get("agreement") is not None:
                agreements.append(1 if test["agreement"] else 0)

        if not agreements:
            return {
                "score": None,
                "tests_with_data": 0,
                "total_tests": len(test_results),
            }

        return {
            "score": sum(agreements) / len(agreements),
            "tests_agreeing": sum(agreements),
            "tests_with_data": len(agreements),
            "total_tests": len(test_results),
        }

    def _interpret_convergence(self, convergence: dict) -> str:
        """Interpret the convergence score."""
        score = convergence.get("score")

        if score is None:
            return "Insufficient data for convergence assessment"
        elif score >= 0.8:
            return "Strong convergence: Methods largely agree, findings are robust"
        elif score >= 0.5:
            return "Moderate convergence: Some agreement, findings need interpretation"
        else:
            return "Weak convergence: Methods diverge, findings may be method-specific"

    def _log_findings(self, results: dict):
        """Log key findings."""
        score = results["convergence_score"].get("score")
        interpretation = results["interpretation"]

        self.log_decision(
            decision="Convergence validation complete",
            rationale=(
                f"Convergence score: {score if score else 'N/A'}. "
                f"{interpretation}"
            ),
        )


def run_phase():
    """Entry point for running this phase."""
    phase = Phase07Convergence()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
