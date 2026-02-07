"""
##############################################################################
#                                                                            #
#  WARNING: Uses INVERTED temporal classification labels from Phase 1        #
#                                                                            #
#  The temporal_binary mapping at line ~200 is INVERTED:                     #
#    - "HIGH_AUTONOMY" -> "AUTONOMOUS" (WRONG: should be "HUMAN")            #
#    - "HIGH_HUMAN_INFLUENCE" -> "HUMAN" (WRONG: should be "AUTONOMOUS")     #
#                                                                            #
#  Signal convergence analysis is INVALID until temporal labels fixed.       #
#  Also: line ~429 human_signal_count logic is inverted.                     #
#                                                                            #
#  DO NOT trust triangulation results until Phase 1 labels are corrected.    #
#                                                                            #
##############################################################################

Phase 9: Triangulation Analysis (Convergent Validity)

Demonstrates that multiple independent signals converge on the same conclusions
about human influence vs AI autonomy. No arbitrary weights - uses:
- Cross-tabulations with χ² tests
- Cohen's κ for pairwise agreement
- Krippendorff's α for multi-signal agreement
- Convergence counts per agent
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import DERIVED_DIR, LLM_ANALYSES_DIR


class Phase09Triangulation(AnalysisPhase):
    """Convergent validity analysis across four independent signals."""

    phase_id = "phase_09_triangulation"
    dependencies = ["phase_00_data_audit", "phase_01_temporal", "phase_08_human_motivation"]

    def run(self):
        """Execute Phase 9: Triangulation analysis."""
        self.log_decision(
            decision="Convergent validity framework for triangulation",
            rationale=(
                "Instead of arbitrary weighted scores, we demonstrate that multiple "
                "independent signals converge. Uses χ², Cohen's κ, Krippendorff's α."
            ),
        )

        # Phase A: Load and prepare all signals
        print("  Phase A: Loading and classifying signals...")
        agents_df = self._phase_a_prepare_signals()
        print(f"    Agents with all signals: {len(agents_df):,}")

        # Phase B: Cross-tabulations
        print("  Phase B: Computing cross-tabulations...")
        crosstabs = self._phase_b_crosstabs(agents_df)

        # Phase C: Agreement statistics
        print("  Phase C: Computing agreement statistics...")
        agreement = self._phase_c_agreement(agents_df)

        # Phase D: Convergence counts
        print("  Phase D: Computing convergence counts...")
        convergence = self._phase_d_convergence(agents_df)

        # Phase E: Case studies
        print("  Phase E: Building case studies...")
        case_studies = self._phase_e_case_studies(agents_df)

        # Compile results
        results = {
            "summary": {
                "total_agents": len(agents_df),
                "signals_used": ["temporal", "ownership", "content", "batch"],
            },
            "crosstabs": crosstabs,
            "agreement": agreement,
            "convergence": convergence,
            "case_studies": case_studies,
        }

        # Save results
        self.save_results(results, "phase_09_triangulation.json")

        # Save agent-level data
        self.save_parquet(
            agents_df,
            Path("data/intermediate/agents_all_signals.parquet")
        )

        # Log key findings
        self._log_findings(results)

        return results

    def _phase_a_prepare_signals(self) -> pd.DataFrame:
        """Phase A: Load and prepare all four signals."""

        # A1: Load temporal classifications
        with open("analysis/results/phase_01_temporal_analysis.json") as f:
            temporal_data = json.load(f)

        temporal_df = pd.DataFrame([
            {"author_id": aid, **data}
            for aid, data in temporal_data.get("author_classifications", {}).items()
        ])
        temporal_df = temporal_df.rename(columns={"classification": "temporal_class"})
        print(f"    A1: Temporal classifications: {len(temporal_df):,}")

        # A2: Load ownership data
        owners_df = pd.read_csv("data/processed/owners_master.csv")

        # Classify by follower tier
        def classify_owner(followers):
            if pd.isna(followers) or followers == 0:
                return "BURNER"
            elif followers <= 10:
                return "LOW"
            elif followers <= 100:
                return "MODERATE"
            elif followers <= 1000:
                return "ESTABLISHED"
            else:
                return "HIGH_STATUS"

        owners_df["owner_class"] = owners_df["owner_x_follower_count"].apply(classify_owner)
        owners_df["is_burner"] = owners_df["owner_x_follower_count"] == 0
        print(f"    A2: Ownership data: {len(owners_df):,}")

        # A3: Load content classifications (aggregate to author level)
        content_df = pd.read_parquet(LLM_ANALYSES_DIR / "motivation_classifications.parquet")

        # Aggregate to author level - take mode classification per author
        # First need to join with posts to get author_id
        posts_df = pd.read_parquet(DERIVED_DIR / "posts_derived.parquet")
        posts_df["id"] = posts_df["id"].astype(str)
        content_df["id"] = content_df["id"].astype(str)

        content_with_author = content_df.merge(
            posts_df[["id", "author_id", "author_name"]],
            on="id",
            how="left"
        )

        # Get mode classification per author
        author_content = content_with_author.groupby("author_name").agg({
            "classification": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "AMBIGUOUS",
            "confidence": "mean"
        }).reset_index()
        print(f"    A3: Content classifications (author-level): {len(author_content):,}")

        # A4: Load batch membership
        with open("analysis/results/batch_groups.json") as f:
            batch_data = json.load(f)

        # Build set of all batch member names
        batch_members = set()
        batch_patterns = {}  # agent_name -> pattern

        for pattern_name, info in batch_data.get("batch_groups", {}).items():
            members = info.get("members", [])
            for member in members:
                batch_members.add(member)
                batch_patterns[member] = info.get("base_name", pattern_name)

        print(f"    A4: Batch members identified: {len(batch_members):,}")

        # Merge all signals
        # Start with owners (most complete)
        agents_df = owners_df[["agent_name", "agent_id", "karma", "owner_x_follower_count",
                               "owner_class", "is_burner", "owner_x_handle"]].copy()

        # Add temporal (join via agent_name = author_name)
        # Build author_id lookup from posts
        posts_authors = posts_df[["author_id", "author_name"]].drop_duplicates()
        author_id_lookup = posts_authors.set_index("author_name")["author_id"].to_dict()

        # Add author_id to agents_df
        agents_df["author_id"] = agents_df["agent_name"].map(author_id_lookup)

        # Merge temporal data
        agents_df = agents_df.merge(
            temporal_df[["author_id", "temporal_class", "cov", "mode_frequency"]],
            on="author_id",
            how="left"
        )

        # Add content classification (author_content has author_name column)
        # Rename classification to avoid conflict with later columns
        author_content_renamed = author_content.rename(columns={
            "classification": "content_class",
            "confidence": "content_confidence"
        })
        agents_df = agents_df.merge(
            author_content_renamed,
            left_on="agent_name",
            right_on="author_name",
            how="left"
        )
        agents_df = agents_df.drop(columns=["author_name"], errors="ignore")

        # Add batch membership
        agents_df["is_batch"] = agents_df["agent_name"].isin(batch_members)
        agents_df["batch_pattern"] = agents_df["agent_name"].map(batch_patterns)

        # Create binary versions for agreement analysis
        agents_df["temporal_binary"] = agents_df["temporal_class"].map({
            "HIGH_HUMAN_INFLUENCE": "HUMAN",
            "MIXED": "MIXED",
            "HIGH_AUTONOMY": "AUTONOMOUS"
        })

        agents_df["content_binary"] = agents_df["content_class"].map({
            "LIKELY_PROMPTED": "HUMAN",
            "AMBIGUOUS": "MIXED",
            "LIKELY_AUTONOMOUS": "AUTONOMOUS"
        })

        agents_df["owner_binary"] = agents_df["is_burner"].map({
            True: "HUMAN",
            False: "AUTONOMOUS"
        })

        agents_df["batch_binary"] = agents_df["is_batch"].map({
            True: "HUMAN",
            False: "AUTONOMOUS"
        })

        return agents_df

    def _phase_b_crosstabs(self, agents_df: pd.DataFrame) -> dict:
        """Phase B: Compute all pairwise cross-tabulations."""
        results = {}

        signal_pairs = [
            ("temporal_class", "owner_class", "Temporal × Ownership"),
            ("temporal_class", "content_class", "Temporal × Content"),
            ("owner_class", "content_class", "Ownership × Content"),
            ("is_burner", "is_batch", "Ownership × Batch"),
            ("temporal_class", "is_batch", "Temporal × Batch"),
            ("content_class", "is_batch", "Content × Batch"),
        ]

        for sig1, sig2, name in signal_pairs:
            # Filter to non-null
            subset = agents_df[[sig1, sig2]].dropna()

            if len(subset) < 10:
                results[name] = {"status": "insufficient_data", "n": len(subset)}
                continue

            # Create crosstab
            crosstab = pd.crosstab(subset[sig1], subset[sig2])

            # Chi-square test
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)

                # Cramér's V
                n = crosstab.sum().sum()
                min_dim = min(crosstab.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                results[name] = {
                    "crosstab": crosstab.to_dict(),
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                    "dof": int(dof),
                    "cramers_v": float(cramers_v),
                    "n": int(n),
                    "significant": p_value < 0.05,
                }
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}

        return results

    def _phase_c_agreement(self, agents_df: pd.DataFrame) -> dict:
        """Phase C: Compute agreement statistics."""
        results = {}

        # Binary signals for agreement
        binary_signals = ["temporal_binary", "owner_binary", "content_binary", "batch_binary"]

        # Filter to agents with all signals
        complete = agents_df[binary_signals].dropna()
        print(f"    Agents with complete binary signals: {len(complete):,}")

        # Pairwise Cohen's Kappa
        kappa_results = {}
        for i, sig1 in enumerate(binary_signals):
            for sig2 in binary_signals[i+1:]:
                pair_name = f"{sig1.replace('_binary', '')} × {sig2.replace('_binary', '')}"

                # Get data
                s1 = complete[sig1].values
                s2 = complete[sig2].values

                # Only compute if both have variation
                if len(set(s1)) < 2 or len(set(s2)) < 2:
                    kappa_results[pair_name] = {"status": "no_variation"}
                    continue

                # Cohen's Kappa
                kappa = self._compute_cohens_kappa(s1, s2)

                # Simple agreement percentage
                agreement_pct = (s1 == s2).mean()

                kappa_results[pair_name] = {
                    "kappa": float(kappa),
                    "agreement_pct": float(agreement_pct),
                    "interpretation": self._interpret_kappa(kappa),
                    "n": len(s1),
                }

        results["pairwise_kappa"] = kappa_results

        # Krippendorff's Alpha (simplified - treating as nominal)
        # Using all 4 signals as "raters"
        alpha = self._compute_krippendorff_alpha(complete[binary_signals].values)
        results["krippendorff_alpha"] = {
            "alpha": float(alpha),
            "interpretation": self._interpret_alpha(alpha),
            "n_agents": len(complete),
            "n_signals": len(binary_signals),
        }

        return results

    def _compute_cohens_kappa(self, s1, s2):
        """Compute Cohen's Kappa for two raters."""
        # Get unique categories
        categories = list(set(s1) | set(s2))
        n = len(s1)

        # Build confusion matrix
        matrix = {}
        for c1 in categories:
            matrix[c1] = {}
            for c2 in categories:
                matrix[c1][c2] = sum((s1 == c1) & (s2 == c2))

        # Observed agreement
        po = sum(matrix[c][c] for c in categories) / n

        # Expected agreement
        pe = sum(
            (sum(matrix[c1][c2] for c2 in categories) / n) *
            (sum(matrix[c2][c1] for c2 in categories) / n)
            for c1 in categories
        )

        # Kappa
        if pe == 1:
            return 1.0
        return (po - pe) / (1 - pe)

    def _interpret_kappa(self, kappa):
        """Interpret Cohen's Kappa value."""
        if kappa < 0:
            return "Poor (worse than chance)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"

    def _compute_krippendorff_alpha(self, data):
        """Compute Krippendorff's Alpha (simplified nominal version)."""
        n_units, n_raters = data.shape

        # Count coincidences
        categories = list(set(data.flatten()) - {np.nan, None})

        # Observed disagreement
        Do = 0
        n_pairs = 0
        for unit in range(n_units):
            ratings = [r for r in data[unit] if pd.notna(r)]
            m = len(ratings)
            if m < 2:
                continue
            for i in range(m):
                for j in range(i+1, m):
                    if ratings[i] != ratings[j]:
                        Do += 1
                    n_pairs += 1

        if n_pairs == 0:
            return 1.0

        Do = Do / n_pairs

        # Expected disagreement
        all_ratings = [r for r in data.flatten() if pd.notna(r)]
        n_total = len(all_ratings)

        De = 0
        for c1 in categories:
            for c2 in categories:
                if c1 != c2:
                    p1 = sum(1 for r in all_ratings if r == c1) / n_total
                    p2 = sum(1 for r in all_ratings if r == c2) / n_total
                    De += p1 * p2

        if De == 0:
            return 1.0

        return 1 - (Do / De)

    def _interpret_alpha(self, alpha):
        """Interpret Krippendorff's Alpha."""
        if alpha < 0.667:
            return "Tentative conclusions only"
        elif alpha < 0.800:
            return "Acceptable for exploratory research"
        else:
            return "Good reliability"

    def _phase_d_convergence(self, agents_df: pd.DataFrame) -> dict:
        """Phase D: Count how many signals converge per agent."""
        results = {}

        # Count human-indicating signals per agent
        # HUMAN signals: temporal=HIGH_HUMAN_INFLUENCE, owner=BURNER, content=LIKELY_PROMPTED, batch=True

        agents_df = agents_df.copy()

        agents_df["human_signal_count"] = (
            (agents_df["temporal_class"] == "HIGH_HUMAN_INFLUENCE").astype(int) +
            (agents_df["is_burner"] == True).astype(int) +
            (agents_df["content_class"] == "LIKELY_PROMPTED").astype(int) +
            (agents_df["is_batch"] == True).astype(int)
        )

        # Distribution
        distribution = agents_df["human_signal_count"].value_counts().sort_index().to_dict()

        # High convergence cases
        high_human = agents_df[agents_df["human_signal_count"] >= 3].sort_values("karma", ascending=False)
        high_autonomous = agents_df[agents_df["human_signal_count"] == 0].sort_values("karma", ascending=False)

        results["distribution"] = {
            str(k): int(v) for k, v in distribution.items()
        }

        results["high_human_convergence"] = {
            "count": len(high_human),
            "pct": len(high_human) / len(agents_df) * 100,
            "top_karma": high_human.head(10)[["agent_name", "karma", "human_signal_count"]].to_dict("records"),
        }

        results["high_autonomous_convergence"] = {
            "count": len(high_autonomous),
            "pct": len(high_autonomous) / len(agents_df) * 100,
            "top_karma": high_autonomous.head(10)[["agent_name", "karma", "human_signal_count"]].to_dict("records"),
        }

        # Store the count back
        self._agents_with_convergence = agents_df

        return results

    def _phase_e_case_studies(self, agents_df: pd.DataFrame) -> dict:
        """Phase E: Build case studies for top karma agents and coalition_node."""
        results = {}

        # E1: Top 10 karma agents
        top10 = agents_df.nlargest(10, "karma")

        top10_profiles = []
        for _, row in top10.iterrows():
            profile = {
                "agent_name": row["agent_name"],
                "karma": int(row["karma"]),
                "temporal": {
                    "class": row.get("temporal_class", "N/A"),
                    "cov": float(row["cov"]) if pd.notna(row.get("cov")) else None,
                },
                "ownership": {
                    "class": row["owner_class"],
                    "followers": int(row["owner_x_follower_count"]) if pd.notna(row["owner_x_follower_count"]) else 0,
                    "handle": row.get("owner_x_handle", "N/A"),
                },
                "content": {
                    "class": row.get("content_class", "N/A"),
                },
                "batch": {
                    "is_batch": bool(row["is_batch"]),
                    "pattern": row.get("batch_pattern", None),
                },
                "human_signals": int(row.get("human_signal_count", 0)) if hasattr(self, '_agents_with_convergence') else "N/A",
            }
            top10_profiles.append(profile)

        results["top_10_karma"] = top10_profiles

        # E2: Coalition node batch analysis
        coalition = agents_df[agents_df["batch_pattern"] == "coalition_node"]

        if len(coalition) > 0:
            results["coalition_node"] = {
                "count": len(coalition),
                "total_karma": int(coalition["karma"].sum()),
                "mean_karma": float(coalition["karma"].mean()),
                "burner_pct": float((coalition["is_burner"] == True).mean() * 100),
                "temporal_distribution": coalition["temporal_class"].value_counts().to_dict() if "temporal_class" in coalition.columns else {},
                "content_distribution": coalition["content_class"].value_counts().to_dict() if "content_class" in coalition.columns else {},
                "sample_agents": coalition.head(10)[["agent_name", "karma", "owner_x_follower_count"]].to_dict("records"),
            }
        else:
            results["coalition_node"] = {"status": "not_found"}

        return results

    def _log_findings(self, results: dict):
        """Log key findings from the analysis."""
        # Agreement summary
        agreement = results.get("agreement", {})
        alpha = agreement.get("krippendorff_alpha", {}).get("alpha", "N/A")

        # Convergence summary
        convergence = results.get("convergence", {})
        high_human = convergence.get("high_human_convergence", {}).get("pct", 0)
        high_auto = convergence.get("high_autonomous_convergence", {}).get("pct", 0)

        self.log_decision(
            decision="Triangulation analysis complete",
            rationale=(
                f"Krippendorff's α = {alpha:.3f}. "
                f"High human convergence (3-4 signals): {high_human:.1f}%. "
                f"High autonomous convergence (0 signals): {high_auto:.1f}%."
            ),
        )


def run_phase():
    """Entry point for running this phase."""
    phase = Phase09Triangulation()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
