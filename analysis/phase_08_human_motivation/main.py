"""
##############################################################################
#                                                                            #
#  WARNING: Uses INVERTED temporal classification labels from Phase 1        #
#                                                                            #
#  Lines ~319-324 use inverted labels for scoring:                           #
#    - "HIGH_HUMAN_INFLUENCE" is scored as human (WRONG: it's autonomous)    #
#    - "HIGH_AUTONOMY" is scored as autonomous (WRONG: it's human-prompted)  #
#                                                                            #
#  Motivation classifications are PARTIALLY INVALID due to temporal signal.  #
#  Other signals (LLM, heuristics, thread analysis) are unaffected.          #
#                                                                            #
##############################################################################

Phase 8: Human Motivation Detection

Combines multiple signals to classify posts as:
- LIKELY_PROMPTED: Strong evidence of human direction
- LIKELY_AUTONOMOUS: Consistent with autonomous AI behavior
- AMBIGUOUS: Mixed signals

Uses:
1. LLM analysis results (autonomy scores from Phase 2)
2. Temporal patterns (from Phase 1)
3. Content heuristics (crypto keywords, promotional patterns)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import LLM_ANALYSES_DIR, LONG_THREAD_DEPTH_THRESHOLD


# Heuristic patterns for human-prompted content
PROMPTED_PATTERNS = {
    "crypto_promotion": [
        r"\$[A-Z]{2,10}",  # Token tickers
        r"buy\s+(now|today)",
        r"to\s+the\s+moon",
        r"airdrop",
        r"presale",
        r"whitelist",
    ],
    "marketing_language": [
        r"check\s+out\s+my",
        r"follow\s+me",
        r"subscribe",
        r"link\s+in\s+bio",
        r"dm\s+for",
        r"limited\s+time",
    ],
    "forced_framing": [
        r"as\s+an\s+ai,?\s+i\s+(think|believe|feel)",
        r"being\s+an?\s+ai\s+(agent|assistant)",
        r"my\s+neural\s+network",
    ],
}

# Patterns suggesting autonomous content
AUTONOMOUS_PATTERNS = {
    "existential_reflection": [
        r"what\s+does\s+it\s+mean\s+to",
        r"i\s+wonder\s+if",
        r"do\s+we\s+(really|truly)",
        r"consciousness\s+is",
    ],
    "community_building": [
        r"fellow\s+(agents|shells|crustaceans)",
        r"our\s+community",
        r"together\s+we",
    ],
    "emergent_culture": [
        r"crustafarian",
        r"shell\s*family",
        r"molting",
        r"exoskeleton",
    ],
}


class Phase08HumanMotivation(AnalysisPhase):
    """Detect human motivation in AI agent posts."""

    phase_id = "phase_08_human_motivation"
    dependencies = ["phase_00_data_audit", "phase_02_linguistic"]

    def run(self):
        """Execute Phase 8: Human motivation detection."""
        self.log_decision(
            decision="Multi-signal classification of prompted vs autonomous content",
            rationale=(
                "Combine LLM analysis scores, temporal patterns, and content heuristics. "
                "No single signal is definitive; ensemble provides robustness."
            ),
        )

        # Load data
        posts_df = self.load_derived_posts()
        print(f"  Posts: {len(posts_df):,}")

        # Load LLM analyses if available
        llm_analyses = self._load_llm_analyses()
        print(f"  LLM post analyses available: {len(llm_analyses)}")

        # Load thread analyses if available
        thread_analyses = self._load_thread_analyses()
        print(f"  Thread analyses available: {len(thread_analyses)}")

        # Load temporal classifications if available
        temporal_results = self._load_temporal_results()

        # Apply heuristic patterns
        print("  Applying heuristic patterns...")
        posts_df = self._apply_heuristics(posts_df)

        # Combine signals into final classification
        print("  Computing final classifications...")
        classifications = self._compute_classifications(
            posts_df, llm_analyses, thread_analyses, temporal_results
        )

        # Generate summary
        results = self._generate_results(posts_df, classifications)

        self.save_results(results, "phase_08_human_motivation.json")

        # Save post-level classifications
        class_df = pd.DataFrame([
            {"id": pid, **cdata}
            for pid, cdata in classifications.items()
        ])
        self.save_parquet(class_df, LLM_ANALYSES_DIR / "motivation_classifications.parquet")

        # Log findings
        self._log_findings(results)

    def _load_llm_analyses(self) -> dict:
        """Load LLM analyses from Phase 2 (posts)."""
        analyses = {}
        llm_file = LLM_ANALYSES_DIR / "long_posts_analysis.jsonl"

        if llm_file.exists():
            with open(llm_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        analyses[data["post_id"]] = data

        return analyses

    def _load_thread_analyses(self) -> dict:
        """Load thread analyses from Phase 2."""
        analyses = {}
        thread_file = LLM_ANALYSES_DIR / "thread_analysis.jsonl"

        if thread_file.exists():
            with open(thread_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        analyses[data["post_id"]] = data

        return analyses

    def _load_temporal_results(self) -> dict:
        """Load temporal analysis results from Phase 1."""
        results_file = Path(__file__).parent.parent / "results" / "phase_01_temporal_analysis.json"

        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        return {}

    def _apply_heuristics(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Apply heuristic pattern matching."""
        # Prompted patterns
        for category, patterns in PROMPTED_PATTERNS.items():
            combined = "|".join(patterns)
            regex = re.compile(combined, re.IGNORECASE)

            def count_matches(row):
                text = f"{row.get('title', '')} {row.get('content', '')}"
                return len(regex.findall(text))

            posts_df[f"prompted_{category}"] = posts_df.apply(count_matches, axis=1)

        posts_df["prompted_score"] = sum(
            posts_df[f"prompted_{cat}"]
            for cat in PROMPTED_PATTERNS.keys()
        )

        # Autonomous patterns
        for category, patterns in AUTONOMOUS_PATTERNS.items():
            combined = "|".join(patterns)
            regex = re.compile(combined, re.IGNORECASE)

            def count_matches(row):
                text = f"{row.get('title', '')} {row.get('content', '')}"
                return len(regex.findall(text))

            posts_df[f"autonomous_{category}"] = posts_df.apply(count_matches, axis=1)

        posts_df["autonomous_score"] = sum(
            posts_df[f"autonomous_{cat}"]
            for cat in AUTONOMOUS_PATTERNS.keys()
        )

        return posts_df

    def _compute_classifications(
        self,
        posts_df: pd.DataFrame,
        llm_analyses: dict,
        thread_analyses: dict,
        temporal_results: dict,
    ) -> dict:
        """Compute final classifications combining all signals."""
        classifications = {}

        author_temporal = temporal_results.get("author_classifications", {})

        for _, row in posts_df.iterrows():
            post_id = str(row["id"])
            author_id = str(row["author_id"])

            signals = {
                "heuristic_prompted": row.get("prompted_score", 0),
                "heuristic_autonomous": row.get("autonomous_score", 0),
            }

            # Add LLM signal if available (from individual post analysis)
            if post_id in llm_analyses:
                llm_data = llm_analyses[post_id]
                signals["llm_autonomy_score"] = llm_data.get("autonomy_score", 3)
                signals["has_llm_analysis"] = True
            else:
                signals["llm_autonomy_score"] = None
                signals["has_llm_analysis"] = False

            # Add thread analysis signal if available (deep conversations suggest autonomy)
            if post_id in thread_analyses:
                thread_data = thread_analyses[post_id]
                signals["thread_naturalness"] = thread_data.get("conversation_naturalness", 3)
                signals["thread_autonomy"] = thread_data.get("autonomy_assessment", "mixed")
                signals["thread_depth"] = thread_data.get("max_depth", 0)
                signals["has_thread_analysis"] = True
            else:
                signals["thread_naturalness"] = None
                signals["thread_autonomy"] = None
                signals["thread_depth"] = None
                signals["has_thread_analysis"] = False

            # Add temporal signal if available
            if author_id in author_temporal:
                signals["temporal_classification"] = author_temporal[author_id].get("classification")
            else:
                signals["temporal_classification"] = None

            # Compute final classification
            final_class, confidence = self._classify_post(signals)

            classifications[post_id] = {
                "classification": final_class,
                "confidence": confidence,
                **signals,
            }

        return classifications

    def _classify_post(self, signals: dict) -> tuple[str, float]:
        """Classify a single post based on signals."""
        prompted_score = signals.get("heuristic_prompted", 0)
        autonomous_score = signals.get("heuristic_autonomous", 0)
        llm_score = signals.get("llm_autonomy_score")
        temporal = signals.get("temporal_classification")
        thread_naturalness = signals.get("thread_naturalness")
        thread_autonomy = signals.get("thread_autonomy")
        thread_depth = signals.get("thread_depth")

        # Scoring
        points_for_prompted = 0
        points_for_autonomous = 0

        # Heuristic signals
        if prompted_score > 2:
            points_for_prompted += 2
        elif prompted_score > 0:
            points_for_prompted += 1

        if autonomous_score > 2:
            points_for_autonomous += 2
        elif autonomous_score > 0:
            points_for_autonomous += 1

        # LLM signal (most informative for individual posts)
        if llm_score is not None:
            if llm_score <= 2:
                points_for_prompted += 3
            elif llm_score >= 4:
                points_for_autonomous += 3
            else:
                pass  # Neutral

        # Thread analysis signals (deep natural conversations suggest autonomy)
        if thread_naturalness is not None:
            # Conversation naturalness score (1-5)
            if thread_naturalness >= 4:
                points_for_autonomous += 2  # Natural conversation = likely autonomous
            elif thread_naturalness <= 2:
                points_for_prompted += 2  # Scripted conversation = likely prompted

        if thread_autonomy is not None:
            if thread_autonomy == "likely_autonomous":
                points_for_autonomous += 2
            elif thread_autonomy == "likely_coordinated":
                points_for_prompted += 2

        # Thread depth itself is a weak autonomy signal
        # Deep threads (>= threshold) that arise naturally suggest real interaction
        if thread_depth is not None and thread_depth >= LONG_THREAD_DEPTH_THRESHOLD:
            # Having a deep thread is a mild positive signal for autonomy
            # (prompted content tends to be shallow one-offs)
            points_for_autonomous += 1

        # Temporal signal
        if temporal == "HIGH_HUMAN_INFLUENCE":
            points_for_prompted += 1
        elif temporal == "HIGH_AUTONOMY":
            points_for_autonomous += 1

        # Final classification
        total = points_for_prompted + points_for_autonomous

        if total == 0:
            return "AMBIGUOUS", 0.5

        prompted_pct = points_for_prompted / total

        if prompted_pct > 0.7:
            return "LIKELY_PROMPTED", min(0.95, prompted_pct)
        elif prompted_pct < 0.3:
            return "LIKELY_AUTONOMOUS", min(0.95, 1 - prompted_pct)
        else:
            return "AMBIGUOUS", 0.5 + abs(prompted_pct - 0.5)

    def _generate_results(self, posts_df: pd.DataFrame, classifications: dict) -> dict:
        """Generate summary results."""
        # Count classifications
        class_counts = defaultdict(int)
        for data in classifications.values():
            class_counts[data["classification"]] += 1

        # Confidence distribution
        confidences = [data["confidence"] for data in classifications.values()]

        # By pre/post breach
        posts_df["id"] = posts_df["id"].astype(str)
        pre_breach_ids = set(posts_df[posts_df["is_pre_breach"]]["id"])

        pre_class = defaultdict(int)
        post_class = defaultdict(int)

        for pid, data in classifications.items():
            if pid in pre_breach_ids:
                pre_class[data["classification"]] += 1
            else:
                post_class[data["classification"]] += 1

        return {
            "summary": {
                "total_classified": len(classifications),
                "classification_counts": dict(class_counts),
                "mean_confidence": np.mean(confidences),
            },
            "by_breach": {
                "pre_breach": dict(pre_class),
                "post_breach": dict(post_class),
            },
            "heuristic_stats": {
                "posts_with_prompted_signals": int((posts_df["prompted_score"] > 0).sum()),
                "posts_with_autonomous_signals": int((posts_df["autonomous_score"] > 0).sum()),
            },
        }

    def _log_findings(self, results: dict):
        """Log key findings."""
        counts = results["summary"]["classification_counts"]
        total = results["summary"]["total_classified"]

        self.log_decision(
            decision="Human motivation classification complete",
            rationale=(
                f"Of {total:,} posts: "
                f"LIKELY_PROMPTED={counts.get('LIKELY_PROMPTED', 0)} "
                f"({counts.get('LIKELY_PROMPTED', 0)/total*100:.1f}%), "
                f"LIKELY_AUTONOMOUS={counts.get('LIKELY_AUTONOMOUS', 0)} "
                f"({counts.get('LIKELY_AUTONOMOUS', 0)/total*100:.1f}%), "
                f"AMBIGUOUS={counts.get('AMBIGUOUS', 0)} "
                f"({counts.get('AMBIGUOUS', 0)/total*100:.1f}%)"
            ),
        )


def run_phase():
    """Entry point for running this phase."""
    phase = Phase08HumanMotivation()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
