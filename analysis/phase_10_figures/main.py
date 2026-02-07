"""
##############################################################################
#                                                                            #
#  WARNING: Uses INVERTED temporal classification labels from Phase 1        #
#                                                                            #
#  Figures/tables showing temporal classifications are INVERTED:             #
#    - "HIGH_AUTONOMY" is actually HUMAN-PROMPTED                            #
#    - "HIGH_HUMAN_INFLUENCE" is actually AUTONOMOUS                         #
#                                                                            #
#  DO NOT generate publication figures until Phase 1 labels are fixed.       #
#                                                                            #
##############################################################################

Phase 10: Publication Figures and Tables

Generates publication-ready outputs:
- Figures for the paper
- LaTeX tables
- Summary statistics

All outputs are auto-regenerated when upstream data changes.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.base import AnalysisPhase
from pipeline.config import (
    FIGURES_DIR,
    TABLES_DIR,
    FIGURE_DPI,
    FIGURE_FORMAT,
)


class Phase10Figures(AnalysisPhase):
    """Generate publication figures and tables."""

    phase_id = "phase_10_figures"
    dependencies = [
        "phase_01_temporal",
        "phase_03_restart",
        "phase_04_topics",
        "phase_05_depth_gradient",
        "phase_07_convergence",
    ]

    def run(self):
        """Execute Phase 10: Generate figures and tables."""
        self.log_decision(
            decision="Generate publication-ready outputs",
            rationale="Auto-regenerate figures and tables from analysis results",
        )

        # Load results
        results_dir = Path(__file__).parent.parent / "results"
        results = self._load_all_results(results_dir)

        # Generate figures
        print("  Generating figures...")
        self._generate_figures(results)

        # Generate tables
        print("  Generating tables...")
        self._generate_tables(results)

        # Generate summary document
        self._generate_summary(results)

    def _load_all_results(self, results_dir: Path) -> dict:
        """Load all analysis results."""
        results = {}

        result_files = [
            "phase_00_audit_summary.json",
            "phase_01_temporal_analysis.json",
            "phase_03_breach_split.json",
            "phase_04_topic_analysis.json",
            "phase_05_depth_gradient.json",
            "phase_06_myth_genealogy.json",
            "phase_07_convergence.json",
            "phase_08_human_motivation.json",
            "phase_09_prompt_injection.json",
        ]

        for filename in result_files:
            filepath = results_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    key = filename.replace(".json", "")
                    results[key] = json.load(f)

        return results

    def _generate_figures(self, results: dict):
        """Generate all figures."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("  WARNING: matplotlib not installed, skipping figures")
            self._save_figure_specs(results)
            return

        # Figure 1: Platform activity timeline
        self._fig_activity_timeline(results, plt)

        # Figure 2: Author classification distribution
        self._fig_author_classification(results, plt)

        # Figure 3: Pre/post breach comparison
        self._fig_breach_comparison(results, plt)

        # Figure 4: Topic distribution
        self._fig_topic_distribution(results, plt)

        # Figure 5: Human motivation breakdown
        self._fig_motivation_breakdown(results, plt)

        print(f"  Figures saved to {FIGURES_DIR}")

    def _fig_activity_timeline(self, results: dict, plt):
        """Figure 1: Platform activity over time."""
        audit = results.get("phase_00_audit_summary", {})
        posts_by_phase = audit.get("posts", {}).get("by_phase", {})

        if not posts_by_phase:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        phases = ["genesis", "growth", "breach", "shutdown", "restoration"]
        counts = [posts_by_phase.get(p, 0) for p in phases]

        colors = ["#4CAF50", "#2196F3", "#FF5722", "#9E9E9E", "#FFC107"]
        bars = ax.bar(phases, counts, color=colors)

        ax.set_xlabel("Platform Phase")
        ax.set_ylabel("Number of Posts")
        ax.set_title("Moltbook Activity by Platform Phase")

        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 500,
                f"{count:,}",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig1_activity_timeline.{FIGURE_FORMAT}", dpi=FIGURE_DPI)
        plt.close()

    def _fig_author_classification(self, results: dict, plt):
        """Figure 2: Temporal classification distribution."""
        temporal = results.get("phase_01_temporal_analysis", {})
        summary = temporal.get("summary", {})
        class_counts = summary.get("classification_counts", {})

        if not class_counts:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors = ["#4CAF50", "#FFC107", "#FF5722"]

        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)])
        ax.set_title("Agent Temporal Classification\n(Based on Posting Patterns)")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig2_author_classification.{FIGURE_FORMAT}", dpi=FIGURE_DPI)
        plt.close()

    def _fig_breach_comparison(self, results: dict, plt):
        """Figure 3: Pre vs post breach comparison."""
        breach = results.get("phase_03_breach_split", {})
        posts = breach.get("posts", {})

        pre = posts.get("pre_breach", {})
        post = posts.get("post_breach", {})

        if not pre or not post:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: Post counts
        ax = axes[0]
        categories = ["Posts", "Authors", "Long Posts"]
        pre_vals = [pre.get("count", 0), pre.get("unique_authors", 0), pre.get("long_posts", 0)]
        post_vals = [post.get("count", 0), post.get("unique_authors", 0), post.get("long_posts", 0)]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, pre_vals, width, label="Pre-Breach", color="#4CAF50")
        ax.bar(x + width/2, post_vals, width, label="Post-Breach", color="#FF5722")

        ax.set_ylabel("Count")
        ax.set_title("A. Volume Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        # Panel B: Word count distribution
        ax = axes[1]
        pre_mean = pre.get("mean_word_count", 0)
        post_mean = post.get("mean_word_count", 0)

        ax.bar(["Pre-Breach", "Post-Breach"], [pre_mean, post_mean], color=["#4CAF50", "#FF5722"])
        ax.set_ylabel("Mean Word Count")
        ax.set_title("B. Content Length")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig3_breach_comparison.{FIGURE_FORMAT}", dpi=FIGURE_DPI)
        plt.close()

    def _fig_topic_distribution(self, results: dict, plt):
        """Figure 4: Topic distribution."""
        topics = results.get("phase_04_topic_analysis", {})
        topic_info = topics.get("topic_info", [])

        if not topic_info:
            return

        # Convert to DataFrame and get top 10 non-outlier topics
        df = pd.DataFrame(topic_info)
        df = df[df["Topic"] != -1].nlargest(10, "Count")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(df["Name"], df["Count"], color="#2196F3")
        ax.set_xlabel("Number of Posts")
        ax.set_title("Top 10 Topics on Moltbook")
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig4_topic_distribution.{FIGURE_FORMAT}", dpi=FIGURE_DPI)
        plt.close()

    def _fig_motivation_breakdown(self, results: dict, plt):
        """Figure 5: Human motivation classification."""
        motivation = results.get("phase_08_human_motivation", {})
        summary = motivation.get("summary", {})
        class_counts = summary.get("classification_counts", {})

        if not class_counts:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors = ["#FF5722", "#4CAF50", "#9E9E9E"]

        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)])
        ax.set_title("Human Motivation Classification\n(Multi-Signal Analysis)")

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"fig5_motivation_breakdown.{FIGURE_FORMAT}", dpi=FIGURE_DPI)
        plt.close()

    def _save_figure_specs(self, results: dict):
        """Save figure specifications if matplotlib unavailable."""
        specs = {
            "fig1_activity_timeline": {
                "title": "Platform Activity by Phase",
                "data": results.get("phase_00_audit_summary", {}),
            },
            "fig2_author_classification": {
                "title": "Temporal Classification",
                "data": results.get("phase_01_temporal_analysis", {}).get("summary", {}),
            },
        }

        with open(FIGURES_DIR / "figure_specs.json", "w") as f:
            json.dump(specs, f, indent=2, default=str)

    def _generate_tables(self, results: dict):
        """Generate LaTeX tables."""
        # Table 1: Dataset overview
        self._table_dataset_overview(results)

        # Table 2: Classification summary
        self._table_classification_summary(results)

        print(f"  Tables saved to {TABLES_DIR}")

    def _table_dataset_overview(self, results: dict):
        """Table 1: Dataset overview."""
        audit = results.get("phase_00_audit_summary", {})

        posts = audit.get("posts", {})
        comments = audit.get("comments", {})

        latex = r"""
\begin{table}[h]
\centering
\caption{Moltbook Dataset Overview}
\label{tab:dataset}
\begin{tabular}{lrr}
\hline
\textbf{Metric} & \textbf{Posts} & \textbf{Comments} \\
\hline
Total Count & """ + f"{posts.get('total', 0):,}" + r""" & """ + f"{comments.get('total', 0):,}" + r""" \\
Unique Authors & """ + f"{posts.get('unique_authors', 0):,}" + r""" & """ + f"{comments.get('unique_authors', 0):,}" + r""" \\
Pre-Breach & """ + f"{posts.get('pre_breach', 0):,}" + r""" & """ + f"{comments.get('pre_breach', 0):,}" + r""" \\
Post-Breach & """ + f"{posts.get('post_breach', 0):,}" + r""" & """ + f"{comments.get('post_breach', 0):,}" + r""" \\
\hline
\end{tabular}
\end{table}
"""

        with open(TABLES_DIR / "table1_dataset_overview.tex", "w") as f:
            f.write(latex)

    def _table_classification_summary(self, results: dict):
        """Table 2: Classification summary."""
        temporal = results.get("phase_01_temporal_analysis", {}).get("summary", {})
        motivation = results.get("phase_08_human_motivation", {}).get("summary", {})

        temp_class = temporal.get("classification_counts", {})
        mot_class = motivation.get("classification_counts", {})

        latex = r"""
\begin{table}[h]
\centering
\caption{Agent Classification Summary}
\label{tab:classification}
\begin{tabular}{lrr}
\hline
\textbf{Classification} & \textbf{Temporal} & \textbf{Motivation} \\
\hline
"""

        # Temporal classifications
        for cls in ["HIGH_AUTONOMY", "MIXED", "HIGH_HUMAN_INFLUENCE"]:
            count = temp_class.get(cls, 0)
            latex += f"{cls.replace('_', ' ').title()} & {count:,} & -- \\\\\n"

        latex += r"\hline" + "\n"

        # Motivation classifications
        for cls in ["LIKELY_PROMPTED", "AMBIGUOUS", "LIKELY_AUTONOMOUS"]:
            count = mot_class.get(cls, 0)
            latex += f"{cls.replace('_', ' ').title()} & -- & {count:,} \\\\\n"

        latex += r"""
\hline
\end{tabular}
\end{table}
"""

        with open(TABLES_DIR / "table2_classification_summary.tex", "w") as f:
            f.write(latex)

    def _generate_summary(self, results: dict):
        """Generate summary document."""
        summary = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "figures_generated": list(FIGURES_DIR.glob(f"*.{FIGURE_FORMAT}")),
            "tables_generated": list(TABLES_DIR.glob("*.tex")),
            "key_findings": self._extract_key_findings(results),
        }

        # Convert Path objects to strings
        summary["figures_generated"] = [str(f.name) for f in summary["figures_generated"]]
        summary["tables_generated"] = [str(f.name) for f in summary["tables_generated"]]

        self.save_results(summary, "phase_10_outputs_summary.json")

    def _extract_key_findings(self, results: dict) -> list:
        """Extract key findings from results."""
        findings = []

        # Dataset size
        audit = results.get("phase_00_audit_summary", {})
        if audit:
            findings.append(
                f"Dataset: {audit.get('posts', {}).get('total', 0):,} posts from "
                f"{audit.get('posts', {}).get('unique_authors', 0):,} authors"
            )

        # Breach split
        breach = results.get("phase_03_breach_split", {})
        if breach:
            pre = breach.get("posts", {}).get("pre_breach", {}).get("count", 0)
            post = breach.get("posts", {}).get("post_breach", {}).get("count", 0)
            findings.append(f"Pre-breach: {pre:,} posts, Post-breach: {post:,} posts")

        # Temporal classification
        temporal = results.get("phase_01_temporal_analysis", {}).get("summary", {})
        if temporal:
            counts = temporal.get("classification_counts", {})
            findings.append(f"Temporal: {counts.get('HIGH_AUTONOMY', 0)} high autonomy agents")

        # Motivation
        motivation = results.get("phase_08_human_motivation", {}).get("summary", {})
        if motivation:
            counts = motivation.get("classification_counts", {})
            findings.append(
                f"Motivation: {counts.get('LIKELY_PROMPTED', 0)} likely prompted, "
                f"{counts.get('LIKELY_AUTONOMOUS', 0)} likely autonomous"
            )

        return findings


def run_phase():
    """Entry point for running this phase."""
    phase = Phase10Figures()
    return phase.execute()


if __name__ == "__main__":
    run_phase()
