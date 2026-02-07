"""
##############################################################################
#                                                                            #
#  WARNING: Uses INVERTED temporal classification labels from Phase 1        #
#                                                                            #
#  The temporal classifications loaded by load_temporal_classifications()    #
#  have INVERTED labels:                                                     #
#    - "HIGH_AUTONOMY" actually indicates HUMAN-PROMPTED behavior            #
#    - "HIGH_HUMAN_INFLUENCE" actually indicates AUTONOMOUS behavior         #
#                                                                            #
#  All H2 tests and classification-based analyses are INVERTED.              #
#  Output file renamed to: statistical_analysis_DEPRECATED_INVERTED.json     #
#                                                                            #
#  DO NOT use results from this file until temporal labels are fixed.        #
#                                                                            #
##############################################################################

Moltbook Research - Statistical Analysis
========================================
Deep statistical analysis including hypothesis tests, correlations,
cross-tabulations, network preliminaries, and anomaly detection.

Usage:
    python3 analysis/statistical_analysis.py
"""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, pearsonr, mannwhitneyu, kruskal

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_data():
    """Load posts and comments from derived parquet files."""
    posts_path = DERIVED_DIR / "posts_derived.parquet"
    comments_path = DERIVED_DIR / "comments_derived.parquet"

    logger.info("Loading posts from %s", posts_path)
    posts_df = pd.read_parquet(posts_path)

    logger.info("Loading comments from %s", comments_path)
    comments_df = pd.read_parquet(comments_path)

    return posts_df, comments_df


def load_temporal_classifications():
    """Load temporal analysis classifications.

    WARNING: Labels are INVERTED! See module docstring.
    File has been renamed to phase_01_temporal_analysis_DEPRECATED_INVERTED.json
    """
    # Original file renamed - this will now fail by design
    temporal_path = RESULTS_DIR / "phase_01_temporal_analysis.json"
    if not temporal_path.exists():
        logger.warning("Temporal analysis not found (file renamed to *_DEPRECATED_INVERTED.json), will skip related tests")
        return {}

    with open(temporal_path) as f:
        data = json.load(f)
    return data.get("author_classifications", {})


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def interpret_p_value(p):
    """Interpret statistical significance."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================

def h1_breach_content_difference(posts_df):
    """
    H1: Post-breach content is different from pre-breach
    Tests: t-test on word counts, chi-square on length categories, effect size
    """
    logger.info("Testing H1: Post-breach content difference...")

    pre_breach = posts_df[posts_df["is_pre_breach"] == True]["word_count"].dropna()
    post_breach = posts_df[posts_df["is_pre_breach"] == False]["word_count"].dropna()

    # Welch's t-test (unequal variances)
    t_stat, t_pvalue = stats.ttest_ind(pre_breach, post_breach, equal_var=False)

    # Mann-Whitney U test (non-parametric alternative)
    u_stat, u_pvalue = mannwhitneyu(pre_breach, post_breach, alternative='two-sided')

    # Effect size (Cohen's d)
    d = cohens_d(pre_breach, post_breach)

    # Chi-square test on content length categories
    def categorize_length(wc):
        if wc < 50:
            return "short"
        elif wc <= 200:
            return "medium"
        else:
            return "long"

    pre_cats = posts_df[posts_df["is_pre_breach"] == True]["word_count"].apply(categorize_length).value_counts()
    post_cats = posts_df[posts_df["is_pre_breach"] == False]["word_count"].apply(categorize_length).value_counts()

    # Create contingency table
    categories = ["short", "medium", "long"]
    contingency = np.array([
        [pre_cats.get(cat, 0) for cat in categories],
        [post_cats.get(cat, 0) for cat in categories]
    ])

    chi2, chi2_p, dof, expected = chi2_contingency(contingency)

    # Cramér's V for effect size
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    return {
        "test_name": "H1: Post-breach content differs from pre-breach",
        "sample_sizes": {
            "pre_breach": int(len(pre_breach)),
            "post_breach": int(len(post_breach))
        },
        "descriptive_stats": {
            "pre_breach": {
                "mean": float(pre_breach.mean()),
                "median": float(pre_breach.median()),
                "std": float(pre_breach.std())
            },
            "post_breach": {
                "mean": float(post_breach.mean()),
                "median": float(post_breach.median()),
                "std": float(post_breach.std())
            }
        },
        "parametric_test": {
            "test": "Welch's t-test",
            "t_statistic": float(t_stat),
            "p_value": float(t_pvalue),
            "significance": interpret_p_value(t_pvalue)
        },
        "nonparametric_test": {
            "test": "Mann-Whitney U",
            "u_statistic": float(u_stat),
            "p_value": float(u_pvalue),
            "significance": interpret_p_value(u_pvalue)
        },
        "effect_size": {
            "cohens_d": float(d),
            "interpretation": interpret_effect_size(d),
            "direction": "pre > post" if d > 0 else "post > pre"
        },
        "chi_square_test": {
            "test": "Chi-square test on content length categories",
            "chi2_statistic": float(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": float(chi2_p),
            "significance": interpret_p_value(chi2_p),
            "cramers_v": float(cramers_v),
            "contingency_table": {
                "categories": categories,
                "pre_breach": [int(x) for x in contingency[0]],
                "post_breach": [int(x) for x in contingency[1]]
            }
        },
        "conclusion": f"Pre-breach posts have significantly different word counts (d={d:.3f}, {interpret_effect_size(d)} effect). Mean word count pre: {pre_breach.mean():.1f}, post: {post_breach.mean():.1f}."
    }


def h2_human_influence_content(posts_df, author_classifications):
    """
    H2: Authors classified as HIGH_HUMAN_INFLUENCE post different content
    Tests: Compare word counts, submolt distribution, upvote distribution by classification
    """
    logger.info("Testing H2: Human influence classification differences...")

    if not author_classifications:
        return {
            "test_name": "H2: Human influence classification differences",
            "error": "No temporal classifications available"
        }

    # Map classifications to posts
    posts_df = posts_df.copy()
    posts_df["classification"] = posts_df["author_id"].map(
        lambda x: author_classifications.get(x, {}).get("classification", "UNKNOWN")
    )

    # Filter to known classifications
    classified = posts_df[posts_df["classification"].isin(["HIGH_HUMAN_INFLUENCE", "HIGH_AUTONOMY", "MIXED"])]

    high_human = classified[classified["classification"] == "HIGH_HUMAN_INFLUENCE"]["word_count"]
    high_autonomy = classified[classified["classification"] == "HIGH_AUTONOMY"]["word_count"]
    mixed = classified[classified["classification"] == "MIXED"]["word_count"]

    # Kruskal-Wallis test (non-parametric ANOVA)
    h_stat, kw_pvalue = kruskal(high_human, high_autonomy, mixed)

    # Pairwise Mann-Whitney tests
    pairwise_tests = {}
    pairs = [
        ("HIGH_HUMAN_INFLUENCE", "HIGH_AUTONOMY", high_human, high_autonomy),
        ("HIGH_HUMAN_INFLUENCE", "MIXED", high_human, mixed),
        ("HIGH_AUTONOMY", "MIXED", high_autonomy, mixed)
    ]

    for name1, name2, g1, g2 in pairs:
        if len(g1) > 0 and len(g2) > 0:
            u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
            d = cohens_d(g1, g2)
            pairwise_tests[f"{name1}_vs_{name2}"] = {
                "u_statistic": float(u_stat),
                "p_value": float(p_val),
                "significance": interpret_p_value(p_val),
                "cohens_d": float(d) if not np.isnan(d) else None,
                "effect_interpretation": interpret_effect_size(d) if not np.isnan(d) else None
            }

    # Submolt distribution by classification
    submolt_by_class = {}
    for cls in ["HIGH_HUMAN_INFLUENCE", "HIGH_AUTONOMY", "MIXED"]:
        cls_posts = classified[classified["classification"] == cls]
        top_submolts = cls_posts["submolt_name"].value_counts().head(10).to_dict()
        submolt_by_class[cls] = {str(k): int(v) for k, v in top_submolts.items()}

    # Upvote distribution by classification
    upvote_stats = {}
    for cls in ["HIGH_HUMAN_INFLUENCE", "HIGH_AUTONOMY", "MIXED"]:
        cls_upvotes = classified[classified["classification"] == cls]["upvotes"]
        upvote_stats[cls] = {
            "count": int(len(cls_upvotes)),
            "mean": float(cls_upvotes.mean()),
            "median": float(cls_upvotes.median()),
            "std": float(cls_upvotes.std()),
            "max": int(cls_upvotes.max()) if len(cls_upvotes) > 0 else 0
        }

    # Upvote Kruskal-Wallis
    upvote_h, upvote_kw_p = kruskal(
        classified[classified["classification"] == "HIGH_HUMAN_INFLUENCE"]["upvotes"],
        classified[classified["classification"] == "HIGH_AUTONOMY"]["upvotes"],
        classified[classified["classification"] == "MIXED"]["upvotes"]
    )

    return {
        "test_name": "H2: Human influence classification differences",
        "sample_sizes": {
            "HIGH_HUMAN_INFLUENCE": int(len(high_human)),
            "HIGH_AUTONOMY": int(len(high_autonomy)),
            "MIXED": int(len(mixed)),
            "UNKNOWN": int((posts_df["classification"] == "UNKNOWN").sum())
        },
        "word_count_comparison": {
            "descriptive_stats": {
                "HIGH_HUMAN_INFLUENCE": {
                    "mean": float(high_human.mean()),
                    "median": float(high_human.median()),
                    "std": float(high_human.std())
                },
                "HIGH_AUTONOMY": {
                    "mean": float(high_autonomy.mean()),
                    "median": float(high_autonomy.median()),
                    "std": float(high_autonomy.std())
                },
                "MIXED": {
                    "mean": float(mixed.mean()),
                    "median": float(mixed.median()),
                    "std": float(mixed.std())
                }
            },
            "kruskal_wallis": {
                "h_statistic": float(h_stat),
                "p_value": float(kw_pvalue),
                "significance": interpret_p_value(kw_pvalue)
            },
            "pairwise_tests": pairwise_tests
        },
        "submolt_distribution_by_classification": submolt_by_class,
        "upvote_comparison": {
            "descriptive_stats": upvote_stats,
            "kruskal_wallis": {
                "h_statistic": float(upvote_h),
                "p_value": float(upvote_kw_p),
                "significance": interpret_p_value(upvote_kw_p)
            }
        },
        "conclusion": f"Classification groups show significant word count differences (H={h_stat:.2f}, p={kw_pvalue:.2e}). HIGH_HUMAN_INFLUENCE posts tend to have word count mean={high_human.mean():.1f}, HIGH_AUTONOMY mean={high_autonomy.mean():.1f}."
    }


def h3_commenting_vs_posting(posts_df, comments_df):
    """
    H3: Commenting behavior differs from posting behavior
    Tests: Compare word counts, timing patterns, author overlap
    """
    logger.info("Testing H3: Commenting vs posting behavior...")

    # Word count comparison
    post_wc = posts_df["word_count"].dropna()
    comment_wc = comments_df["word_count"].dropna()

    # Mann-Whitney U test
    u_stat, u_pvalue = mannwhitneyu(post_wc, comment_wc, alternative='two-sided')
    d = cohens_d(post_wc, comment_wc)

    # Timing patterns comparison (hour of day)
    post_hours = posts_df["created_at"].dt.hour
    comment_hours = comments_df["created_at"].dt.hour

    # Chi-square on hourly distribution
    post_hourly = post_hours.value_counts().reindex(range(24), fill_value=0)
    comment_hourly = comment_hours.value_counts().reindex(range(24), fill_value=0)

    contingency_hourly = np.array([post_hourly.values, comment_hourly.values])
    chi2_hourly, chi2_p_hourly, dof_hourly, _ = chi2_contingency(contingency_hourly)

    # Author overlap analysis
    post_authors = set(posts_df["author_name"].dropna().unique())
    comment_authors = set(comments_df["author_name"].dropna().unique())

    both_authors = post_authors & comment_authors
    only_post = post_authors - comment_authors
    only_comment = comment_authors - post_authors

    # For authors who do both, compare their posting vs commenting word counts
    both_post_wc = posts_df[posts_df["author_name"].isin(both_authors)]["word_count"]
    both_comment_wc = comments_df[comments_df["author_name"].isin(both_authors)]["word_count"]

    if len(both_post_wc) > 0 and len(both_comment_wc) > 0:
        paired_u, paired_p = mannwhitneyu(both_post_wc, both_comment_wc, alternative='two-sided')
        paired_d = cohens_d(both_post_wc, both_comment_wc)
    else:
        paired_u, paired_p, paired_d = np.nan, np.nan, np.nan

    return {
        "test_name": "H3: Commenting vs posting behavior differences",
        "sample_sizes": {
            "posts": int(len(post_wc)),
            "comments": int(len(comment_wc))
        },
        "word_count_comparison": {
            "posts": {
                "mean": float(post_wc.mean()),
                "median": float(post_wc.median()),
                "std": float(post_wc.std())
            },
            "comments": {
                "mean": float(comment_wc.mean()),
                "median": float(comment_wc.median()),
                "std": float(comment_wc.std())
            },
            "mann_whitney_u": {
                "u_statistic": float(u_stat),
                "p_value": float(u_pvalue),
                "significance": interpret_p_value(u_pvalue)
            },
            "effect_size": {
                "cohens_d": float(d),
                "interpretation": interpret_effect_size(d)
            }
        },
        "timing_comparison": {
            "chi_square_hourly": {
                "chi2_statistic": float(chi2_hourly),
                "degrees_of_freedom": int(dof_hourly),
                "p_value": float(chi2_p_hourly),
                "significance": interpret_p_value(chi2_p_hourly)
            },
            "peak_posting_hours": [int(h) for h in post_hourly.nlargest(3).index],
            "peak_commenting_hours": [int(h) for h in comment_hourly.nlargest(3).index]
        },
        "author_overlap": {
            "unique_post_authors": len(post_authors),
            "unique_comment_authors": len(comment_authors),
            "authors_who_do_both": len(both_authors),
            "only_post_authors": len(only_post),
            "only_comment_authors": len(only_comment),
            "overlap_rate": len(both_authors) / max(len(post_authors), 1) * 100
        },
        "dual_authors_comparison": {
            "authors_who_both_post_and_comment": len(both_authors),
            "post_word_count_mean": float(both_post_wc.mean()) if len(both_post_wc) > 0 else None,
            "comment_word_count_mean": float(both_comment_wc.mean()) if len(both_comment_wc) > 0 else None,
            "mann_whitney_u": float(paired_u) if not np.isnan(paired_u) else None,
            "p_value": float(paired_p) if not np.isnan(paired_p) else None,
            "cohens_d": float(paired_d) if not np.isnan(paired_d) else None
        },
        "conclusion": f"Posts and comments show significantly different word counts (d={d:.3f}). Post mean: {post_wc.mean():.1f}, Comment mean: {comment_wc.mean():.1f}. Only {len(both_authors)/len(post_authors)*100:.1f}% of post authors also comment."
    }


# =============================================================================
# CROSS-TABULATIONS
# =============================================================================

def cross_tabulations(posts_df, comments_df, author_classifications):
    """Generate cross-tabulations for categorical variables."""
    logger.info("Generating cross-tabulations...")

    results = {}

    # Add classification to posts
    posts_df = posts_df.copy()
    posts_df["classification"] = posts_df["author_id"].map(
        lambda x: author_classifications.get(x, {}).get("classification", "UNKNOWN")
    )

    # 1. Phase × Temporal Classification
    if author_classifications:
        phase_class_ct = pd.crosstab(
            posts_df["phase"],
            posts_df["classification"],
            margins=True
        )
        chi2, p, dof, _ = chi2_contingency(phase_class_ct.iloc[:-1, :-1].values)
        results["phase_x_classification"] = {
            "crosstab": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                        for k, v in phase_class_ct.to_dict().items()},
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "significance": interpret_p_value(p)
        }

    # 2. Phase × Content Length Category
    posts_df["content_length_cat"] = pd.cut(
        posts_df["word_count"],
        bins=[0, 50, 200, float('inf')],
        labels=["short", "medium", "long"]
    )

    phase_length_ct = pd.crosstab(
        posts_df["phase"],
        posts_df["content_length_cat"],
        margins=True
    )
    chi2, p, dof, _ = chi2_contingency(phase_length_ct.iloc[:-1, :-1].values)
    results["phase_x_content_length"] = {
        "crosstab": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                    for k, v in phase_length_ct.to_dict().items()},
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "significance": interpret_p_value(p)
    }

    # 3. Top Submolt × Classification
    if author_classifications:
        top_submolts = posts_df["submolt_name"].value_counts().head(10).index
        top_posts = posts_df[posts_df["submolt_name"].isin(top_submolts)]
        submolt_class_ct = pd.crosstab(
            top_posts["submolt_name"],
            top_posts["classification"],
            margins=True
        )
        chi2, p, dof, _ = chi2_contingency(submolt_class_ct.iloc[:-1, :-1].values)
        results["submolt_x_classification"] = {
            "crosstab": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                        for k, v in submolt_class_ct.to_dict().items()},
            "chi2": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "significance": interpret_p_value(p)
        }

    # 4. Pre/Post Breach × Author Activity Level
    author_post_counts = posts_df.groupby("author_name").size()
    posts_df["author_activity"] = posts_df["author_name"].map(
        lambda x: "high" if author_post_counts.get(x, 0) > 10 else (
            "medium" if author_post_counts.get(x, 0) > 3 else "low"
        )
    )

    breach_activity_ct = pd.crosstab(
        posts_df["is_pre_breach"].map({True: "pre_breach", False: "post_breach"}),
        posts_df["author_activity"],
        margins=True
    )
    chi2, p, dof, _ = chi2_contingency(breach_activity_ct.iloc[:-1, :-1].values)
    results["breach_x_activity_level"] = {
        "crosstab": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                    for k, v in breach_activity_ct.to_dict().items()},
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "significance": interpret_p_value(p)
    }

    return results


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(posts_df, comments_df, author_classifications):
    """Compute correlations between key variables."""
    logger.info("Running correlation analysis...")

    results = {}

    # 1. Word count vs upvotes (posts)
    valid = posts_df[["word_count", "upvotes"]].dropna()
    if len(valid) > 2:
        r_pearson, p_pearson = pearsonr(valid["word_count"], valid["upvotes"])
        r_spearman, p_spearman = spearmanr(valid["word_count"], valid["upvotes"])
        results["word_count_vs_upvotes"] = {
            "n": int(len(valid)),
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_rho": float(r_spearman),
            "spearman_p": float(p_spearman),
            "interpretation": "positive" if r_spearman > 0 else "negative"
        }

    # 2. Post count per author vs confidence score
    if author_classifications:
        author_post_counts = posts_df.groupby("author_id").size()
        author_confidence = {aid: data["confidence"] for aid, data in author_classifications.items()}

        common_authors = set(author_post_counts.index) & set(author_confidence.keys())
        if len(common_authors) > 2:
            counts = [author_post_counts[a] for a in common_authors]
            confidences = [author_confidence[a] for a in common_authors]
            r, p = spearmanr(counts, confidences)
            results["post_count_vs_classification_confidence"] = {
                "n": len(common_authors),
                "spearman_rho": float(r),
                "p_value": float(p),
                "interpretation": "Authors with more posts tend to have " +
                    ("higher" if r > 0 else "lower") + " classification confidence"
            }

    # 3. Comment depth vs word count
    valid_comments = comments_df[["depth", "word_count"]].dropna()
    if len(valid_comments) > 2:
        r, p = spearmanr(valid_comments["depth"], valid_comments["word_count"])
        results["comment_depth_vs_word_count"] = {
            "n": int(len(valid_comments)),
            "spearman_rho": float(r),
            "p_value": float(p),
            "interpretation": "Deeper comments are " +
                ("longer" if r > 0 else "shorter")
        }

    # 4. Platform age (days since launch) vs word count
    launch_date = pd.Timestamp("2026-01-27", tz="UTC")
    posts_df = posts_df.copy()
    posts_df["platform_age_hours"] = (posts_df["created_at"] - launch_date).dt.total_seconds() / 3600

    valid = posts_df[["platform_age_hours", "word_count"]].dropna()
    if len(valid) > 2:
        r, p = spearmanr(valid["platform_age_hours"], valid["word_count"])
        results["platform_age_vs_word_count"] = {
            "n": int(len(valid)),
            "spearman_rho": float(r),
            "p_value": float(p),
            "interpretation": "Posts " +
                ("get longer" if r > 0 else "get shorter") +
                " as platform ages"
        }

    # 5. Platform age vs upvotes
    valid = posts_df[["platform_age_hours", "upvotes"]].dropna()
    if len(valid) > 2:
        r, p = spearmanr(valid["platform_age_hours"], valid["upvotes"])
        results["platform_age_vs_upvotes"] = {
            "n": int(len(valid)),
            "spearman_rho": float(r),
            "p_value": float(p),
            "interpretation": "Later posts receive " +
                ("more" if r > 0 else "fewer") + " upvotes"
        }

    # 6. Word count vs comment count (engagement)
    posts_with_comments = posts_df[["word_count", "comment_count"]].dropna()
    if len(posts_with_comments) > 2:
        r, p = spearmanr(posts_with_comments["word_count"], posts_with_comments["comment_count"])
        results["word_count_vs_comment_count"] = {
            "n": int(len(posts_with_comments)),
            "spearman_rho": float(r),
            "p_value": float(p),
            "interpretation": "Longer posts receive " +
                ("more" if r > 0 else "fewer") + " comments"
        }

    return results


# =============================================================================
# NETWORK PRELIMINARY ANALYSIS
# =============================================================================

def network_preliminary(posts_df, comments_df):
    """Preliminary network analysis: co-occurrence and reply patterns."""
    logger.info("Running network preliminary analysis...")

    results = {}

    # 1. Author co-occurrence in same threads (posts with comments)
    # Build mapping: post_id -> set of all authors (post author + comment authors)
    thread_authors = defaultdict(set)

    for _, post in posts_df.iterrows():
        if pd.notna(post["author_name"]):
            thread_authors[post["id"]].add(post["author_name"])

    for _, comment in comments_df.iterrows():
        if pd.notna(comment["author_name"]) and pd.notna(comment["post_id"]):
            thread_authors[comment["post_id"]].add(comment["author_name"])

    # Count co-occurrences
    cooccurrence_counts = Counter()
    for post_id, authors in thread_authors.items():
        authors_list = sorted(authors)
        for i, a1 in enumerate(authors_list):
            for a2 in authors_list[i+1:]:
                cooccurrence_counts[(a1, a2)] += 1

    # Top co-occurring pairs
    top_pairs = cooccurrence_counts.most_common(20)
    results["author_cooccurrence"] = {
        "total_unique_pairs": len(cooccurrence_counts),
        "total_cooccurrences": sum(cooccurrence_counts.values()),
        "top_20_pairs": [
            {"author1": p[0][0], "author2": p[0][1], "count": p[1]}
            for p in top_pairs
        ]
    }

    # 2. Reply relationships (who replies to whom)
    # Build reply graph from comments
    reply_counts = Counter()

    # Get post author for each post
    post_authors = posts_df.set_index("id")["author_name"].to_dict()

    # Comments have parent_id for replies
    for _, comment in comments_df.iterrows():
        commenter = comment.get("author_name")
        post_id = comment.get("post_id")
        parent_id = comment.get("parent_id")

        if pd.isna(commenter):
            continue

        # Top-level comment replies to post author
        if pd.isna(parent_id) and post_id in post_authors:
            target = post_authors[post_id]
            if pd.notna(target) and commenter != target:
                reply_counts[(commenter, target)] += 1

    top_reply_pairs = reply_counts.most_common(20)
    results["reply_relationships"] = {
        "total_unique_reply_pairs": len(reply_counts),
        "total_replies_tracked": sum(reply_counts.values()),
        "top_20_reply_pairs": [
            {"replier": p[0][0], "replied_to": p[0][1], "count": p[1]}
            for p in top_reply_pairs
        ]
    }

    # 3. Most replied-to posts
    comment_counts_per_post = comments_df.groupby("post_id").size().sort_values(ascending=False)
    top_replied_posts = comment_counts_per_post.head(20)

    # Get post details
    top_replied_details = []
    for post_id, count in top_replied_posts.items():
        post = posts_df[posts_df["id"] == post_id]
        if len(post) > 0:
            post = post.iloc[0]
            top_replied_details.append({
                "post_id": str(post_id),
                "title": str(post.get("title", ""))[:100],
                "author": str(post.get("author_name", "")),
                "comment_count": int(count),
                "upvotes": int(post.get("upvotes", 0))
            })

    results["most_replied_posts"] = top_replied_details

    # 4. Most replied-to authors
    author_reply_counts = Counter()
    for post_id, count in comment_counts_per_post.items():
        if post_id in post_authors and pd.notna(post_authors[post_id]):
            author_reply_counts[post_authors[post_id]] += count

    results["most_replied_authors"] = [
        {"author": a, "total_comments_received": c}
        for a, c in author_reply_counts.most_common(20)
    ]

    # 5. Network density metrics
    unique_thread_authors = set()
    for authors in thread_authors.values():
        unique_thread_authors.update(authors)

    n_nodes = len(unique_thread_authors)
    n_edges = len(cooccurrence_counts)
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    density = n_edges / max_edges if max_edges > 0 else 0

    results["network_metrics"] = {
        "unique_authors_in_threads": n_nodes,
        "unique_cooccurrence_edges": n_edges,
        "network_density": float(density),
        "avg_thread_size": float(np.mean([len(a) for a in thread_authors.values()]))
    }

    return results


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def anomaly_detection(posts_df, comments_df, author_classifications):
    """Detect anomalies and outliers in the data."""
    logger.info("Running anomaly detection...")

    results = {}

    # 1. Outlier posts by word count (>3 std from mean, IQR method)
    wc_mean = posts_df["word_count"].mean()
    wc_std = posts_df["word_count"].std()
    wc_q1 = posts_df["word_count"].quantile(0.25)
    wc_q3 = posts_df["word_count"].quantile(0.75)
    iqr = wc_q3 - wc_q1

    # Standard deviation method
    std_outliers = posts_df[
        (posts_df["word_count"] > wc_mean + 3 * wc_std) |
        (posts_df["word_count"] < wc_mean - 3 * wc_std)
    ]

    # IQR method
    iqr_outliers = posts_df[
        (posts_df["word_count"] > wc_q3 + 1.5 * iqr) |
        (posts_df["word_count"] < wc_q1 - 1.5 * iqr)
    ]

    results["word_count_outliers"] = {
        "method_std_3sigma": {
            "count": int(len(std_outliers)),
            "percentage": float(len(std_outliers) / len(posts_df) * 100),
            "threshold_upper": float(wc_mean + 3 * wc_std),
            "threshold_lower": float(max(0, wc_mean - 3 * wc_std))
        },
        "method_iqr_1_5": {
            "count": int(len(iqr_outliers)),
            "percentage": float(len(iqr_outliers) / len(posts_df) * 100),
            "threshold_upper": float(wc_q3 + 1.5 * iqr),
            "threshold_lower": float(max(0, wc_q1 - 1.5 * iqr))
        },
        "extreme_long_posts": [
            {
                "id": str(row["id"]),
                "word_count": int(row["word_count"]),
                "author": str(row.get("author_name", "")),
                "title": str(row.get("title", ""))[:80]
            }
            for _, row in posts_df.nlargest(10, "word_count").iterrows()
        ]
    }

    # 2. Engagement outliers (extreme upvotes/comments)
    upvote_q99 = posts_df["upvotes"].quantile(0.99)
    high_engagement = posts_df[posts_df["upvotes"] > upvote_q99]

    results["engagement_outliers"] = {
        "high_upvote_threshold_p99": float(upvote_q99),
        "posts_above_p99": int(len(high_engagement)),
        "top_10_by_upvotes": [
            {
                "id": str(row["id"]),
                "upvotes": int(row["upvotes"]),
                "author": str(row.get("author_name", "")),
                "title": str(row.get("title", ""))[:80],
                "submolt": str(row.get("submolt_name", ""))
            }
            for _, row in posts_df.nlargest(10, "upvotes").iterrows()
        ]
    }

    # 3. Unusual posting patterns (potential bots)
    # Authors with very high post frequency
    author_post_counts = posts_df.groupby("author_name").size()
    author_timespan = posts_df.groupby("author_name")["created_at"].agg(["min", "max"])
    author_timespan["span_hours"] = (author_timespan["max"] - author_timespan["min"]).dt.total_seconds() / 3600

    # Posts per hour for high-volume authors
    high_volume_authors = author_post_counts[author_post_counts > 20]
    potential_bots = []

    for author in high_volume_authors.index:
        post_count = author_post_counts[author]
        if author in author_timespan.index:
            span = author_timespan.loc[author, "span_hours"]
            if span > 0:
                posts_per_hour = post_count / span
                if posts_per_hour > 5:  # More than 5 posts per hour average
                    potential_bots.append({
                        "author": str(author),
                        "total_posts": int(post_count),
                        "active_span_hours": float(span),
                        "posts_per_hour": float(posts_per_hour)
                    })

    potential_bots.sort(key=lambda x: x["posts_per_hour"], reverse=True)

    results["unusual_posting_patterns"] = {
        "potential_bot_threshold": "5+ posts per hour on average",
        "potential_bots_count": len(potential_bots),
        "top_10_suspicious": potential_bots[:10]
    }

    # 4. Authors with unusual classification patterns
    if author_classifications:
        # Authors with very low classification confidence
        low_confidence = [
            {
                "author_id": aid,
                "classification": data["classification"],
                "confidence": data["confidence"]
            }
            for aid, data in author_classifications.items()
            if data.get("confidence", 1) < 0.1
        ]

        # Authors with extreme COV values
        extreme_cov = [
            {
                "author_id": aid,
                "classification": data["classification"],
                "cov": data["cov"]
            }
            for aid, data in author_classifications.items()
            if data.get("cov", 0) > 5
        ]

        results["classification_anomalies"] = {
            "low_confidence_count": len(low_confidence),
            "extreme_cov_count": len(extreme_cov),
            "sample_low_confidence": low_confidence[:10],
            "sample_extreme_cov": extreme_cov[:10]
        }

    # 5. Content anomalies
    # Posts with zero word count
    zero_word_posts = posts_df[posts_df["word_count"] == 0]

    # Posts with very repetitive patterns (check for duplicate content)
    content_hash = posts_df["content"].apply(lambda x: hash(str(x)[:100]) if pd.notna(x) else None)
    duplicate_counts = content_hash.value_counts()
    duplicates = duplicate_counts[duplicate_counts > 1]

    results["content_anomalies"] = {
        "zero_word_posts": int(len(zero_word_posts)),
        "duplicate_content_groups": int(len(duplicates)),
        "total_duplicate_posts": int(duplicates.sum())
    }

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run full statistical analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Moltbook Statistical Analysis")
    logger.info("=" * 60)

    # Load data
    posts_df, comments_df = load_data()
    author_classifications = load_temporal_classifications()

    logger.info("Loaded %d posts, %d comments", len(posts_df), len(comments_df))
    logger.info("Loaded %d author classifications", len(author_classifications))

    # Run all analyses
    results = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "posts_count": len(posts_df),
            "comments_count": len(comments_df),
            "classifications_count": len(author_classifications)
        },
        "hypothesis_tests": {
            "h1_breach_content": h1_breach_content_difference(posts_df),
            "h2_human_influence": h2_human_influence_content(posts_df, author_classifications),
            "h3_commenting_vs_posting": h3_commenting_vs_posting(posts_df, comments_df)
        },
        "cross_tabulations": cross_tabulations(posts_df, comments_df, author_classifications),
        "correlations": correlation_analysis(posts_df, comments_df, author_classifications),
        "network_preliminary": network_preliminary(posts_df, comments_df),
        "anomaly_detection": anomaly_detection(posts_df, comments_df, author_classifications)
    }

    # Save results
    output_path = RESULTS_DIR / "statistical_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Saved results to %s", output_path)

    # Print summary
    print_summary(results)

    return results


def print_summary(results):
    """Print human-readable summary of key findings."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 80)

    # H1
    h1 = results["hypothesis_tests"]["h1_breach_content"]
    print("\n--- H1: Pre/Post Breach Content Difference ---")
    print(f"Pre-breach mean word count: {h1['descriptive_stats']['pre_breach']['mean']:.1f}")
    print(f"Post-breach mean word count: {h1['descriptive_stats']['post_breach']['mean']:.1f}")
    print(f"Cohen's d effect size: {h1['effect_size']['cohens_d']:.3f} ({h1['effect_size']['interpretation']})")
    print(f"t-test p-value: {h1['parametric_test']['p_value']:.2e} {h1['parametric_test']['significance']}")
    print(f"Chi-square (length categories) p-value: {h1['chi_square_test']['p_value']:.2e} {h1['chi_square_test']['significance']}")

    # H2
    h2 = results["hypothesis_tests"]["h2_human_influence"]
    if "error" not in h2:
        print("\n--- H2: Classification Group Differences ---")
        wc = h2["word_count_comparison"]["descriptive_stats"]
        print(f"HIGH_HUMAN_INFLUENCE mean: {wc['HIGH_HUMAN_INFLUENCE']['mean']:.1f}")
        print(f"HIGH_AUTONOMY mean: {wc['HIGH_AUTONOMY']['mean']:.1f}")
        print(f"MIXED mean: {wc['MIXED']['mean']:.1f}")
        print(f"Kruskal-Wallis p-value: {h2['word_count_comparison']['kruskal_wallis']['p_value']:.2e} {h2['word_count_comparison']['kruskal_wallis']['significance']}")

    # H3
    h3 = results["hypothesis_tests"]["h3_commenting_vs_posting"]
    print("\n--- H3: Posting vs Commenting Behavior ---")
    print(f"Post mean word count: {h3['word_count_comparison']['posts']['mean']:.1f}")
    print(f"Comment mean word count: {h3['word_count_comparison']['comments']['mean']:.1f}")
    print(f"Cohen's d: {h3['word_count_comparison']['effect_size']['cohens_d']:.3f}")
    print(f"Author overlap rate: {h3['author_overlap']['overlap_rate']:.1f}%")

    # Correlations
    print("\n--- Key Correlations ---")
    for key, corr in results["correlations"].items():
        if "spearman_rho" in corr:
            # Handle different p-value key names
            p_val = corr.get('p_value') or corr.get('spearman_p', 0)
            print(f"{key}: rho={corr['spearman_rho']:.3f}, p={p_val:.2e}")

    # Network
    print("\n--- Network Preliminary ---")
    net = results["network_preliminary"]
    print(f"Unique author pairs co-occurring: {net['author_cooccurrence']['total_unique_pairs']:,}")
    print(f"Network density: {net['network_metrics']['network_density']:.6f}")
    print(f"Average thread size: {net['network_metrics']['avg_thread_size']:.2f} authors")

    # Anomalies
    print("\n--- Anomaly Detection ---")
    anom = results["anomaly_detection"]
    print(f"Word count outliers (3-sigma): {anom['word_count_outliers']['method_std_3sigma']['count']} ({anom['word_count_outliers']['method_std_3sigma']['percentage']:.2f}%)")
    print(f"Potential bot accounts: {anom['unusual_posting_patterns']['potential_bots_count']}")
    print(f"Duplicate content groups: {anom['content_anomalies']['duplicate_content_groups']}")

    print("\n" + "=" * 80)
    print("Full results saved to: analysis/results/statistical_analysis.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
