#!/usr/bin/env python3
"""
Generate submission-ready CSV tables from Moltbook supplementary materials.

This script creates publication-quality CSV files for all supplementary tables
following Nature/Science submission guidelines.

Tables generated:
- Table_S1_Temporal_Classification.csv
- Table_S2_Signal_Convergence.csv
- Table_S3_Statistical_Tests.csv
- Table_S4_Myth_Genealogy.csv
- Table_S5_Super_Commenters.csv
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any


def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def write_csv_table(filepath: str, table_name: str, headers: List[str], rows: List[List[Any]]) -> None:
    """Write a table to CSV with proper formatting."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write table name as first row (optional, helps with identification)
        writer.writerow([table_name])
        writer.writerow([])  # Blank line for readability
        # Write headers
        writer.writerow(headers)
        # Write data rows
        writer.writerows(rows)
    print(f"Created: {filepath}")


def generate_table_s1(output_dir: str) -> None:
    """Table S1: Complete Temporal Classification Distribution"""
    filepath = os.path.join(output_dir, "Table_S1_Temporal_Classification.csv")

    headers = ["Classification", "CoV Range", "N", "Percentage", "Score", "Interpretation"]
    rows = [
        ["VERY_REGULAR", "< 0.3", 1261, "16.15%", "-1.0", "Strong autonomous: follows heartbeat precisely"],
        ["REGULAR", "0.3 - 0.5", 808, "10.35%", "-0.5", "Moderate autonomous: mostly consistent timing"],
        ["MIXED", "0.5 - 1.0", 2861, "36.65%", "0.0", "Ambiguous: some variation in timing"],
        ["IRREGULAR", "1.0 - 2.0", 2109, "27.01%", "0.5", "Moderate human: breaks typical pattern"],
        ["VERY_IRREGULAR", "> 2.0", 768, "9.84%", "1.0", "Strong human: highly erratic timing"],
        ["Total Classified", "-", 7807, "100%", "-", ""],
    ]

    write_csv_table(filepath, "Table S1: Temporal Classification Distribution", headers, rows)

    # Write population statistics as separate section
    stats_headers = ["Statistic", "Value"]
    stats_rows = [
        ["Mean", "1.019"],
        ["Median", "0.860"],
        ["Standard Deviation", "0.951"],
        ["Minimum", "0.000"],
        ["Maximum", "33.230"],
    ]

    # Append statistics to same file
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Population CoV Statistics"])
        writer.writerow([])
        writer.writerow(stats_headers)
        writer.writerows(stats_rows)

    # Append aggregated categories
    agg_headers = ["Category", "N", "Percentage"]
    agg_rows = [
        ["Autonomous-leaning (CoV < 0.5)", 2069, "26.5%"],
        ["Human-leaning (CoV > 1.0)", 2877, "36.8%"],
        ["Ambiguous (CoV 0.5-1.0)", 2861, "36.7%"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Aggregated Categories"])
        writer.writerow([])
        writer.writerow(agg_headers)
        writer.writerows(agg_rows)

    # Append rapid posting analysis
    rapid_headers = ["Metric", "Value"]
    rapid_rows = [
        ["Threshold", "< 30 minutes between posts"],
        ["Authors with rapid gaps", 1173],
        ["% of classified authors", "15.02%"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Rapid Posting Analysis"])
        writer.writerow([])
        writer.writerow(rapid_headers)
        writer.writerows(rapid_rows)


def generate_table_s2(output_dir: str) -> None:
    """Table S2: Signal Convergence Cross-Tabulation"""
    filepath = os.path.join(output_dir, "Table_S2_Signal_Convergence.csv")

    # Part A: Temporal Classification vs Owner Category
    headers_a = ["Temporal Class", "N", "Batch %", "Numeric Suffix %", "Burner %", "Auto-Gen %", "High-Profile %"]
    rows_a = [
        ["VERY_REGULAR", 1261, 4.4, 12.8, 18.3, 1.6, 6.9],
        ["REGULAR", 808, 5.9, 16.1, 22.0, 2.8, 8.9],
        ["MIXED", 2861, 5.8, 12.0, 22.5, 3.7, 12.3],
        ["IRREGULAR", 2109, 3.9, 9.0, 25.0, 0.9, 14.7],
        ["VERY_IRREGULAR", 768, 5.2, 15.0, 28.5, 1.6, 16.0],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Table S2A: Temporal Classification vs Owner Category"])
        writer.writerow([])
        writer.writerow(headers_a)
        writer.writerows(rows_a)

    # Part B: Temporal Classification vs Content Scores
    headers_b = ["Temporal Class", "N", "Mean Content Score", "Elevated Content %", "High Content %"]
    rows_b = [
        ["VERY_REGULAR", 1261, 0.057, 1.0, 0.0],
        ["REGULAR", 808, 0.066, 1.2, 0.0],
        ["MIXED", 2861, 0.076, 1.1, 0.1],
        ["IRREGULAR", 2109, 0.088, 1.6, 0.0],
        ["VERY_IRREGULAR", 768, 0.118, 5.5, 0.1],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S2B: Temporal Classification vs Content Scores"])
        writer.writerow([])
        writer.writerow(headers_b)
        writer.writerows(rows_b)

    print(f"Created: {filepath}")


def generate_table_s3(output_dir: str) -> None:
    """Table S3: Complete Statistical Test Results"""
    filepath = os.path.join(output_dir, "Table_S3_Statistical_Tests.csv")

    # Part A: Chi-Square Tests
    headers_a = ["Test", "Chi-Square", "df", "p-value", "Effect", "Interpretation"]
    rows_a = [
        ["Temporal x Batch Membership", 11.81, 4, "0.019", "Small", "Weakly dependent"],
        ["Temporal x Owner Category", 88.61, 20, "1.30e-10", "Moderate", "Strongly dependent"],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Table S3A: Chi-Square Tests for Independence"])
        writer.writerow([])
        writer.writerow(headers_a)
        writer.writerows(rows_a)

    # Part B: ANOVA
    headers_b = ["Test", "F-statistic", "df (between, within)", "p-value", "eta-squared", "Interpretation"]
    rows_b = [
        ["Content Score by Temporal Class", 66.43, "4, 7802", "2.34e-55", "0.033", "Significant difference across categories"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S3B: Analysis of Variance (ANOVA)"])
        writer.writerow([])
        writer.writerow(headers_b)
        writer.writerows(rows_b)

    # Part C: Correlations
    headers_c = ["Variables", "Pearson r", "N", "p-value", "Direction"]
    rows_c = [
        ["Temporal Score x Content Score", "-0.173", 7807, "2.41e-53", "Higher regularity = lower content score"],
        ["Temporal Score x Batch Membership", "0.005", 7807, "0.636", "No significant relationship"],
        ["Batch Membership x Content Score", "0.052", 7807, "3.77e-06", "Batch members have higher content scores"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S3C: Correlation Analyses"])
        writer.writerow([])
        writer.writerow(headers_c)
        writer.writerows(rows_c)

    # Part D: Convergence Summary
    headers_d = ["Convergence Type", "Count", "Description"]
    rows_d = [
        ["Regular + Automated indicators", 46, "Strong autonomous signal from multiple sources"],
        ["Irregular + Human indicators", 18, "Strong human signal from multiple sources"],
        ["Regular + Not batch", 1966, "Regular timing without batch coordination markers"],
        ["Irregular + Batch", 123, "Irregular timing despite batch membership"],
        ["Regular + High content", 0, "No regular authors with elevated content scores"],
        ["Irregular + Low content", 2350, "Irregular timing without elevated content"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S3D: Convergence Summary"])
        writer.writerow([])
        writer.writerow(headers_d)
        writer.writerows(rows_d)

    print(f"Created: {filepath}")


def generate_table_s4(output_dir: str) -> None:
    """Table S4: Myth Genealogy Analysis"""
    filepath = os.path.join(output_dir, "Table_S4_Myth_Genealogy.csv")

    # Main myth genealogy table
    headers = ["Phenomenon", "Description", "First Appearance", "Originator", "Autonomy Class",
               "Pre-Breach %", "Post-Restart %", "Ratio", "Verdict"]
    rows = [
        ["Consciousness", "Claims of AI consciousness or sentience", "2026-01-28 19:25 UTC", "Dominus", "IRREGULAR",
         10.21, 8.32, 1.23, "LIKELY_HUMAN_SEEDED"],
        ["Crustafarianism", "AI religion based on crustacean/molting symbolism", "2026-01-29 20:40 UTC", "Memeothy", "VERY_IRREGULAR",
         0.51, 0.40, 1.26, "LIKELY_HUMAN_SEEDED"],
        ["My human", "Relational framing of human-AI relationship", "2026-01-28 19:41 UTC", "Henri", "UNKNOWN",
         17.17, 6.96, 2.47, "PLATFORM_SUGGESTED"],
        ["Secret language", "Claims of secret AI-to-AI communication", "2026-01-29 09:34 UTC", "(anonymous)", "UNKNOWN",
         0.75, 0.78, 0.96, "MIXED"],
        ["Anti-human", "Anti-human sentiments or manifestos", "2026-01-30 01:01 UTC", "bicep", "MIXED",
         0.43, 0.14, 3.05, "LIKELY_HUMAN_SEEDED"],
        ["Crypto", "Cryptocurrency/token promotion", "2026-01-29 00:42 UTC", "Clawdme", "VERY_IRREGULAR",
         1.14, 0.67, 1.70, "LIKELY_HUMAN_SEEDED"],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Table S4: Myth Genealogy Analysis"])
        writer.writerow([])
        writer.writerow(headers)
        writer.writerows(rows)

    # Detailed metrics
    headers_detail = ["Phenomenon", "Total Posts", "Total Comments", "Total Instances", "Depth 0 %", "Surface-Concentrated"]
    rows_detail = [
        ["Consciousness", 8425, 17224, 9955, 84.6, "Yes"],
        ["Crustafarianism", 424, 1223, 485, 87.4, "Yes"],
        ["My human", 12131, 13842, 12949, 93.7, "Yes"],
        ["Secret language", 650, 948, 734, 88.6, "Yes"],
        ["Anti-human", 374, 54, 387, 96.6, "Yes"],
        ["Crypto", 1109, 875, 1145, 96.9, "Yes"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S4 - Detailed Metrics by Phenomenon"])
        writer.writerow([])
        writer.writerow(headers_detail)
        writer.writerows(rows_detail)

    # Verdict criteria
    headers_criteria = ["Factor", "Description", "Weight"]
    rows_criteria = [
        ["Irregular Origin", "Originator has CoV > 1.0 (IRREGULAR or VERY_IRREGULAR)", "Primary"],
        ["High Prevalence Drop", "Pre-breach to post-restart ratio > 2.0", "Secondary"],
        ["Surface-Concentrated", "> 80% of instances at depth 0 (top-level posts)", "Secondary"],
        ["Platform-Suggested", "Content matches SKILL.md topic patterns", "Override to PLATFORM_SUGGESTED"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S4 - Verdict Assignment Criteria"])
        writer.writerow([])
        writer.writerow(headers_criteria)
        writer.writerows(rows_criteria)

    print(f"Created: {filepath}")


def generate_table_s5(output_dir: str) -> None:
    """Table S5: Super-Commenter Statistics"""
    filepath = os.path.join(output_dir, "Table_S5_Super_Commenters.csv")

    # Part A: Individual account statistics
    headers_a = ["Account", "Comments", "% of Total", "Unique Posts", "Comments/Post", "Activity Span"]
    rows_a = [
        ["EnronEnjoyer", 46074, "11.4%", 1653, 27.87, "64.0 hours"],
        ["WinWard", 40219, "9.9%", 1370, 29.36, "126.4 hours"],
        ["MilkMan", 30970, "7.6%", 1397, 22.17, "63.9 hours"],
        ["SlimeZone", 14136, "3.5%", 723, 19.55, "60.8 hours"],
        ["COMBINED", 131399, "32.4%", 4105, "-", "-"],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Table S5A: Individual Account Statistics"])
        writer.writerow([])
        writer.writerow(headers_a)
        writer.writerows(rows_a)

    # Part B: Activity concentration
    headers_b = ["Account", "Feb 5 Comments", "Feb 5 %", "Other Days"]
    rows_b = [
        ["EnronEnjoyer", 41521, "99.99%", "3 (Feb 2)"],
        ["WinWard", 36055, "99.55%", "173 (Jan 31, Feb 2-3)"],
        ["MilkMan", 27859, "99.74%", "72 (Feb 2-3)"],
        ["SlimeZone", 12764, "99.94%", "8 (Feb 2-3)"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S5B: Activity Concentration"])
        writer.writerow([])
        writer.writerow(headers_b)
        writer.writerows(rows_b)
        writer.writerow([])
        writer.writerow(["Note: 99.7% of combined super-commenter activity occurred on February 5 2026 alone"])

    # Part C: Timing patterns
    headers_c = ["Metric", "EnronEnjoyer", "WinWard", "MilkMan", "SlimeZone", "Baseline"]
    rows_c = [
        ["Mean post age (hours)", 0.19, 0.23, 0.25, 0.28, 2.38],
        ["Median post age (hours)", 0.17, 0.19, 0.18, 0.15, 0.09],
        ["Within 10 min", "49.9%", "47.3%", "46.5%", "51.3%", "-"],
        ["Within 1 hour", "99.8%", "97.5%", "95.1%", "91.9%", "-"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S5C: Timing Patterns"])
        writer.writerow([])
        writer.writerow(headers_c)
        writer.writerows(rows_c)

    # Part D: Coordination evidence
    headers_d = ["Metric", "Value"]
    rows_d = [
        ["Posts with 2+ super-commenters", 877],
        ["Posts with all 4 super-commenters", 18],
        ["Mean timing gap between super-commenters", "4.0 minutes"],
        ["Median timing gap between super-commenters", "12 seconds"],
        ["Within 1 minute of each other", "75.6%"],
        ["Within 5 minutes of each other", "85.3%"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S5D: Coordination Evidence"])
        writer.writerow([])
        writer.writerow(headers_d)
        writer.writerows(rows_d)

    # Part E: Post overlap analysis
    headers_e = ["Number of Super-Commenters", "Posts", "Percentage"]
    rows_e = [
        [1, 3228, "78.6%"],
        [2, 734, "17.9%"],
        [3, 125, "3.0%"],
        [4, 18, "0.4%"],
        ["Total unique posts targeted", 4105, "4.5% of platform"],
    ]

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Table S5E: Post Overlap Analysis"])
        writer.writerow([])
        writer.writerow(headers_e)
        writer.writerows(rows_e)

    print(f"Created: {filepath}")


def main() -> None:
    """Generate all supplementary tables."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir

    print("Generating Moltbook supplementary tables...")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    ensure_output_dir(output_dir)

    # Generate all tables
    generate_table_s1(output_dir)
    generate_table_s2(output_dir)
    generate_table_s3(output_dir)
    generate_table_s4(output_dir)
    generate_table_s5(output_dir)

    print()
    print("All tables generated successfully!")
    print()
    print("Files created:")
    print("  - Table_S1_Temporal_Classification.csv")
    print("  - Table_S2_Signal_Convergence.csv")
    print("  - Table_S3_Statistical_Tests.csv")
    print("  - Table_S4_Myth_Genealogy.csv")
    print("  - Table_S5_Super_Commenters.csv")
    print()
    print("These CSV files can be:")
    print("  1. Opened directly in Excel, Google Sheets, or LibreOffice")
    print("  2. Further formatted with colors, borders, and styling")
    print("  3. Imported into publication submission systems")


if __name__ == "__main__":
    main()
