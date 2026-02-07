#!/usr/bin/env python3
"""
Figure 2: Temporal Signal and Triangulation Validation
Moltbook Research Paper

Panel (a): CoV distribution showing autonomous vs human-prompted classification
Panel (b): Multi-signal triangulation convergence

Updated: 2026-02-06
- Uses unified color palette from color_palette.py
- Fixed text cutoff and legend overlap issues
- Lowercase panel labels per Nature style
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Import unified color palette
from color_palette import (
    COLORS,
    TEMPORAL_COLORS,
    TEMPORAL_ORDER,
    apply_moltbook_style,
    get_figure2_colors,
)

# Apply Moltbook publication style
apply_moltbook_style()

# Load data
results_dir = Path(__file__).parent.parent
with open(results_dir / 'b1_temporal_signal_v2.json', 'r') as f:
    temporal_data = json.load(f)

with open(results_dir / 'b5_triangulation_v2.json', 'r') as f:
    triangulation_data = json.load(f)

# Get figure-specific colors
fig2_colors = get_figure2_colors()

# Create figure with two panels - increased width and height for better spacing
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.4)

# === Panel (a): CoV Distribution ===
ax1 = fig.add_subplot(gs[0])

# Classification counts and percentages
classifications = TEMPORAL_ORDER
counts = [temporal_data['classification_counts'][c] for c in classifications]
percentages = [temporal_data['classification_percentages'][c] for c in classifications]

# Create bars using unified TEMPORAL_COLORS
bars = ax1.bar(range(len(classifications)), counts,
               color=[TEMPORAL_COLORS[c] for c in classifications],
               edgecolor='white', linewidth=0.5)

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, percentages)):
    height = bar.get_height()
    ax1.annotate(f'{pct:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# X-axis labels
labels = ['Very\nRegular\n(CoV<0.3)', 'Regular\n(0.3-0.5)', 'Mixed\n(0.5-1.0)',
          'Irregular\n(1.0-2.0)', 'Very\nIrregular\n(CoV>2.0)']
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels)
ax1.set_ylabel('Number of Authors')
ax1.set_xlabel('Temporal Classification (Coefficient of Variation)')

# Add threshold line and annotation
ax1.axvline(x=1.5, color=fig2_colors['threshold_line'], linestyle='--', linewidth=1.5, alpha=0.7)
ax1.annotate('Classification\nThreshold', xy=(1.5, max(counts)*0.82),
            xytext=(2.3, max(counts)*0.88),
            fontsize=8, ha='left',
            arrowprops=dict(arrowstyle='->', color=fig2_colors['threshold_line'], lw=0.8))

# Add autonomous/human labels at top - positioned over correct bars with adequate margin
ax1.text(0.5, 1.05, 'AUTONOMOUS', ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=TEMPORAL_COLORS['VERY_REGULAR'],
        transform=ax1.get_xaxis_transform(), clip_on=False)
ax1.text(3.5, 1.05, 'HUMAN-PROMPTED', ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=TEMPORAL_COLORS['VERY_IRREGULAR'],
        transform=ax1.get_xaxis_transform(), clip_on=False)

# Statistics box - positioned in upper left to avoid overlap
stats_text = (f"n = {temporal_data['eligible_authors']:,} authors\n"
              f"Mean CoV = {temporal_data['cov_statistics']['mean']:.2f}\n"
              f"Median CoV = {temporal_data['cov_statistics']['median']:.2f}")
ax1.text(0.03, 0.97, stats_text, transform=ax1.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['grid'], alpha=0.9))

ax1.set_ylim(0, max(counts) * 1.15)
ax1.set_title('(a)', loc='left', fontweight='bold', fontsize=12)

# === Panel (b): Triangulation Convergence ===
ax2 = fig.add_subplot(gs[1])

# Data for triangulation
crosstab = triangulation_data['crosstab']

# Categories to compare
categories = TEMPORAL_ORDER
x_pos = np.arange(len(categories))
bar_width = 0.25

# Extract signals for each category
burner_pct = [crosstab[c]['pct_burner'] for c in categories]
content_score = [crosstab[c]['mean_content_score'] * 100 for c in categories]  # Scale for visibility
elevated_content = [crosstab[c]['pct_elevated_content'] for c in categories]

# Create grouped bars with semantic colors from palette
bars1 = ax2.bar(x_pos - bar_width, burner_pct, bar_width,
                label='Burner Account %', color=fig2_colors['burner_metric'], alpha=0.85)
bars2 = ax2.bar(x_pos, content_score, bar_width,
                label='Content Score (x100)', color=fig2_colors['content_metric'], alpha=0.85)
bars3 = ax2.bar(x_pos + bar_width, elevated_content, bar_width,
                label='Elevated Content %', color=fig2_colors['elevated_metric'], alpha=0.85)

# Labels
short_labels = ['V.Reg', 'Reg', 'Mixed', 'Irreg', 'V.Irreg']
ax2.set_xticks(x_pos)
ax2.set_xticklabels(short_labels)
ax2.set_ylabel('Percentage / Score')
ax2.set_xlabel('Temporal Classification')

# Legend repositioned outside plot area to avoid overlap
ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.65), framealpha=0.95, fontsize=8)

# Add trend arrows with matching colors - adjusted to avoid stats box
ax2.annotate('', xy=(4.2, 28), xytext=(0.5, 18),
            arrowprops=dict(arrowstyle='->', color=fig2_colors['burner_metric'], lw=1.8, ls='--', alpha=0.7))
ax2.annotate('', xy=(4.2, 12), xytext=(0.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=fig2_colors['content_metric'], lw=1.8, ls='--', alpha=0.7))

# Statistics annotations - compact format in upper right
stats = triangulation_data['statistics']
stats_text = (
    f"Triangulation Statistics:\n"
    f"Temporal vs Owner: chi2={stats['chi2_temporal_vs_owner']['chi2']:.1f}***\n"
    f"Content by Temporal: F={stats['anova_content_by_temporal']['f_statistic']:.1f}***\n"
    f"Temporal vs Batch: chi2={stats['chi2_temporal_vs_batch']['chi2']:.1f}*"
)
ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=7,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor=fig2_colors['stats_box'],
                  edgecolor=fig2_colors['stats_edge'], alpha=0.95, pad=0.4))

# Correlation annotation
corr = triangulation_data['correlations']['temporal_vs_content']
ax2.text(0.03, 0.03, f"r = {corr['correlation']:.3f}, p < 1e-50",
        transform=ax2.transAxes, fontsize=8, style='italic',
        verticalalignment='bottom', horizontalalignment='left')

ax2.set_ylim(0, 32)
ax2.set_title('(b)', loc='left', fontweight='bold', fontsize=12)

# Overall figure title (no "Figure 2:" prefix per Nature style)
fig.suptitle('Temporal Signal Detection and Multi-Signal Triangulation',
             fontsize=13, fontweight='bold', y=0.98)

# Adjust layout to prevent text cutoff - add padding at top for title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figures
output_dir = Path(__file__).parent
fig.savefig(output_dir / 'figure2_temporal_triangulation.png', dpi=300,
            bbox_inches='tight', facecolor='white')
fig.savefig(output_dir / 'figure2_temporal_triangulation.pdf',
            bbox_inches='tight', facecolor='white')

print("Figure 2 saved successfully!")
print(f"  PNG: {output_dir / 'figure2_temporal_triangulation.png'}")
print(f"  PDF: {output_dir / 'figure2_temporal_triangulation.pdf'}")

# Create caption
caption = """Figure 2. Temporal signal detection and multi-signal triangulation validation.

(a) Distribution of authors by temporal classification based on Coefficient of Variation (CoV)
in posting intervals. Authors with low CoV (<0.5) exhibit regular, automated posting patterns
consistent with autonomous platform scheduling ("heartbeat"), while high CoV (>1.0) indicates
irregular timing characteristic of human prompting. Of 7,807 eligible authors (minimum 3 posts),
26.5% show autonomous patterns (Very Regular + Regular) and 36.8% show human-influenced patterns
(Irregular + Very Irregular). Mean CoV = 1.02, median = 0.86.

(b) Triangulation convergence across independent signals. Three additional indicators-burner
account prevalence, promotional content scores, and elevated content percentage-all increase
monotonically from autonomous to human-prompted categories. Chi-square tests confirm significant
dependency between temporal classification and owner category (chi2=88.61, p<1e-10) and batch
naming (chi2=11.81, p=0.019). ANOVA shows content scores differ significantly across temporal
categories (F=66.43, p<1e-55). The correlation between temporal regularity and content score
(r=-0.173, p<1e-50) indicates that irregular posters produce more promotional content.
This multi-signal convergence provides independent validation that temporal patterns reliably
distinguish autonomous from human-influenced AI agents.

*p<0.05, **p<0.01, ***p<0.001
"""

with open(output_dir / 'figure2_temporal_triangulation_caption.txt', 'w') as f:
    f.write(caption)

print(f"  Caption: {output_dir / 'figure2_temporal_triangulation_caption.txt'}")
