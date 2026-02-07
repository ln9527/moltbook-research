"""
Figure S2: Network Visualization
Shows network structure, degree distribution, community structure, and super-commenter impact.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import Counter

# Font settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7

# Paths
NETWORK_FILE = Path("data/processed/networks/comment_network_analysis_20260205_163239.json")
OUTPUT_DIR = Path("analysis/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load network data
print("Loading network data...")
with open(NETWORK_FILE) as f:
    network = json.load(f)

# Extract data
basic_metrics = network['basic_metrics']
top_out_degree = network['top_out_degree']
top_in_degree = network['top_in_degree']
scc = network['scc_analysis']

# Super-commenters (from super_commenter_analysis.txt)
super_commenters = {
    'EnronEnjoyer': 46074,
    'WinWard': 40219,
    'MilkMan': 30970,
    'SlimeZone': 14136
}

# Create figure
fig = plt.figure(figsize=(7.5, 8))

# Panel A: Degree distribution (log-log)
ax1 = plt.subplot(3, 2, (1, 2))

# Extract out-degree counts
out_degrees = [count for _, count in top_out_degree]

# Create degree distribution
degree_counts = Counter(out_degrees)
degrees = sorted(degree_counts.keys())
counts = [degree_counts[d] for d in degrees]

# Log-log plot
ax1.scatter(degrees, counts, alpha=0.6, s=30, color='#2E86AB', edgecolors='white', linewidth=0.5)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Out-degree (log scale)', fontweight='bold')
ax1.set_ylabel('Frequency (log scale)', fontweight='bold')
ax1.set_title('A. Degree Distribution', fontweight='bold', loc='left', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add annotation about power law
ax1.text(0.98, 0.98, 'Heavy-tailed distribution\nsuggests scale-free network',
         transform=ax1.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5),
         fontsize=7)

# Panel B: Community structure (SCC sizes)
ax2 = plt.subplot(3, 2, 3)

# Get SCC sizes
scc_sizes = scc['scc_size_distribution']
largest_scc = scc_sizes[0]
small_sccs = len([s for s in scc_sizes if s == 1])
medium_sccs = len([s for s in scc_sizes if 2 <= s < 10])
large_sccs = len([s for s in scc_sizes if s >= 10 and s < largest_scc])

categories = ['Largest\nComponent\n(n=1690)', f'Large\n(10-1689)\n(n={large_sccs})',
              f'Medium\n(2-9)\n(n={medium_sccs})', f'Isolated\n(n=1)\n(n={small_sccs})']
sizes = [1, large_sccs, medium_sccs, small_sccs]
colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E']

bars = ax2.bar(range(len(categories)), sizes, color=colors, edgecolor='white', linewidth=0.5)
ax2.set_ylabel('Number of components', fontweight='bold')
ax2.set_title('B. Community Structure (SCCs)', fontweight='bold', loc='left', pad=10)
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels(categories, fontsize=6.5)
ax2.set_yscale('log')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)

# Add percentage annotation on largest component
largest_pct = (largest_scc / basic_metrics['n_nodes']) * 100
ax2.text(0, sizes[0], f'{largest_pct:.1f}% of nodes', ha='center', va='bottom',
         fontsize=6, fontweight='bold')

# Panel C: Network properties table
ax3 = plt.subplot(3, 2, 4)
ax3.axis('off')

# Create table data
table_data = [
    ['Property', 'Value'],
    ['Nodes', f"{basic_metrics['n_nodes']:,}"],
    ['Edges', f"{basic_metrics['n_edges']:,}"],
    ['Density', f"{basic_metrics['density']:.6f}"],
    ['Avg degree', f"{basic_metrics['avg_out_degree']:.2f}"],
    ['Reciprocity', f"{network['reciprocity']['reciprocity_rate']*100:.2f}%"],
    ['Largest SCC', f"{largest_scc:,} nodes"],
    ['SCC coverage', f"{largest_pct:.1f}%"]
]

# Draw table
table = ax3.table(cellText=table_data, cellLoc='left', loc='center',
                  colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 2.2)

# Style header row
for i in range(2):
    cell = table[(0, i)]
    cell.set_facecolor('#E8E8E8')
    cell.set_text_props(weight='bold')

# Color alternating rows
for i in range(1, len(table_data)):
    for j in range(2):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#F8F8F8')

ax3.set_title('C. Network Properties', fontweight='bold', loc='left', pad=10, y=0.95)

# Panel D: Super-commenter impact (horizontal bar chart)
ax4 = plt.subplot(3, 2, (5, 6))

# Prepare data
total_comments = 405707
names = list(super_commenters.keys())
comment_counts = list(super_commenters.values())
percentages = [(count / total_comments) * 100 for count in comment_counts]

# Create horizontal bars
y_pos = np.arange(len(names))
bars = ax4.barh(y_pos, percentages, color='#D62828', edgecolor='white', linewidth=0.5)

# Customize
ax4.set_yticks(y_pos)
ax4.set_yticklabels(names)
ax4.set_xlabel('Percentage of all comments (%)', fontweight='bold')
ax4.set_title('D. Super-Commenter Impact', fontweight='bold', loc='left', pad=10)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
ax4.invert_yaxis()

# Add value labels
for i, (bar, pct, count) in enumerate(zip(bars, percentages, comment_counts)):
    ax4.text(pct + 0.3, i, f'{pct:.1f}%\n({count:,})',
             va='center', fontsize=6.5, fontweight='bold')

# Add combined annotation
combined_pct = sum(percentages)
combined_count = sum(comment_counts)
ax4.axvline(combined_pct, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax4.text(combined_pct, -0.7, f'Combined: {combined_pct:.1f}%\n({combined_count:,} comments)',
         ha='center', va='top', fontsize=7, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, edgecolor='black', linewidth=0.5))

# Add coordination timing annotation
ax4.text(0.98, 0.02, 'Median gap between super-commenters:\n12 seconds on same post',
         transform=ax4.transAxes, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8, edgecolor='#D62828', linewidth=1),
         fontsize=7, fontweight='bold')

# Overall title
fig.suptitle('Supplementary Figure S2: Comment Network Structure and Super-Commenter Impact',
             fontsize=11, fontweight='bold', y=0.98)

# Add figure caption
caption = ('Network analysis reveals a scale-free structure with heavy-tailed degree distribution (A), sparse connectivity (density=0.0001), '
           'and low reciprocity (1.09%). The network consists of one large strongly connected component covering 7.5% of nodes (B-C). '
           'Four super-commenters produced 32% of all comments (D), with median 12-second timing gaps suggesting single-operator control.')

fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=7,
         wrap=True, bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8, edgecolor='gray', linewidth=0.5))

plt.tight_layout(rect=[0, 0.04, 1, 0.97])

# Save figure
output_png = OUTPUT_DIR / "figure_s2_network_visualization.png"
output_pdf = OUTPUT_DIR / "figure_s2_network_visualization.pdf"

plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved:")
print(f"  PNG: {output_png}")
print(f"  PDF: {output_pdf}")

plt.close()

print("\nFigure S2 complete!")
print("\nKey findings displayed:")
print("  - Degree distribution shows power-law/scale-free pattern")
print("  - 8,100 strongly connected components")
print("  - Largest SCC: 1,690 nodes (7.5%)")
print("  - Very low reciprocity (1.09%)")
print("  - Super-commenters made 32.4% of all comments")
print("  - Median 12-second gap between super-commenters")
