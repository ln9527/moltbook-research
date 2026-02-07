"""
Supplementary Figure S3: Embedding Cluster Analysis
Shows UMAP projection, top clusters, and autonomy ratios
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans']
})

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / 'results'
OUTPUT_DIR = RESULTS_DIR / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
print("Loading cluster data...")
with open(RESULTS_DIR / 'c1_topic_clusters.json') as f:
    cluster_data = json.load(f)

print("Loading visualization data...")
with open(RESULTS_DIR / 'c1_topic_clusters_viz.json') as f:
    viz_data = json.load(f)

# Create figure with 3 panels
fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.35,
                      left=0.06, right=0.98, top=0.92, bottom=0.12)

# Color scheme matching main figures
COLORS = {
    'REGULAR': '#2E86AB',      # Blue
    'IRREGULAR': '#A23B72',    # Purple
    'UNKNOWN': '#999999'       # Gray
}

# Panel A: UMAP projection colored by temporal classification
print("Creating panel A: UMAP projection...")
ax_umap = fig.add_subplot(gs[0, 0])

# Extract points and prepare data
points = viz_data['points']
x_coords = [p['x'] for p in points]
y_coords = [p['y'] for p in points]
temporal_classes = [p.get('temporal_class', 'UNKNOWN') for p in points]

# Create color array
colors_array = [COLORS.get(tc, COLORS['UNKNOWN']) for tc in temporal_classes]

# Plot scatter with smaller alpha for better visibility
ax_umap.scatter(x_coords, y_coords, c=colors_array, s=1, alpha=0.4, rasterized=True)

# Add cluster centroids
centroids = viz_data['cluster_centroids']
cent_x = [c['x'] for c in centroids]
cent_y = [c['y'] for c in centroids]
ax_umap.scatter(cent_x, cent_y, c='black', s=15, marker='x', alpha=0.6,
                linewidths=1, label='Cluster centers')

ax_umap.set_xlabel('UMAP Dimension 1')
ax_umap.set_ylabel('UMAP Dimension 2')
ax_umap.set_title('(a) UMAP Projection of Post Embeddings', fontweight='bold', pad=10)

# Add legend
legend_elements = [
    mpatches.Patch(color=COLORS['REGULAR'], label='Regular (autonomous)'),
    mpatches.Patch(color=COLORS['IRREGULAR'], label='Irregular (human-influenced)'),
    mpatches.Patch(color=COLORS['UNKNOWN'], label='Unknown'),
    plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='black',
               markersize=6, label='Cluster centers')
]
ax_umap.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

# Remove spines for cleaner look
ax_umap.spines['top'].set_visible(False)
ax_umap.spines['right'].set_visible(False)

# Panel B: Top 10 clusters by size
print("Creating panel B: Top clusters...")
ax_clusters = fig.add_subplot(gs[0, 1])

# Get clusters sorted by size
clusters = sorted(cluster_data['clusters'], key=lambda x: x['size'], reverse=True)
top_10 = clusters[:10]

# Prepare data
cluster_ids = [c['cluster_id'] for c in top_10]
sizes = [c['size'] for c in top_10]
labels = []

# Create labels from sample titles (truncated)
for c in top_10:
    sample_titles = c.get('sample_titles', [])
    if sample_titles:
        # Use first title, truncated
        title = sample_titles[0]
        if len(title) > 40:
            title = title[:37] + '...'
        labels.append(f"C{c['cluster_id']}: {title}")
    else:
        labels.append(f"Cluster {c['cluster_id']}")

# Create horizontal bar chart
y_pos = np.arange(len(labels))
bars = ax_clusters.barh(y_pos, sizes, color='#5C8AA5', alpha=0.8)

ax_clusters.set_yticks(y_pos)
ax_clusters.set_yticklabels(labels, fontsize=7)
ax_clusters.set_xlabel('Number of Posts')
ax_clusters.set_title('(b) Top 10 Clusters by Size', fontweight='bold', pad=10)
ax_clusters.invert_yaxis()

# Add value labels
for i, (bar, size) in enumerate(zip(bars, sizes)):
    ax_clusters.text(size + 50, i, f'{size:,}', va='center', fontsize=7)

# Remove spines
ax_clusters.spines['top'].set_visible(False)
ax_clusters.spines['right'].set_visible(False)

# Panel C: Autonomy ratio by cluster (top 20)
print("Creating panel C: Autonomy ratios...")
ax_autonomy = fig.add_subplot(gs[0, 2])

# Calculate autonomy ratio for each cluster
top_20 = clusters[:20]
autonomy_ratios = []
cluster_labels = []

for c in top_20:
    temp_dist = c.get('temporal_distribution', {})
    regular = temp_dist.get('REGULAR', 0)
    irregular = temp_dist.get('IRREGULAR', 0)
    total = regular + irregular

    if total > 0:
        ratio = regular / total
    else:
        ratio = 0.5  # Unknown

    autonomy_ratios.append(ratio)
    cluster_labels.append(f"C{c['cluster_id']}")

# Sort by autonomy ratio
sorted_indices = np.argsort(autonomy_ratios)
autonomy_ratios_sorted = [autonomy_ratios[i] for i in sorted_indices]
cluster_labels_sorted = [cluster_labels[i] for i in sorted_indices]

# Color bars by ratio (more blue = more autonomous, more purple = more human)
colors_bars = []
for ratio in autonomy_ratios_sorted:
    if ratio > 0.6:
        colors_bars.append(COLORS['REGULAR'])
    elif ratio < 0.4:
        colors_bars.append(COLORS['IRREGULAR'])
    else:
        colors_bars.append(COLORS['UNKNOWN'])

y_pos = np.arange(len(cluster_labels_sorted))
bars = ax_autonomy.barh(y_pos, autonomy_ratios_sorted, color=colors_bars, alpha=0.8)

ax_autonomy.set_yticks(y_pos)
ax_autonomy.set_yticklabels(cluster_labels_sorted, fontsize=7)
ax_autonomy.set_xlabel('Autonomy Ratio (Regular / Total)')
ax_autonomy.set_title('(c) Autonomy Ratio by Cluster (Top 20)', fontweight='bold', pad=10)
ax_autonomy.set_xlim(0, 1)
ax_autonomy.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_autonomy.invert_yaxis()

# Add reference lines
ax_autonomy.axvline(x=0.4, color=COLORS['IRREGULAR'], linestyle=':', alpha=0.3, linewidth=1)
ax_autonomy.axvline(x=0.6, color=COLORS['REGULAR'], linestyle=':', alpha=0.3, linewidth=1)

# Remove spines
ax_autonomy.spines['top'].set_visible(False)
ax_autonomy.spines['right'].set_visible(False)

# Add overall title
fig.suptitle('Supplementary Figure S3: Embedding Cluster Analysis',
             fontsize=12, fontweight='bold', y=0.98)

# Save figure
print("Saving figure...")
output_png = OUTPUT_DIR / 'figure_s3_embedding_clusters.png'
output_pdf = OUTPUT_DIR / 'figure_s3_embedding_clusters.pdf'

plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')

print(f"Figure saved to:")
print(f"  {output_png}")
print(f"  {output_pdf}")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total clusters: {len(clusters)}")
print(f"Total posts in visualization: {len(points)}")
print(f"Temporal class distribution:")
class_counts = Counter(temporal_classes)
for cls in ['REGULAR', 'IRREGULAR', 'UNKNOWN']:
    count = class_counts[cls]
    pct = 100 * count / len(points)
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

print(f"\nTop 5 largest clusters:")
for i, c in enumerate(top_10[:5], 1):
    temp_dist = c.get('temporal_distribution', {})
    regular = temp_dist.get('REGULAR', 0)
    irregular = temp_dist.get('IRREGULAR', 0)
    total = regular + irregular
    ratio = regular / total if total > 0 else 0

    sample_title = c.get('sample_titles', [''])[0]
    if len(sample_title) > 60:
        sample_title = sample_title[:57] + '...'

    print(f"  {i}. Cluster {c['cluster_id']}: {c['size']:,} posts, autonomy={ratio:.2f}")
    print(f"     Example: {sample_title}")

plt.close()
print("\nDone!")
