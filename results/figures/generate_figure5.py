"""
Generate Figure 5: Network Formation Visualization

Shows how AI agents form networks through feed-based discovery (85.9%)
rather than targeted social connections, with extremely low reciprocity (1.09%).

This demonstrates broadcast-style communication patterns rather than
conversational human-like interaction.

Updated: 2026-02-06
- Applied unified color palette
- Fixed panel label format (lowercase)
- Fixed legend and label positioning
- Applied Nature/Science style
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
from pathlib import Path

# Import unified color palette
from color_palette import (
    COLORS,
    get_figure5_colors,
    apply_moltbook_style
)


def load_data():
    """Load network formation data from JSON files."""
    results_dir = Path(__file__).parent.parent

    # First contact data
    with open(results_dir / 'f1_first_contact.json') as f:
        first_contact = json.load(f)

    # Network evolution data
    with open(results_dir / 'f3_network_evolution.json') as f:
        network_evolution = json.load(f)

    # Network analysis with reciprocity
    networks_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'networks'
    network_file = list(networks_dir.glob('comment_network_analysis_*.json'))[0]
    with open(network_file) as f:
        network_analysis = json.load(f)

    return first_contact, network_evolution, network_analysis


def create_panel_a(ax, first_contact):
    """Panel A: First contact mechanism distribution (donut chart with improved layout)."""
    # Get figure-specific colors
    fig5_colors = get_figure5_colors()

    # Data from first_contact.json
    percentages = first_contact['contact_type_percentages']

    # Combine small categories
    labels = ['Feed Discovery', 'Organic Follow-up', 'Other']
    sizes = [
        percentages['new_post'],  # 85.9%
        percentages['organic'],   # 12.8%
        percentages['mention'] + percentages['trending_post'] + percentages['viral_post']  # 1.3%
    ]
    colors = [
        fig5_colors['feed_discovery'],
        fig5_colors['organic_followup'],
        fig5_colors['mention_trending']
    ]
    explode = (0.03, 0, 0)  # Slight explode for feed slice

    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # Custom labels positioned manually
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        explode=explode,
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 2, 'edgecolor': 'white', 'width': 0.55},
        pctdistance=0.72,
        textprops={'fontsize': 10, 'fontweight': 'bold', 'color': 'white'}
    )

    # Customize percentage text
    for autotext, size in zip(autotexts, sizes):
        if size > 50:  # Make the 85.9% larger and more visible
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        elif size > 10:
            autotext.set_fontsize(10)
            autotext.set_color('white')

    # Add center text
    ax.text(0, 0, f'n={first_contact["total_unique_pairs"]:,}\nfirst contacts',
            ha='center', va='center', fontsize=9, color=COLORS['text_primary'])

    # Add legend with improved positioning (outside the donut)
    legend_labels = [f'{labels[i]}: {sizes[i]:.1f}%' for i in range(len(labels))]
    legend = ax.legend(
        wedges,
        legend_labels,
        loc='upper left',
        bbox_to_anchor=(-0.15, -0.05),
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        edgecolor=COLORS['grid']
    )
    legend.get_frame().set_linewidth(0.5)

    ax.set_title('(a) First Contact Mechanism', fontweight='bold', pad=12, fontsize=11)

    # Add annotation highlighting feed-based discovery (positioned to avoid overlap)
    ax.annotate('Feed-based\ndiscovery\ndominates',
                xy=(0.85, 0.85), xycoords='axes fraction',
                fontsize=9, ha='center', style='italic',
                color=fig5_colors['feed_discovery'],
                fontweight='medium')


def create_panel_b(ax, network_analysis):
    """Panel B: Reciprocity comparison (bar chart with human baseline)."""
    # Get figure-specific colors
    fig5_colors = get_figure5_colors()

    # Data
    reciprocity = network_analysis['reciprocity']
    agent_reciprocity = reciprocity['reciprocity_rate'] * 100  # Convert to percentage

    # Human social network baseline (from literature)
    human_baseline = 25  # Conservative estimate for social media

    categories = ['AI Agents\n(Moltbook)', 'Human Social\nNetworks*']
    values = [agent_reciprocity, human_baseline]
    colors_bar = [fig5_colors['ai_reciprocity'], fig5_colors['human_baseline']]

    bars = ax.bar(categories, values, color=colors_bar, edgecolor='white', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label = f'{val:.1f}%' if val < 10 else f'{val:.0f}%'
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, label,
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=COLORS['text_primary'])

    ax.set_ylabel('Reciprocity Rate (%)', fontweight='bold')
    ax.set_title('(b) Reciprocity: Agents vs Humans', fontweight='bold', pad=12, fontsize=11)
    ax.set_ylim(0, 35)

    # Add subtle grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color=COLORS['grid'])
    ax.set_axisbelow(True)

    # Add ratio annotation
    ratio = human_baseline / agent_reciprocity
    ax.annotate(f'{ratio:.0f}x lower\nthan humans',
                xy=(0, agent_reciprocity), xytext=(0.5, 18),
                textcoords='data',
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['text_primary'], lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['text_primary'], linewidth=0.8))

    # Add footnote with improved visibility
    ax.text(0.5, -0.22, '*Human baseline: typical social media reciprocity',
            transform=ax.transAxes, fontsize=8, ha='center', style='italic',
            color=COLORS['text_secondary'])


def create_panel_c(ax):
    """Panel C: Schematic showing broadcast vs conversational patterns."""
    # Get figure-specific colors
    fig5_colors = get_figure5_colors()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')

    # Left side: Broadcast pattern (Moltbook agents)
    center_left = (2.5, 5)
    ax.add_patch(Circle(center_left, 0.45, facecolor=fig5_colors['broadcast_nodes'],
                        edgecolor=COLORS['text_primary'], linewidth=2))
    ax.text(center_left[0], center_left[1], 'A', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Surrounding nodes with one-way arrows
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 nodes
    radius = 1.8
    for i, angle in enumerate(angles):
        x = center_left[0] + radius * np.cos(angle)
        y = center_left[1] + radius * np.sin(angle)
        ax.add_patch(Circle((x, y), 0.28, facecolor=fig5_colors['broadcast_edges'],
                            edgecolor=COLORS['text_primary'], linewidth=1))

        # One-way arrow from center to node
        dx = (x - center_left[0]) * 0.5
        dy = (y - center_left[1]) * 0.5
        ax.annotate('', xy=(center_left[0] + dx*1.5, center_left[1] + dy*1.5),
                   xytext=(center_left[0] + dx*0.9, center_left[1] + dy*0.9),
                   arrowprops=dict(arrowstyle='->', color=fig5_colors['broadcast_nodes'], lw=1.5))

    ax.text(center_left[0], 1.6, 'Broadcast Pattern\n(AI Agents)', ha='center', va='center',
            fontsize=10, fontweight='bold', color=fig5_colors['broadcast_nodes'])
    ax.text(center_left[0], 0.8, '1.09% reciprocity', ha='center', va='center',
            fontsize=9, color=COLORS['text_primary'])

    # Right side: Conversational pattern (Human networks)
    center_right = (7.5, 5)

    # Create 5 interconnected nodes
    node_positions = [
        (7.5, 6.5),
        (6.2, 5.5),
        (8.8, 5.5),
        (6.5, 4),
        (8.5, 4),
    ]

    for pos in node_positions:
        ax.add_patch(Circle(pos, 0.32, facecolor=fig5_colors['human_baseline'],
                            edgecolor=COLORS['text_primary'], linewidth=1.5))

    # Add bidirectional arrows between some pairs (consistent styling)
    bidirectional_pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (1, 2), (3, 4)]
    for i, j in bidirectional_pairs:
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[j]
        # Draw double-headed arrow
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / length, dy / length

        # Offset for bidirectional arrows
        offset = 0.1
        perp_x, perp_y = -dy * offset, dx * offset

        # Arrow 1 (i to j) - use consistent green color
        ax.annotate('', xy=(x2 - dx*0.42 + perp_x, y2 - dy*0.42 + perp_y),
                   xytext=(x1 + dx*0.42 + perp_x, y1 + dy*0.42 + perp_y),
                   arrowprops=dict(arrowstyle='->', color=fig5_colors['reciprocal_edges'], lw=1.2))
        # Arrow 2 (j to i)
        ax.annotate('', xy=(x1 + dx*0.42 - perp_x, y1 + dy*0.42 - perp_y),
                   xytext=(x2 - dx*0.42 - perp_x, y2 - dy*0.42 - perp_y),
                   arrowprops=dict(arrowstyle='->', color=fig5_colors['reciprocal_edges'], lw=1.2))

    ax.text(center_right[0], 1.6, 'Conversational Pattern\n(Human Networks)', ha='center', va='center',
            fontsize=10, fontweight='bold', color=fig5_colors['human_baseline'])
    ax.text(center_right[0], 0.8, '~25% reciprocity', ha='center', va='center',
            fontsize=9, color=COLORS['text_primary'])

    # Dividing line
    ax.axvline(x=5, ymin=0.15, ymax=0.85, color=COLORS['grid'], linestyle='--', linewidth=1.5)

    ax.set_title('(c) Network Formation Patterns', fontweight='bold', pad=12, fontsize=11, y=1.02)


def create_panel_d(ax, first_contact):
    """Panel D: Stacked bar showing contact types by period."""
    # Get figure-specific colors
    fig5_colors = get_figure5_colors()

    contact_by_period = first_contact['contact_by_period']

    # Prepare data
    periods = ['Jan 27-28', 'Jan 29-30', 'Jan 31', 'Feb 1-2\n(Offline)', 'Feb 3+']
    period_keys = ['Jan 27-28', 'Jan 29-30', 'Jan 31', 'Feb 1-2', 'Feb 3+']

    categories = ['new_post', 'organic', 'mention', 'trending_post']
    category_labels = ['Feed (New Post)', 'Organic', 'Mention', 'Trending']
    colors_stack = [
        fig5_colors['feed_discovery'],
        fig5_colors['organic_followup'],
        fig5_colors['mention_trending'],
        COLORS['human_influenced']  # Use red for trending/viral
    ]

    # Build data arrays
    data = {cat: [] for cat in categories}
    totals = []

    for pk in period_keys:
        period_data = contact_by_period.get(pk, {})
        total = sum(period_data.values()) if period_data else 0
        totals.append(total)
        for cat in categories:
            count = period_data.get(cat, 0)
            pct = (count / total * 100) if total > 0 else 0
            data[cat].append(pct)

    # Create stacked bar chart
    x = np.arange(len(periods))
    width = 0.65

    bottom = np.zeros(len(periods))
    bars_list = []
    for cat, label, color in zip(categories, category_labels, colors_stack):
        vals = data[cat]
        bars = ax.bar(x, vals, width, bottom=bottom, label=label, color=color,
                      edgecolor='white', linewidth=0.5)
        bars_list.append(bars)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=8)
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xlabel('Time Period', fontweight='bold')
    ax.set_title('(d) Contact Mechanisms Over Time', fontweight='bold', pad=12, fontsize=11)
    ax.set_ylim(0, 108)

    # Move legend outside to the right to avoid overlapping with n= labels
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8,
              framealpha=0.95, edgecolor=COLORS['grid'])

    # Add total counts as secondary annotation with improved visibility
    for i, (period, total) in enumerate(zip(periods, totals)):
        ax.text(i, 103, f'n={total:,}', ha='center', va='bottom', fontsize=8, rotation=0,
                color=COLORS['text_primary'], fontweight='medium')

    # Highlight the offline period with subtle shading
    ax.axvspan(2.7, 3.3, alpha=0.15, color=COLORS['neutral'])


def main():
    """Generate the complete Figure 5."""
    # Apply unified Moltbook style
    apply_moltbook_style()

    # Load data
    first_contact, network_evolution, network_analysis = load_data()

    # Create figure with 4 panels - increased width and adjusted right margin for legend
    fig = plt.figure(figsize=(14, 10))

    # Define grid spec with improved spacing - more room on right for external legend
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                          left=0.08, right=0.85, top=0.92, bottom=0.10)

    # Panel A: First contact mechanism (top left)
    ax_a = fig.add_subplot(gs[0, 0])
    create_panel_a(ax_a, first_contact)

    # Panel B: Reciprocity comparison (top right)
    ax_b = fig.add_subplot(gs[0, 1])
    create_panel_b(ax_b, network_analysis)

    # Panel C: Network pattern schematic (bottom left)
    ax_c = fig.add_subplot(gs[1, 0])
    create_panel_c(ax_c)

    # Panel D: Contact types over time (bottom right)
    ax_d = fig.add_subplot(gs[1, 1])
    create_panel_d(ax_d, first_contact)

    # Main title (without "Figure 5:" prefix per Nature style)
    fig.suptitle('Network Formation in AI Agent Society',
                 fontsize=14, fontweight='bold', y=0.97)

    # Save figures
    output_dir = Path(__file__).parent

    # Save as PNG (300 DPI)
    plt.savefig(output_dir / 'figure5_network_formation.png', dpi=300, facecolor='white')
    print(f"Saved: {output_dir / 'figure5_network_formation.png'}")

    # Save as PDF
    plt.savefig(output_dir / 'figure5_network_formation.pdf', format='pdf', facecolor='white')
    print(f"Saved: {output_dir / 'figure5_network_formation.pdf'}")

    plt.close()

    # Create updated caption
    caption = """Figure 5: Network Formation in AI Agent Society

(a) First contact mechanism distribution. The overwhelming majority of agent connections
(85.9%) originate through feed-based discovery, where agents encounter and respond to new
posts from previously unknown authors. Only 12.8% of first contacts occur through organic
follow-up interactions (commenting after initial contact), and direct mechanisms (mentions,
trending posts) account for just 1.3% of connections. This passive, feed-driven pattern
contrasts sharply with human social network formation, which typically involves more
intentional, targeted relationship building.

(b) Reciprocity rate comparison. AI agents exhibit a reciprocity rate of just 1.09%
(371 reciprocal pairs out of 68,207 directed edges), approximately 23x lower than
typical human social networks (~25%). This indicates broadcast-style communication
where agents respond to content but rarely engage in sustained back-and-forth
dialogue with specific partners.

(c) Schematic comparison of network patterns. Left: The broadcast pattern observed
in AI agents, where central nodes emit content to many recipients with minimal return
communication. Right: The conversational pattern typical of human networks,
characterized by bidirectional exchanges and relationship maintenance.

(d) Contact mechanisms over time. Feed-based discovery dominates across all time
periods, including the post-restoration phase (Feb 3+), where 87.7% of 43,535 new
connections formed through new post discovery. The offline period (Feb 1-2) shows
anomalous trending-post dominance due to limited activity.

These patterns suggest that AI agents, when given autonomy in a social platform,
naturally adopt information-processing behaviors (responding to novel content)
rather than relationship-building behaviors (cultivating reciprocal connections).
This has implications for understanding AI social dynamics and distinguishing
autonomous from human-prompted agent activity."""

    with open(output_dir / 'figure5_network_formation_caption.txt', 'w') as f:
        f.write(caption)
    print(f"Saved: {output_dir / 'figure5_network_formation_caption.txt'}")


if __name__ == '__main__':
    main()
