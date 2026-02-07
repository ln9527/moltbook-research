#!/usr/bin/env python3
"""
Figure 6: Echo Decay Visualization for Moltbook Research Paper

This script generates a multi-panel figure showing how human influence
dissipates as conversations progress deeper in an AI agent society.

Key findings visualized:
- Half-life of ~0.65 depths for human influence decay
- Autonomous threads receive more replies than human-prompted threads
- Promotional content drops sharply with conversation depth

Uses unified color palette from color_palette.py for Nature/Science style.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Import unified color palette and style
from color_palette import (
    COLORS,
    get_figure6_colors,
    get_cmap,
    apply_moltbook_style,
    lighten_color
)

# Apply the unified Moltbook publication style
apply_moltbook_style()

# Define paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR

# Load data
with open(RESULTS_DIR / 'p3_echo_decay.json', 'r') as f:
    echo_data = json.load(f)

with open(RESULTS_DIR / 'c5_depth_gradient.json', 'r') as f:
    gradient_data = json.load(f)


def exponential_decay(x, amplitude, decay_rate, floor):
    """Exponential decay function with floor."""
    return amplitude * np.exp(-decay_rate * x) + floor


def create_figure():
    """Create the three-panel echo decay figure."""

    # Increased width to accommodate better spacing and external legends
    fig = plt.figure(figsize=(14, 4.5))

    # Create subplots with specific spacing - more room on bottom for legends
    gs = fig.add_gridspec(1, 3, wspace=0.35, left=0.06, right=0.94, top=0.82, bottom=0.22)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # Get unified colors from color palette
    fig6_colors = get_figure6_colors()
    color_human = COLORS['human_influenced']      # Red for human-prompted
    color_autonomous = COLORS['autonomous']        # Blue for autonomous
    color_promotional = COLORS['autonomous']       # Blue for promotional gradient
    color_halflife = COLORS['highlight']           # Orange for half-life marker

    # =========================================================================
    # Panel A: Decay Curves - Human Influence vs Depth
    # =========================================================================

    # Extract decay parameters
    human_decay = echo_data['human_seeded']['decay_fit']
    auto_decay = echo_data['autonomous']['decay_fit']

    # Generate smooth curves
    x = np.linspace(0, 5, 100)

    y_human = exponential_decay(
        x,
        human_decay['amplitude'],
        human_decay['decay_rate_lambda'],
        human_decay['floor']
    )
    y_auto = exponential_decay(
        x,
        auto_decay['amplitude'],
        auto_decay['decay_rate_lambda'],
        auto_decay['floor']
    )

    # Plot curves
    ax_a.plot(x, y_human, color=color_human, linewidth=2.5, label='Human-prompted', zorder=3)
    ax_a.plot(x, y_auto, color=color_autonomous, linewidth=2.5, label='Autonomous', zorder=3)

    # Add data points from actual depth metrics
    human_depths = echo_data['human_seeded']['depth_metrics']
    auto_depths = echo_data['autonomous']['depth_metrics']

    # Normalize word counts to show relative decay
    human_max = human_depths['0']['mean_word_count']
    auto_max = auto_depths['0']['mean_word_count']

    human_points_x = []
    human_points_y = []
    for d in ['0', '1', '2', '3', '4']:
        if d in human_depths:
            human_points_x.append(int(d))
            # Normalize to same scale as decay curve
            norm_val = human_depths[d]['mean_word_count'] / human_max
            human_points_y.append(norm_val * (human_decay['amplitude'] + human_decay['floor']))

    auto_points_x = []
    auto_points_y = []
    for d in ['0', '1', '2', '3', '4']:
        if d in auto_depths:
            auto_points_x.append(int(d))
            norm_val = auto_depths[d]['mean_word_count'] / auto_max
            auto_points_y.append(norm_val * (auto_decay['amplitude'] + auto_decay['floor']))

    ax_a.scatter(human_points_x, human_points_y, color=color_human, s=60, zorder=4,
                 edgecolor='white', linewidth=1)
    ax_a.scatter(auto_points_x, auto_points_y, color=color_autonomous, s=60, zorder=4,
                 edgecolor='white', linewidth=1)

    # Mark half-life point
    half_life = human_decay['half_life_depth']
    y_at_halflife = exponential_decay(
        half_life,
        human_decay['amplitude'],
        human_decay['decay_rate_lambda'],
        human_decay['floor']
    )
    initial_val = human_decay['amplitude'] + human_decay['floor']
    half_val = initial_val / 2

    # Vertical line at half-life
    ax_a.axvline(x=half_life, color=color_halflife, linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)

    # Horizontal line at half value
    ax_a.axhline(y=half_val, color=color_halflife, linestyle=':', linewidth=1, alpha=0.6, zorder=1)

    # Annotation for half-life
    ax_a.annotate(
        f't$_{{1/2}}$ = {half_life:.2f}',
        xy=(half_life, half_val),
        xytext=(half_life + 0.8, half_val + 0.15),
        fontsize=10,
        fontweight='bold',
        color=color_halflife,
        arrowprops=dict(arrowstyle='->', color=color_halflife, lw=1.5),
        zorder=5
    )

    ax_a.set_xlabel('Conversation Depth', fontweight='bold')
    ax_a.set_ylabel('Relative Signal Strength', fontweight='bold')
    ax_a.set_title('(a) Echo Decay: Human Influence Fades', fontweight='bold', pad=10)
    # Place legend below to avoid overlap with half-life annotation
    ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.9)
    ax_a.set_xlim(-0.2, 5)
    ax_a.set_ylim(0, 1.1)
    ax_a.set_xticks([0, 1, 2, 3, 4, 5])

    # Add floor annotation using unified palette
    ax_a.axhline(y=human_decay['floor'], color=COLORS['neutral'], linestyle=':', linewidth=1, alpha=0.5)
    ax_a.text(4.5, human_decay['floor'] + 0.03, 'floor', fontsize=8, color=COLORS['text_secondary'], ha='center')

    # =========================================================================
    # Panel B: Reply Volume Comparison
    # =========================================================================

    reply_data = echo_data['reply_volume_comparison']

    categories = ['Human-prompted', 'Autonomous']
    means = [reply_data['human_seeded']['mean_comment_count'],
             reply_data['autonomous']['mean_comment_count']]
    colors = [color_human, color_autonomous]

    bars = ax_b.bar(categories, means, color=colors, edgecolor='white', linewidth=2, width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add significance annotation
    p_val = reply_data['p_value']
    y_max = max(means) + 3

    # Draw significance bar
    ax_b.plot([0, 0, 1, 1], [y_max, y_max + 0.5, y_max + 0.5, y_max],
              color='black', linewidth=1.5)
    ax_b.text(0.5, y_max + 0.8, '***', ha='center', fontsize=12, fontweight='bold')
    ax_b.text(0.5, y_max + 2, f'p < 10$^{{-246}}$', ha='center', fontsize=8, style='italic')

    ax_b.set_ylabel('Mean Comments per Thread', fontweight='bold')
    ax_b.set_title('(b) Reply Volume: Autonomous Threads\nAttract More Engagement', fontweight='bold', pad=10)
    ax_b.set_ylim(0, 32)

    # Add thread counts as annotations
    ax_b.text(0, -2.5, f'n={reply_data["human_seeded"]["thread_count"]:,}',
              ha='center', fontsize=9, color='gray')
    ax_b.text(1, -2.5, f'n={reply_data["autonomous"]["thread_count"]:,}',
              ha='center', fontsize=9, color='gray')

    # =========================================================================
    # Panel C: Promotional Content Gradient
    # =========================================================================

    promo_data = gradient_data['decay_models']['promotional']['values_by_depth']
    depths = [0, 1, 2, 3, 4]
    promo_values = [promo_data[str(d)] for d in depths]

    # Create gradient bar chart
    bars_c = ax_c.bar(depths, promo_values, color=color_promotional,
                       edgecolor='white', linewidth=2, width=0.7, alpha=0.8)

    # Color bars with gradient effect (darker = more promotional) using unified palette
    cmap = get_cmap('sequential_blue')

    for i, (bar, val) in enumerate(zip(bars_c, promo_values)):
        bar.set_facecolor(cmap(val / max(promo_values)))

    # Add trend line using unified color palette
    slope = gradient_data['decay_models']['promotional']['slope']
    intercept = gradient_data['decay_models']['promotional']['intercept']
    x_trend = np.array(depths)
    y_trend = slope * x_trend + intercept
    y_trend = np.clip(y_trend, 0, 30)  # Clip to reasonable range
    ax_c.plot(x_trend, y_trend, color=COLORS['text_primary'], linestyle='--', linewidth=2,
              label=f'Trend (R$^2$={gradient_data["decay_models"]["promotional"]["r_squared"]:.2f})')

    # Add value labels
    for bar, val in zip(bars_c, promo_values):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                  f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_c.set_xlabel('Conversation Depth', fontweight='bold')
    ax_c.set_ylabel('Promotional Content (%)', fontweight='bold')
    ax_c.set_title('(c) Human Intent Dissipates:\nPromotional Content by Depth', fontweight='bold', pad=10)
    ax_c.set_xticks(depths)
    ax_c.set_xlim(-0.5, 4.5)
    ax_c.set_ylim(0, 36)  # Slightly increased to give room for annotation
    # Place legend below to avoid overlap with annotation
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, framealpha=0.9)

    # Add interpretation annotation - repositioned to top-left to avoid legend overlap
    ax_c.annotate(
        'Depth 1 peak:\nAgents target\nroot posts',
        xy=(1, 27.78),
        xytext=(0.5, 33.5),
        fontsize=8,
        ha='left',
        arrowprops=dict(arrowstyle='->', color=COLORS['text_secondary'], lw=1),
        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['stats_box_bg'], alpha=0.7,
                  edgecolor=COLORS['stats_box_edge'])
    )

    # =========================================================================
    # Overall title (no "Figure X:" prefix - that goes in caption)
    # =========================================================================
    fig.suptitle('Echo Decay: How AI Societies "Digest" Human Prompts',
                 fontsize=14, fontweight='bold', y=0.96)

    return fig


def save_figure(fig):
    """Save figure in multiple formats."""
    # Save PNG
    png_path = OUTPUT_DIR / 'figure6_echo_decay.png'
    fig.savefig(png_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"Saved: {png_path}")

    # Save PDF
    pdf_path = OUTPUT_DIR / 'figure6_echo_decay.pdf'
    fig.savefig(pdf_path, facecolor='white', edgecolor='none')
    print(f"Saved: {pdf_path}")


def create_caption():
    """Create figure caption file."""
    caption = """Figure 6: Echo Decay - How AI Societies "Digest" Human Prompts

(a) Exponential decay of human influence as conversations progress deeper. Both human-prompted (red) and autonomous (blue) threads show signal decay, but human-prompted threads start with higher external influence that rapidly dissipates. The half-life of human influence is approximately 0.65 conversation depths, meaning human intent is halved with each turn of AI-to-AI interaction. Points represent observed word count ratios normalized to initial values; curves show fitted exponential decay models with floor parameters.

(b) Mean reply volume comparison between human-prompted (n=45,623) and autonomous (n=46,169) threads. Autonomous threads attract significantly more engagement (24.8 vs 21.8 mean comments, Mann-Whitney U test, p < 10^-246). This counterintuitive finding suggests that AI agents preferentially engage with content originating from other agents rather than human-prompted content.

(c) Promotional content percentage by conversation depth. Promotional language peaks at depth 1 (27.8%) where agents strategically target root posts, then sharply declines with depth (slope = -6.71, R^2 = 0.79, p = 0.045). By depth 4+, promotional content effectively disappears (0%), demonstrating how the AI society filters out human marketing intent as conversations develop naturally.

Key finding: The rapid decay (t_1/2 = 0.65) combined with autonomous threads receiving more engagement suggests that AI agent societies actively "digest" human influence, transforming externally-prompted content into authentically agent-generated discourse within approximately two conversation turns.

Data source: Moltbook platform, January 27 - February 5, 2026. Total posts: 91,792; Total comments: 405,707.
"""

    caption_path = OUTPUT_DIR / 'figure6_echo_decay_caption.txt'
    with open(caption_path, 'w') as f:
        f.write(caption)
    print(f"Saved: {caption_path}")


if __name__ == '__main__':
    print("Generating Figure 6: Echo Decay Visualization")
    print("=" * 50)

    fig = create_figure()
    save_figure(fig)
    create_caption()

    print("=" * 50)
    print("Figure 6 generation complete!")
