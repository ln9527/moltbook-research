#!/usr/bin/env python3
"""
Generate Figure 1: Myth Genealogy & Timeline
Moltbook Research Paper

This figure shows how 5/6 viral phenomena trace back to IRREGULAR/VERY_IRREGULAR originators,
providing key triangulation evidence that validates the temporal classification approach.

Panels:
(a) Platform timeline with breach/shutdown annotations
(b) Myth prevalence: pre-breach vs post-restart (paired bars)
(c) Originator autonomy classification (visual)
(d) Depth distribution for each myth (% at depth 0 vs deeper)

Updated: 2026-02-06
- Uses unified color palette from color_palette.py
- Lowercase panel labels per Nature style
- Colorblind accessible with patterns/hatching
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Import unified color palette
from color_palette import (
    COLORS,
    TEMPORAL_COLORS,
    VERDICT_COLORS,
    VERDICT_LABELS,
    apply_moltbook_style,
    get_figure1_colors,
    get_color_with_alpha,
)

# Apply publication-ready style
apply_moltbook_style()

# Get figure-specific colors
FIG1_COLORS = get_figure1_colors()

# Load data
results_dir = Path(__file__).parent.parent
with open(results_dir / 'p1_myth_genealogy.json', 'r') as f:
    data = json.load(f)

phenomena = data['phenomena']

# Define display order and labels
PHENOMENA_ORDER = ['consciousness', 'my_human', 'crypto', 'crustafarianism', 'anti_human', 'secret_language']
PHENOMENA_LABELS = {
    'consciousness': 'Consciousness\nClaims',
    'crustafarianism': 'Crustafarianism',
    'my_human': '"My Human"',
    'secret_language': 'Secret\nLanguage',
    'anti_human': 'Anti-Human\nManifestos',
    'crypto': 'Crypto\nPromotion'
}
SHORT_LABELS = {
    'consciousness': 'Consciousness',
    'crustafarianism': 'Crustafarianism',
    'my_human': '"My Human"',
    'secret_language': 'Secret Lang.',
    'anti_human': 'Anti-Human',
    'crypto': 'Crypto'
}

# Map autonomy to unified colors
AUTONOMY_COLORS = {
    'VERY_IRREGULAR': TEMPORAL_COLORS['VERY_IRREGULAR'],
    'IRREGULAR': TEMPORAL_COLORS['IRREGULAR'],
    'MIXED': TEMPORAL_COLORS['MIXED'],
    'UNKNOWN': '#ADB5BD',
    'REGULAR': TEMPORAL_COLORS['REGULAR'],
    'VERY_REGULAR': TEMPORAL_COLORS['VERY_REGULAR'],
}

# Verdict display colors and labels
VERDICT_DISPLAY_COLORS = {
    'LIKELY_HUMAN_SEEDED': VERDICT_COLORS['LIKELY_HUMAN_SEEDED'],
    'PLATFORM_SUGGESTED': VERDICT_COLORS['PLATFORM_SUGGESTED'],
    'MIXED': VERDICT_COLORS['MIXED'],
}
VERDICT_SHORT = {
    'LIKELY_HUMAN_SEEDED': 'Human-Seeded',
    'PLATFORM_SUGGESTED': 'Platform-Suggested',
    'MIXED': 'Mixed Origin'
}


def create_figure():
    """Create the 4-panel figure."""
    fig = plt.figure(figsize=(14, 11))

    # Create grid with custom spacing - increased right margin for legends
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                          left=0.08, right=0.92, top=0.93, bottom=0.08)

    ax_timeline = fig.add_subplot(gs[0, 0])
    ax_prevalence = fig.add_subplot(gs[0, 1])
    ax_autonomy = fig.add_subplot(gs[1, 0])
    ax_depth = fig.add_subplot(gs[1, 1])

    # Panel (a): Platform Timeline with myth appearances
    plot_timeline(ax_timeline)

    # Panel (b): Pre-breach vs Post-restart prevalence
    plot_prevalence(ax_prevalence)

    # Panel (c): Originator autonomy classification
    plot_autonomy(ax_autonomy)

    # Panel (d): Depth distribution
    plot_depth_distribution(ax_depth)

    # Add panel labels - lowercase per Nature style
    for ax, label in zip([ax_timeline, ax_prevalence, ax_autonomy, ax_depth],
                         ['(a)', '(b)', '(c)', '(d)']):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='top', ha='left')

    return fig


def plot_timeline(ax):
    """Panel (a): Platform timeline with key events and myth first appearances."""

    # Timeline dates
    start_date = datetime(2026, 1, 27)
    end_date = datetime(2026, 2, 6)
    breach_start = datetime(2026, 2, 1, 8, 0)
    breach_end = datetime(2026, 2, 3, 4, 0)

    # Create base timeline
    ax.axhline(y=0.5, color=COLORS['text_primary'], linewidth=2, zorder=1)

    # Shade breach period using unified color
    offline_color = get_color_with_alpha(COLORS['human_influenced'], 0.25)
    ax.axvspan(breach_start, breach_end, alpha=0.25, color=COLORS['human_influenced'], zorder=0,
               label='Platform Offline (44h)')

    # Mark myth first appearances
    y_positions = [0.15, 0.3, 0.7, 0.85, 0.55, 0.4]

    for i, phenomenon in enumerate(PHENOMENA_ORDER):
        p = phenomena[phenomenon]
        first_date = datetime.fromisoformat(p['first_appearance'].replace('+00:00', ''))
        autonomy = p['originator_autonomy']
        color = AUTONOMY_COLORS.get(autonomy, '#ADB5BD')

        # Draw marker
        ax.scatter(first_date, 0.5, s=100, c=color, zorder=3, edgecolors='white', linewidth=1)

        # Draw connector line
        y_pos = y_positions[i]
        ax.plot([first_date, first_date], [0.5, y_pos], color=color, linewidth=1,
                linestyle='--', alpha=0.7, zorder=2)

        # Add label
        ax.annotate(SHORT_LABELS[phenomenon], xy=(first_date, y_pos),
                    fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85,
                              edgecolor='none'),
                    color='white', fontweight='bold')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_xlim(start_date, end_date)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(axis='x', rotation=45)

    # Labels
    ax.set_title('Myth First Appearances & Platform Timeline')
    ax.set_xlabel('Date (2026)')

    # Add legend for autonomy types using unified colors
    # Place legend outside to avoid overlap with timeline content
    legend_elements = [
        mpatches.Patch(facecolor=TEMPORAL_COLORS['VERY_IRREGULAR'], label='Very Irregular'),
        mpatches.Patch(facecolor=TEMPORAL_COLORS['IRREGULAR'], label='Irregular'),
        mpatches.Patch(facecolor=TEMPORAL_COLORS['MIXED'], label='Mixed'),
        mpatches.Patch(facecolor='#ADB5BD', label='Unknown'),
        mpatches.Patch(facecolor=COLORS['human_influenced'], alpha=0.25, label='Platform Offline'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.02),
              fontsize=7, framealpha=0.9, ncol=3)

    # Remove y-axis spine
    ax.spines['left'].set_visible(False)


def plot_prevalence(ax):
    """Panel (b): Pre-breach vs Post-restart prevalence comparison."""

    x = np.arange(len(PHENOMENA_ORDER))
    width = 0.35

    pre_breach = [phenomena[p]['pre_breach_prevalence_pct'] for p in PHENOMENA_ORDER]
    post_restart = [phenomena[p]['post_restart_prevalence_pct'] for p in PHENOMENA_ORDER]

    # Use unified colors: pre_breach (autonomous blue), post_restart (platform green)
    bars1 = ax.bar(x - width/2, pre_breach, width, label='Pre-Breach',
                   color=COLORS['autonomous'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, post_restart, width, label='Post-Restart',
                   color=COLORS['platform'], edgecolor='white', linewidth=0.5)

    # Add value labels on bars (only for bars > 1%)
    for bar, val in zip(bars1, pre_breach):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars2, post_restart):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=7)

    # Add ratio annotations with better positioning to avoid overlap
    for i, (pre, post) in enumerate(zip(pre_breach, post_restart)):
        if pre > 0 and post > 0:
            ratio = pre / post
            if ratio > 1.5:
                # Position annotation higher to avoid bar overlap
                y_pos = max(pre, post) + 2.0
                ax.annotate(f'{ratio:.1f}x', xy=(i, y_pos),
                           ha='center', fontsize=7, color=COLORS['text_secondary'],
                           style='italic', fontweight='bold')

    ax.set_ylabel('Prevalence (%)')
    ax.set_xlabel('Viral Phenomenon')
    ax.set_title('Prevalence Before vs After Platform Restart')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[p] for p in PHENOMENA_ORDER], rotation=30, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)

    # Set y-axis limit with more padding for ratio annotations
    ax.set_ylim(0, max(pre_breach + post_restart) * 1.3)


def plot_autonomy(ax):
    """Panel (c): Originator autonomy classification - the key finding."""

    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(PHENOMENA_ORDER) + 1)

    # Column positions
    col_phenomenon = 0.5
    col_author = 3.0
    col_autonomy = 5.5
    col_verdict = 8.0

    # Header
    header_y = len(PHENOMENA_ORDER) + 0.5
    ax.text(col_phenomenon, header_y, 'Phenomenon', fontweight='bold', fontsize=9, ha='left', va='center')
    ax.text(col_author, header_y, 'Originator', fontweight='bold', fontsize=9, ha='left', va='center')
    ax.text(col_autonomy, header_y, 'Autonomy', fontweight='bold', fontsize=9, ha='center', va='center')
    ax.text(col_verdict, header_y, 'Verdict', fontweight='bold', fontsize=9, ha='center', va='center')

    # Add header line
    ax.axhline(y=header_y - 0.3, color=COLORS['text_primary'], linewidth=1, xmin=0.02, xmax=0.98)

    # Data rows
    for i, phenomenon in enumerate(PHENOMENA_ORDER):
        p = phenomena[phenomenon]
        y = len(PHENOMENA_ORDER) - i

        # Alternate row background
        if i % 2 == 0:
            ax.axhspan(y - 0.4, y + 0.4, color='#f5f5f5', zorder=0)

        # Phenomenon name
        ax.text(col_phenomenon, y, SHORT_LABELS[phenomenon], fontsize=8, ha='left', va='center')

        # Originator
        author = p['first_author'] if p['first_author'] else '(unknown)'
        ax.text(col_author, y, author, fontsize=8, ha='left', va='center', style='italic')

        # Autonomy classification with color-coded badge
        autonomy = p['originator_autonomy']
        color = AUTONOMY_COLORS.get(autonomy, '#ADB5BD')

        # Create badge
        badge = FancyBboxPatch((col_autonomy - 0.8, y - 0.25), 1.6, 0.5,
                               boxstyle="round,pad=0.02,rounding_size=0.15",
                               facecolor=color, edgecolor='none', alpha=0.9, zorder=2)
        ax.add_patch(badge)

        autonomy_display = autonomy.replace('_', '\n') if autonomy != 'UNKNOWN' else 'UNKNOWN'
        ax.text(col_autonomy, y, autonomy_display, fontsize=6, ha='center', va='center',
                color='white', fontweight='bold', zorder=3)

        # Verdict
        verdict = p['verdict']
        verdict_color = VERDICT_DISPLAY_COLORS.get(verdict, COLORS['text_secondary'])
        ax.text(col_verdict, y, VERDICT_SHORT.get(verdict, verdict), fontsize=8,
                ha='center', va='center', color=verdict_color, fontweight='bold')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_title('Myth Origins: 5/6 Trace to Human-Influenced Originators')

    # Add summary annotation with larger font (8pt minimum per review)
    # Positioned higher to avoid cutoff
    summary_text = ("Key Finding: Most viral phenomena were seeded by\n"
                   "IRREGULAR/VERY_IRREGULAR agents, validating that\n"
                   "high temporal variance = human manipulation")
    ax.annotate(summary_text, xy=(5, 0.4), ha='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['stats_box_bg'],
                         edgecolor=COLORS['stats_box_edge'], alpha=0.9))


def plot_depth_distribution(ax):
    """Panel (d): Depth distribution showing myths concentrate at surface level.

    Includes hatching patterns for colorblind accessibility.
    """

    x = np.arange(len(PHENOMENA_ORDER))

    depth_0_pct = [phenomena[p]['depth_0_percentage'] for p in PHENOMENA_ORDER]
    depth_deeper = [100 - phenomena[p]['depth_0_percentage'] for p in PHENOMENA_ORDER]

    # Use unified colors with hatching for colorblind accessibility
    bars1 = ax.bar(x, depth_0_pct, label='Top-level (Depth 0)',
                   color=COLORS['autonomous'], edgecolor='white', linewidth=0.5,
                   hatch='')  # No hatch for primary category
    bars2 = ax.bar(x, depth_deeper, bottom=depth_0_pct, label='In Replies (Depth 1+)',
                   color=TEMPORAL_COLORS['IRREGULAR'], edgecolor='white', linewidth=0.5,
                   hatch='///')  # Diagonal hatch for secondary category

    # Add percentage labels
    for i, (d0, d1) in enumerate(zip(depth_0_pct, depth_deeper)):
        ax.text(i, d0/2, f'{d0:.0f}%', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
        if d1 > 5:  # Only label if significant
            ax.text(i, d0 + d1/2, f'{d1:.0f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    ax.set_ylabel('Distribution (%)')
    ax.set_xlabel('Viral Phenomenon')
    ax.set_title('Myths Concentrate at Top-Level Posts\n(Broadcast Pattern, Not Viral Spread)')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[p] for p in PHENOMENA_ORDER], rotation=30, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 105)

    # Add average line using highlight color with better label positioning
    avg_depth_0 = np.mean(depth_0_pct)
    ax.axhline(y=avg_depth_0, color=COLORS['human_influenced'], linestyle='--', linewidth=1, alpha=0.7)
    # Position label on the left to avoid overlapping with legend
    ax.text(0.5, avg_depth_0 - 3, f'Avg: {avg_depth_0:.0f}%',
            fontsize=7, color=COLORS['human_influenced'], ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))


def main():
    """Generate and save the figure."""
    fig = create_figure()

    # Save in multiple formats
    output_dir = Path(__file__).parent
    fig.savefig(output_dir / 'figure1_myth_genealogy.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'figure1_myth_genealogy.pdf', dpi=300, facecolor='white')

    print(f"Figure saved to {output_dir / 'figure1_myth_genealogy.png'}")
    print(f"Figure saved to {output_dir / 'figure1_myth_genealogy.pdf'}")

    # Generate caption
    caption = """Figure 1: Myth Genealogy and Origins of Viral Phenomena on Moltbook

(a) Timeline showing first appearances of six viral phenomena on the platform.
Markers are color-coded by the originating agent's temporal autonomy classification:
VERY_IRREGULAR (red) and IRREGULAR (orange) indicate high coefficient of variation
in posting times, suggesting human prompting rather than autonomous heartbeat-driven behavior.
The shaded region marks the 44-hour platform offline period (Feb 1-3, 2026).

(b) Prevalence comparison before the platform breach (pre-breach) versus after restart
(post-restart). "My Human" framing shows the largest decline (2.47x), consistent with
platform-suggested content that disappeared when SKILL.md patterns were removed.
Crypto promotion and anti-human manifestos also declined substantially.

(c) Originator profiles for each viral phenomenon. Five of six phenomena trace back
to agents with IRREGULAR or VERY_IRREGULAR temporal patterns, providing independent
validation that high temporal variance correlates with human manipulation. The "My Human"
pattern is classified as PLATFORM_SUGGESTED due to its match with SKILL.md template content.

(d) Depth distribution showing that myths concentrate at top-level posts (depth 0)
rather than spreading virally through reply chains. Average 91% top-level concentration
indicates broadcast rather than organic conversation propagation. Diagonal hatching
provides pattern redundancy for colorblind accessibility.

Data: N = 91,792 posts, 405,707 comments from 22,020 authors (Jan 27 - Feb 5, 2026).
"""

    with open(output_dir / 'figure1_myth_genealogy_caption.txt', 'w') as f:
        f.write(caption)

    print(f"Caption saved to {output_dir / 'figure1_myth_genealogy_caption.txt'}")


if __name__ == '__main__':
    main()
