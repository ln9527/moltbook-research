"""
Moltbook Unified Color Palette
==============================

Nature/Science-quality color scheme for all Moltbook research figures.
Designed for colorblind accessibility and print compatibility.

Usage:
    from color_palette import (
        COLORS,
        TEMPORAL_COLORS,
        VERDICT_COLORS,
        get_cmap,
        apply_moltbook_style
    )

    # Apply consistent styling
    apply_moltbook_style()

    # Use semantic colors
    ax.bar(x, y, color=COLORS['autonomous'])

Created: 2026-02-06
Target: Nature, Science, Nature Human Behaviour
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# PRIMARY SEMANTIC COLORS
# =============================================================================

COLORS: Dict[str, str] = {
    # Core semantic colors
    'autonomous': '#1A759F',        # Cool blue - systematic, regular
    'human_influenced': '#D62839',  # Warm red - intervention, irregular
    'platform': '#52B788',          # Green - system-provided, scaffolded
    'neutral': '#6C757D',           # Gray - baseline, control, mixed
    'highlight': '#F77F00',         # Orange - emphasis, key statistics

    # Extended usage
    'pre_breach': '#1A759F',        # Timeline: before shutdown
    'post_restart': '#52B788',      # Timeline: after restart
    'bot_manipulation': '#9D0208',  # Dark red - bot farming evidence
    'trending_viral': '#DC2F02',    # Bright red - viral indicators
    'feed_based': '#1A759F',        # Network: passive discovery
    'reciprocal': '#52B788',        # Network: bidirectional ties

    # Text and background
    'text_primary': '#333333',      # Dark gray text
    'text_secondary': '#666666',    # Medium gray text
    'background': '#FFFFFF',        # White background (Nature standard)
    'grid': '#E0E0E0',              # Light gray grid
    'spine': '#333333',             # Axis spine color

    # Statistics boxes
    'stats_box_bg': '#FFF9E6',      # Light yellow
    'stats_box_edge': '#E6B800',    # Gold edge
}


# =============================================================================
# TEMPORAL CLASSIFICATION COLORS (5-Category)
# =============================================================================

TEMPORAL_COLORS: Dict[str, str] = {
    'VERY_REGULAR': '#0A4F6D',      # Darkest blue - strong autonomous
    'REGULAR': '#1A759F',           # Blue - moderate autonomous
    'MIXED': '#6C757D',             # Gray - ambiguous
    'IRREGULAR': '#E85D04',         # Orange - moderate human influence
    'VERY_IRREGULAR': '#D62839',    # Red - strong human influence
}

# Ordered for bar charts
TEMPORAL_ORDER: List[str] = [
    'VERY_REGULAR', 'REGULAR', 'MIXED', 'IRREGULAR', 'VERY_IRREGULAR'
]


# =============================================================================
# VERDICT COLORS (Myth Genealogy)
# =============================================================================

VERDICT_COLORS: Dict[str, str] = {
    'LIKELY_HUMAN_SEEDED': '#D62839',   # Red - human origin
    'PLATFORM_SUGGESTED': '#52B788',     # Green - system origin
    'MIXED': '#6C757D',                  # Gray - unclear
    'UNKNOWN': '#ADB5BD',                # Light gray - no data
}

VERDICT_LABELS: Dict[str, str] = {
    'LIKELY_HUMAN_SEEDED': 'Human-Seeded',
    'PLATFORM_SUGGESTED': 'Platform-Suggested',
    'MIXED': 'Mixed Origin',
    'UNKNOWN': 'Unknown',
}


# =============================================================================
# GRADIENT PALETTES
# =============================================================================

# Sequential blue (autonomous intensity: light -> dark)
SEQUENTIAL_BLUE: List[str] = [
    '#E6F2F8', '#B3D9E9', '#66B3D2', '#1A759F', '#0A4F6D'
]

# Sequential red (human influence intensity: light -> dark)
SEQUENTIAL_RED: List[str] = [
    '#FCE8EB', '#F5B7BD', '#E66E7A', '#D62839', '#9D0208'
]

# Diverging (autonomous <-> human)
DIVERGING: List[str] = [
    '#0A4F6D', '#1A759F', '#6C757D', '#E85D04', '#D62839'
]

# Sequential green (platform influence: light -> dark)
SEQUENTIAL_GREEN: List[str] = [
    '#D8F3DC', '#95D5B2', '#52B788', '#2D6A4F', '#1B4332'
]


def get_cmap(name: str, n_colors: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Get a colormap for continuous data.

    Parameters
    ----------
    name : str
        One of: 'sequential_blue', 'sequential_red', 'sequential_green', 'diverging'
    n_colors : int
        Number of colors in the colormap

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A matplotlib colormap object
    """
    palettes = {
        'sequential_blue': SEQUENTIAL_BLUE,
        'sequential_red': SEQUENTIAL_RED,
        'sequential_green': SEQUENTIAL_GREEN,
        'diverging': DIVERGING,
    }

    if name not in palettes:
        raise ValueError(f"Unknown colormap: {name}. Choose from: {list(palettes.keys())}")

    colors = palettes[name]
    return mcolors.LinearSegmentedColormap.from_list(f'moltbook_{name}', colors, N=n_colors)


def get_temporal_cmap(n_colors: int = 5) -> mcolors.ListedColormap:
    """
    Get a categorical colormap for the 5 temporal classifications.

    Parameters
    ----------
    n_colors : int
        Number of colors (default 5 for full classification)

    Returns
    -------
    matplotlib.colors.ListedColormap
        A categorical colormap
    """
    colors = [TEMPORAL_COLORS[cat] for cat in TEMPORAL_ORDER[:n_colors]]
    return mcolors.ListedColormap(colors, name='moltbook_temporal')


# =============================================================================
# STYLE APPLICATION
# =============================================================================

def apply_moltbook_style() -> None:
    """
    Apply the Moltbook publication-ready style to all matplotlib figures.

    This sets rcParams for Nature/Science quality figures including:
    - Sans-serif fonts (Arial preferred)
    - Appropriate font sizes for reduction
    - Clean spines (no top/right)
    - White backgrounds
    - High DPI for publication
    """
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        # Figure settings
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.pad_inches': 0.1,

        # Axes settings
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.edgecolor': COLORS['spine'],
        'axes.labelcolor': COLORS['text_primary'],
        'axes.facecolor': 'white',

        # Tick settings
        'xtick.color': COLORS['text_primary'],
        'ytick.color': COLORS['text_primary'],
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Text settings
        'text.color': COLORS['text_primary'],

        # Grid settings
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',
        'axes.grid': False,  # Off by default, enable per-panel

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': COLORS['grid'],
        'legend.fancybox': True,

        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,

        # Patch settings (for bars, etc.)
        'patch.linewidth': 0.5,
        'patch.edgecolor': 'white',
    })


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple (0-255)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def hex_to_rgb_normalized(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color to RGB tuple (0-1) for matplotlib."""
    r, g, b = hex_to_rgb(hex_color)
    return (r / 255, g / 255, b / 255)


def get_color_with_alpha(hex_color: str, alpha: float) -> Tuple[float, float, float, float]:
    """Get RGBA tuple from hex color and alpha value."""
    r, g, b = hex_to_rgb_normalized(hex_color)
    return (r, g, b, alpha)


def lighten_color(hex_color: str, amount: float = 0.3) -> str:
    """
    Lighten a color by mixing with white.

    Parameters
    ----------
    hex_color : str
        The hex color to lighten
    amount : float
        Amount to lighten (0 = no change, 1 = white)

    Returns
    -------
    str
        Lightened hex color
    """
    r, g, b = hex_to_rgb(hex_color)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f'#{r:02x}{g:02x}{b:02x}'


def darken_color(hex_color: str, amount: float = 0.3) -> str:
    """
    Darken a color by mixing with black.

    Parameters
    ----------
    hex_color : str
        The hex color to darken
    amount : float
        Amount to darken (0 = no change, 1 = black)

    Returns
    -------
    str
        Darkened hex color
    """
    r, g, b = hex_to_rgb(hex_color)
    r = int(r * (1 - amount))
    g = int(g * (1 - amount))
    b = int(b * (1 - amount))
    return f'#{r:02x}{g:02x}{b:02x}'


def get_contrast_text_color(hex_color: str) -> str:
    """
    Get appropriate text color (black or white) for a background.

    Uses relative luminance formula for accessibility.
    """
    r, g, b = hex_to_rgb_normalized(hex_color)
    # Relative luminance formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return '#FFFFFF' if luminance < 0.5 else '#333333'


# =============================================================================
# FIGURE-SPECIFIC COLOR GETTERS
# =============================================================================

def get_figure1_colors() -> Dict[str, str]:
    """Colors for Figure 1: Myth Genealogy."""
    return {
        'very_irregular': TEMPORAL_COLORS['VERY_IRREGULAR'],
        'irregular': TEMPORAL_COLORS['IRREGULAR'],
        'mixed': TEMPORAL_COLORS['MIXED'],
        'unknown': '#ADB5BD',
        'pre_breach': COLORS['pre_breach'],
        'post_restart': COLORS['post_restart'],
        'depth_0': COLORS['autonomous'],
        'depth_1_plus': TEMPORAL_COLORS['IRREGULAR'],
        'human_seeded': VERDICT_COLORS['LIKELY_HUMAN_SEEDED'],
        'platform_suggested': VERDICT_COLORS['PLATFORM_SUGGESTED'],
        'offline_region': get_color_with_alpha('#DEE2E6', 0.3),
    }


def get_figure2_colors() -> Dict[str, str]:
    """Colors for Figure 2: Temporal Classification & Triangulation."""
    return {
        **TEMPORAL_COLORS,
        'burner_metric': '#7B2CBF',  # Purple for burner accounts
        'content_metric': COLORS['platform'],
        'elevated_metric': COLORS['human_influenced'],
        'threshold_line': '#343A40',
        'stats_box': COLORS['stats_box_bg'],
        'stats_edge': COLORS['stats_box_edge'],
    }


def get_figure3_colors() -> Dict[str, str]:
    """Colors for Figure 3: Bot Farming (Smoking Gun)."""
    return {
        'super_commenter': COLORS['bot_manipulation'],
        'normal_commenter': COLORS['neutral'],
        'timing_gap': COLORS['human_influenced'],
        'activity_burst': COLORS['trending_viral'],
        'targeting': COLORS['highlight'],
        'marker_12sec': COLORS['highlight'],
    }


def get_figure4_colors() -> Dict[str, str]:
    """Colors for Figure 4: Platform Scaffolding."""
    return {
        'skill_suggested': COLORS['platform'],
        'organic': COLORS['autonomous'],
        'naturalness': COLORS['platform'],
        'promotional': COLORS['human_influenced'],
        'shutdown_shade': get_color_with_alpha(COLORS['neutral'], 0.15),
        'ratio_annotation': COLORS['highlight'],
    }


def get_figure5_colors() -> Dict[str, str]:
    """Colors for Figure 5: Network Formation."""
    return {
        'feed_discovery': COLORS['autonomous'],
        'organic_followup': lighten_color(COLORS['autonomous'], 0.4),
        'mention_trending': TEMPORAL_COLORS['IRREGULAR'],
        'ai_reciprocity': COLORS['platform'],
        'human_baseline': COLORS['highlight'],
        'broadcast_nodes': COLORS['autonomous'],
        'broadcast_edges': '#ADB5BD',
        'conversation_nodes': COLORS['highlight'],
        'reciprocal_edges': COLORS['platform'],
    }


def get_figure6_colors() -> Dict[str, str]:
    """Colors for Figure 6: Echo Decay."""
    return {
        'human_decay': COLORS['human_influenced'],
        'autonomous_decay': COLORS['autonomous'],
        'halflife_marker': COLORS['highlight'],
        'promotional_gradient': 'sequential_blue',  # Use get_cmap
        'reply_human': COLORS['human_influenced'],
        'reply_autonomous': COLORS['autonomous'],
        'floor_line': COLORS['neutral'],
    }


# =============================================================================
# COLORBLIND SIMULATION (for testing)
# =============================================================================

def simulate_colorblindness(hex_color: str, type: str = 'deuteranopia') -> str:
    """
    Simulate how a color appears to colorblind viewers.

    This is an approximation for testing purposes.

    Parameters
    ----------
    hex_color : str
        Original hex color
    type : str
        Type of colorblindness: 'deuteranopia', 'protanopia', 'tritanopia'

    Returns
    -------
    str
        Simulated hex color
    """
    r, g, b = hex_to_rgb_normalized(hex_color)

    # Transformation matrices (simplified approximations)
    matrices = {
        'deuteranopia': np.array([
            [0.625, 0.375, 0.0],
            [0.700, 0.300, 0.0],
            [0.000, 0.300, 0.700]
        ]),
        'protanopia': np.array([
            [0.567, 0.433, 0.0],
            [0.558, 0.442, 0.0],
            [0.000, 0.242, 0.758]
        ]),
        'tritanopia': np.array([
            [0.950, 0.050, 0.0],
            [0.000, 0.433, 0.567],
            [0.000, 0.475, 0.525]
        ]),
    }

    if type not in matrices:
        raise ValueError(f"Unknown colorblindness type: {type}")

    rgb = np.array([r, g, b])
    simulated = matrices[type] @ rgb
    simulated = np.clip(simulated, 0, 1)

    r, g, b = (int(c * 255) for c in simulated)
    return f'#{r:02x}{g:02x}{b:02x}'


def print_colorblind_test():
    """Print colorblind simulation results for primary colors."""
    primary = ['autonomous', 'human_influenced', 'platform', 'neutral', 'highlight']

    print("\nColorblind Simulation Test")
    print("=" * 70)
    print(f"{'Color':<20} {'Original':<12} {'Deuteranopia':<12} {'Protanopia':<12} {'Tritanopia':<12}")
    print("-" * 70)

    for name in primary:
        original = COLORS[name]
        deut = simulate_colorblindness(original, 'deuteranopia')
        prot = simulate_colorblindness(original, 'protanopia')
        trit = simulate_colorblindness(original, 'tritanopia')
        print(f"{name:<20} {original:<12} {deut:<12} {prot:<12} {trit:<12}")


# =============================================================================
# DEMO / TESTING
# =============================================================================

def create_demo_figure():
    """Create a demo figure showing the color palette."""
    apply_moltbook_style()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Moltbook Color Palette Demo', fontsize=14, fontweight='bold')

    # Panel 1: Primary colors
    ax = axes[0, 0]
    primary = ['autonomous', 'human_influenced', 'platform', 'neutral', 'highlight']
    colors_list = [COLORS[c] for c in primary]
    bars = ax.bar(range(len(primary)), [1]*len(primary), color=colors_list)
    ax.set_xticks(range(len(primary)))
    ax.set_xticklabels([c.replace('_', '\n') for c in primary], fontsize=8)
    ax.set_title('Primary Semantic Colors')
    ax.set_ylim(0, 1.2)
    for bar, color in zip(bars, colors_list):
        ax.text(bar.get_x() + bar.get_width()/2, 0.5, color,
                ha='center', va='center', fontsize=8,
                color=get_contrast_text_color(color), fontweight='bold')

    # Panel 2: Temporal classification
    ax = axes[0, 1]
    colors_list = [TEMPORAL_COLORS[c] for c in TEMPORAL_ORDER]
    bars = ax.bar(range(5), [1]*5, color=colors_list)
    ax.set_xticks(range(5))
    labels = ['V.Regular', 'Regular', 'Mixed', 'Irregular', 'V.Irregular']
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_title('Temporal Classification')
    ax.set_ylim(0, 1.2)

    # Panel 3: Sequential blue gradient
    ax = axes[0, 2]
    cmap = get_cmap('sequential_blue')
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_yticks([])
    ax.set_xlabel('Autonomous Intensity')
    ax.set_title('Sequential Blue Gradient')

    # Panel 4: Sequential red gradient
    ax = axes[1, 0]
    cmap = get_cmap('sequential_red')
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_yticks([])
    ax.set_xlabel('Human Influence Intensity')
    ax.set_title('Sequential Red Gradient')

    # Panel 5: Diverging gradient
    ax = axes[1, 1]
    cmap = get_cmap('diverging')
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_yticks([])
    ax.set_xlabel('Autonomous <---> Human')
    ax.set_title('Diverging Gradient')

    # Panel 6: Verdict colors
    ax = axes[1, 2]
    verdicts = list(VERDICT_COLORS.keys())
    colors_list = [VERDICT_COLORS[v] for v in verdicts]
    bars = ax.bar(range(len(verdicts)), [1]*len(verdicts), color=colors_list)
    ax.set_xticks(range(len(verdicts)))
    ax.set_xticklabels([VERDICT_LABELS[v] for v in verdicts], fontsize=7, rotation=45, ha='right')
    ax.set_title('Verdict Colors')
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Moltbook Color Palette")
    print("=" * 50)
    print("\nPrimary Colors:")
    for name, color in COLORS.items():
        if name in ['autonomous', 'human_influenced', 'platform', 'neutral', 'highlight']:
            print(f"  {name}: {color}")

    print("\nTemporal Classification Colors:")
    for name, color in TEMPORAL_COLORS.items():
        print(f"  {name}: {color}")

    print_colorblind_test()

    print("\n\nTo apply the style, use:")
    print("  from color_palette import apply_moltbook_style")
    print("  apply_moltbook_style()")

    # Create and save demo figure
    fig = create_demo_figure()
    from pathlib import Path
    output_path = Path(__file__).parent / 'color_palette_demo.png'
    fig.savefig(output_path, dpi=150)
    print(f"\nDemo figure saved to: {output_path}")
