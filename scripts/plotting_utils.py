"""
Plotting Utilities for SuSteelAible Capstone Project
==================================================================

This script provides standardized plotting functions for visualizing
steel industry emissions data across multiple analysis notebooks.

Author: Irene Polgar
Date: December 2025
Version: 1.0
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# =============================================================================
# COLOR SCHEMES & STYLING
# =============================================================================

# Technology color scheme (consistent across all plots)
TECH_COLORS = {
    'BF-BOF': '#e41a1c',           # Red (carbon-intensive)
    'EAF': '#4daf4a',              # Green (cleaner)
    'EAF Stainless': '#377eb8'     # Blue (stainless steel)
}


def get_tech_color(tech):
    """
    Assign color based on technology string matching.
    
    Handles variations in technology naming (e.g., 'EAF Stainless', 
    'BF-BOF → H₂-DRI') by checking if keywords are present in the string.
    
    Parameters
    ----------
    tech : str
        Technology name (e.g., 'BF-BOF', 'EAF', 'EAF Stainless')
    
    Returns
    -------
    str
        Hex color code
    
    Examples
    --------
    >>> get_tech_color('BF-BOF')
    '#e41a1c'
    >>> get_tech_color('BF-BOF → H₂-DRI')
    '#e41a1c'
    >>> get_tech_color('EAF Stainless')
    '#377eb8'
    """
    # Check for BF-BOF first (most specific)
    if 'BF-BOF' in str(tech):
        return TECH_COLORS['BF-BOF']
    # Check for Stainless (more specific than general EAF)
    elif 'Stainless' in str(tech):
        return TECH_COLORS['EAF Stainless']
    # Check for EAF (general electric arc furnace)
    elif 'EAF' in str(tech):
        return TECH_COLORS['EAF']
    # Fallback (shouldn't happen with clean data)
    else:
        return '#999999'  # Gray


def set_plot_style():
    """
    Set consistent matplotlib/seaborn style for all plots.
    
    Call this at the start of notebooks to ensure visual consistency.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('husl')
    
    # Set default figure parameters
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


# =============================================================================
# TECHNOLOGY COMPARISON PLOT
# =============================================================================

def plot_technology_boxplot(df, intensity_col='scope1_intensity', 
                            technology_col='technology',
                            title='Emission Intensity by Technology',
                            ylabel='Scope 1 Intensity (tCO$_2$e/t steel)',
                            show_stats=False,
                            show_mean=False,
                            figsize=(8, 6),
                            save_path=None):
    """
    Create boxplot comparing emission intensity across technologies.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with emission intensity and technology columns
    intensity_col : str
        Name of column containing emission intensity values
        Default: 'scope1_intensity'
    technology_col : str
        Name of column containing technology classifications
        Default: 'technology'
    title : str
        Plot title
    ylabel : str
        Y-axis label (use CO$_2$ for subscript, not CO₂)
        Default: 'Scope 1 Intensity (tCO$_2$e/t steel)'
    show_stats : bool
        If True, adds annotation showing percentage difference between technologies
        Only works for binary comparison (2 technologies)
        Default: False
    show_mean : bool
        If True, plots mean values as diamond markers in addition to median line
        Default: False
    figsize : tuple
        Figure size (width, height)
        Default: (8, 6)
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects
    dict or None
        If show_stats=True, returns dictionary with statistics:
        {'means': [mean1, mean2, ...], 'pct_difference': float}
        Otherwise returns None
    
    Examples
    --------
    >>> # Simple boxplot
    >>> fig, ax, _ = plot_technology_boxplot(df_analysis)
    
    >>> # With percentage annotation (for baseline model)
    >>> fig, ax, stats = plot_technology_boxplot(
    ...     df_analysis, 
    ...     intensity_col='total_intensity_best',
    ...     ylabel='Operational Intensity (Scope 1+2, tCO$_2$e/t)',
    ...     show_stats=True,
    ...     save_path='../outputs/tech_comparison.png'
    ... )
    >>> print(f"Gap: {stats['pct_difference']:.0f}% lower")
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Prepare data - filter to non-null values
    plot_data = df[df[intensity_col].notna()].copy()
    technologies = sorted(plot_data[technology_col].unique())
    
    # Prepare data for boxplot
    data_by_tech = [
        plot_data[plot_data[technology_col] == tech][intensity_col].dropna()
        for tech in technologies
    ]
    
    # Create boxplot
    bp = ax.boxplot(
        data_by_tech,
        labels=technologies,
        patch_artist=True,
        widths=0.6
    )
    
    # Color boxes using shared color scheme
    for patch, tech in zip(bp['boxes'], technologies):
        patch.set_facecolor(get_tech_color(tech))
        patch.set_alpha(0.7)
    
    # Calculate means if needed
    means = [data.mean() for data in data_by_tech]
    
    # Optionally add mean markers
    if show_mean:
        ax.plot(range(1, len(technologies) + 1), means, 'D', 
                color='black', markersize=10, label='Mean', zorder=3)
        ax.legend(fontsize=10)
    
    # Styling
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(labelsize=12, pad=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Optionally add statistics annotation (for binary comparison)
    stats_dict = None
    if show_stats and len(technologies) == 2:
        # Calculate percentage difference (assumes lower is better)
        mean_low = min(means)
        mean_high = max(means)
        pct_lower = ((mean_high - mean_low) / mean_high) * 100
        
        # Add annotation
        ax.annotate(
            f'{pct_lower:.0f}% lower',
            xy=(1.5, (mean_low + mean_high) / 2),
            fontsize=12,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
        )
        
        # Create dict mapping technology to mean
        tech_means = dict(zip(technologies, means))
        
        # Return stats with clear keys
        stats_dict = {
            'eaf_mean': tech_means.get('EAF', None),
            'bof_mean': tech_means.get('BF-BOF', None),
            'difference': mean_high - mean_low,
            'pct_lower': pct_lower,
            'all_means': tech_means  # Include full mapping for debugging
        }
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, ax, stats_dict

# =============================================================================
# TESTING CODE
# =============================================================================

if __name__ == "__main__":
    print("Testing plotting_utils.py...")
    print("=" * 80)
    
    # Test color scheme
    print("\n1. Testing technology color scheme:")
    test_techs = ['BF-BOF', 'EAF', 'EAF Stainless', 'BF-BOF → H₂-DRI']
    for tech in test_techs:
        color = get_tech_color(tech)
        print(f"   {tech:<20} → {color}")
    
    print("\n✓ All functions defined successfully!")
    print("=" * 80)
