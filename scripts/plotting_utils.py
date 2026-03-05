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
# MAPPINGS FOR ANIMATION (Action Score Temporal Comparison)
# =============================================================================

# Maps action score company names to ClimateBERT company names
ANIMATION_NAME_MAPPING = {
    'ArcelorMittal': 'ArcelorMittal',
    'SSAB': 'SSAB',
    'Salzgitter AG': 'Salzgitter',
    'Outokumpu': 'Outokumpu',
    'Celsa Group': 'Celsa',
    'SIDENOR Group': 'SIDENOR',
    'Feralpi Group': 'Feralpi',
    'Voestalpine': 'Voestalpine',
    'Acerinox EU': 'Acerinox',
    'Tata Steel Nederland': 'TataSteelNederland',
    'Tata Steel UK': 'TataSteelUK',
    'SHS Group': 'Dillinger',
    "Acciaierie d'Italia Holding": 'AcciaieriedItalia'
}

# Shortens long names for display (avoid label overlap)
ANIMATION_DISPLAY_NAMES = {
    'TataSteelNederland': 'Tata NL',
    'TataSteelUK': 'Tata UK',
    'AcciaieriedItalia': 'Acciaierie',
    'Dillinger': 'SHS'
}

# Technology mapping (uses short names)
ANIMATION_TECH_MAP = {
    'Celsa': 'EAF',
    'SIDENOR': 'EAF',
    'Feralpi': 'EAF',
    'Outokumpu': 'EAF',
    'Acerinox': 'EAF',
    'SSAB': 'BF-BOF',
    'Salzgitter': 'BF-BOF',
    'Tata NL': 'BF-BOF',
    'Tata UK': 'BF-BOF',
    'SHS': 'BF-BOF',
    'ArcelorMittal': 'BF-BOF',
    'Voestalpine': 'BF-BOF',
    'Acciaierie': 'BF-BOF'
}


def prepare_animation_data(action_scores_df, talk_scores_df):
    """
    Prepare data for animation using hardcoded SuSteelAible project mappings.
    
    Parameters
    ----------
    action_scores_df : pd.DataFrame
        Action scores with 'company' column
    talk_scores_df : pd.DataFrame
        Talk scores with 'company' column
    
    Returns
    -------
    tuple
        (action_scores_mapped, talk_scores_mapped)
    
    Examples
    --------
    >>> action_ready, talk_ready = prepare_animation_data(
    ...     action_scores_temporal, 
    ...     climate_agg_df
    ... )
    """
    # Copy to avoid modifying originals
    action_df = action_scores_df.copy()
    talk_df = talk_scores_df.copy()
    
    # Step 1: Map action score names to match talk score names
    action_df['company'] = action_df['company'].map(ANIMATION_NAME_MAPPING)
    
    # Step 2: Shorten names for display
    action_df['company'] = action_df['company'].replace(ANIMATION_DISPLAY_NAMES)
    talk_df['company'] = talk_df['company'].replace(ANIMATION_DISPLAY_NAMES)
    
    print("✓ Animation data prepared with SuSteelAible project mappings")
    
    return action_df, talk_df

# =============================================================================
# ANIMATED TALK VS ACTION MATRIX
# =============================================================================

def create_animated_talk_action_matrix(action_scores_df, talk_scores_df, 
                                       technology_map, output_path='talk_vs_action.mp4',
                                       total_frames=200, fps=15,
                                       pause_start=20, pause_end=100):
    """
    Create animated Talk vs Action matrix showing company movement from 2019 to 2024.
    
    This function creates a PowerPoint-compatible MP4 animation showing how companies'
    climate discussion (Talk) and operational progress (Action) evolved between the
    pre-COVID (2019) and post-COVID (2024) periods.
    
    Parameters
    ----------
    action_scores_df : pd.DataFrame
        Action scores with columns: company, period, total_score
        Must have 'pre2020' and 'post2020' periods
    talk_scores_df : pd.DataFrame
        Talk scores (from ClimateBERT) with columns: company, period, climate_pct_mean
        Must have 'pre2020' and 'post2020' periods
    technology_map : dict
        Mapping of company names to technology types
        Example: {'Celsa Group': 'EAF', 'SSAB': 'BF-BOF', ...}
    output_path : str, optional
        Path to save the MP4 file
        Default: 'talk_vs_action.mp4'
    total_frames : int, optional
        Total number of animation frames
        Default: 200 (~13 seconds at 15 FPS)
    fps : int, optional
        Frames per second
        Default: 15
    pause_start : int, optional
        Number of frames to pause at start (showing 2019)
        Default: 20
    pause_end : int, optional
        Number of frames to pause at end (showing 2024)
        Default: 100
    
    Returns
    -------
    str
        Path to saved MP4 file
    
    Examples
    --------
    >>> # Basic usage
    >>> tech_map = {
    ...     'Celsa Group': 'EAF',
    ...     'SSAB': 'BF-BOF',
    ...     'Outokumpu': 'EAF'
    ... }
    >>> create_animated_talk_action_matrix(
    ...     action_scores_df=action_scores,
    ...     talk_scores_df=climate_bert_scores,
    ...     technology_map=tech_map
    ... )
    'talk_vs_action.mp4'
    
    >>> # Custom animation settings
    >>> create_animated_talk_action_matrix(
    ...     action_scores_df=action_scores,
    ...     talk_scores_df=climate_bert_scores,
    ...     technology_map=tech_map,
    ...     output_path='../outputs/animation.mp4',
    ...     total_frames=150,
    ...     fps=20
    ... )
    '../outputs/animation.mp4'
    
    Notes
    -----
    - Requires ffmpeg to be installed (auto-installs if missing)
    - Colors: EAF = Green (#2E7D32), BF-BOF = Orange (#D25A19)
    - New reporters (companies only in 2024) are marked with *
    - Arrows show movement for companies with distance > 5 points
    - Output format: MP4 (H.264, yuv420p) compatible with PowerPoint
    """
    from matplotlib.animation import FuncAnimation
    import subprocess
    
    # Check ffmpeg availability
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ ffmpeg found")
    except:
        print("\n⚠️  ffmpeg not found. Installing...")
        try:
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], check=True)
            print("✓ ffmpeg installed")
        except:
            print("❌ Could not install ffmpeg automatically.")
            print("   Please install manually: sudo apt-get install ffmpeg")
            raise
    
    # Animation-specific technology colors (different from main color scheme)
    tech_colors = {
        'EAF': '#2E7D32',      # Dark green
        'BF-BOF': '#D25A19'    # Orange
    }
    
    # Prepare animation data
    animation_data = []
    
    for company in action_scores_df['company'].unique():
        # Get action scores
        action_pre = action_scores_df[
            (action_scores_df['company'] == company) & 
            (action_scores_df['period'] == 'pre2020')
        ]
        action_post = action_scores_df[
            (action_scores_df['company'] == company) & 
            (action_scores_df['period'] == 'post2020')
        ]
        
        # Get talk scores
        climate_pre = talk_scores_df[
            (talk_scores_df['company'] == company) & 
            (talk_scores_df['period'] == 'pre2020')
        ]
        climate_post = talk_scores_df[
            (talk_scores_df['company'] == company) & 
            (talk_scores_df['period'] == 'post2020')
        ]
        
        # Determine technology
        tech = technology_map.get(company, 'Unknown')
        # Simplify technology for color mapping
        if 'BF-BOF' in tech:
            tech_simplified = 'BF-BOF'
        elif 'EAF' in tech:
            tech_simplified = 'EAF'
        else:
            tech_simplified = 'Unknown'
        
        # 2019 data point
        if len(action_pre) > 0 and len(climate_pre) > 0:
            animation_data.append({
                'company': company,
                'year': 2019,
                'action_score': action_pre['total_score'].values[0],
                'talk_score': climate_pre['climate_pct_mean'].values[0],
                'technology': tech_simplified,
                'color': tech_colors.get(tech_simplified, 'gray')
            })
        
        # 2024 data point
        if len(action_post) > 0 and len(climate_post) > 0:
            has_2019 = len(action_pre) > 0 and len(climate_pre) > 0
            animation_data.append({
                'company': company,
                'year': 2024,
                'action_score': action_post['total_score'].values[0],
                'talk_score': climate_post['climate_pct_mean'].values[0],
                'technology': tech_simplified,
                'color': tech_colors.get(tech_simplified, 'gray'),
                'is_new': not has_2019
            })
    
    anim_df = pd.DataFrame(animation_data)
    
    # Identify company sets
    companies_2019 = set(anim_df[anim_df['year'] == 2019]['company'].unique())
    companies_2024 = set(anim_df[anim_df['year'] == 2024]['company'].unique())
    companies_both = companies_2019 & companies_2024
    companies_new = companies_2024 - companies_2019
    
    print(f"\n✓ Animation data prepared:")
    print(f"  Companies in 2019: {len(companies_2019)}")
    print(f"  Companies in 2024: {len(companies_2024)}")
    print(f"  New reporters: {len(companies_new)}")
    if len(companies_new) > 0:
        print(f"    {', '.join(sorted(companies_new))}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    def smoothstep(t):
        """Smooth easing function for animation transitions"""
        return t * t * (3 - 2 * t)
    
    def update(frame):
        """Update function for each animation frame"""
        ax.clear()
        
        # Calculate animation progress
        if frame < pause_start:
            progress = 0
            period_label = "2019"
        elif frame >= total_frames - pause_end:
            progress = 1.0
            period_label = "2024"
        else:
            t = (frame - pause_start) / (total_frames - pause_start - pause_end)
            progress = smoothstep(t)
            period_label = None
        
        # Get data for each year
        df_2019 = anim_df[anim_df['year'] == 2019]
        df_2024 = anim_df[anim_df['year'] == 2024]
        
        # Plot companies with both years (transitioning)
        for company in companies_both:
            data_2019 = df_2019[df_2019['company'] == company].iloc[0]
            data_2024 = df_2024[df_2024['company'] == company].iloc[0]
            
            # Interpolate positions
            action_pos = (
                data_2019['action_score'] * (1 - progress) + 
                data_2024['action_score'] * progress
            )
            talk_pos = (
                data_2019['talk_score'] * (1 - progress) + 
                data_2024['talk_score'] * progress
            )
            
            ax.scatter(
                action_pos, talk_pos,
                c=data_2024['color'],
                s=400,
                alpha=0.8,
                edgecolors='black',
                linewidth=2.5,
                zorder=10
            )
            
            ax.annotate(
                company,
                (action_pos, talk_pos),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                zorder=11
            )
            
            # Draw arrow at end
            if progress == 1.0:
                distance = np.sqrt(
                    (data_2024['action_score'] - data_2019['action_score'])**2 +
                    (data_2024['talk_score'] - data_2019['talk_score'])**2
                )
                if distance > 5:
                    ax.annotate(
                        '',
                        xy=(data_2024['action_score'], data_2024['talk_score']),
                        xytext=(data_2019['action_score'], data_2019['talk_score']),
                        arrowprops=dict(
                            arrowstyle='->',
                            color=data_2024['color'],
                            lw=2.5,
                            alpha=0.6
                        ),
                        zorder=5
                    )
        
        # Plot new reporters (fade in)
        for company in companies_new:
            if progress > 0:
                data_2024 = df_2024[df_2024['company'] == company].iloc[0]
                
                ax.scatter(
                    data_2024['action_score'],
                    data_2024['talk_score'],
                    c=data_2024['color'],
                    s=400,
                    alpha=0.8 * progress,
                    edgecolors='black',
                    linewidth=2.5,
                    zorder=10
                )
                
                label = company + ' *'
                
                ax.annotate(
                    label,
                    (data_2024['action_score'], data_2024['talk_score']),
                    xytext=(6, 6),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold',
                    alpha=progress,
                    zorder=11
                )
        
        # Reference lines
        ax.axhline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)
        ax.axvline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)
        
        # Labels
        ax.set_xlabel('ACTION SCORE (Operational Progress)', 
                     fontsize=18,
                     fontweight='bold')
        ax.set_ylabel('CLIMATE COMMUNICATION (% of Report)', 
                     fontsize=18,
                     fontweight='bold')
        
        # Title
        if period_label:
            title_text = f'Talk vs Action — {period_label}'
        else:
            title_text = 'Talk vs Action'
        
        ax.set_title(title_text,
                    fontsize=22,
                    fontweight='bold', 
                    pad=20)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=tech_colors['EAF'], markersize=15,
                      markeredgecolor='black', markeredgewidth=2,
                      label='EAF', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=tech_colors['BF-BOF'], markersize=15,
                      markeredgecolor='black', markeredgewidth=2,
                      label='BF-BOF', linestyle='None'),
        ]
        
        if len(companies_new) > 0:
            legend_elements.append(
                plt.Line2D([0], [0], marker='', color='w',
                          label='* = New reporter', linestyle='None')
            )
        
        ax.legend(handles=legend_elements, 
                 loc='upper left', 
                 fontsize=13,
                 framealpha=0.95)
        
        # Grid
        ax.grid(True, alpha=0.3, zorder=0, linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Limits
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # Tick labels
        ax.tick_params(axis='both', labelsize=14)
        
        return []
    
    # Create animation
    print(f"\nGenerating animation ({total_frames} frames at {fps} FPS)...")
    anim = FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=1000/fps,
        blit=False,
        repeat=False
    )
    
    # Save as MP4
    print(f"Saving as MP4 (this may take ~1 minute)...")
    anim.save(
        output_path,
        writer='ffmpeg',
        fps=fps,
        dpi=120,
        codec='libx264',
        bitrate=5000,
        extra_args=['-pix_fmt', 'yuv420p']
    )
    
    plt.close()
    
    print(f"\n✅ Saved: {output_path}")
    print(f"\nVideo details:")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    print(f"  Resolution: 1920x1440 (16:12 aspect ratio)")
    print(f"  Format: MP4 (H.264)")
    
    return output_path


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
