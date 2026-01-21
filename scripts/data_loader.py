"""
Data Loading Functions for SuSteelAible Capstone Project
==================================================================

This script provides standardized data loading and preparation functions
for use across multiple analysis notebooks (EDA, trends, baseline models).

Author: Irene Polgar
Date: December 2025
Version: 2.0 (Updated for Excel format and new folder structure)
"""

import pandas as pd
import numpy as np


def load_company_data(filepath='../data/emissions_and_production_technology.xlsx',
                      sheet_name='emissions_steel_production',
                      fix_apostrophes=True, filter_region=None):
    """
    Load company-level emissions and production data from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to Excel file containing emissions data
        Default: '../data/emissions_and_production_technology.xlsx'
    sheet_name : str
        Name of sheet containing emissions data
        Default: 'emissions_steel_production'
    fix_apostrophes : bool
        Fix encoding issues with apostrophes (default: True)
    filter_region : str, optional
        Filter to specific region: 'Europe', 'Asia', or None (all companies)
    
    Returns
    -------
    pd.DataFrame
        Company emissions data with columns:
        - company, country, technology, year
        - production, scope1, scope2_location, scope2_market
        - scope_1_2_location, intensity_location_co2e, intensity_market_co2e
        - data_quality, notes
    
    Notes
    -----
    Excel file uses German setup (comma as decimal separator) but pandas
    handles this automatically with openpyxl engine.
    """
    # Read Excel file with openpyxl engine
    df = pd.read_excel(
        filepath, 
        sheet_name=sheet_name,
        engine='openpyxl'
    )
    
    # Fix apostrophe encoding issues (e.g., Acciaierie d'Italia)
    if fix_apostrophes and 'company' in df.columns:
        df['company'] = df['company'].str.replace('\x92', "'", regex=False)
        df['company'] = df['company'].str.replace("'", "'", regex=False)  # Curly apostrophe
    
    # Ensure year is integer
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)
    
    # Filter by region if specified
    if filter_region == 'Europe':
        european_countries = [
            'Spain', 'Finland', 'Sweden', 'Luxembourg', 'Germany', 
            'Netherlands', 'UK', 'Austria', 'Bulgaria-Greece', 
            'Italy', 'France', 'Europe (Multi-country)'
        ]
        df = df[df['country'].isin(european_countries)].copy()
    elif filter_region == 'Asia':
        asian_countries = ['Japan', 'China', 'India']
        df = df[df['country'].isin(asian_countries)].copy()
    
    return df


def load_eu_data(filepath='../data/external_drivers.xlsx',
                 sheet_name='Sheet1',
                 years=(2013, 2024)):
    """
    Load and prepare EU-wide steel industry data from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to Excel file containing external drivers data
        Default: '../data/EDA/external_drivers.xlsx'
    sheet_name : str
        Name of sheet containing external drivers
        Default: 'Sheet1'
    years : tuple
        Year range to filter (start, end) inclusive
    
    Returns
    -------
    pd.DataFrame
        EU-wide data with columns:
        - year, carbon_price, electricity_price_eu, coal_price_australia
        - iron_ore_price, natural_gas_price_eu, crude_steel_production_eu
        - ETS_iron_steel, eu_intensity_external
    
    Notes
    -----
    Calculates EU-wide emission intensity from ETS data and production.
    """
    # Read Excel file
    external = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        engine='openpyxl'
    )
    
    # Filter to Europe-wide data only
    eu_data = external[external['country'] == 'Europe (Multi-country)'].copy()
    
    # Select relevant columns (check which columns actually exist)
    available_cols = [
        'year',
        'carbon_price',
        'electricity_price_eu',
        'coal_price_australia',
        'iron_ore_price',
        'natural_gas_price_eu',
        'crude_steel_production_eu',
        'ETS_iron_steel'
    ]
    
    # Only select columns that exist in the dataframe
    cols_to_select = [col for col in available_cols if col in eu_data.columns]
    eu_data = eu_data[cols_to_select].copy()
    
    # Calculate EU-wide emission intensity if data available
    if 'ETS_iron_steel' in eu_data.columns and 'crude_steel_production_eu' in eu_data.columns:
        # ETS_iron_steel is in tonnes, production in Mt
        eu_data['eu_intensity_external'] = (
            (eu_data['ETS_iron_steel'] / 1_000_000) / 
            eu_data['crude_steel_production_eu']
        )
    
    # Filter to specified year range
    if 'year' in eu_data.columns:
        eu_data = eu_data[
            (eu_data['year'] >= years[0]) & 
            (eu_data['year'] <= years[1])
        ]
    
    return eu_data


def load_global_data(filepath='../data/global_steel_trend.xlsx',
                     sheet_name='global_steel_trend'):
    """
    Load global steel production and emissions trend data from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to Excel file containing global steel trend data
        Default: '../data/EDA/global_steel_trend.xlsx'
    sheet_name : str
        Name of sheet containing global trend data
        Default: 'global_steel_trend'
    
    Returns
    -------
    pd.DataFrame
        Global steel data (for context/visualization only)
    
    Notes
    -----
    This data provides global context but is not used in primary analysis.
    """
    global_data = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        engine='openpyxl'
    )
    return global_data


def load_transparency_scores(filepath='../data/emissions_and_production_technology.xlsx',
                              sheet_name='transparency_scores'):
    """
    Load sustainability certification and transparency data from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to Excel file
    sheet_name : str
        Name of sheet containing transparency scores
        Default: 'transparency_scores'
    
    Returns
    -------
    pd.DataFrame
        Transparency data with CDP ratings, ISO certifications, SBTi validation, etc.
    """
    transparency = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        engine='openpyxl'
    )
    return transparency


def load_technology_scores(filepath='../data/emissions_and_production_technology.xlsx',
                            sheet_name='technology_scores'):
    """
    Load technology assessment scores from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to Excel file
    sheet_name : str
        Name of sheet containing technology scores
        Default: 'technology_scores'
    
    Returns
    -------
    pd.DataFrame
        Technology scores with transformation plans and readiness assessment.
    """
    tech_scores = pd.read_excel(
        filepath,
        sheet_name=sheet_name,
        engine='openpyxl'
    )
    return tech_scores


def filter_complete_data(df, min_years=4):
    """
    Filter dataset to companies with sufficient data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Company emissions dataframe
    min_years : int
        Minimum number of years with complete scope1 data required
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only companies meeting completeness threshold
    """
    # Count years with non-null scope1 per company
    completeness = df.groupby('company')['scope1'].count()
    valid_companies = completeness[completeness >= min_years].index
    
    df_filtered = df[df['company'].isin(valid_companies)].copy()
    
    return df_filtered


def select_best_scope2_method(df):
    """
    Select best available Scope 2 methodology per company.
    
    For each company, compares data availability for location-based vs 
    market-based Scope 2 and selects the method with more complete data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Company emissions dataframe with scope2_location and scope2_market
    
    Returns
    -------
    pd.DataFrame
        Extended dataframe with:
        - scope2_intensity_location: Location-based Scope 2 intensity
        - scope2_intensity_market: Market-based Scope 2 intensity  
        - scope2_intensity_best: Best available method per company
        - scope2_method_used: Which method was selected ('location' or 'market')
    
    Notes
    -----
    Selection criteria: method with most non-null observations per company.
    Ties favor location-based for consistency.
    """
    df = df.copy()
    
    # Calculate both intensity types
    df['scope2_intensity_location'] = df['scope2_location'] / df['production']
    df['scope2_intensity_market'] = df['scope2_market'] / df['production']
    
    # Determine best method per company
    df['scope2_method_used'] = df.groupby('company').apply(
        lambda x: 'location' if x['scope2_intensity_location'].notna().sum() >= 
                                 x['scope2_intensity_market'].notna().sum() 
                            else 'market'
    ).loc[df['company']].values
    
    # Select best value
    df['scope2_intensity_best'] = df.apply(
        lambda row: (row['scope2_intensity_location'] 
                    if row['scope2_method_used'] == 'location'
                    else row['scope2_intensity_market']),
        axis=1
    )
    
    return df


def prepare_analysis_dataset(df):
    """
    Prepare dataset for analysis by calculating additional metrics.
    
    NOTE: For EDA notebook, these calculations are shown explicitly 
    for educational purposes. For other notebooks, this function 
    provides convenient pre-calculated columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw company emissions dataframe
    
    Returns
    -------
    pd.DataFrame
        Extended dataframe with:
        - scope1_intensity: Scope 1 emissions / production
        - scope2_intensity_location: Location-based Scope 2 / production
        - scope2_intensity_market: Market-based Scope 2 / production
        - scope2_intensity_best: Best available Scope 2 method per company
        - scope2_method_used: Method selected ('location' or 'market')
        - technology_group: Simplified categories (EAF, BF-BOF, Other)
    """
    df = df.copy()
    
    # Calculate Scope 1 intensity
    if 'scope1' in df.columns and 'production' in df.columns:
        if 'scope1_intensity' not in df.columns:
            df['scope1_intensity'] = df['scope1'] / df['production']
    
    # Calculate Scope 2 intensities (both methods + best selection)
    if 'scope2_location' in df.columns and 'production' in df.columns:
        df = select_best_scope2_method(df)
    
    # Create simplified technology groups
    if 'technology' in df.columns:
        df['technology_group'] = df['technology'].apply(
            lambda x: 'EAF' if 'EAF' in str(x) 
            else 'BF-BOF' if 'BF-BOF' in str(x)
            else 'Other'
        )
    
    return df


def get_data_summary(df):
    """
    Generate summary statistics about the loaded dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Company emissions dataframe
    
    Returns
    -------
    dict
        Dictionary with summary statistics
    """
    summary = {
        'n_companies': df['company'].nunique(),
        'n_rows': len(df),
        'year_range': (df['year'].min(), df['year'].max()),
        'companies': sorted(df['company'].unique()),
        'technologies': sorted(df['technology'].unique()) if 'technology' in df.columns else [],
        'countries': sorted(df['country'].unique()) if 'country' in df.columns else [],
        'completeness_scope1': df['scope1'].notna().sum() / len(df) * 100 if 'scope1' in df.columns else 0,
        'completeness_scope2': df['scope2_location'].notna().sum() / len(df) * 100 if 'scope2_location' in df.columns else 0
    }
    
    return summary


def print_data_summary(df):
    """
    Print formatted summary of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Company emissions dataframe
    """
    summary = get_data_summary(df)
    
    print("-"*80)
    print("DATASET SUMMARY")
    print("-"*80)
    print(f"\nCompanies: {summary['n_companies']}")
    print(f"Total rows: {summary['n_rows']}")
    print(f"Year range: {summary['year_range'][0]}-{summary['year_range'][1]}")
    
    if summary['technologies']:
        print(f"\nTechnologies: {', '.join(summary['technologies'])}")
    if summary['countries']:
        print(f"Countries: {', '.join(summary['countries'])}")
    print("-"*80)


# =============================================================================
# TESTING CODE - Only runs when script is executed directly
# =============================================================================

if __name__ == "__main__":
    print("Testing data_loader.py with Excel files...")
    print("="*80)
    
    try:
        # 1. Load main company emissions data
        print("\n1. Loading company emissions data...")
        df = load_company_data()
        print(f"   ✓ Loaded {len(df)} rows from emissions_steel_production sheet")
        print_data_summary(df)
        
        # 2. Load external datasets
        print("\n2. Loading EU-wide data...")
        eu_data = load_eu_data()
        print(f"   ✓ Loaded {len(eu_data)} rows of EU data")
        
        print("\n3. Loading global steel trends...")
        global_data = load_global_data()
        print(f"   ✓ Loaded {len(global_data)} rows of global data")
        
        print("\n4. Loading transparency scores...")
        transparency = load_transparency_scores()
        print(f"   ✓ Loaded {len(transparency)} rows from transparency_scores sheet")
        
        print("\n5. Loading technology scores...")
        tech = load_technology_scores()
        print(f"   ✓ Loaded {len(tech)} rows from technology_scores sheet")
        
        # 3. Test data filtering
        print("\n6. Testing filter_complete_data (min 5 years)...")
        df_complete = filter_complete_data(df, min_years=5)
        print(f"   ✓ Filtered to {df_complete['company'].nunique()} companies")
        print(f"   ✓ {len(df_complete)} rows remaining")
        
        # 4. Test best Scope 2 method selection
        print("\n7. Testing select_best_scope2_method...")
        df_with_scope2 = select_best_scope2_method(df_complete)
        print(f"   ✓ Added scope2_intensity_best column")
        
        # Show which method was selected per company
        method_summary = df_with_scope2.groupby('company')['scope2_method_used'].first()
        location_count = (method_summary == 'location').sum()
        market_count = (method_summary == 'market').sum()
        print(f"   ✓ Location-based: {location_count} companies")
        print(f"   ✓ Market-based: {market_count} companies")
        
        # 5. Test full analysis dataset preparation
        print("\n8. Testing prepare_analysis_dataset...")
        df_analysis = prepare_analysis_dataset(df_complete)
        print(f"   ✓ Analysis dataset ready: {len(df_analysis)} rows")
        
        # Show what columns were added
        new_cols = [col for col in df_analysis.columns if col not in df.columns]
        if new_cols:
            print(f"   ✓ Added columns: {', '.join(new_cols)}")
        
        # Final success message
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED - All functions working correctly!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: File not found - {e}")
        print("\n   Make sure Excel files are in the correct locations:")
        print("   - ../data/emissions_and_production_technology.xlsx")
        print("   - ../data/external_drivers.xlsx")
        print("   - ../data/global_steel_trend.xlsx")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()