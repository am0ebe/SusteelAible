import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
from linearmodels.panel import PanelOLS, RandomEffects
from datetime import datetime

def fix_column_names(df):
    df.columns = (df.columns
            .str.replace(r'[\s\n\-’\'\"\‘\’\“\”\[\]\(\)\{\}\<\>\&\+]+', '_', regex=True)
            .str.replace(r'_{2,}', '_', regex=True) 
            .str.strip('_')
            .str.lower())
    
def overview_data(df):
    print("shape: ",df.shape)
    display(df.head(2))
    print("-"*100)
    print("Duplicates:")
    print(df.duplicated().value_counts())
    print("-"*100)
    print("Unique values:")
    display(df.nunique().to_frame().T)
    print("-"*100)
    print("DataFrame info:")
    print(df.info())
    print("-"*100)
    print("Descriptive statistics:")
    display(df.describe().T.round(2))
    print("-"*100)


def plot_panel_timeseries(df, value_col, color_by='company', group_by='company', time_col='year'):
    """Plot time series colored by any variable"""
    df_reset = df.reset_index()
    
    # Map color_by values to colors
    unique_colors = df_reset[color_by].unique()
    color_map = {val: plt.cm.tab10(i) for i, val in enumerate(unique_colors)}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for company in df_reset[group_by].unique():
        company_data = df_reset[df_reset[group_by] == company]
        color_val = company_data[color_by].iloc[0]  # Get color category
        color = color_map[color_val]
        
        ax.plot(company_data[time_col], company_data[value_col], 
                marker='o', label=f"{company} ({color_val})", 
                color=color, linewidth=2)
    
    ax.set_xlabel(time_col.title())
    ax.set_ylabel(value_col.title())
    ax.set_title(f'{value_col.title()} Over Time (Color: {color_by})')
    ax.legend(title=f'Company ({color_by})', bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
def EffectsModel(df, y, x_vars,entity = 'company', time = 'year', cluster_entity=True, cluster_time=True, 
                            weights=None, check_rank=True, drop_absorbed=True,
                            threshold_obs=3, model ='fixed', save = False):
    """
    Fixed Effects model for panel data.
    Parameters:
    data: pd.DataFrame
    y : str
        dependant variabke
    x : list of str
        independant variables
    entity : str
        Name of entity, default 'company'
    time : str
        Name of time identifier, default 'year'
    cluster_entity : bool
        default=True
    cluster_time : bool
        defuault=True
    weights : str, default = None
        Name of weights variable if any
    model : str, 
        default = 'fixed', other options = 'twoway', 'random'
    save : bool
        option to save dataframe as csv output
    --- 

    Returns:
    results : PanelEffectsResults
        linearmodels results object with methods:
        - .summary: print detailed results
        - .params: coefficients
        - .std_errors: standard errors
        - .tstats: t-statistics
        - .pvalues: p-values
        - .rsquared: R-squared
        - .resids: residuals
        - .predict(): fitted values
    """

    cols_needed = [y] + x_vars + [entity, time]
    df = df[cols_needed].copy()
    df = df[cols_needed].dropna()

    df = df.groupby(entity).filter(lambda x: x[y].notna().sum() >= threshold_obs)

    w = df[weights] if weights else None
    df = df.set_index([entity, time]).sort_index()

    # Build formula string
    if model == 'fixed':
        if isinstance(x_vars, str):
            formula = f"{y} ~ {x_vars} + EntityEffects"
        else:
            formula = f"{y} ~ {' + '.join(x_vars)} + EntityEffects"
    elif model == 'random':
        if isinstance(x_vars, str):
            formula = f"{y} ~ {x_vars}"
        else:
            formula = f"{y} ~ {' + '.join(x_vars)}"
    elif model == 'twoway':
        if isinstance(x_vars, str):
            formula = f"{y} ~ {x_vars} + TimeEffects + EntityEffects"
        else:
            formula = f"{y} ~ {' + '.join(x_vars)} + TimeEffects + EntityEffects"
    elif model== 'time':
        if isinstance(x_vars, str):
            formula = f"{y} ~ {x_vars} + TimeEffects"
        else:
            formula = f"{y} ~ {' + '.join(x_vars)} + TimeEffects"
    else:
        print("No model defined. Please specify")

    if model == 'random':
        mod = RandomEffects.from_formula(
            formula, 
            data=df, 
            weights=w,
            check_rank=check_rank
            )
    else:
        mod = PanelOLS.from_formula(
            formula, 
            data=df, 
            weights=w,
            check_rank=check_rank, drop_absorbed=drop_absorbed
            )

    cov_type = 'clustered' if (cluster_entity or cluster_time) else 'robust'
    result = mod.fit(
        cov_type=cov_type,
        cluster_entity=cluster_entity,
        cluster_time=cluster_time
        )
    if save == True: 
        df.to_csv(f'model_df_{datetime.now().strftime("%m-%d_%H-%M")}.csv')
    return result
 

def mixed_model(df, y, x_vars,entity = 'company', time = 'year', cluster_entity=True, cluster_time=True, 
                            weights=None, check_rank=True, drop_absorbed=True, auto_drop_invariant=False, 
                            threshold_obs=3):
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    df = df.groupby(entity).filter(lambda x: x[y].notna().sum() >= threshold_obs)
    formula = " + ".join(x_vars)
    mod = MixedLM.from_formula(f"{y} ~ {time} + {formula}", data=df, groups=entity)
    result = mod.fit()
    return result


def lag_model(df, y, x,entity = 'company', time = 'year',):
    df = df[[entity,time,y,x]].copy()
    df = df.set_index([entity, time]).sort_index()
    df[f"{y}_lag1"] = df.groupby(level=entity)[y].shift(1)
    df[f"{y}_lag2"] = df.groupby(level=entity)[y].shift(2)
    df[f"{x}_lag1"] = df.groupby(level=entity)[x].shift(1)
    df[f"{x}_lag2"] = df.groupby(level=entity)[x].shift(2)
    
    model = PanelOLS.from_formula(f'{y} ~ {y}_lag1 + {y}_lag2 + {x}_lag1 +{x}_lag2', data=df)
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    # Joint F-test for Granger causality: H0: x_lag1 = x_lag2 = 0
    try:
        f_test = result.wald_test(formula=f'{x}_lag1 = 0; {x}_lag2 = 0')
        f_pvalue = f_test.pval
    except:
        # Fallback: manual calculation if wald_test fails
        f_pvalue = None
    
    # Extract key statistics
    x_lag1_tstat = result.tstats[f'{x}_lag1']
    x_lag2_tstat = result.tstats[f'{x}_lag2']
    x_lag1_pval = result.pvalues[f'{x}_lag1']
    x_lag2_pval = result.pvalues[f'{x}_lag2']
    
    # Determine if Granger causality holds
    if f_pvalue is not None:
        granger_result = "✅ SIGNIFICANT" if f_pvalue < 0.05 else "❌ NOT SIGNIFICANT"
    else:
        # Fallback: at least one lag significant
        granger_result = "⚠️ CHECK MANUALLY" if (x_lag1_pval < 0.05 or x_lag2_pval < 0.05) else "❌ NOT SIGNIFICANT"
    
    return {
        'result': result,
        'f_test': f_test if f_pvalue is not None else None,
        'f_pvalue': f_pvalue,
        'x_lag1_tstat': x_lag1_tstat,
        'x_lag2_tstat': x_lag2_tstat,
        'x_lag1_pval': x_lag1_pval,
        'x_lag2_pval': x_lag2_pval,
        'granger': granger_result
    }


def coef_plot(coeff, ci, title=''):
    y_cats = {"Scope 1":'scope1',
    "Production Volume":'production',
    "Age":'age',
    "BF-BOF":'technology_bf_bof',
    "EAF":'technology_eaf',
    "EAF Stainless":'technology_eaf_stainless',
    "Carbon Price":'carbon_price',
    "Electricity Price":'electricity_price',
    "Coal Price":'coal_price_australia',
    "Iron Ore Price":'iron_ore_price',
    "Natural Gas Price":'natural_gas_price',
    "ETS":'ets',
    "CBAM":'cbam',
    "IED update":'ied_update',
    "Green Deal":'green_deal',
    "NZIA":'nzia',
    "GPP":'gpp',
    "CSR":'csr',
    "Paris":'paris',
    "Government Change":'govt_change',
    "SCI":'sci'}
    label_map = {v: k for k, v in y_cats.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(coeff))
    ax.barh(y_pos, coeff, xerr=[coeff - ci.iloc[:,0], ci.iloc[:,1] - coeff],
            capsize=5, color='steelblue', alpha=0.8,
            error_kw={'elinewidth': 1, 'ecolor': 'gray'})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label_map.get(idx, idx) for idx in coeff.index])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient estimate')
    ax.set_title(f'External Drivers Coefficients {title}') # (R² within: {fe_result.rsquared_within:.3f})
    plt.tight_layout()
    plt.show()
    # fig.savefig("coef.png", transparent = True)


# Small Multiples (facet Plots)
def facet_panel_plots(df, value_col, n_cols=2):
    """Create small multiples for each company"""
    df_reset = df.reset_index()
    entities = df_reset['company'].unique()
    n_entities = len(entities)
    
    n_rows = int(np.ceil(n_entities / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, company in enumerate(entities):
        company_data = df_reset[df_reset['company'] == company]
        ax = axes[idx]
        ax.plot(company_data['year'], company_data[value_col], 
                marker='o', linewidth=2, color='steelblue')
        ax.set_title(f'company: {company}')
        ax.set_xlabel('Year')
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(entities), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{value_col} by company', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
