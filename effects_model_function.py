import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects


def identify_time_invariant(data, x_vars, entity):
    """Identify variables that don't vary within entities (absorbed by entity FE)."""
    time_invariant = []
    for var in x_vars:
        # Check if variable varies within each entity
        varies = data.groupby(entity)[var].nunique() > 1
        if not varies.any():
            time_invariant.append(var)
    return time_invariant

def EffectsModel(df, y, x_vars, entity = 'company', time = 'year', cluster_entity=True, cluster_time=True, 
                            weights=None, check_rank=True, drop_absorbed=True, auto_drop_invariant=True, 
                            threshold_obs=3, model ='fixed'):
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

    # Handle time-invariant variables if requested
    if auto_drop_invariant and not isinstance(x_vars, str):
        time_invariant = identify_time_invariant(df, x_vars, entity)
        if time_invariant:
            print(f"\nWarning: Dropping time-invariant variables (absorbed by entity FE):")
            print(f"  {', '.join(time_invariant)}\n")
            x_vars = [v for v in x_vars if v not in time_invariant]
            if not x_vars:
                raise ValueError("All X variables are time-invariant and absorbed by entity FE")
            
    df = df.set_index([entity, time])

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
    else:
        print("No model defined. Please specify")

    if model == 'random':
        mod = RandomEffects.from_formula(
            formula, 
            data=df, 
            weights=w
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
    return result
 

