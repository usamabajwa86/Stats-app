"""
Agriculture-Specific Analysis Module
Includes RCBD, Latin Square, Yield Analysis, and Field Trial Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st

# ==================== RANDOMIZED COMPLETE BLOCK DESIGN (RCBD) ====================

def rcbd_analysis(df, treatment_col, block_col, response_col, alpha=0.05):
    """
    Analyze Randomized Complete Block Design (RCBD)
    """
    try:
        # Create formula for two-way ANOVA
        formula = f'{response_col} ~ C({treatment_col}) + C({block_col})'

        # Fit model
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)

        # Extract treatment p-value
        treatment_p = anova_table.loc[f'C({treatment_col})', 'PR(>F)']
        block_p = anova_table.loc[f'C({block_col})', 'PR(>F)']

        # Calculate treatment means
        treatment_means = df.groupby(treatment_col)[response_col].agg(['mean', 'std', 'count']).reset_index()
        treatment_means.columns = ['Treatment', 'Mean', 'Std Dev', 'N']

        # Calculate block means
        block_means = df.groupby(block_col)[response_col].agg(['mean', 'std', 'count']).reset_index()
        block_means.columns = ['Block', 'Mean', 'Std Dev', 'N']

        # Calculate coefficient of variation
        overall_mean = df[response_col].mean()
        overall_std = df[response_col].std()
        cv = (overall_std / overall_mean * 100) if overall_mean != 0 else 0

        # Post-hoc test for treatments if significant
        posthoc_results = None
        if treatment_p < alpha:
            tukey = pairwise_tukeyhsd(endog=df[response_col],
                                     groups=df[treatment_col],
                                     alpha=alpha)
            posthoc_df = pd.DataFrame(data=tukey.summary().data[1:],
                                     columns=tukey.summary().data[0])
            posthoc_results = posthoc_df

        results = {
            'test_type': 'RCBD Analysis',
            'anova_table': anova_table,
            'treatment_means': treatment_means,
            'block_means': block_means,
            'treatment_significant': treatment_p < alpha,
            'block_significant': block_p < alpha,
            'treatment_p_value': treatment_p,
            'block_p_value': block_p,
            'cv_percent': cv,
            'posthoc_results': posthoc_results,
            'model_summary': model.summary()
        }

        return results
    except Exception as e:
        st.error(f"Error performing RCBD analysis: {str(e)}")
        return None

# ==================== LATIN SQUARE DESIGN ====================

def latin_square_analysis(df, treatment_col, row_col, col_col, response_col, alpha=0.05):
    """
    Analyze Latin Square Design
    """
    try:
        # Create formula for three-way ANOVA
        formula = f'{response_col} ~ C({treatment_col}) + C({row_col}) + C({col_col})'

        # Fit model
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)

        # Extract p-values
        treatment_p = anova_table.loc[f'C({treatment_col})', 'PR(>F)']
        row_p = anova_table.loc[f'C({row_col})', 'PR(>F)']
        col_p = anova_table.loc[f'C({col_col})', 'PR(>F)']

        # Calculate treatment means
        treatment_means = df.groupby(treatment_col)[response_col].agg(['mean', 'std', 'count']).reset_index()
        treatment_means.columns = ['Treatment', 'Mean', 'Std Dev', 'N']

        # Calculate CV
        overall_mean = df[response_col].mean()
        overall_std = df[response_col].std()
        cv = (overall_std / overall_mean * 100) if overall_mean != 0 else 0

        # Post-hoc test for treatments if significant
        posthoc_results = None
        if treatment_p < alpha:
            tukey = pairwise_tukeyhsd(endog=df[response_col],
                                     groups=df[treatment_col],
                                     alpha=alpha)
            posthoc_df = pd.DataFrame(data=tukey.summary().data[1:],
                                     columns=tukey.summary().data[0])
            posthoc_results = posthoc_df

        results = {
            'test_type': 'Latin Square Design Analysis',
            'anova_table': anova_table,
            'treatment_means': treatment_means,
            'treatment_significant': treatment_p < alpha,
            'row_significant': row_p < alpha,
            'col_significant': col_p < alpha,
            'treatment_p_value': treatment_p,
            'row_p_value': row_p,
            'col_p_value': col_p,
            'cv_percent': cv,
            'posthoc_results': posthoc_results,
            'model_summary': model.summary()
        }

        return results
    except Exception as e:
        st.error(f"Error performing Latin Square analysis: {str(e)}")
        return None

# ==================== YIELD COMPARISON ====================

def yield_comparison_analysis(df, treatment_col, yield_col, control_name=None):
    """
    Specialized yield comparison analysis for agricultural trials
    """
    try:
        # Calculate yield statistics by treatment
        yield_stats = df.groupby(treatment_col)[yield_col].agg([
            ('N', 'count'),
            ('Mean Yield', 'mean'),
            ('Std Dev', 'std'),
            ('SE', 'sem'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('CV %', lambda x: (x.std() / x.mean() * 100) if x.mean() != 0 else 0)
        ]).reset_index()

        # If control is specified, calculate relative yields
        if control_name and control_name in df[treatment_col].values:
            control_mean = df[df[treatment_col] == control_name][yield_col].mean()

            relative_yields = []
            for _, row in yield_stats.iterrows():
                rel_yield = ((row['Mean Yield'] - control_mean) / control_mean * 100) if control_mean != 0 else 0
                relative_yields.append(rel_yield)

            yield_stats['Relative Yield (%)'] = relative_yields
            yield_stats['Yield Advantage'] = yield_stats['Mean Yield'] - control_mean

        # Perform ANOVA
        groups = [group[yield_col].values for name, group in df.groupby(treatment_col)]
        f_stat, p_value = stats.f_oneway(*groups)

        # Post-hoc test if significant
        posthoc_results = None
        if p_value < 0.05:
            tukey = pairwise_tukeyhsd(endog=df[yield_col],
                                     groups=df[treatment_col],
                                     alpha=0.05)
            posthoc_df = pd.DataFrame(data=tukey.summary().data[1:],
                                     columns=tukey.summary().data[0])
            posthoc_results = posthoc_df

        results = {
            'test_type': 'Yield Comparison Analysis',
            'yield_statistics': yield_stats,
            'anova_f_statistic': f_stat,
            'anova_p_value': p_value,
            'significant_difference': p_value < 0.05,
            'posthoc_results': posthoc_results,
            'control_treatment': control_name
        }

        return results
    except Exception as e:
        st.error(f"Error performing yield comparison: {str(e)}")
        return None

# ==================== TREATMENT EFFECT ANALYSIS ====================

def treatment_effect_analysis(df, control_group, treatment_col, response_col):
    """
    Analyze treatment effects relative to control
    """
    try:
        # Separate control and treatment groups
        control_data = df[df[treatment_col] == control_group][response_col]
        treatments = [t for t in df[treatment_col].unique() if t != control_group]

        effect_results = []

        for treatment in treatments:
            treatment_data = df[df[treatment_col] == treatment][response_col]

            # Calculate means and difference
            control_mean = control_data.mean()
            treatment_mean = treatment_data.mean()
            diff = treatment_mean - control_mean
            percent_change = (diff / control_mean * 100) if control_mean != 0 else 0

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data)

            # Calculate effect size
            pooled_std = np.sqrt(((len(control_data)-1)*control_data.std()**2 +
                                (len(treatment_data)-1)*treatment_data.std()**2) /
                               (len(control_data)+len(treatment_data)-2))
            cohens_d = diff / pooled_std if pooled_std != 0 else 0

            effect_results.append({
                'Treatment': treatment,
                'Control Mean': control_mean,
                'Treatment Mean': treatment_mean,
                'Difference': diff,
                'Percent Change (%)': percent_change,
                't-statistic': t_stat,
                'p-value': p_value,
                'Cohen\'s d': cohens_d,
                'Significant': p_value < 0.05
            })

        effect_df = pd.DataFrame(effect_results)

        results = {
            'test_type': 'Treatment Effect Analysis',
            'control_group': control_group,
            'effect_summary': effect_df,
            'control_n': len(control_data),
            'control_mean': control_data.mean(),
            'control_std': control_data.std()
        }

        return results
    except Exception as e:
        st.error(f"Error performing treatment effect analysis: {str(e)}")
        return None

# ==================== FIELD TRIAL SUMMARY ====================

def field_trial_summary(df, treatment_col, response_cols, block_col=None):
    """
    Create comprehensive field trial summary
    """
    try:
        summary_results = {}

        for response in response_cols:
            # Basic statistics by treatment
            stats_table = df.groupby(treatment_col)[response].agg([
                ('Count', 'count'),
                ('Mean', 'mean'),
                ('Median', 'median'),
                ('Std Dev', 'std'),
                ('SE', 'sem'),
                ('Min', 'min'),
                ('Max', 'max'),
                ('CV%', lambda x: (x.std() / x.mean() * 100) if x.mean() != 0 else 0)
            ]).reset_index()

            # ANOVA or RCBD depending on whether blocks are present
            if block_col:
                formula = f'{response} ~ C({treatment_col}) + C({block_col})'
                model = ols(formula, data=df).fit()
                anova_table = anova_lm(model, typ=2)
                treatment_p = anova_table.loc[f'C({treatment_col})', 'PR(>F)']
            else:
                groups = [group[response].values for name, group in df.groupby(treatment_col)]
                f_stat, treatment_p = stats.f_oneway(*groups)
                anova_table = None

            summary_results[response] = {
                'statistics': stats_table,
                'anova_table': anova_table,
                'p_value': treatment_p,
                'significant': treatment_p < 0.05
            }

        results = {
            'test_type': 'Field Trial Summary',
            'response_variables': response_cols,
            'summary_by_variable': summary_results
        }

        return results
    except Exception as e:
        st.error(f"Error creating field trial summary: {str(e)}")
        return None

# ==================== MULTI-LOCATION TRIAL ANALYSIS ====================

def multi_location_analysis(df, treatment_col, location_col, response_col, alpha=0.05):
    """
    Analyze multi-location trials with treatment × location interaction
    """
    try:
        # Create formula with interaction
        formula = f'{response_col} ~ C({treatment_col}) * C({location_col})'

        # Fit model
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)

        # Extract p-values
        treatment_p = anova_table.loc[f'C({treatment_col})', 'PR(>F)']
        location_p = anova_table.loc[f'C({location_col})', 'PR(>F)']
        interaction_p = anova_table.loc[f'C({treatment_col}):C({location_col})', 'PR(>F)']

        # Treatment means across locations
        overall_means = df.groupby(treatment_col)[response_col].agg(['mean', 'std', 'count']).reset_index()
        overall_means.columns = ['Treatment', 'Mean', 'Std Dev', 'N']

        # Treatment × Location means
        interaction_means = df.groupby([treatment_col, location_col])[response_col].mean().reset_index()
        interaction_pivot = interaction_means.pivot(index=treatment_col,
                                                   columns=location_col,
                                                   values=response_col)

        results = {
            'test_type': 'Multi-Location Analysis',
            'anova_table': anova_table,
            'overall_means': overall_means,
            'interaction_table': interaction_pivot,
            'treatment_significant': treatment_p < alpha,
            'location_significant': location_p < alpha,
            'interaction_significant': interaction_p < alpha,
            'treatment_p': treatment_p,
            'location_p': location_p,
            'interaction_p': interaction_p,
            'model_summary': model.summary()
        }

        return results
    except Exception as e:
        st.error(f"Error performing multi-location analysis: {str(e)}")
        return None

# ==================== SPLIT PLOT DESIGN ====================

def split_plot_analysis(df, main_plot_col, sub_plot_col, block_col, response_col, alpha=0.05):
    """
    Analyze Split-Plot Design
    """
    try:
        # This is a simplified split-plot analysis
        # For complete analysis, consider using mixed models

        formula = f'{response_col} ~ C({main_plot_col}) * C({sub_plot_col}) + C({block_col})'

        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)

        # Extract p-values
        main_p = anova_table.loc[f'C({main_plot_col})', 'PR(>F)']
        sub_p = anova_table.loc[f'C({sub_plot_col})', 'PR(>F)']
        interaction_p = anova_table.loc[f'C({main_plot_col}):C({sub_plot_col})', 'PR(>F)']

        # Means
        main_means = df.groupby(main_plot_col)[response_col].mean()
        sub_means = df.groupby(sub_plot_col)[response_col].mean()
        interaction_means = df.groupby([main_plot_col, sub_plot_col])[response_col].mean().reset_index()

        results = {
            'test_type': 'Split-Plot Design Analysis',
            'anova_table': anova_table,
            'main_plot_means': main_means,
            'sub_plot_means': sub_means,
            'interaction_means': interaction_means,
            'main_plot_significant': main_p < alpha,
            'sub_plot_significant': sub_p < alpha,
            'interaction_significant': interaction_p < alpha,
            'model_summary': model.summary()
        }

        return results
    except Exception as e:
        st.error(f"Error performing split-plot analysis: {str(e)}")
        return None
