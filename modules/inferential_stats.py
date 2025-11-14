"""
Inferential Statistics Module
Includes t-tests, ANOVA, chi-square, and non-parametric tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import streamlit as st

# ==================== T-TESTS ====================

def independent_t_test(group1, group2, equal_var=True, alpha=0.05):
    """
    Perform independent samples t-test
    """
    # Remove NaN values
    group1 = pd.Series(group1).dropna()
    group2 = pd.Series(group2).dropna()

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1)+len(group2)-2))
    cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std != 0 else 0

    # Calculate confidence interval
    se = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
    dof = len(group1) + len(group2) - 2
    ci = stats.t.interval(1-alpha, dof, loc=(group1.mean() - group2.mean()), scale=se)

    results = {
        'test_type': 'Independent Samples T-Test',
        'group1_mean': group1.mean(),
        'group1_std': group1.std(),
        'group1_n': len(group1),
        'group2_mean': group2.mean(),
        'group2_std': group2.std(),
        'group2_n': len(group2),
        'mean_difference': group1.mean() - group2.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'cohens_d': cohens_d,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'significant': p_value < alpha
    }

    return results

def paired_t_test(group1, group2, alpha=0.05):
    """
    Perform paired samples t-test
    """
    # Remove NaN values
    group1 = pd.Series(group1).dropna()
    group2 = pd.Series(group2).dropna()

    if len(group1) != len(group2):
        st.error("Paired t-test requires equal sample sizes")
        return None

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(group1, group2)

    # Calculate effect size
    differences = group1 - group2
    cohens_d = differences.mean() / differences.std() if differences.std() != 0 else 0

    # Confidence interval
    se = differences.sem()
    dof = len(differences) - 1
    ci = stats.t.interval(1-alpha, dof, loc=differences.mean(), scale=se)

    results = {
        'test_type': 'Paired Samples T-Test',
        'group1_mean': group1.mean(),
        'group2_mean': group2.mean(),
        'mean_difference': differences.mean(),
        'std_difference': differences.std(),
        'n_pairs': len(group1),
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'cohens_d': cohens_d,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'significant': p_value < alpha
    }

    return results

def one_sample_t_test(data, pop_mean, alpha=0.05):
    """
    Perform one-sample t-test
    """
    data = pd.Series(data).dropna()

    # Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(data, pop_mean)

    # Effect size
    cohens_d = (data.mean() - pop_mean) / data.std() if data.std() != 0 else 0

    # Confidence interval
    se = data.sem()
    dof = len(data) - 1
    ci = stats.t.interval(1-alpha, dof, loc=data.mean(), scale=se)

    results = {
        'test_type': 'One-Sample T-Test',
        'sample_mean': data.mean(),
        'population_mean': pop_mean,
        'sample_std': data.std(),
        'sample_n': len(data),
        'mean_difference': data.mean() - pop_mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'cohens_d': cohens_d,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'significant': p_value < alpha
    }

    return results

# ==================== ANOVA ====================

def one_way_anova(df, group_col, value_col, alpha=0.05):
    """
    Perform one-way ANOVA
    """
    try:
        groups = [group[value_col].values for name, group in df.groupby(group_col)]

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate effect size (eta-squared)
        grand_mean = df[value_col].mean()
        ss_between = sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups])
        ss_total = sum([(x - grand_mean)**2 for g in groups for x in g])
        eta_squared = ss_between / ss_total if ss_total != 0 else 0

        # Group statistics
        group_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std']).reset_index()

        results = {
            'test_type': 'One-Way ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'df_between': len(groups) - 1,
            'df_within': len(df) - len(groups),
            'significant': p_value < alpha,
            'group_stats': group_stats
        }

        return results
    except Exception as e:
        st.error(f"Error performing ANOVA: {str(e)}")
        return None

def two_way_anova(df, factor1_col, factor2_col, value_col, alpha=0.05):
    """
    Perform two-way ANOVA
    """
    try:
        # Create formula
        formula = f'{value_col} ~ C({factor1_col}) + C({factor2_col}) + C({factor1_col}):C({factor2_col})'

        # Fit model
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)

        results = {
            'test_type': 'Two-Way ANOVA',
            'anova_table': anova_table,
            'model_summary': model.summary(),
            'significant': any(anova_table['PR(>F)'].dropna() < alpha)
        }

        return results
    except Exception as e:
        st.error(f"Error performing two-way ANOVA: {str(e)}")
        return None

# ==================== POST-HOC TESTS ====================

def tukey_hsd_test(df, group_col, value_col, alpha=0.05):
    """
    Perform Tukey's HSD post-hoc test
    """
    try:
        tukey = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col], alpha=alpha)

        # Convert to dataframe
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

        results = {
            'test_type': "Tukey's HSD Post-Hoc Test",
            'results_table': tukey_df,
            'summary': str(tukey.summary())
        }

        return results
    except Exception as e:
        st.error(f"Error performing Tukey's HSD: {str(e)}")
        return None

def lsd_test(df, group_col, value_col, alpha=0.05):
    """
    Perform Fisher's LSD post-hoc test
    """
    try:
        groups = df[group_col].unique()
        n_groups = len(groups)

        # Get group data
        group_data = {group: df[df[group_col] == group][value_col].values for group in groups}

        # Calculate MSE from one-way ANOVA
        all_groups = [group_data[g] for g in groups]
        f_stat, p_value = stats.f_oneway(*all_groups)

        # Calculate MSE
        ss_within = sum([sum((g - np.mean(g))**2) for g in all_groups])
        df_within = len(df) - n_groups
        mse = ss_within / df_within

        # Perform pairwise comparisons
        comparisons = []
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                data1 = group_data[group1]
                data2 = group_data[group2]

                mean_diff = np.mean(data1) - np.mean(data2)
                se = np.sqrt(mse * (1/len(data1) + 1/len(data2)))
                t_crit = stats.t.ppf(1 - alpha/2, df_within)
                lsd = t_crit * se

                comparisons.append({
                    'Group 1': group1,
                    'Group 2': group2,
                    'Mean Difference': mean_diff,
                    'LSD': lsd,
                    'Significant': abs(mean_diff) > lsd
                })

        lsd_df = pd.DataFrame(comparisons)

        results = {
            'test_type': "Fisher's LSD Post-Hoc Test",
            'results_table': lsd_df,
            'mse': mse
        }

        return results
    except Exception as e:
        st.error(f"Error performing LSD test: {str(e)}")
        return None

# ==================== CHI-SQUARE TEST ====================

def chi_square_test(df, col1, col2, alpha=0.05):
    """
    Perform chi-square test of independence
    """
    try:
        # Create contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])

        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1]) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        results = {
            'test_type': 'Chi-Square Test of Independence',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'observed': contingency_table,
            'expected': pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns),
            'significant': p_value < alpha
        }

        return results
    except Exception as e:
        st.error(f"Error performing chi-square test: {str(e)}")
        return None

# ==================== NON-PARAMETRIC TESTS ====================

def mann_whitney_u_test(group1, group2, alpha=0.05):
    """
    Perform Mann-Whitney U test (non-parametric alternative to independent t-test)
    """
    group1 = pd.Series(group1).dropna()
    group2 = pd.Series(group2).dropna()

    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Calculate effect size (rank-biserial correlation)
    n1, n2 = len(group1), len(group2)
    r = 1 - (2*u_stat) / (n1 * n2)

    results = {
        'test_type': 'Mann-Whitney U Test',
        'group1_median': group1.median(),
        'group1_n': n1,
        'group2_median': group2.median(),
        'group2_n': n2,
        'u_statistic': u_stat,
        'p_value': p_value,
        'effect_size_r': r,
        'significant': p_value < alpha
    }

    return results

def kruskal_wallis_test(df, group_col, value_col, alpha=0.05):
    """
    Perform Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA)
    """
    try:
        groups = [group[value_col].values for name, group in df.groupby(group_col)]

        # Perform Kruskal-Wallis test
        h_stat, p_value = stats.kruskal(*groups)

        # Calculate effect size (eta-squared)
        n = len(df)
        k = len(groups)
        eta_squared = (h_stat - k + 1) / (n - k) if (n - k) != 0 else 0

        # Group statistics
        group_stats = df.groupby(group_col)[value_col].agg(['count', 'median', 'mean']).reset_index()

        results = {
            'test_type': 'Kruskal-Wallis H Test',
            'h_statistic': h_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'degrees_of_freedom': k - 1,
            'significant': p_value < alpha,
            'group_stats': group_stats
        }

        return results
    except Exception as e:
        st.error(f"Error performing Kruskal-Wallis test: {str(e)}")
        return None

def wilcoxon_signed_rank_test(group1, group2, alpha=0.05):
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
    """
    group1 = pd.Series(group1).dropna()
    group2 = pd.Series(group2).dropna()

    if len(group1) != len(group2):
        st.error("Wilcoxon signed-rank test requires equal sample sizes")
        return None

    # Perform Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(group1, group2)

    # Calculate effect size
    n = len(group1)
    r = w_stat / (n * (n + 1) / 2)

    results = {
        'test_type': 'Wilcoxon Signed-Rank Test',
        'group1_median': group1.median(),
        'group2_median': group2.median(),
        'n_pairs': n,
        'w_statistic': w_stat,
        'p_value': p_value,
        'effect_size_r': r,
        'significant': p_value < alpha
    }

    return results

# ==================== NORMALITY TESTS ====================

def shapiro_wilk_test(data, alpha=0.05):
    """
    Perform Shapiro-Wilk test for normality
    """
    data = pd.Series(data).dropna()

    if len(data) < 3:
        st.error("Shapiro-Wilk test requires at least 3 observations")
        return None

    # Perform Shapiro-Wilk test
    w_stat, p_value = stats.shapiro(data)

    results = {
        'test_type': 'Shapiro-Wilk Normality Test',
        'w_statistic': w_stat,
        'p_value': p_value,
        'sample_size': len(data),
        'normal': p_value > alpha,
        'interpretation': 'Data appears normally distributed' if p_value > alpha else 'Data does not appear normally distributed'
    }

    return results

def kolmogorov_smirnov_test(data, alpha=0.05):
    """
    Perform Kolmogorov-Smirnov test for normality
    """
    data = pd.Series(data).dropna()

    # Standardize data
    standardized = (data - data.mean()) / data.std()

    # Perform KS test against normal distribution
    ks_stat, p_value = stats.kstest(standardized, 'norm')

    results = {
        'test_type': 'Kolmogorov-Smirnov Normality Test',
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'sample_size': len(data),
        'normal': p_value > alpha,
        'interpretation': 'Data appears normally distributed' if p_value > alpha else 'Data does not appear normally distributed'
    }

    return results
