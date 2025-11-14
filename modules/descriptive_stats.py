"""
Descriptive Statistics Module
Calculates various descriptive statistics
"""

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

def calculate_descriptive_stats(data, column_name=None):
    """
    Calculate comprehensive descriptive statistics
    """
    if isinstance(data, pd.Series) or isinstance(data, np.ndarray):
        series = pd.Series(data)
    elif isinstance(data, pd.DataFrame) and column_name:
        series = data[column_name]
    else:
        st.error("Invalid data format")
        return None

    # Remove NaN values
    series = series.dropna()

    if len(series) == 0:
        st.error("No valid data points")
        return None

    stats_dict = {
        'Count': len(series),
        'Mean': series.mean(),
        'Median': series.median(),
        'Mode': series.mode().values[0] if not series.mode().empty else np.nan,
        'Standard Deviation': series.std(),
        'Variance': series.var(),
        'Standard Error': series.sem(),
        'Coefficient of Variation (%)': (series.std() / series.mean() * 100) if series.mean() != 0 else np.nan,
        'Minimum': series.min(),
        'Maximum': series.max(),
        'Range': series.max() - series.min(),
        'Q1 (25th percentile)': series.quantile(0.25),
        'Q2 (50th percentile)': series.quantile(0.50),
        'Q3 (75th percentile)': series.quantile(0.75),
        'IQR': series.quantile(0.75) - series.quantile(0.25),
        'Skewness': series.skew(),
        'Kurtosis': series.kurtosis(),
        'Sum': series.sum()
    }

    return stats_dict

def grouped_descriptive_stats(df, group_col, value_col):
    """
    Calculate descriptive statistics for grouped data
    """
    try:
        grouped = df.groupby(group_col)[value_col].agg([
            ('Count', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std Dev', 'std'),
            ('Std Error', 'sem'),
            ('CV%', lambda x: (x.std() / x.mean() * 100) if x.mean() != 0 else np.nan),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Range', lambda x: x.max() - x.min())
        ]).round(4)

        return grouped
    except Exception as e:
        st.error(f"Error calculating grouped statistics: {str(e)}")
        return None

def summary_statistics_table(df, columns=None):
    """
    Create a summary statistics table for multiple columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    summary_data = []

    for col in columns:
        stats_dict = calculate_descriptive_stats(df, col)
        if stats_dict:
            stats_dict['Variable'] = col
            summary_data.append(stats_dict)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Reorder columns to have Variable first
        cols = ['Variable'] + [c for c in summary_df.columns if c != 'Variable']
        summary_df = summary_df[cols]
        return summary_df

    return None

def frequency_table(data, column_name=None):
    """
    Create a frequency table for categorical or discrete data
    """
    if isinstance(data, pd.DataFrame) and column_name:
        series = data[column_name]
    else:
        series = pd.Series(data)

    freq_table = pd.DataFrame({
        'Value': series.value_counts().index,
        'Frequency': series.value_counts().values,
        'Relative Frequency': series.value_counts(normalize=True).values,
        'Percentage': series.value_counts(normalize=True).values * 100,
        'Cumulative Frequency': series.value_counts().sort_index().cumsum().values
    })

    return freq_table

def confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the mean
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    data = pd.Series(data).dropna()

    n = len(data)
    mean = data.mean()
    se = data.sem()

    # Calculate confidence interval
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)

    return {
        'mean': mean,
        'confidence_level': confidence,
        'lower_bound': ci[0],
        'upper_bound': ci[1],
        'margin_of_error': ci[1] - mean
    }

def percentiles(data, percentile_list=None):
    """
    Calculate custom percentiles
    """
    if percentile_list is None:
        percentile_list = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    data = pd.Series(data).dropna()

    percentile_dict = {}
    for p in percentile_list:
        percentile_dict[f'{p}th percentile'] = data.quantile(p/100)

    return percentile_dict

def describe_distribution(data):
    """
    Provide description of data distribution
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    data = pd.Series(data).dropna()

    skewness = data.skew()
    kurtosis = data.kurtosis()

    distribution_info = {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'interpretation': ''
    }

    # Interpret skewness
    if abs(skewness) < 0.5:
        skew_interp = "approximately symmetric"
    elif skewness > 0.5:
        skew_interp = "positively skewed (right-tailed)"
    else:
        skew_interp = "negatively skewed (left-tailed)"

    # Interpret kurtosis
    if abs(kurtosis) < 0.5:
        kurt_interp = "mesokurtic (normal-like)"
    elif kurtosis > 0.5:
        kurt_interp = "leptokurtic (heavy-tailed)"
    else:
        kurt_interp = "platykurtic (light-tailed)"

    distribution_info['interpretation'] = f"The distribution is {skew_interp} and {kurt_interp}."

    return distribution_info
