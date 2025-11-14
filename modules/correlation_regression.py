"""
Correlation and Regression Analysis Module
Includes correlation tests and various regression models
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st

# ==================== CORRELATION ====================

def pearson_correlation(x, y, alpha=0.05):
    """
    Calculate Pearson correlation coefficient
    """
    # Remove NaN values
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    x_clean = df['x']
    y_clean = df['y']

    if len(x_clean) < 3:
        st.error("Need at least 3 data points for correlation")
        return None

    # Calculate Pearson correlation
    r, p_value = stats.pearsonr(x_clean, y_clean)

    # Confidence interval for correlation
    fisher_z = np.arctanh(r)
    se = 1 / np.sqrt(len(x_clean) - 3)
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_lower = np.tanh(fisher_z - z_crit * se)
    ci_upper = np.tanh(fisher_z + z_crit * se)

    # Interpret strength
    if abs(r) < 0.3:
        strength = "weak"
    elif abs(r) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    results = {
        'test_type': 'Pearson Correlation',
        'correlation_coefficient': r,
        'r_squared': r**2,
        'p_value': p_value,
        'n': len(x_clean),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha,
        'strength': strength,
        'direction': 'positive' if r > 0 else 'negative'
    }

    return results

def spearman_correlation(x, y, alpha=0.05):
    """
    Calculate Spearman rank correlation coefficient
    """
    # Remove NaN values
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    x_clean = df['x']
    y_clean = df['y']

    if len(x_clean) < 3:
        st.error("Need at least 3 data points for correlation")
        return None

    # Calculate Spearman correlation
    rho, p_value = stats.spearmanr(x_clean, y_clean)

    # Interpret strength
    if abs(rho) < 0.3:
        strength = "weak"
    elif abs(rho) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    results = {
        'test_type': 'Spearman Rank Correlation',
        'correlation_coefficient': rho,
        'rho_squared': rho**2,
        'p_value': p_value,
        'n': len(x_clean),
        'significant': p_value < alpha,
        'strength': strength,
        'direction': 'positive' if rho > 0 else 'negative'
    }

    return results

def correlation_matrix(df, method='pearson'):
    """
    Calculate correlation matrix for multiple variables
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for correlation matrix")
        return None

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = df[numeric_cols].corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = df[numeric_cols].corr(method='spearman')
    else:
        st.error("Invalid correlation method")
        return None

    # Calculate p-values
    n = len(df)
    p_values = pd.DataFrame(np.zeros((len(numeric_cols), len(numeric_cols))),
                           index=numeric_cols, columns=numeric_cols)

    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i != j:
                if method == 'pearson':
                    _, p_val = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                else:
                    _, p_val = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                p_values.iloc[i, j] = p_val

    results = {
        'correlation_matrix': corr_matrix,
        'p_values': p_values,
        'method': method
    }

    return results

# ==================== REGRESSION ====================

def simple_linear_regression(x, y, alpha=0.05):
    """
    Perform simple linear regression
    """
    # Remove NaN values
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    X = df['x'].values.reshape(-1, 1)
    y_clean = df['y'].values

    if len(X) < 3:
        st.error("Need at least 3 data points for regression")
        return None

    # Fit model
    model = LinearRegression()
    model.fit(X, y_clean)

    # Predictions
    y_pred = model.predict(X)

    # Calculate statistics
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = r2_score(y_clean, y_pred)
    mse = mean_squared_error(y_clean, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_clean, y_pred)

    # Calculate correlation and p-value
    r, p_value = stats.pearsonr(df['x'], df['y'])

    # Standard error of slope
    n = len(X)
    dof = n - 2
    residuals = y_clean - y_pred
    s_residuals = np.sqrt(np.sum(residuals**2) / dof)
    x_mean = X.mean()
    se_slope = s_residuals / np.sqrt(np.sum((X - x_mean)**2))
    t_stat = slope / se_slope
    p_value_slope = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    # Confidence interval for slope
    t_crit = stats.t.ppf(1 - alpha/2, dof)
    ci_slope_lower = slope - t_crit * se_slope
    ci_slope_upper = slope + t_crit * se_slope

    results = {
        'test_type': 'Simple Linear Regression',
        'equation': f'y = {slope:.4f}x + {intercept:.4f}',
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'adjusted_r_squared': 1 - (1 - r_squared) * (n - 1) / (n - 2),
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'n': n,
        'p_value': p_value_slope,
        'se_slope': se_slope,
        't_statistic': t_stat,
        'ci_slope_lower': ci_slope_lower,
        'ci_slope_upper': ci_slope_upper,
        'significant': p_value_slope < alpha,
        'predictions': y_pred,
        'residuals': residuals
    }

    return results, model

def multiple_linear_regression(df, dependent_var, independent_vars, alpha=0.05):
    """
    Perform multiple linear regression
    """
    try:
        # Prepare data
        data = df[[dependent_var] + independent_vars].dropna()
        X = data[independent_vars].values
        y = data[dependent_var].values

        if len(X) < len(independent_vars) + 2:
            st.error(f"Need at least {len(independent_vars) + 2} data points for this regression")
            return None

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Predictions
        y_pred = model.predict(X)

        # Calculate statistics
        n = len(X)
        k = len(independent_vars)
        r_squared = r2_score(y, y_pred)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)

        # F-statistic
        f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
        p_value_f = 1 - stats.f.cdf(f_stat, k, n - k - 1)

        # Coefficient statistics
        residuals = y - y_pred
        s_residuals = np.sqrt(np.sum(residuals**2) / (n - k - 1))

        # Create coefficient table
        coefficients = pd.DataFrame({
            'Variable': ['Intercept'] + independent_vars,
            'Coefficient': [model.intercept_] + list(model.coef_),
        })

        results = {
            'test_type': 'Multiple Linear Regression',
            'coefficients': coefficients,
            'intercept': model.intercept_,
            'r_squared': r_squared,
            'adjusted_r_squared': adjusted_r_squared,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'n': n,
            'k': k,
            'f_statistic': f_stat,
            'p_value_f': p_value_f,
            'significant': p_value_f < alpha,
            'predictions': y_pred,
            'residuals': residuals
        }

        return results, model
    except Exception as e:
        st.error(f"Error performing multiple regression: {str(e)}")
        return None

def polynomial_regression(x, y, degree=2, alpha=0.05):
    """
    Perform polynomial regression
    """
    # Remove NaN values
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    X = df['x'].values.reshape(-1, 1)
    y_clean = df['y'].values

    if len(X) < degree + 2:
        st.error(f"Need at least {degree + 2} data points for degree {degree} polynomial")
        return None

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y_clean)

    # Predictions
    y_pred = model.predict(X_poly)

    # Calculate statistics
    r_squared = r2_score(y_clean, y_pred)
    n = len(X)
    k = degree
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    mse = mean_squared_error(y_clean, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_clean, y_pred)

    # Create equation string
    terms = []
    for i, coef in enumerate(model.coef_):
        if i == 0:
            terms.append(f'{model.intercept_:.4f}')
        elif i == 1:
            terms.append(f'{coef:.4f}x')
        else:
            terms.append(f'{coef:.4f}x^{i}')
    equation = ' + '.join(terms)

    # Residuals
    residuals = y_clean - y_pred

    results = {
        'test_type': f'Polynomial Regression (Degree {degree})',
        'equation': equation,
        'degree': degree,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'r_squared': r_squared,
        'adjusted_r_squared': adjusted_r_squared,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'n': n,
        'predictions': y_pred,
        'residuals': residuals,
        'X_poly': X_poly
    }

    return results, model, poly_features

def regression_diagnostics(residuals, predictions):
    """
    Perform regression diagnostics
    """
    residuals = pd.Series(residuals).dropna()

    # Normality test for residuals
    _, p_normal = stats.shapiro(residuals)

    # Homoscedasticity (constant variance)
    # Use Breusch-Pagan test approximation
    residuals_squared = residuals ** 2
    correlation = np.corrcoef(predictions, residuals_squared)[0, 1]

    diagnostics = {
        'residuals_mean': residuals.mean(),
        'residuals_std': residuals.std(),
        'residuals_normal': p_normal > 0.05,
        'shapiro_p_value': p_normal,
        'homoscedasticity_note': 'Check residual plot for constant variance'
    }

    return diagnostics

def covariance(x, y):
    """
    Calculate covariance between two variables
    """
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    x_clean = df['x']
    y_clean = df['y']

    cov = np.cov(x_clean, y_clean)[0, 1]

    results = {
        'covariance': cov,
        'x_mean': x_clean.mean(),
        'y_mean': y_clean.mean(),
        'n': len(x_clean)
    }

    return results
