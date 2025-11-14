"""
Visualization Module
Creates various charts and plots for statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==================== BASIC PLOTS ====================

def create_histogram(data, column_name=None, bins=30, title=None, show_normal=True):
    """
    Create histogram with optional normal distribution overlay
    """
    if isinstance(data, pd.DataFrame) and column_name:
        series = data[column_name].dropna()
        title = title or f'Histogram of {column_name}'
    else:
        series = pd.Series(data).dropna()
        title = title or 'Histogram'

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n, bins_edges, patches = ax.hist(series, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Add normal distribution overlay if requested
    if show_normal:
        mu, sigma = series.mean(), series.std()
        x = np.linspace(series.min(), series.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
        ax.legend()

    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    plt.tight_layout()

    return fig

def create_boxplot(df, columns=None, title='Box Plot', orientation='vertical'):
    """
    Create box plot for one or multiple columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    if orientation == 'vertical':
        df[columns].boxplot(ax=ax)
        ax.set_ylabel('Value')
    else:
        df[columns].boxplot(ax=ax, vert=False)
        ax.set_xlabel('Value')

    ax.set_title(title)
    plt.tight_layout()

    return fig

def create_violin_plot(df, group_col, value_col, title='Violin Plot'):
    """
    Create violin plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x=group_col, y=value_col, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

def create_bar_chart(df, x_col, y_col, title='Bar Chart', error_col=None):
    """
    Create bar chart with optional error bars
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if error_col is not None:
        ax.bar(df[x_col], df[y_col], yerr=df[error_col], capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    else:
        ax.bar(df[x_col], df[y_col], alpha=0.7, color='steelblue', edgecolor='black')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

def create_line_plot(df, x_col, y_col, title='Line Plot', group_col=None):
    """
    Create line plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if group_col:
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group]
            ax.plot(group_data[x_col], group_data[y_col], marker='o', label=group)
        ax.legend()
    else:
        ax.plot(df[x_col], df[y_col], marker='o', color='steelblue')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    plt.tight_layout()

    return fig

def create_scatter_plot(x, y, title='Scatter Plot', xlabel='X', ylabel='Y', add_regression=False):
    """
    Create scatter plot with optional regression line
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove NaN
    df = pd.DataFrame({'x': x, 'y': y}).dropna()

    ax.scatter(df['x'], df['y'], alpha=0.6, color='steelblue', edgecolors='black')

    # Add regression line if requested
    if add_regression:
        z = np.polyfit(df['x'], df['y'], 1)
        p = np.poly1d(z)
        ax.plot(df['x'], p(df['x']), "r--", linewidth=2, label='Regression Line')
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    return fig

# ==================== ADVANCED PLOTS ====================

def create_correlation_heatmap(corr_matrix, title='Correlation Heatmap', annot=True):
    """
    Create correlation heatmap
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f', ax=ax)

    ax.set_title(title)
    plt.tight_layout()

    return fig

def create_qq_plot(data, title='Q-Q Plot'):
    """
    Create Q-Q plot to check normality
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    data = pd.Series(data).dropna()

    fig, ax = plt.subplots(figsize=(8, 8))

    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()

    return fig

def create_residual_plot(predictions, residuals, title='Residual Plot'):
    """
    Create residual plot for regression diagnostics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Residuals vs Fitted
    ax1.scatter(predictions, residuals, alpha=0.6, color='steelblue', edgecolors='black')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True)

    # Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot of Residuals')
    ax2.grid(True)

    plt.tight_layout()

    return fig

def create_pca_biplot(pca_results, pc_x=0, pc_y=1):
    """
    Create PCA biplot
    """
    try:
        # Extract data
        scores = pca_results['principal_components'].iloc[:, [pc_x, pc_y]].values
        loadings = pca_results['loadings'].iloc[:, [pc_x, pc_y]].values
        feature_names = pca_results['loadings'].index.tolist()

        # Variance explained
        var_exp = pca_results['variance_explained']['Variance Explained (%)'].values

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot scores
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.6, s=50, color='steelblue', edgecolors='black')

        # Plot loadings as arrows
        scale_factor = 0.8 * max(np.abs(scores).max(axis=0))
        for i, feature in enumerate(feature_names):
            ax.arrow(0, 0, loadings[i, 0] * scale_factor[0], loadings[i, 1] * scale_factor[1],
                    color='red', alpha=0.7, head_width=0.1, head_length=0.1)
            ax.text(loadings[i, 0] * scale_factor[0] * 1.15,
                   loadings[i, 1] * scale_factor[1] * 1.15,
                   feature, color='red', fontsize=10)

        ax.set_xlabel(f'PC{pc_x+1} ({var_exp[pc_x]:.1f}%)')
        ax.set_ylabel(f'PC{pc_y+1} ({var_exp[pc_y]:.1f}%)')
        ax.set_title('PCA Biplot')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig
    except Exception as e:
        st.error(f"Error creating PCA biplot: {str(e)}")
        return None

def create_scree_plot(pca_results):
    """
    Create scree plot for PCA
    """
    variance_df = pca_results['variance_explained']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(range(1, len(variance_df) + 1), variance_df['Variance Explained (%)'],
           marker='o', color='steelblue', linewidth=2, markersize=8)

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Scree Plot')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def create_cluster_plot(clustered_data, x_col, y_col, cluster_col='Cluster', title='Cluster Plot'):
    """
    Create 2D cluster visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique clusters
    clusters = clustered_data[cluster_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    for i, cluster in enumerate(sorted(clusters)):
        cluster_data = clustered_data[clustered_data[cluster_col] == cluster]
        ax.scatter(cluster_data[x_col], cluster_data[y_col],
                  label=f'Cluster {cluster}', alpha=0.6, s=50,
                  color=colors[i], edgecolors='black')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def create_dendrogram(linkage_matrix, title='Dendrogram'):
    """
    Create dendrogram for hierarchical clustering
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    scipy_dendrogram(linkage_matrix, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    plt.tight_layout()

    return fig

# ==================== INTERACTIVE PLOTS (PLOTLY) ====================

def create_interactive_scatter(x, y, title='Interactive Scatter Plot', xlabel='X', ylabel='Y', color=None):
    """
    Create interactive scatter plot using Plotly
    """
    df = pd.DataFrame({'x': x, 'y': y})
    if color is not None:
        df['color'] = color

    if color is not None:
        fig = px.scatter(df, x='x', y='y', color='color', title=title,
                        labels={'x': xlabel, 'y': ylabel})
    else:
        fig = px.scatter(df, x='x', y='y', title=title,
                        labels={'x': xlabel, 'y': ylabel})

    return fig

def create_interactive_boxplot(df, y_cols, title='Interactive Box Plot'):
    """
    Create interactive box plot using Plotly
    """
    fig = go.Figure()

    for col in y_cols:
        fig.add_trace(go.Box(y=df[col], name=col))

    fig.update_layout(title=title, yaxis_title='Value')

    return fig

def create_interactive_histogram(data, title='Interactive Histogram', nbins=30):
    """
    Create interactive histogram using Plotly
    """
    fig = px.histogram(pd.DataFrame({'value': data}), x='value', nbins=nbins, title=title)
    fig.update_layout(xaxis_title='Value', yaxis_title='Frequency')

    return fig

# ==================== EXPORT FUNCTIONS ====================

def save_figure(fig, filename, format='png', dpi=300):
    """
    Save figure to file
    """
    try:
        if format.lower() in ['png', 'jpg', 'jpeg', 'svg', 'pdf']:
            fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
            return True
        else:
            st.error(f"Unsupported format: {format}")
            return False
    except Exception as e:
        st.error(f"Error saving figure: {str(e)}")
        return False
