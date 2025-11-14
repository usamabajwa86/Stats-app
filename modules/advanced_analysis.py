"""
Advanced Analysis Module
Includes PCA, Cluster Analysis, and other advanced methods
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import streamlit as st

# ==================== PRINCIPAL COMPONENT ANALYSIS ====================

def perform_pca(df, n_components=None, standardize=True):
    """
    Perform Principal Component Analysis
    """
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for PCA")
            return None

        # Prepare data
        X = df[numeric_cols].dropna()

        if len(X) < 2:
            st.error("Need at least 2 observations for PCA")
            return None

        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Determine number of components
        if n_components is None:
            n_components = min(len(numeric_cols), len(X))

        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)

        # Create PC labels
        pc_labels = [f'PC{i+1}' for i in range(n_components)]

        # PC scores
        pc_df = pd.DataFrame(
            data=principal_components,
            columns=pc_labels
        )

        # Loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=pc_labels,
            index=numeric_cols
        )

        # Variance explained
        variance_explained = pd.DataFrame({
            'Principal Component': pc_labels,
            'Eigenvalue': pca.explained_variance_,
            'Variance Explained (%)': pca.explained_variance_ratio_ * 100,
            'Cumulative Variance (%)': np.cumsum(pca.explained_variance_ratio_) * 100
        })

        results = {
            'test_type': 'Principal Component Analysis',
            'n_components': n_components,
            'principal_components': pc_df,
            'loadings': loadings,
            'variance_explained': variance_explained,
            'total_variance_explained': np.sum(pca.explained_variance_ratio_) * 100,
            'pca_model': pca,
            'scaler': scaler if standardize else None
        }

        return results
    except Exception as e:
        st.error(f"Error performing PCA: {str(e)}")
        return None

# ==================== CLUSTER ANALYSIS ====================

def kmeans_clustering(df, n_clusters=3, standardize=True, random_state=42):
    """
    Perform K-Means Clustering
    """
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for clustering")
            return None

        # Prepare data
        X = df[numeric_cols].dropna()

        if len(X) < n_clusters:
            st.error(f"Need at least {n_clusters} observations for {n_clusters} clusters")
            return None

        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Perform K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Add cluster labels to dataframe
        clustered_df = X.copy()
        clustered_df['Cluster'] = cluster_labels

        # Cluster centers (in original scale if standardized)
        if standardize:
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            centers = kmeans.cluster_centers_

        centers_df = pd.DataFrame(
            centers,
            columns=numeric_cols,
            index=[f'Cluster {i}' for i in range(n_clusters)]
        )

        # Cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        cluster_sizes.index = [f'Cluster {i}' for i in cluster_sizes.index]

        # Inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_

        results = {
            'test_type': 'K-Means Clustering',
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'clustered_data': clustered_df,
            'cluster_centers': centers_df,
            'cluster_sizes': cluster_sizes,
            'inertia': inertia,
            'model': kmeans,
            'scaler': scaler if standardize else None
        }

        return results
    except Exception as e:
        st.error(f"Error performing K-Means clustering: {str(e)}")
        return None

def hierarchical_clustering(df, n_clusters=3, linkage_method='ward', standardize=True):
    """
    Perform Hierarchical Clustering
    """
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for clustering")
            return None

        # Prepare data
        X = df[numeric_cols].dropna()

        if len(X) < n_clusters:
            st.error(f"Need at least {n_clusters} observations for {n_clusters} clusters")
            return None

        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Perform Hierarchical Clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        cluster_labels = hierarchical.fit_predict(X_scaled)

        # Add cluster labels to dataframe
        clustered_df = X.copy()
        clustered_df['Cluster'] = cluster_labels

        # Create linkage matrix for dendrogram
        linkage_matrix = linkage(X_scaled, method=linkage_method)

        # Cluster sizes
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        cluster_sizes.index = [f'Cluster {i}' for i in cluster_sizes.index]

        # Calculate cluster centers
        centers = []
        for i in range(n_clusters):
            cluster_data = X_scaled[cluster_labels == i]
            centers.append(cluster_data.mean(axis=0))

        if standardize:
            centers = scaler.inverse_transform(centers)

        centers_df = pd.DataFrame(
            centers,
            columns=numeric_cols,
            index=[f'Cluster {i}' for i in range(n_clusters)]
        )

        results = {
            'test_type': 'Hierarchical Clustering',
            'n_clusters': n_clusters,
            'linkage_method': linkage_method,
            'cluster_labels': cluster_labels,
            'clustered_data': clustered_df,
            'cluster_centers': centers_df,
            'cluster_sizes': cluster_sizes,
            'linkage_matrix': linkage_matrix,
            'model': hierarchical,
            'scaler': scaler if standardize else None
        }

        return results
    except Exception as e:
        st.error(f"Error performing hierarchical clustering: {str(e)}")
        return None

def elbow_method(df, max_clusters=10, standardize=True):
    """
    Perform elbow method to determine optimal number of clusters
    """
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].dropna()

        # Standardize if requested
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Calculate inertia for different numbers of clusters
        inertias = []
        K_range = range(1, min(max_clusters + 1, len(X)))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        elbow_df = pd.DataFrame({
            'Number of Clusters': list(K_range),
            'Inertia': inertias
        })

        results = {
            'test_type': 'Elbow Method',
            'elbow_data': elbow_df
        }

        return results
    except Exception as e:
        st.error(f"Error performing elbow method: {str(e)}")
        return None

# ==================== DISTANCE MATRIX ====================

def distance_matrix(df, metric='euclidean'):
    """
    Calculate distance matrix between observations
    """
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].dropna()

        # Calculate pairwise distances
        distances = pdist(X, metric=metric)
        dist_matrix = squareform(distances)

        # Create dataframe
        dist_df = pd.DataFrame(
            dist_matrix,
            index=range(len(X)),
            columns=range(len(X))
        )

        results = {
            'distance_matrix': dist_df,
            'metric': metric
        }

        return results
    except Exception as e:
        st.error(f"Error calculating distance matrix: {str(e)}")
        return None

# ==================== FACTOR ANALYSIS ====================

def factor_analysis_summary(df, n_factors=None):
    """
    Provide summary for factor analysis (simplified version using PCA)
    """
    # Use PCA as a proxy for factor analysis
    pca_results = perform_pca(df, n_components=n_factors, standardize=True)

    if pca_results:
        pca_results['test_type'] = 'Factor Analysis (via PCA)'

    return pca_results
