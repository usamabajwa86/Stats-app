"""
Interpretation Module
Provides plain language interpretations of statistical results
"""

import numpy as np

# ==================== T-TEST INTERPRETATIONS ====================

def interpret_t_test(results):
    """
    Interpret t-test results in plain language
    """
    test_type = results.get('test_type', 'T-Test')
    p_value = results.get('p_value', 0)
    significant = results.get('significant', False)
    cohens_d = results.get('cohens_d', 0)

    interpretation = f"**{test_type} Results:**\n\n"

    # Statistical significance
    if significant:
        interpretation += f"✓ The test is **statistically significant** (p = {p_value:.4f}, p < 0.05).\n\n"
    else:
        interpretation += f"✗ The test is **not statistically significant** (p = {p_value:.4f}, p ≥ 0.05).\n\n"

    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect = "negligible"
    elif abs_d < 0.5:
        effect = "small"
    elif abs_d < 0.8:
        effect = "medium"
    else:
        effect = "large"

    interpretation += f"**Effect Size:** Cohen's d = {cohens_d:.3f} ({effect} effect)\n\n"

    # Practical interpretation
    if 'mean_difference' in results:
        mean_diff = results['mean_difference']
        interpretation += f"**Mean Difference:** {mean_diff:.4f}\n\n"

        if significant:
            interpretation += "**Conclusion:** There is a statistically significant difference between the groups. "
            interpretation += f"The effect size is {effect}, indicating a {effect} practical difference.\n\n"
        else:
            interpretation += "**Conclusion:** There is no statistically significant difference between the groups. "
            interpretation += "Any observed difference could be due to random chance.\n\n"

    # Confidence interval
    if 'ci_lower' in results and 'ci_upper' in results:
        interpretation += f"**95% Confidence Interval:** [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]\n\n"

    # Recommendations
    interpretation += "**Recommendations:**\n"
    if not significant:
        interpretation += "- Consider increasing sample size for more statistical power\n"
        interpretation += "- Check if assumptions are met (normality, equal variances)\n"
        interpretation += "- Consider using non-parametric alternatives if assumptions are violated\n"

    return interpretation

# ==================== ANOVA INTERPRETATIONS ====================

def interpret_anova(results):
    """
    Interpret ANOVA results in plain language
    """
    test_type = results.get('test_type', 'ANOVA')
    p_value = results.get('p_value', 0)
    significant = results.get('significant', False)
    eta_squared = results.get('eta_squared', 0)

    interpretation = f"**{test_type} Results:**\n\n"

    # Statistical significance
    if significant:
        interpretation += f"✓ The ANOVA is **statistically significant** (p = {p_value:.4f}, p < 0.05).\n\n"
        interpretation += "This indicates that **at least one group mean is significantly different** from the others.\n\n"
    else:
        interpretation += f"✗ The ANOVA is **not statistically significant** (p = {p_value:.4f}, p ≥ 0.05).\n\n"
        interpretation += "This suggests that **there are no significant differences** among the group means.\n\n"

    # Effect size interpretation
    if eta_squared < 0.01:
        effect = "negligible"
    elif eta_squared < 0.06:
        effect = "small"
    elif eta_squared < 0.14:
        effect = "medium"
    else:
        effect = "large"

    interpretation += f"**Effect Size:** η² = {eta_squared:.3f} ({effect} effect)\n"
    interpretation += f"This means that approximately {eta_squared*100:.1f}% of the variance in the outcome is explained by group membership.\n\n"

    # Recommendations
    interpretation += "**Recommendations:**\n"
    if significant:
        interpretation += "- Conduct post-hoc tests (e.g., Tukey's HSD) to identify which specific groups differ\n"
        interpretation += "- Create visualization (box plots, bar charts) to illustrate group differences\n"
    else:
        interpretation += "- Consider the practical significance beyond statistical significance\n"
        interpretation += "- Evaluate whether sample size was adequate for detecting meaningful differences\n"

    return interpretation

# ==================== CORRELATION INTERPRETATIONS ====================

def interpret_correlation(results):
    """
    Interpret correlation results in plain language
    """
    test_type = results.get('test_type', 'Correlation')
    r = results.get('correlation_coefficient', 0)
    p_value = results.get('p_value', 0)
    significant = results.get('significant', False)
    strength = results.get('strength', '')
    direction = results.get('direction', '')

    interpretation = f"**{test_type} Results:**\n\n"

    # Correlation coefficient
    interpretation += f"**Correlation Coefficient:** r = {r:.3f}\n\n"

    # Statistical significance
    if significant:
        interpretation += f"✓ The correlation is **statistically significant** (p = {p_value:.4f}, p < 0.05).\n\n"
    else:
        interpretation += f"✗ The correlation is **not statistically significant** (p = {p_value:.4f}, p ≥ 0.05).\n\n"

    # Interpretation
    interpretation += f"**Interpretation:** There is a **{strength} {direction} relationship** between the variables.\n\n"

    if abs(r) > 0.7:
        interpretation += "The strong correlation suggests that the variables move together consistently.\n"
    elif abs(r) > 0.3:
        interpretation += "The moderate correlation suggests some relationship, but other factors are also at play.\n"
    else:
        interpretation += "The weak correlation suggests little linear relationship between the variables.\n"

    interpretation += "\n"

    # R-squared
    if 'r_squared' in results:
        r_sq = results['r_squared']
        interpretation += f"**Coefficient of Determination (R²):** {r_sq:.3f}\n"
        interpretation += f"Approximately {r_sq*100:.1f}% of the variance in one variable can be explained by the other.\n\n"

    # Important notes
    interpretation += "**Important Notes:**\n"
    interpretation += "- Correlation does not imply causation\n"
    interpretation += "- The relationship may not be linear\n"
    interpretation += "- Outliers can strongly influence correlation\n"

    return interpretation

# ==================== REGRESSION INTERPRETATIONS ====================

def interpret_regression(results):
    """
    Interpret regression results in plain language
    """
    test_type = results.get('test_type', 'Regression')
    r_squared = results.get('r_squared', 0)
    p_value = results.get('p_value', 0)
    significant = results.get('significant', False)

    interpretation = f"**{test_type} Results:**\n\n"

    # Equation
    if 'equation' in results:
        interpretation += f"**Regression Equation:** {results['equation']}\n\n"

    # Model fit
    interpretation += f"**R² (Coefficient of Determination):** {r_squared:.3f}\n"
    interpretation += f"The model explains {r_squared*100:.1f}% of the variance in the dependent variable.\n\n"

    if r_squared > 0.7:
        interpretation += "This indicates a **strong model fit** with good predictive power.\n\n"
    elif r_squared > 0.4:
        interpretation += "This indicates a **moderate model fit** with reasonable predictive power.\n\n"
    else:
        interpretation += "This indicates a **weak model fit** with limited predictive power.\n\n"

    # Statistical significance
    if significant:
        interpretation += f"✓ The regression model is **statistically significant** (p = {p_value:.4f}, p < 0.05).\n\n"
    else:
        interpretation += f"✗ The regression model is **not statistically significant** (p = {p_value:.4f}, p ≥ 0.05).\n\n"

    # Slope interpretation (for simple regression)
    if 'slope' in results:
        slope = results['slope']
        interpretation += f"**Slope Interpretation:** For every one-unit increase in X, Y changes by {slope:.4f} units on average.\n\n"

    # Model quality metrics
    if 'rmse' in results:
        interpretation += f"**RMSE (Root Mean Square Error):** {results['rmse']:.4f}\n"
        interpretation += "Lower RMSE indicates better model predictions.\n\n"

    # Recommendations
    interpretation += "**Recommendations:**\n"
    interpretation += "- Check residual plots to verify model assumptions\n"
    interpretation += "- Examine Q-Q plot to check normality of residuals\n"
    interpretation += "- Look for patterns in residuals that might suggest model improvements\n"

    return interpretation

# ==================== CHI-SQUARE INTERPRETATIONS ====================

def interpret_chi_square(results):
    """
    Interpret chi-square test results in plain language
    """
    chi2 = results.get('chi2_statistic', 0)
    p_value = results.get('p_value', 0)
    significant = results.get('significant', False)
    cramers_v = results.get('cramers_v', 0)

    interpretation = "**Chi-Square Test of Independence Results:**\n\n"

    # Statistical significance
    if significant:
        interpretation += f"✓ The test is **statistically significant** (χ² = {chi2:.3f}, p = {p_value:.4f}, p < 0.05).\n\n"
        interpretation += "This indicates that **there is a significant association** between the two categorical variables.\n\n"
    else:
        interpretation += f"✗ The test is **not statistically significant** (χ² = {chi2:.3f}, p = {p_value:.4f}, p ≥ 0.05).\n\n"
        interpretation += "This suggests that **there is no significant association** between the two categorical variables.\n\n"

    # Effect size
    if cramers_v < 0.1:
        effect = "negligible"
    elif cramers_v < 0.3:
        effect = "small"
    elif cramers_v < 0.5:
        effect = "medium"
    else:
        effect = "large"

    interpretation += f"**Effect Size (Cramér's V):** {cramers_v:.3f} ({effect} effect)\n\n"

    # Interpretation
    if significant:
        interpretation += "**Conclusion:** The variables are not independent. Knowledge of one variable provides information about the other.\n\n"
    else:
        interpretation += "**Conclusion:** The variables appear to be independent. They do not significantly influence each other.\n\n"

    return interpretation

# ==================== NON-PARAMETRIC TEST INTERPRETATIONS ====================

def interpret_nonparametric(results):
    """
    Interpret non-parametric test results in plain language
    """
    test_type = results.get('test_type', 'Non-Parametric Test')
    p_value = results.get('p_value', 0)
    significant = results.get('significant', False)

    interpretation = f"**{test_type} Results:**\n\n"

    # Statistical significance
    if significant:
        interpretation += f"✓ The test is **statistically significant** (p = {p_value:.4f}, p < 0.05).\n\n"
        interpretation += "This indicates a **significant difference** between the groups or conditions.\n\n"
    else:
        interpretation += f"✗ The test is **not statistically significant** (p = {p_value:.4f}, p ≥ 0.05).\n\n"
        interpretation += "This suggests **no significant difference** between the groups or conditions.\n\n"

    # Test-specific information
    if 'Mann-Whitney' in test_type:
        interpretation += "**About this test:** Mann-Whitney U test is used when comparing two independent groups "
        interpretation += "without assuming normal distribution.\n\n"
    elif 'Kruskal-Wallis' in test_type:
        interpretation += "**About this test:** Kruskal-Wallis H test is used when comparing three or more independent groups "
        interpretation += "without assuming normal distribution.\n\n"
        if significant:
            interpretation += "**Recommendation:** Conduct post-hoc pairwise comparisons to identify which groups differ.\n\n"
    elif 'Wilcoxon' in test_type:
        interpretation += "**About this test:** Wilcoxon signed-rank test is used for paired samples "
        interpretation += "without assuming normal distribution.\n\n"

    # Effect size if available
    if 'effect_size_r' in results:
        r = abs(results['effect_size_r'])
        if r < 0.1:
            effect = "negligible"
        elif r < 0.3:
            effect = "small"
        elif r < 0.5:
            effect = "medium"
        else:
            effect = "large"
        interpretation += f"**Effect Size:** r = {results['effect_size_r']:.3f} ({effect} effect)\n\n"

    return interpretation

# ==================== PCA INTERPRETATIONS ====================

def interpret_pca(results):
    """
    Interpret PCA results in plain language
    """
    n_components = results.get('n_components', 0)
    total_var = results.get('total_variance_explained', 0)

    interpretation = "**Principal Component Analysis Results:**\n\n"

    interpretation += f"**Number of Components:** {n_components}\n"
    interpretation += f"**Total Variance Explained:** {total_var:.2f}%\n\n"

    # Interpretation
    if total_var > 80:
        interpretation += f"The first {n_components} principal components capture a **large proportion** of the total variance, "
        interpretation += "indicating effective dimensionality reduction.\n\n"
    elif total_var > 60:
        interpretation += f"The first {n_components} principal components capture a **moderate proportion** of the total variance. "
        interpretation += "Consider using more components for better representation.\n\n"
    else:
        interpretation += f"The first {n_components} principal components capture a **small proportion** of the total variance. "
        interpretation += "More components may be needed to adequately represent the data.\n\n"

    interpretation += "**Interpretation Guide:**\n"
    interpretation += "- Principal components are linear combinations of original variables\n"
    interpretation += "- Each component explains a portion of the total variance\n"
    interpretation += "- Loadings show how original variables contribute to each component\n"
    interpretation += "- Use scree plot to determine optimal number of components\n\n"

    interpretation += "**Applications:**\n"
    interpretation += "- Data visualization in reduced dimensions\n"
    interpretation += "- Feature extraction for machine learning\n"
    interpretation += "- Identifying patterns and relationships in multivariate data\n"

    return interpretation

# ==================== CLUSTER ANALYSIS INTERPRETATIONS ====================

def interpret_clustering(results):
    """
    Interpret clustering results in plain language
    """
    test_type = results.get('test_type', 'Cluster Analysis')
    n_clusters = results.get('n_clusters', 0)

    interpretation = f"**{test_type} Results:**\n\n"

    interpretation += f"**Number of Clusters:** {n_clusters}\n\n"

    # Cluster sizes
    if 'cluster_sizes' in results:
        interpretation += "**Cluster Sizes:**\n"
        for cluster, size in results['cluster_sizes'].items():
            interpretation += f"- {cluster}: {size} observations\n"
        interpretation += "\n"

    # Interpretation
    interpretation += "**Interpretation:**\n"
    interpretation += f"The data has been partitioned into {n_clusters} distinct groups based on similarity.\n\n"

    interpretation += "**How to Use Results:**\n"
    interpretation += "1. Examine cluster centers to understand what characterizes each group\n"
    interpretation += "2. Compare cluster sizes - very small clusters may be outliers\n"
    interpretation += "3. Visualize clusters in 2D/3D to assess separation\n"
    interpretation += "4. Consider the practical meaning of each cluster in your domain\n\n"

    interpretation += "**Validation:**\n"
    interpretation += "- Use elbow method or silhouette analysis to validate cluster count\n"
    interpretation += "- Check if clusters are well-separated and compact\n"
    interpretation += "- Verify that clustering makes sense in the context of your research\n"

    return interpretation

# ==================== NORMALITY TEST INTERPRETATIONS ====================

def interpret_normality_test(results):
    """
    Interpret normality test results in plain language
    """
    test_type = results.get('test_type', 'Normality Test')
    p_value = results.get('p_value', 0)
    normal = results.get('normal', False)

    interpretation = f"**{test_type} Results:**\n\n"

    # Result
    if normal:
        interpretation += f"✓ The data **appears normally distributed** (p = {p_value:.4f}, p > 0.05).\n\n"
        interpretation += "**Implication:** Parametric tests (t-test, ANOVA, regression) are appropriate.\n\n"
    else:
        interpretation += f"✗ The data **does not appear normally distributed** (p = {p_value:.4f}, p ≤ 0.05).\n\n"
        interpretation += "**Implication:** Consider using non-parametric tests or data transformation.\n\n"

    # Recommendations
    interpretation += "**Recommendations:**\n"
    if not normal:
        interpretation += "- Consider data transformation (log, square root, etc.)\n"
        interpretation += "- Use non-parametric alternatives (Mann-Whitney, Kruskal-Wallis)\n"
        interpretation += "- Check for outliers that might affect normality\n"
        interpretation += "- For large samples (n > 30), parametric tests are often robust to non-normality\n"
    else:
        interpretation += "- Proceed with parametric statistical tests\n"
        interpretation += "- Visual inspection (histogram, Q-Q plot) is also recommended\n"

    return interpretation

# ==================== GENERAL REPORT GENERATION ====================

def generate_apa_format(results):
    """
    Generate APA-style results statement
    """
    test_type = results.get('test_type', '')

    if 'T-Test' in test_type:
        t = results.get('t_statistic', 0)
        df = results.get('degrees_of_freedom', 0)
        p = results.get('p_value', 0)
        d = results.get('cohens_d', 0)

        apa = f"t({df}) = {t:.2f}, p = {p:.3f}, d = {d:.2f}"

    elif 'ANOVA' in test_type:
        f = results.get('f_statistic', 0)
        df1 = results.get('df_between', 0)
        df2 = results.get('df_within', 0)
        p = results.get('p_value', 0)
        eta = results.get('eta_squared', 0)

        apa = f"F({df1}, {df2}) = {f:.2f}, p = {p:.3f}, η² = {eta:.3f}"

    elif 'Correlation' in test_type:
        r = results.get('correlation_coefficient', 0)
        p = results.get('p_value', 0)
        n = results.get('n', 0)

        apa = f"r({n-2}) = {r:.2f}, p = {p:.3f}"

    elif 'Chi-Square' in test_type:
        chi2 = results.get('chi2_statistic', 0)
        df = results.get('degrees_of_freedom', 0)
        p = results.get('p_value', 0)

        apa = f"χ²({df}) = {chi2:.2f}, p = {p:.3f}"

    else:
        apa = "APA format not available for this test"

    return apa
