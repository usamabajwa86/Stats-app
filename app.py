"""
Statistical Analysis Web Application
A comprehensive platform for research data analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import warnings

warnings.filterwarnings('ignore')

# Import custom modules
from modules import data_handler
from modules import descriptive_stats
from modules import inferential_stats
from modules import correlation_regression
from modules import advanced_analysis
from modules import visualizations
from modules import interpretations
from modules import agriculture_analysis

# Page configuration
st.set_page_config(
    page_title="Statistical Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        padding: 0.5rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Main header
st.markdown('<div class="main-header">📊 Statistical Analysis Web Application</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray;">A comprehensive platform for research data analysis</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")

# Main workflow steps
workflow_step = st.sidebar.radio(
    "Select Workflow Step:",
    ["🏠 Home", "📁 Data Upload", "📊 Analysis", "📈 Visualization", "📄 Report"],
    index=0
)

st.sidebar.markdown("---")

# ==================== HOME PAGE ====================
if workflow_step == "🏠 Home":
    st.header("Welcome to the Statistical Analysis Platform!")

    st.markdown("""
    ### 🎯 Purpose
    This application is designed specifically for research students (particularly in agriculture and life sciences)
    to analyze their thesis/research data without needing multiple tools or extensive statistical knowledge.

    ### ✨ Key Features
    - **Easy Data Upload**: Support for CSV, Excel, and TXT formats
    - **Comprehensive Analysis**: Descriptive, inferential, and advanced statistics
    - **Specialized Agricultural Tools**: RCBD, Latin Square, yield analysis
    - **Auto-Interpretation**: Plain language explanations of results
    - **Professional Visualizations**: Publication-ready charts
    - **Export Capabilities**: Download results and reports

    ### 📚 Available Statistical Tests

    #### Descriptive Statistics
    - Mean, Median, Mode, Standard Deviation
    - Variance, Standard Error, CV
    - Range, IQR, Skewness, Kurtosis

    #### Inferential Statistics
    - T-tests (Independent, Paired, One-sample)
    - ANOVA (One-way, Two-way)
    - Post-hoc tests (Tukey HSD, LSD)
    - Chi-square test
    - Mann-Whitney U, Kruskal-Wallis

    #### Correlation & Regression
    - Pearson & Spearman correlation
    - Simple & Multiple Linear Regression
    - Polynomial Regression

    #### Advanced Analysis
    - Principal Component Analysis (PCA)
    - Cluster Analysis (K-Means, Hierarchical)
    - Normality Tests

    #### Agriculture-Specific
    - RCBD Analysis
    - Latin Square Design
    - Yield Comparison
    - Treatment Effect Analysis
    - Multi-location Trials

    ### 🚀 Getting Started
    1. Click **"📁 Data Upload"** in the sidebar
    2. Upload your data file or use example datasets
    3. Preview and clean your data
    4. Go to **"📊 Analysis"** to select statistical tests
    5. View results and interpretations
    6. Generate visualizations and reports

    ### 💡 Tips
    - Start with descriptive statistics to understand your data
    - Check normality before using parametric tests
    - Use post-hoc tests after significant ANOVA results
    - Review interpretations for guidance on reporting results
    """)

    st.info("👈 Use the sidebar to navigate through the workflow steps!")

# ==================== DATA UPLOAD PAGE ====================
elif workflow_step == "📁 Data Upload":
    st.header("📁 Data Upload & Preparation")

    # Data source selection
    data_source = st.radio("Select Data Source:", ["Upload File", "Use Example Dataset", "Manual Entry"])

    if data_source == "Upload File":
        st.subheader("Upload Your Data")

        uploaded_file = st.file_uploader(
            "Choose a file (CSV, Excel, or TXT)",
            type=['csv', 'xlsx', 'xls', 'txt'],
            help="Upload your dataset in CSV, Excel, or TXT format"
        )

        if uploaded_file is not None:
            df = data_handler.load_data(uploaded_file)

            if df is not None:
                st.session_state.df = df
                st.success("✅ Data loaded successfully!")

                # Display basic info
                st.subheader("Dataset Preview")
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.dataframe(df.head(10))

                # Data validation
                if data_handler.validate_data(df):
                    # Data cleaning options
                    if st.checkbox("Show Data Cleaning Options"):
                        df_cleaned = data_handler.clean_data(df)
                        st.session_state.df = df_cleaned

                    # Data transformation options
                    if st.checkbox("Show Data Transformation Options"):
                        df_transformed = data_handler.transform_data(st.session_state.df)
                        st.session_state.df = df_transformed

                    # Download cleaned data
                    if st.button("Export Cleaned Data"):
                        csv = data_handler.export_data(st.session_state.df)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="cleaned_data.csv",
                            mime="text/csv"
                        )

    elif data_source == "Use Example Dataset":
        st.subheader("Example Datasets")

        example_choice = st.selectbox(
            "Select Example Dataset:",
            ["Field Trial Data", "Treatment Comparison", "Yield Data"]
        )

        if st.button("Load Example Dataset"):
            # Create example datasets
            if example_choice == "Field Trial Data":
                np.random.seed(42)
                df = pd.DataFrame({
                    'Treatment': np.repeat(['Control', 'Treatment1', 'Treatment2', 'Treatment3'], 12),
                    'Block': np.tile(np.repeat(['Block1', 'Block2', 'Block3', 'Block4'], 3), 4),
                    'Yield': np.concatenate([
                        np.random.normal(45, 5, 12),  # Control
                        np.random.normal(52, 5, 12),  # Treatment1
                        np.random.normal(48, 5, 12),  # Treatment2
                        np.random.normal(55, 5, 12)   # Treatment3
                    ]),
                    'Plant_Height': np.concatenate([
                        np.random.normal(85, 8, 12),
                        np.random.normal(95, 8, 12),
                        np.random.normal(88, 8, 12),
                        np.random.normal(98, 8, 12)
                    ])
                })
            elif example_choice == "Treatment Comparison":
                np.random.seed(42)
                df = pd.DataFrame({
                    'Group': np.repeat(['Control', 'Treatment_A', 'Treatment_B'], 30),
                    'Value': np.concatenate([
                        np.random.normal(100, 15, 30),
                        np.random.normal(115, 15, 30),
                        np.random.normal(108, 15, 30)
                    ])
                })
            else:  # Yield Data
                np.random.seed(42)
                df = pd.DataFrame({
                    'Variety': np.repeat(['Var1', 'Var2', 'Var3', 'Var4', 'Var5'], 20),
                    'Location': np.tile(np.repeat(['Loc1', 'Loc2', 'Loc3', 'Loc4'], 5), 5),
                    'Yield_kg_ha': np.random.normal(3500, 500, 100),
                    'Protein_percent': np.random.normal(12.5, 1.5, 100)
                })

            st.session_state.df = df
            st.success("✅ Example dataset loaded!")
            st.dataframe(df.head(10))

    elif data_source == "Manual Entry":
        st.subheader("Manual Data Entry")
        st.info("This feature allows you to input small datasets manually.")
        st.write("Enter data in the format below and separate values with commas.")

        df_manual = data_handler.manual_data_entry()
        if df_manual is not None and not df_manual.empty:
            st.session_state.df = df_manual
            st.success("✅ Manual data entered successfully!")
            st.dataframe(df_manual)

# ==================== ANALYSIS PAGE ====================
elif workflow_step == "📊 Analysis":
    st.header("📊 Statistical Analysis")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section!")
    else:
        df = st.session_state.df

        # Display current dataset info
        with st.expander("📋 Current Dataset Info"):
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("**Numeric Columns:**", data_handler.get_numeric_columns(df))
            st.write("**Categorical Columns:**", data_handler.get_categorical_columns(df))

        # Analysis category selection
        st.subheader("Select Analysis Type")

        analysis_category = st.selectbox(
            "Choose Analysis Category:",
            ["Descriptive Statistics", "Inferential Statistics",
             "Correlation & Regression", "Advanced Analysis", "Agriculture-Specific"]
        )

        st.markdown("---")

        # ==================== DESCRIPTIVE STATISTICS ====================
        if analysis_category == "Descriptive Statistics":
            st.subheader("📊 Descriptive Statistics")

            analysis_type = st.radio(
                "Select Analysis:",
                ["Summary Statistics", "Grouped Statistics", "Frequency Table",
                 "Confidence Interval", "Distribution Analysis"]
            )

            if analysis_type == "Summary Statistics":
                numeric_cols = data_handler.get_numeric_columns(df)

                if len(numeric_cols) == 0:
                    st.error("No numeric columns found in dataset!")
                else:
                    selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:3])

                    if st.button("Calculate Statistics"):
                        if selected_cols:
                            results_df = descriptive_stats.summary_statistics_table(df, selected_cols)

                            if results_df is not None:
                                st.success("✅ Analysis complete!")
                                st.dataframe(results_df.round(4))

                                # Download option
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    "Download Results",
                                    data=csv,
                                    file_name="descriptive_stats.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.warning("Please select at least one column!")

            elif analysis_type == "Grouped Statistics":
                numeric_cols = data_handler.get_numeric_columns(df)
                cat_cols = data_handler.get_categorical_columns(df)

                if len(numeric_cols) == 0 or len(cat_cols) == 0:
                    st.error("Need both numeric and categorical columns!")
                else:
                    group_col = st.selectbox("Select grouping variable:", cat_cols)
                    value_col = st.selectbox("Select value variable:", numeric_cols)

                    if st.button("Calculate Grouped Statistics"):
                        results = descriptive_stats.grouped_descriptive_stats(df, group_col, value_col)

                        if results is not None:
                            st.success("✅ Analysis complete!")
                            st.dataframe(results)

                            # Download option
                            csv = results.to_csv()
                            st.download_button(
                                "Download Results",
                                data=csv,
                                file_name="grouped_stats.csv",
                                mime="text/csv"
                            )

            elif analysis_type == "Frequency Table":
                all_cols = df.columns.tolist()
                selected_col = st.selectbox("Select column:", all_cols)

                if st.button("Generate Frequency Table"):
                    freq_table = descriptive_stats.frequency_table(df, selected_col)
                    st.success("✅ Frequency table generated!")
                    st.dataframe(freq_table)

            elif analysis_type == "Confidence Interval":
                numeric_cols = data_handler.get_numeric_columns(df)
                selected_col = st.selectbox("Select column:", numeric_cols)
                confidence = st.slider("Confidence Level:", 0.90, 0.99, 0.95, 0.01)

                if st.button("Calculate Confidence Interval"):
                    ci_results = descriptive_stats.confidence_interval(df[selected_col], confidence)
                    st.success("✅ Confidence interval calculated!")

                    st.write(f"**Mean:** {ci_results['mean']:.4f}")
                    st.write(f"**{confidence*100}% Confidence Interval:** [{ci_results['lower_bound']:.4f}, {ci_results['upper_bound']:.4f}]")
                    st.write(f"**Margin of Error:** ±{ci_results['margin_of_error']:.4f}")

            elif analysis_type == "Distribution Analysis":
                numeric_cols = data_handler.get_numeric_columns(df)
                selected_col = st.selectbox("Select column:", numeric_cols)

                if st.button("Analyze Distribution"):
                    dist_info = descriptive_stats.describe_distribution(df[selected_col])
                    st.success("✅ Distribution analyzed!")

                    st.write(f"**Skewness:** {dist_info['skewness']:.4f}")
                    st.write(f"**Kurtosis:** {dist_info['kurtosis']:.4f}")
                    st.write(f"**Interpretation:** {dist_info['interpretation']}")

        # ==================== INFERENTIAL STATISTICS ====================
        elif analysis_category == "Inferential Statistics":
            st.subheader("📊 Inferential Statistics")

            analysis_type = st.selectbox(
                "Select Test:",
                ["Independent T-Test", "Paired T-Test", "One-Sample T-Test",
                 "One-Way ANOVA", "Two-Way ANOVA", "Chi-Square Test",
                 "Mann-Whitney U Test", "Kruskal-Wallis Test",
                 "Shapiro-Wilk Normality Test", "Post-Hoc Tests"]
            )

            alpha = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, 0.01)

            if analysis_type == "Independent T-Test":
                numeric_cols = data_handler.get_numeric_columns(df)
                cat_cols = data_handler.get_categorical_columns(df)

                if len(cat_cols) > 0:
                    group_col = st.selectbox("Select grouping variable:", cat_cols)
                    value_col = st.selectbox("Select value variable:", numeric_cols)

                    groups = df[group_col].unique()
                    if len(groups) == 2:
                        equal_var = st.checkbox("Assume equal variances", value=True)

                        if st.button("Run T-Test"):
                            group1 = df[df[group_col] == groups[0]][value_col]
                            group2 = df[df[group_col] == groups[1]][value_col]

                            results = inferential_stats.independent_t_test(group1, group2, equal_var, alpha)

                            if results:
                                st.success("✅ T-Test complete!")

                                # Display results
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Group 1 Mean", f"{results['group1_mean']:.4f}")
                                    st.metric("Group 2 Mean", f"{results['group2_mean']:.4f}")
                                with col2:
                                    st.metric("t-statistic", f"{results['t_statistic']:.4f}")
                                    st.metric("p-value", f"{results['p_value']:.4f}")

                                st.write(f"**Mean Difference:** {results['mean_difference']:.4f}")
                                st.write(f"**Cohen's d:** {results['cohens_d']:.4f}")
                                st.write(f"**95% CI:** [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")

                                if results['significant']:
                                    st.success("✅ Result is statistically significant!")
                                else:
                                    st.info("ℹ️ Result is not statistically significant.")

                                # Interpretation
                                st.markdown("---")
                                st.subheader("📝 Interpretation")
                                interpretation = interpretations.interpret_t_test(results)
                                st.markdown(interpretation)

                                # APA format
                                st.markdown("---")
                                st.subheader("📄 APA Format")
                                apa = interpretations.generate_apa_format(results)
                                st.code(apa)

                                st.session_state.analysis_results = results
                    else:
                        st.error("Independent T-Test requires exactly 2 groups!")

            elif analysis_type == "One-Way ANOVA":
                numeric_cols = data_handler.get_numeric_columns(df)
                cat_cols = data_handler.get_categorical_columns(df)

                if len(cat_cols) > 0:
                    group_col = st.selectbox("Select grouping variable:", cat_cols)
                    value_col = st.selectbox("Select value variable:", numeric_cols)

                    if st.button("Run One-Way ANOVA"):
                        results = inferential_stats.one_way_anova(df, group_col, value_col, alpha)

                        if results:
                            st.success("✅ ANOVA complete!")

                            # Display results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("F-statistic", f"{results['f_statistic']:.4f}")
                            with col2:
                                st.metric("p-value", f"{results['p_value']:.4f}")
                            with col3:
                                st.metric("η²", f"{results['eta_squared']:.4f}")

                            if results['significant']:
                                st.success("✅ Result is statistically significant!")
                            else:
                                st.info("ℹ️ Result is not statistically significant.")

                            # Group statistics
                            st.subheader("Group Statistics")
                            st.dataframe(results['group_stats'])

                            # Interpretation
                            st.markdown("---")
                            st.subheader("📝 Interpretation")
                            interpretation = interpretations.interpret_anova(results)
                            st.markdown(interpretation)

                            # APA format
                            st.markdown("---")
                            st.subheader("📄 APA Format")
                            apa = interpretations.generate_apa_format(results)
                            st.code(apa)

                            st.session_state.analysis_results = results

            elif analysis_type == "Post-Hoc Tests":
                st.info("Run this after a significant ANOVA result")

                numeric_cols = data_handler.get_numeric_columns(df)
                cat_cols = data_handler.get_categorical_columns(df)

                if len(cat_cols) > 0:
                    group_col = st.selectbox("Select grouping variable:", cat_cols)
                    value_col = st.selectbox("Select value variable:", numeric_cols)

                    posthoc_type = st.radio("Select Post-Hoc Test:", ["Tukey HSD", "Fisher's LSD"])

                    if st.button("Run Post-Hoc Test"):
                        if posthoc_type == "Tukey HSD":
                            results = inferential_stats.tukey_hsd_test(df, group_col, value_col, alpha)
                        else:
                            results = inferential_stats.lsd_test(df, group_col, value_col, alpha)

                        if results:
                            st.success("✅ Post-hoc test complete!")
                            st.dataframe(results['results_table'])
                            st.session_state.analysis_results = results

            elif analysis_type == "Shapiro-Wilk Normality Test":
                numeric_cols = data_handler.get_numeric_columns(df)
                selected_col = st.selectbox("Select column to test:", numeric_cols)

                if st.button("Run Normality Test"):
                    results = inferential_stats.shapiro_wilk_test(df[selected_col], alpha)

                    if results:
                        st.success("✅ Normality test complete!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("W-statistic", f"{results['w_statistic']:.4f}")
                        with col2:
                            st.metric("p-value", f"{results['p_value']:.4f}")

                        if results['normal']:
                            st.success("✅ Data appears normally distributed!")
                        else:
                            st.warning("⚠️ Data does not appear normally distributed!")

                        # Interpretation
                        st.markdown("---")
                        st.subheader("📝 Interpretation")
                        interpretation = interpretations.interpret_normality_test(results)
                        st.markdown(interpretation)

                        st.session_state.analysis_results = results

        # ==================== CORRELATION & REGRESSION ====================
        elif analysis_category == "Correlation & Regression":
            st.subheader("📈 Correlation & Regression Analysis")

            analysis_type = st.selectbox(
                "Select Analysis:",
                ["Pearson Correlation", "Spearman Correlation", "Correlation Matrix",
                 "Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]
            )

            numeric_cols = data_handler.get_numeric_columns(df)

            if analysis_type in ["Pearson Correlation", "Spearman Correlation"]:
                if len(numeric_cols) < 2:
                    st.error("Need at least 2 numeric columns!")
                else:
                    col1 = st.selectbox("Select X variable:", numeric_cols, index=0)
                    col2 = st.selectbox("Select Y variable:", numeric_cols, index=min(1, len(numeric_cols)-1))
                    alpha = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, 0.01)

                    if st.button("Calculate Correlation"):
                        if analysis_type == "Pearson Correlation":
                            results = correlation_regression.pearson_correlation(df[col1], df[col2], alpha)
                        else:
                            results = correlation_regression.spearman_correlation(df[col1], df[col2], alpha)

                        if results:
                            st.success("✅ Correlation analysis complete!")

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Correlation", f"{results['correlation_coefficient']:.4f}")
                            with col_b:
                                st.metric("p-value", f"{results['p_value']:.4f}")
                            with col_c:
                                st.metric("R²", f"{results.get('r_squared', results.get('rho_squared', 0)):.4f}")

                            if results['significant']:
                                st.success(f"✅ {results['strength'].title()} {results['direction']} correlation!")
                            else:
                                st.info("ℹ️ Correlation is not statistically significant.")

                            # Interpretation
                            st.markdown("---")
                            st.subheader("📝 Interpretation")
                            interpretation = interpretations.interpret_correlation(results)
                            st.markdown(interpretation)

                            st.session_state.analysis_results = results

            elif analysis_type == "Correlation Matrix":
                if len(numeric_cols) < 2:
                    st.error("Need at least 2 numeric columns!")
                else:
                    method = st.radio("Correlation Method:", ["pearson", "spearman"])

                    if st.button("Calculate Correlation Matrix"):
                        results = correlation_regression.correlation_matrix(df, method)

                        if results:
                            st.success("✅ Correlation matrix calculated!")
                            st.dataframe(results['correlation_matrix'].round(3))

                            st.session_state.analysis_results = results

            elif analysis_type == "Simple Linear Regression":
                if len(numeric_cols) < 2:
                    st.error("Need at least 2 numeric columns!")
                else:
                    x_col = st.selectbox("Select X variable (independent):", numeric_cols, index=0)
                    y_col = st.selectbox("Select Y variable (dependent):", numeric_cols, index=min(1, len(numeric_cols)-1))
                    alpha = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, 0.01)

                    if st.button("Run Regression"):
                        result = correlation_regression.simple_linear_regression(df[x_col], df[y_col], alpha)

                        if result:
                            results, model = result
                            st.success("✅ Regression analysis complete!")

                            st.write(f"**Equation:** {results['equation']}")

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("R²", f"{results['r_squared']:.4f}")
                            with col_b:
                                st.metric("RMSE", f"{results['rmse']:.4f}")
                            with col_c:
                                st.metric("p-value", f"{results['p_value']:.4f}")

                            if results['significant']:
                                st.success("✅ Regression is statistically significant!")
                            else:
                                st.info("ℹ️ Regression is not statistically significant.")

                            # Interpretation
                            st.markdown("---")
                            st.subheader("📝 Interpretation")
                            interpretation = interpretations.interpret_regression(results)
                            st.markdown(interpretation)

                            st.session_state.analysis_results = results

        # ==================== ADVANCED ANALYSIS ====================
        elif analysis_category == "Advanced Analysis":
            st.subheader("🔬 Advanced Analysis")

            analysis_type = st.selectbox(
                "Select Analysis:",
                ["Principal Component Analysis (PCA)", "K-Means Clustering",
                 "Hierarchical Clustering", "Elbow Method"]
            )

            numeric_cols = data_handler.get_numeric_columns(df)

            if analysis_type == "Principal Component Analysis (PCA)":
                if len(numeric_cols) < 2:
                    st.error("Need at least 2 numeric columns for PCA!")
                else:
                    max_components = min(len(numeric_cols), 10)
                    if max_components > 2:
                        n_components = st.slider("Number of components:", 2, max_components, 2)
                    else:
                        n_components = st.number_input("Number of components:", min_value=1, max_value=max_components, value=2)
                    standardize = st.checkbox("Standardize variables", value=True)

                    if st.button("Run PCA"):
                        results = advanced_analysis.perform_pca(df, n_components, standardize)

                        if results:
                            st.success("✅ PCA complete!")

                            st.write(f"**Total Variance Explained:** {results['total_variance_explained']:.2f}%")

                            st.subheader("Variance Explained by Component")
                            st.dataframe(results['variance_explained'])

                            st.subheader("Component Loadings")
                            st.dataframe(results['loadings'].round(3))

                            # Interpretation
                            st.markdown("---")
                            st.subheader("📝 Interpretation")
                            interpretation = interpretations.interpret_pca(results)
                            st.markdown(interpretation)

                            st.session_state.analysis_results = results

            elif analysis_type == "K-Means Clustering":
                if len(numeric_cols) < 2:
                    st.error("Need at least 2 numeric columns for clustering!")
                else:
                    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                    standardize = st.checkbox("Standardize variables", value=True)

                    if st.button("Run K-Means"):
                        results = advanced_analysis.kmeans_clustering(df, n_clusters, standardize)

                        if results:
                            st.success("✅ K-Means clustering complete!")

                            st.subheader("Cluster Sizes")
                            st.write(results['cluster_sizes'])

                            st.subheader("Cluster Centers")
                            st.dataframe(results['cluster_centers'].round(3))

                            st.write(f"**Inertia (Within-cluster sum of squares):** {results['inertia']:.2f}")

                            # Interpretation
                            st.markdown("---")
                            st.subheader("📝 Interpretation")
                            interpretation = interpretations.interpret_clustering(results)
                            st.markdown(interpretation)

                            st.session_state.analysis_results = results

        # ==================== AGRICULTURE-SPECIFIC ANALYSIS ====================
        elif analysis_category == "Agriculture-Specific":
            st.subheader("🌾 Agriculture-Specific Analysis")

            analysis_type = st.selectbox(
                "Select Analysis:",
                ["RCBD Analysis", "Latin Square Design", "Yield Comparison",
                 "Treatment Effect Analysis", "Multi-Location Analysis"]
            )

            if analysis_type == "RCBD Analysis":
                st.info("Randomized Complete Block Design (RCBD) Analysis")

                numeric_cols = data_handler.get_numeric_columns(df)
                cat_cols = data_handler.get_categorical_columns(df)

                if len(cat_cols) < 2 or len(numeric_cols) < 1:
                    st.error("Need at least 2 categorical columns and 1 numeric column!")
                else:
                    treatment_col = st.selectbox("Select treatment column:", cat_cols)
                    block_col = st.selectbox("Select block column:", cat_cols)
                    response_col = st.selectbox("Select response variable:", numeric_cols)
                    alpha = st.slider("Significance Level (α):", 0.01, 0.10, 0.05, 0.01)

                    if st.button("Run RCBD Analysis"):
                        results = agriculture_analysis.rcbd_analysis(df, treatment_col, block_col, response_col, alpha)

                        if results:
                            st.success("✅ RCBD analysis complete!")

                            # ANOVA table
                            st.subheader("ANOVA Table")
                            st.dataframe(results['anova_table'])

                            # Treatment means
                            st.subheader("Treatment Means")
                            st.dataframe(results['treatment_means'])

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Treatment p-value", f"{results['treatment_p_value']:.4f}")
                            with col2:
                                st.metric("Block p-value", f"{results['block_p_value']:.4f}")
                            with col3:
                                st.metric("CV %", f"{results['cv_percent']:.2f}")

                            if results['treatment_significant']:
                                st.success("✅ Treatment effect is significant!")

                                if results['posthoc_results'] is not None:
                                    st.subheader("Post-Hoc Comparisons (Tukey HSD)")
                                    st.dataframe(results['posthoc_results'])
                            else:
                                st.info("ℹ️ Treatment effect is not significant.")

                            st.session_state.analysis_results = results

            elif analysis_type == "Yield Comparison":
                st.info("Compare yields across treatments")

                numeric_cols = data_handler.get_numeric_columns(df)
                cat_cols = data_handler.get_categorical_columns(df)

                if len(cat_cols) < 1 or len(numeric_cols) < 1:
                    st.error("Need at least 1 categorical and 1 numeric column!")
                else:
                    treatment_col = st.selectbox("Select treatment column:", cat_cols)
                    yield_col = st.selectbox("Select yield column:", numeric_cols)

                    treatments = df[treatment_col].unique().tolist()
                    control_name = st.selectbox("Select control treatment:", ["None"] + treatments)
                    control_name = None if control_name == "None" else control_name

                    if st.button("Run Yield Comparison"):
                        results = agriculture_analysis.yield_comparison_analysis(
                            df, treatment_col, yield_col, control_name
                        )

                        if results:
                            st.success("✅ Yield comparison complete!")

                            st.subheader("Yield Statistics")
                            st.dataframe(results['yield_statistics'].round(2))

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("F-statistic", f"{results['anova_f_statistic']:.4f}")
                            with col2:
                                st.metric("p-value", f"{results['anova_p_value']:.4f}")

                            if results['significant_difference']:
                                st.success("✅ Significant yield differences detected!")

                                if results['posthoc_results'] is not None:
                                    st.subheader("Post-Hoc Comparisons")
                                    st.dataframe(results['posthoc_results'])
                            else:
                                st.info("ℹ️ No significant yield differences.")

                            st.session_state.analysis_results = results

# ==================== VISUALIZATION PAGE ====================
elif workflow_step == "📈 Visualization":
    st.header("📈 Data Visualization")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df

        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart",
             "Correlation Heatmap", "Q-Q Plot", "Line Plot"]
        )

        numeric_cols = data_handler.get_numeric_columns(df)
        cat_cols = data_handler.get_categorical_columns(df)

        if viz_type == "Histogram":
            if len(numeric_cols) > 0:
                col = st.selectbox("Select column:", numeric_cols)
                bins = st.slider("Number of bins:", 10, 100, 30)
                show_normal = st.checkbox("Show normal distribution overlay", value=True)

                if st.button("Generate Histogram"):
                    fig = visualizations.create_histogram(df, col, bins, show_normal=show_normal)
                    st.pyplot(fig)

                    # Download option
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("Download Plot", data=buf, file_name="histogram.png", mime="image/png")

        elif viz_type == "Box Plot":
            if len(numeric_cols) > 0:
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:3])

                if st.button("Generate Box Plot"):
                    if selected_cols:
                        fig = visualizations.create_boxplot(df, selected_cols)
                        st.pyplot(fig)

                        # Download option
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button("Download Plot", data=buf, file_name="boxplot.png", mime="image/png")

        elif viz_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X variable:", numeric_cols, index=0)
                y_col = st.selectbox("Select Y variable:", numeric_cols, index=1)
                add_regression = st.checkbox("Add regression line", value=True)

                if st.button("Generate Scatter Plot"):
                    fig = visualizations.create_scatter_plot(
                        df[x_col], df[y_col],
                        xlabel=x_col, ylabel=y_col,
                        add_regression=add_regression
                    )
                    st.pyplot(fig)

                    # Download option
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("Download Plot", data=buf, file_name="scatterplot.png", mime="image/png")

        elif viz_type == "Correlation Heatmap":
            if len(numeric_cols) >= 2:
                if st.button("Generate Correlation Heatmap"):
                    corr_matrix = df[numeric_cols].corr()
                    fig = visualizations.create_correlation_heatmap(corr_matrix)
                    st.pyplot(fig)

                    # Download option
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("Download Plot", data=buf, file_name="heatmap.png", mime="image/png")

        elif viz_type == "Q-Q Plot":
            if len(numeric_cols) > 0:
                col = st.selectbox("Select column:", numeric_cols)

                if st.button("Generate Q-Q Plot"):
                    fig = visualizations.create_qq_plot(df[col])
                    st.pyplot(fig)

                    # Download option
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button("Download Plot", data=buf, file_name="qqplot.png", mime="image/png")

# ==================== REPORT PAGE ====================
elif workflow_step == "📄 Report":
    st.header("📄 Analysis Report")

    if st.session_state.analysis_results is None:
        st.warning("⚠️ No analysis results available. Please run an analysis first!")
    else:
        st.success("✅ Analysis results are available!")

        st.info("Report generation feature - Export your analysis results")

        # Display current results summary
        st.subheader("Current Analysis Summary")
        results = st.session_state.analysis_results

        st.write(f"**Analysis Type:** {results.get('test_type', 'Unknown')}")

        # Export options
        st.subheader("Export Options")

        if st.button("Download Results as JSON"):
            import json
            # Convert results to JSON-serializable format
            json_data = {}
            for key, value in results.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    json_data[key] = value.to_dict()
                elif isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    json_data[key] = value

            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                "Download JSON",
                data=json_str,
                file_name="analysis_results.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Statistical Analysis Web Application v1.0</p>
    <p>Designed for research students in agriculture and life sciences</p>
</div>
""", unsafe_allow_html=True)
