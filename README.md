# Statistical Analysis Web Application

A comprehensive, user-friendly platform for research data analysis, designed specifically for students and researchers in agriculture and life sciences.

## Overview

This application provides an all-in-one solution for statistical analysis without requiring multiple tools or extensive statistical knowledge. It features an intuitive interface, comprehensive statistical tests, automatic result interpretation, and publication-ready visualizations.

## Features

### Data Management
- **Multiple Format Support**: Upload CSV, Excel (.xls, .xlsx), and TXT files
- **Data Validation**: Automatic data type detection and validation
- **Data Cleaning**: Handle missing values, outliers, and data transformations
- **Manual Entry**: Input small datasets directly in the application
- **Example Datasets**: Pre-loaded sample data for learning and testing

### Statistical Analysis

#### Descriptive Statistics
- Mean, Median, Mode
- Standard Deviation, Variance
- Standard Error, Coefficient of Variation
- Range, IQR (Interquartile Range)
- Skewness and Kurtosis
- Confidence Intervals
- Frequency Tables

#### Inferential Statistics
- **T-Tests**:
  - Independent samples t-test
  - Paired samples t-test
  - One-sample t-test
- **ANOVA**:
  - One-way ANOVA
  - Two-way ANOVA
  - Post-hoc tests (Tukey HSD, Fisher's LSD)
- **Non-Parametric Tests**:
  - Mann-Whitney U test
  - Kruskal-Wallis test
  - Wilcoxon signed-rank test
- **Other Tests**:
  - Chi-square test of independence
  - Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)

#### Correlation & Regression
- Pearson correlation
- Spearman correlation
- Correlation matrices with p-values
- Simple linear regression
- Multiple linear regression
- Polynomial regression
- Regression diagnostics

#### Advanced Analysis
- Principal Component Analysis (PCA)
- K-Means clustering
- Hierarchical clustering
- Elbow method for optimal clusters
- Distance matrices

#### Agriculture-Specific Analysis
- Randomized Complete Block Design (RCBD)
- Latin Square Design
- Yield comparison analysis
- Treatment effect analysis
- Multi-location trial analysis
- Split-plot design analysis

### Visualization
- Histograms with normal distribution overlay
- Box plots and Violin plots
- Scatter plots with regression lines
- Bar charts with error bars
- Line plots
- Correlation heatmaps
- Q-Q plots for normality assessment
- Residual plots for regression diagnostics
- PCA biplots and scree plots
- Cluster visualizations
- Dendrograms for hierarchical clustering

### Interpretation & Reporting
- Plain language interpretations of all results
- Statistical significance indicators
- Effect size calculations and interpretations
- APA-style result formatting
- Assumption checking recommendations
- Export results in multiple formats

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the repository**:
   ```bash
   cd "stats app"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Navigate to the application directory**:
   ```bash
   cd "stats app"
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Access the application**:
   - The application will automatically open in your default web browser
   - If not, navigate to: `http://localhost:8501`

### Workflow

#### Step 1: Data Upload
1. Click on "📁 Data Upload" in the sidebar
2. Choose your data source:
   - **Upload File**: Browse and select your data file
   - **Use Example Dataset**: Load pre-configured sample data
   - **Manual Entry**: Input data directly
3. Preview your data
4. (Optional) Clean and transform data as needed

#### Step 2: Statistical Analysis
1. Navigate to "📊 Analysis" in the sidebar
2. Select analysis category:
   - Descriptive Statistics
   - Inferential Statistics
   - Correlation & Regression
   - Advanced Analysis
   - Agriculture-Specific
3. Choose specific test or method
4. Configure parameters (e.g., significance level)
5. Select variables
6. Run analysis
7. Review results and interpretations

#### Step 3: Visualization
1. Go to "📈 Visualization"
2. Select visualization type
3. Choose variables
4. Customize plot parameters
5. Generate and download visualizations

#### Step 4: Export Results
1. Navigate to "📄 Report"
2. Review analysis summary
3. Export results in desired format

## Example Use Cases

### Example 1: Comparing Treatment Effects (RCBD)

```
Scenario: Comparing yields of 4 fertilizer treatments across 4 blocks

1. Upload data with columns: Treatment, Block, Yield
2. Go to Analysis → Agriculture-Specific → RCBD Analysis
3. Select:
   - Treatment column: Treatment
   - Block column: Block
   - Response variable: Yield
4. Set significance level (α = 0.05)
5. Run analysis
6. Review ANOVA table and post-hoc comparisons
7. Generate visualizations (box plots, bar charts)
```

### Example 2: Correlation Analysis

```
Scenario: Analyzing relationship between plant height and yield

1. Upload data with numeric columns
2. Go to Analysis → Correlation & Regression → Pearson Correlation
3. Select:
   - X variable: Plant_Height
   - Y variable: Yield
4. Run correlation analysis
5. Review correlation coefficient, p-value, and interpretation
6. Create scatter plot with regression line
```

### Example 3: Multiple Group Comparison

```
Scenario: Comparing growth rates across 3 treatments

1. Upload data with Group and Growth_Rate columns
2. Go to Analysis → Inferential Statistics → One-Way ANOVA
3. Select:
   - Grouping variable: Group
   - Value variable: Growth_Rate
4. Run ANOVA
5. If significant, run post-hoc test (Tukey HSD)
6. Generate box plots for visualization
```

## Sample Datasets

The application includes three sample datasets in the `sample_data` folder:

1. **example_field_trial.csv**: RCBD data with treatments, blocks, and multiple response variables
2. **example_treatment_comparison.csv**: Treatment comparison data for basic statistical tests
3. **example_yield_data.csv**: Multi-location variety trial data

## Statistical Test Selection Guide

### Choosing the Right Test

| Research Question | Recommended Test |
|-------------------|------------------|
| Compare 2 independent groups | Independent t-test |
| Compare 2 paired/related groups | Paired t-test |
| Compare 3+ independent groups | One-way ANOVA |
| Compare groups with 2 factors | Two-way ANOVA |
| Relationship between 2 variables | Correlation (Pearson/Spearman) |
| Predict Y from X | Simple Linear Regression |
| Predict Y from multiple X's | Multiple Linear Regression |
| Compare categorical variables | Chi-square test |
| Non-normal data, 2 groups | Mann-Whitney U test |
| Non-normal data, 3+ groups | Kruskal-Wallis test |
| Field trial with blocks | RCBD Analysis |
| Reduce dimensions | PCA |
| Group similar observations | Cluster Analysis |

### Assumption Checking

Before parametric tests:
1. **Check normality**: Use Shapiro-Wilk test or Q-Q plots
2. **Check homogeneity of variance**: Use Levene's test or examine box plots
3. If assumptions violated: Consider non-parametric alternatives or data transformation

## Interpreting Results

### Statistical Significance
- **p < 0.05**: Result is statistically significant (reject null hypothesis)
- **p ≥ 0.05**: Result is not statistically significant (fail to reject null hypothesis)

### Effect Sizes
- **Cohen's d**: 0.2 (small), 0.5 (medium), 0.8 (large)
- **η² (eta-squared)**: 0.01 (small), 0.06 (medium), 0.14 (large)
- **Correlation (r)**: 0.3 (weak), 0.5 (moderate), 0.7 (strong)

### Reporting Results
The application provides:
- Plain language interpretations
- APA-style formatted results
- Confidence intervals
- Effect sizes
- Recommendations for further analysis

## Troubleshooting

### Common Issues

1. **"Module not found" error**:
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - Activate virtual environment if using one

2. **Data not loading**:
   - Check file format (CSV, Excel, TXT)
   - Ensure data has column headers
   - Check for special characters in file name

3. **Statistical test fails**:
   - Verify sufficient sample size
   - Check that variables are correct type (numeric vs categorical)
   - Ensure no missing values if not handled

4. **Visualization not displaying**:
   - Ensure numeric data for numerical plots
   - Check that selected columns exist
   - Try refreshing the browser

### Getting Help

If you encounter issues:
1. Check the user guide in the `docs` folder
2. Review example datasets for proper format
3. Ensure all assumptions are met for statistical tests
4. Check Python and package versions

## Technical Details

### Project Structure
```
stats_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── modules/
│   ├── __init__.py
│   ├── data_handler.py            # Data loading and preprocessing
│   ├── descriptive_stats.py       # Descriptive statistics
│   ├── inferential_stats.py       # Inferential tests
│   ├── correlation_regression.py  # Correlation and regression
│   ├── advanced_analysis.py       # PCA, clustering
│   ├── visualizations.py          # Chart generation
│   ├── interpretations.py         # Result interpretation
│   └── agriculture_analysis.py    # Agricultural designs
├── sample_data/                   # Example datasets
│   ├── example_field_trial.csv
│   ├── example_treatment_comparison.csv
│   └── example_yield_data.csv
└── docs/                          # Documentation
    └── user_guide.md
```

### Key Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scipy**: Statistical functions
- **statsmodels**: Statistical models
- **matplotlib/seaborn/plotly**: Visualizations
- **scikit-learn**: Machine learning algorithms

## Contributing

This is an educational tool designed for research students. Suggestions for improvements are welcome.

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Designed specifically for research students in agriculture and life sciences to facilitate data analysis and promote statistical literacy.

## Version History

- **v1.0.0** (2025): Initial release
  - Comprehensive statistical tests
  - Agriculture-specific analyses
  - Interactive visualizations
  - Automated interpretations

## Contact & Support

For questions or feedback about the application, please refer to the user guide or consult with your research advisor.

---

**Happy Analyzing! 📊🌾**
