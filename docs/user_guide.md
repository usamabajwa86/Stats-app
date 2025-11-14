# Statistical Analysis Web Application - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Preparation](#data-preparation)
4. [Statistical Tests Guide](#statistical-tests-guide)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)
7. [Frequently Asked Questions](#frequently-asked-questions)

## Introduction

Welcome to the Statistical Analysis Web Application! This guide will help you make the most of the application's features for analyzing your research data.

### Who Is This For?
- Research students (especially in agriculture and life sciences)
- Researchers conducting field trials
- Anyone needing statistical analysis without complex software

### What Can You Do?
- Perform comprehensive statistical analyses
- Generate publication-ready visualizations
- Get plain-language interpretations of results
- Export results and reports

## Getting Started

### First Time Using the App

1. **Launch the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Explore the Interface**:
   - Home page provides overview
   - Sidebar navigation for different sections
   - Workflow progresses through: Upload → Analysis → Visualization → Report

3. **Try an Example**:
   - Go to "Data Upload"
   - Select "Use Example Dataset"
   - Load "Field Trial Data"
   - Proceed to Analysis section

## Data Preparation

### Data Format Requirements

#### Column Headers
- First row should contain variable names
- Use descriptive names (e.g., "Treatment", "Yield_kg_ha")
- Avoid special characters (except underscore)

#### Data Types
- **Numeric variables**: Numbers only (e.g., 45.2, 100, 3.14)
- **Categorical variables**: Text labels (e.g., "Control", "Treatment1", "Block1")

#### Example Format (CSV):
```csv
Treatment,Block,Yield,Height
Control,B1,45.2,82
Control,B2,44.8,85
Treatment1,B1,52.3,95
Treatment1,B2,51.9,92
```

### Data Cleaning Checklist

Before analysis:
- ✓ Check for missing values
- ✓ Identify and handle outliers
- ✓ Verify data types are correct
- ✓ Ensure adequate sample size
- ✓ Check for data entry errors

### Handling Missing Data

The app provides several options:
1. **Remove rows**: Best when few missing values
2. **Fill with mean/median**: For numeric data
3. **Fill with mode**: For categorical data
4. **Keep as is**: If analysis handles missing values

## Statistical Tests Guide

### Descriptive Statistics

#### When to Use
- Summarizing data characteristics
- Before any inferential analysis
- Reporting basic data properties

#### What You Get
- Central tendency (mean, median, mode)
- Variability (SD, variance, range)
- Distribution shape (skewness, kurtosis)

#### Example Use Case
*"I want to describe the average yield and variability in my experiment"*
1. Go to Analysis → Descriptive Statistics → Summary Statistics
2. Select yield column
3. Review mean, SD, CV%, min, max

### T-Tests

#### Independent T-Test

**When to Use**: Compare means of 2 independent groups

**Example**: Comparing yield between Control and Treatment

**Steps**:
1. Analysis → Inferential Statistics → Independent T-Test
2. Select grouping variable (e.g., Treatment)
3. Select value variable (e.g., Yield)
4. Check "equal variances" if appropriate
5. Run test

**What to Look For**:
- p-value < 0.05: Groups are significantly different
- Cohen's d: Effect size (how big is the difference?)
- Confidence interval: Range of plausible differences

#### Paired T-Test

**When to Use**: Compare measurements from same subjects/plots at different times

**Example**: Before and after measurements on same plants

**Requirements**:
- Equal number of observations in both groups
- Each observation in group 1 paired with one in group 2

#### One-Sample T-Test

**When to Use**: Compare sample mean to a known population value

**Example**: Testing if average yield differs from expected 50 kg/ha

### ANOVA (Analysis of Variance)

#### One-Way ANOVA

**When to Use**: Compare means of 3 or more independent groups

**Example**: Comparing 4 fertilizer treatments

**Steps**:
1. Analysis → Inferential Statistics → One-Way ANOVA
2. Select grouping variable (e.g., Treatment)
3. Select value variable (e.g., Yield)
4. Run ANOVA

**If Significant**:
- Run post-hoc test (Tukey HSD) to see which groups differ
- Create box plots for visualization

**What You Get**:
- F-statistic and p-value
- η² (effect size)
- Group means and standard deviations

#### Two-Way ANOVA

**When to Use**: Analyze effect of 2 factors and their interaction

**Example**: Testing effect of Treatment AND Location on yield

**Interpreting Results**:
- Main effect of Factor 1: Does factor 1 affect outcome?
- Main effect of Factor 2: Does factor 2 affect outcome?
- Interaction: Does effect of one factor depend on the other?

### Post-Hoc Tests

**When to Use**: After significant ANOVA (p < 0.05)

**Purpose**: Identify which specific groups differ

**Options**:
1. **Tukey HSD**: More conservative, controls family-wise error rate
2. **Fisher's LSD**: Less conservative, more powerful

**Reading Results**:
- Look for "Significant" column
- Compare mean differences
- Note confidence intervals

### Chi-Square Test

**When to Use**: Test association between 2 categorical variables

**Example**: Is disease incidence independent of treatment?

**Data Format**:
```
Treatment | Disease_Present | Disease_Absent
Control   | 10             | 40
Treatment | 3              | 47
```

**Interpreting Results**:
- p < 0.05: Variables are associated (not independent)
- Cramér's V: Strength of association

### Non-Parametric Tests

Use when:
- Data not normally distributed
- Small sample sizes
- Ordinal data

#### Mann-Whitney U Test
- Non-parametric alternative to independent t-test
- Compares medians of 2 groups

#### Kruskal-Wallis Test
- Non-parametric alternative to one-way ANOVA
- Compares medians of 3+ groups

#### Wilcoxon Signed-Rank Test
- Non-parametric alternative to paired t-test
- Compares paired observations

### Correlation Analysis

#### Pearson Correlation

**When to Use**: Assess linear relationship between 2 continuous variables

**Requirements**:
- Both variables normally distributed
- Relationship is linear

**Interpreting r**:
- r = 0: No correlation
- r = ±0.3: Weak correlation
- r = ±0.5: Moderate correlation
- r = ±0.7: Strong correlation
- r = ±1: Perfect correlation

**Important**: Correlation ≠ Causation!

#### Spearman Correlation

**When to Use**:
- Non-linear relationships
- Non-normal data
- Ordinal data

**Advantage**: More robust to outliers than Pearson

### Regression Analysis

#### Simple Linear Regression

**When to Use**: Predict Y from X (one predictor)

**Example**: Predict yield from fertilizer amount

**What You Get**:
- Regression equation: y = mx + b
- R²: How much variance explained (0-1)
- p-value: Is relationship significant?
- RMSE: Average prediction error

**Checking Assumptions**:
1. Review residual plot
2. Check Q-Q plot for normality
3. Look for patterns suggesting non-linearity

#### Multiple Linear Regression

**When to Use**: Predict Y from multiple X variables

**Example**: Predict yield from fertilizer, irrigation, and temperature

**Interpreting Coefficients**:
- Each coefficient shows effect of that variable holding others constant
- Check p-values for significance of each predictor

#### Polynomial Regression

**When to Use**: Relationship is curved, not straight

**Example**: Diminishing returns of fertilizer

**Choosing Degree**:
- Degree 2: Quadratic (one curve)
- Degree 3: Cubic (two curves)
- Higher degrees: Risk of overfitting

### Agriculture-Specific Analyses

#### RCBD (Randomized Complete Block Design)

**When to Use**: Field trial with blocks to control spatial variation

**Example Structure**:
```
Block 1: Control, Trt1, Trt2, Trt3
Block 2: Control, Trt1, Trt2, Trt3
Block 3: Control, Trt1, Trt2, Trt3
```

**What Analysis Does**:
- Tests treatment effect (controlling for blocks)
- Tests block effect (is blocking effective?)
- Provides post-hoc comparisons if significant

**CV% Guideline** (Coefficient of Variation):
- <10%: Excellent precision
- 10-20%: Good precision
- 20-30%: Acceptable
- >30%: Consider data quality

#### Latin Square Design

**When to Use**: Control variation in 2 directions (rows and columns)

**Example**: Greenhouse with row and column effects

**Requirements**:
- Equal number of treatments, rows, and columns
- Each treatment appears once in each row and column

#### Yield Comparison

**When to Use**: Compare yields across varieties/treatments

**Features**:
- Calculates relative yields (% of control)
- Shows yield advantage/disadvantage
- Statistical testing included

**Setting Control**:
- Select standard variety/treatment as control
- Results show differences relative to this

#### Treatment Effect Analysis

**When to Use**: Compare each treatment to control separately

**Useful For**:
- Identifying which treatments outperform control
- Calculating effect sizes for each treatment
- Showing percent improvements

#### Multi-Location Analysis

**When to Use**: Same treatments tested at multiple locations

**What to Examine**:
- Treatment effect: Do treatments differ overall?
- Location effect: Do locations differ?
- Interaction: Do treatment rankings change across locations?

**If Significant Interaction**:
- Treatment performance depends on location
- May need location-specific recommendations

### Advanced Analysis

#### Principal Component Analysis (PCA)

**When to Use**:
- Reduce many variables to few components
- Visualize multivariate data
- Identify patterns in complex datasets

**Example**: 10 plant traits reduced to 2-3 main components

**Interpreting Results**:
- Scree plot: How many components to keep?
- Loadings: How variables relate to components
- Biplot: Visualize samples and variables together

**Rule of Thumb**: Keep components explaining >70% variance

#### Cluster Analysis

**When to Use**: Group similar observations

**Example**: Group varieties by trait similarity

**K-Means vs Hierarchical**:
- K-Means: Specify number of clusters upfront
- Hierarchical: Explore different cluster numbers

**Determining K (Number of Clusters)**:
1. Use elbow method
2. Consider practical meaning
3. Validate with domain knowledge

## Interpreting Results

### Understanding P-Values

**What p-value means**:
- Probability of getting results this extreme if no real effect exists
- NOT the probability that the null hypothesis is true

**Decision Rules**:
- p < 0.05: Reject null hypothesis (result is "significant")
- p ≥ 0.05: Fail to reject null hypothesis (result is "not significant")

**Important Notes**:
- Significant ≠ Important (consider effect size)
- Not significant ≠ No effect (could be insufficient power)

### Effect Sizes

**Why They Matter**:
- Statistical significance depends on sample size
- Effect size shows practical importance
- Essential for meta-analyses

**Common Effect Sizes**:

| Test | Effect Size | Small | Medium | Large |
|------|------------|-------|--------|-------|
| T-test | Cohen's d | 0.2 | 0.5 | 0.8 |
| ANOVA | η² | 0.01 | 0.06 | 0.14 |
| Correlation | r | 0.1 | 0.3 | 0.5 |

### Confidence Intervals

**What They Show**: Range of plausible values for the true effect

**Example**: 95% CI for mean difference: [2.3, 8.7]
- We're 95% confident the true difference is between 2.3 and 8.7
- If CI doesn't include 0, difference is significant at α = 0.05

**Interpreting Width**:
- Narrow CI: Precise estimate
- Wide CI: Uncertain estimate (need more data)

## Best Practices

### Before Analysis

1. **Visualize Your Data**:
   - Create histograms, box plots
   - Check for outliers and patterns
   - Assess normality visually

2. **Check Assumptions**:
   - Normality (Shapiro-Wilk test, Q-Q plot)
   - Homogeneity of variance (box plots)
   - Independence of observations

3. **Plan Your Analysis**:
   - Define hypotheses clearly
   - Choose appropriate test
   - Decide on significance level (usually α = 0.05)

### During Analysis

1. **Use Appropriate Tests**:
   - Match test to data structure
   - Check sample size requirements
   - Consider non-parametric alternatives if needed

2. **Report Completely**:
   - Test statistic and p-value
   - Effect size
   - Confidence intervals
   - Sample sizes

3. **Don't P-Hack**:
   - Don't try multiple tests until you get significance
   - Don't remove "inconvenient" data points
   - Don't change hypotheses after seeing results

### After Analysis

1. **Interpret in Context**:
   - Consider biological/practical significance
   - Relate to existing literature
   - Acknowledge limitations

2. **Visualize Results**:
   - Show means with error bars
   - Use appropriate plot types
   - Make figures publication-ready

3. **Report Honestly**:
   - Report all analyses conducted
   - Include non-significant results
   - Discuss alternative explanations

## Frequently Asked Questions

### Q1: How much data do I need?

**A**: Minimum sample sizes:
- T-test: 10-15 per group minimum, 30+ ideal
- ANOVA: 20 per group minimum
- Correlation: 30 minimum, 50+ ideal
- Regression: 10-20 observations per predictor

### Q2: What if my data isn't normal?

**A**: Options:
1. Transform data (log, square root)
2. Use non-parametric tests
3. For large samples (n > 30), parametric tests often robust

### Q3: Should I remove outliers?

**A**: Consider:
- Is it a data entry error? (If yes, correct or remove)
- Is it a legitimate extreme value? (Keep it!)
- Does it affect conclusions? (Report with and without)

Never remove outliers just to get significance!

### Q4: Multiple comparisons - do I need correction?

**A**: Yes, if:
- Testing many hypotheses
- Multiple post-hoc comparisons

Use: Tukey HSD for ANOVA post-hocs (built into app)

### Q5: What's the difference between one-tailed and two-tailed tests?

**A**:
- **Two-tailed**: Test if groups differ (in either direction)
- **One-tailed**: Test if one group is specifically greater/less

**Default**: Use two-tailed (more conservative, appropriate for most research)

### Q6: How do I report results in my thesis?

**A**: The app provides:
1. APA-style formatted results
2. Plain language interpretations
3. Effect sizes and confidence intervals

**Example**: "Yield was significantly higher in Treatment A (M = 52.1, SD = 3.2) than Control (M = 45.3, SD = 2.8), t(38) = 7.82, p < .001, d = 2.31, 95% CI [4.9, 8.7]."

### Q7: What if my ANOVA is significant but post-hoc shows no differences?

**A**: Can happen due to:
- Multiple comparison correction
- Specific pattern of differences
- Suggestion: Examine means carefully, consider planned contrasts

### Q8: Can I analyze data with unequal sample sizes?

**A**: Yes, most tests handle unequal n, but:
- Try to keep groups reasonably balanced
- Large imbalances reduce power
- Some tests assume equal variances (can adjust)

### Q9: What does "control for blocks" mean in RCBD?

**A**:
- Blocks represent environmental variation
- Analysis removes block effects before testing treatments
- Increases power to detect treatment differences

### Q10: How do I choose between parametric and non-parametric tests?

**Decision Tree**:
1. Data normally distributed? → Parametric
2. Large sample (n > 30)? → Parametric often OK
3. Small sample + non-normal? → Non-parametric
4. Ordinal data? → Non-parametric

## Tips for Success

1. **Start Simple**: Begin with descriptive statistics and visualizations

2. **Check Assumptions**: Don't skip assumption checking

3. **Use Example Data**: Practice with provided examples first

4. **Understand Don't Just Click**: Read interpretations, understand what tests do

5. **Keep Records**: Download results, save visualizations

6. **Seek Help**: Consult with advisors on complex analyses

7. **Learn Gradually**: Master basic tests before advanced methods

8. **Think Critically**: Statistical significance isn't everything

## Additional Resources

### Learning More
- Read interpretation sections carefully
- Try different analyses on example data
- Compare results from related tests
- Consult introductory statistics textbooks

### Getting Help
- Review this user guide
- Check README.md for technical issues
- Discuss with research advisor
- Verify results with statistical consultant for publication

---

**Remember**: This application is a tool to assist your research. Understanding the statistical principles behind the tests is crucial for proper interpretation and reporting of results.

**Good luck with your research! 📊🌾**
