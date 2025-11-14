# Quick Start Guide

## Installation (5 minutes)

1. Open terminal/command prompt
2. Navigate to project folder:
   ```bash
   cd "stats app"
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Your First Analysis (10 minutes)

### Example 1: Compare Two Groups (T-Test)

1. **Upload Data**
   - Click "📁 Data Upload" in sidebar
   - Select "Use Example Dataset" → "Treatment Comparison"
   - Click "Load Example Dataset"

2. **Run Analysis**
   - Click "📊 Analysis" in sidebar
   - Select "Inferential Statistics"
   - Choose "Independent T-Test"
   - Grouping variable: `Group`
   - Value variable: `Growth_Rate`
   - Click "Run T-Test"

3. **View Results**
   - Check p-value (significant if < 0.05)
   - Read interpretation section
   - Note APA-formatted results

4. **Visualize**
   - Go to "📈 Visualization"
   - Select "Box Plot"
   - Choose `Growth_Rate`
   - Generate and download

### Example 2: Field Trial Analysis (RCBD)

1. **Load Data**
   - Data Upload → Example Dataset → "Field Trial Data"

2. **Analyze**
   - Analysis → Agriculture-Specific → RCBD Analysis
   - Treatment: `Treatment`
   - Block: `Block`
   - Response: `Yield_kg_ha`
   - Run analysis

3. **Interpret**
   - Check if treatment is significant
   - Review post-hoc comparisons
   - Note CV% (should be < 20%)

### Example 3: Correlation Analysis

1. **Load Data**
   - Use "Yield Data" example

2. **Analyze**
   - Analysis → Correlation & Regression → Pearson Correlation
   - X: `Yield_kg_ha`
   - Y: `Protein_percent`
   - Calculate correlation

3. **Visualize**
   - Visualization → Scatter Plot
   - Add regression line
   - Download plot

## Common Workflows

### Workflow 1: Treatment Comparison
```
Upload Data → Descriptive Stats → Normality Test →
(if normal) ANOVA → Post-hoc → Visualization → Report
(if not normal) Kruskal-Wallis → Visualization → Report
```

### Workflow 2: Relationship Between Variables
```
Upload Data → Scatter Plot → Normality Check →
(if normal) Pearson Correlation → Regression → Report
(if not normal) Spearman Correlation → Report
```

### Workflow 3: Field Trial
```
Upload Data → Descriptive by Group → RCBD Analysis →
Post-hoc → Box Plots → Bar Charts → Report
```

## Test Selection Cheat Sheet

| Goal | Test |
|------|------|
| Compare 2 groups | Independent T-Test |
| Compare 3+ groups | One-Way ANOVA |
| Compare groups over time | Repeated Measures (use Paired T-Test for 2 timepoints) |
| Relationship between 2 variables | Pearson/Spearman Correlation |
| Predict outcome | Linear Regression |
| Field trial analysis | RCBD Analysis |
| Categorical association | Chi-Square Test |
| Non-normal data, 2 groups | Mann-Whitney U |
| Non-normal data, 3+ groups | Kruskal-Wallis |

## Keyboard Shortcuts

- `Ctrl + R` / `Cmd + R`: Refresh app
- `Ctrl + S` / `Cmd + S`: Save (in edit mode)

## Quick Tips

✅ **DO**:
- Check normality before parametric tests
- Use descriptive statistics first
- Read interpretations carefully
- Download results regularly
- Visualize your data

❌ **DON'T**:
- Remove outliers without justification
- Run tests without checking assumptions
- Interpret p-values as effect size
- Test repeatedly until significant
- Skip the interpretation sections

## Getting Help

- **Technical Issues**: Check README.md
- **Statistical Questions**: See docs/user_guide.md
- **Examples**: Use sample datasets in sample_data/

## File Structure
```
stats app/
├── app.py              ← Main application
├── requirements.txt    ← Dependencies
├── README.md          ← Full documentation
├── QUICKSTART.md      ← This file
├── modules/           ← Analysis modules
├── sample_data/       ← Example datasets
└── docs/              ← Detailed guides
```

## Troubleshooting

**App won't start?**
```bash
pip install --upgrade streamlit pandas numpy scipy statsmodels matplotlib seaborn plotly
```

**Import errors?**
```bash
pip install -r requirements.txt --force-reinstall
```

**Data won't load?**
- Check file format (CSV, Excel)
- Ensure first row has column names
- No special characters in file path

**Test fails?**
- Check minimum sample size
- Verify variable types (numeric/categorical)
- Remove or handle missing values

## Next Steps

1. ✅ Complete Example 1, 2, and 3 above
2. 📖 Read the User Guide (docs/user_guide.md)
3. 📊 Try with your own data
4. 💾 Save and export results
5. 📈 Create publication-ready figures

---

**You're ready to start analyzing! Good luck! 🎉**
