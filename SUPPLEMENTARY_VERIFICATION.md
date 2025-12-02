# Supplementary Material Verification

**Status**: ✅ CONFIRMED - Supplementary material is complete and properly referenced

## File Location
`journalPaper/supplementary_benchmarking.tex`

## Contents

### Table: Complete Model Benchmarking Results
The supplementary file contains a comprehensive table with **all 7 models**:

1. **Ridge** - R² = 0.763 (±0.133) - Best model
2. **Elastic Net** - R² = 0.753 (±0.082) - p_adj = 0.2099 (Indistinguishable from best)
3. **Linear Regression** - R² = 0.744 (±0.144) - p_adj < 0.001
4. **Gradient Boosting** - R² = 0.610 (±0.150) - p_adj = 0.0003
5. **Random Forest** - R² = 0.585 (±0.160) - p_adj = 0.0008
6. **Decision Tree** - R² = -0.012 (±0.533) - p_adj < 0.001
7. **SVR** - R² = -0.103 (±0.123) - p_adj < 0.001

### Statistical Information Included
- ✅ Mean R² for each model
- ✅ Standard deviation for each model
- ✅ Raw p-values (Wilcoxon signed-rank test)
- ✅ Holm-Bonferroni corrected p-values
- ✅ Methodology description (5 folds × 5 repeats = 25 rounds)

### Interpretation Section
- ✅ Explains why Ridge was best
- ✅ Explains why Elastic Net was chosen despite Ridge being slightly better
- ✅ Lists statistical significance for all comparisons
- ✅ Justifies model selection based on interpretability

## Reference in Main Text
**Line 356** of `main_new.tex`:
```latex
\multicolumn{4}{l}{\footnotesize Full results for all 7 models available in supplementary material} \\
```

## Recommendation
The supplementary material is **complete and ready for submission**. It provides full transparency for the model selection process and allows readers to verify all statistical claims made in the main text.
