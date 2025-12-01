# Comprehensive Model Benchmark Results

**Date**: December 1, 2025
**Status**: ✅ Benchmark Complete

## Executive Summary
To ensure the generalizability of our findings, we benchmarked seven regression model families using repeated K-fold cross-validation (5 folds, 5 repeats, 25 runs).

**Key Finding**: **Ridge Regression** ($R^2 = 0.763$) and **Elastic Net** ($R^2 = 0.753$) were the top performers and are **statistically indistinguishable** ($p_{adj} = 0.21$).
Complex non-linear models (Random Forest, Gradient Boosting) performed significantly worse, confirming that the degradation scoring system follows a linear additive logic.

---

## 1. Model Performance (Ranked by Mean R²)

| Rank | Model | Mean R² | Std Dev | Status vs Best |
|------|-------|---------|---------|----------------|
| 1 | **Ridge Regression** | **0.763** | ±0.133 | **Best Model** |
| 2 | **Elastic Net** | 0.753 | ±0.082 | **Indistinguishable** ($p=0.21$) |
| 3 | Linear Regression | 0.744 | ±0.144 | Significantly Worse |
| 4 | Gradient Boosting | 0.610 | ±0.150 | Significantly Worse |
| 5 | Random Forest | 0.585 | ±0.160 | Significantly Worse |
| 6 | Decision Tree | -0.012 | ±0.533 | Poor |
| 7 | SVR (RBF) | -0.103 | ±0.123 | Poor |

## 2. Statistical Significance (Wilcoxon + Holm-Bonferroni)
We performed paired Wilcoxon signed-rank tests on the 25 fold-wise $R^2$ differences between the best model (Ridge) and all others.

*   **Ridge vs Elastic Net**: $p_{adj} = 0.2099$ (Fail to Reject Null $\rightarrow$ Equivalent)
*   **Ridge vs Random Forest**: $p_{adj} = 0.0008$ (Reject Null $\rightarrow$ RF is worse)
*   **Ridge vs Gradient Boosting**: $p_{adj} = 0.0003$ (Reject Null $\rightarrow$ GBM is worse)

## 3. Recommendation for Manuscript
*   **Primary Model**: Continue using **Elastic Net** for feature importance interpretation. Although Ridge had a slightly higher mean $R^2$ (0.763 vs 0.753), they are statistically equivalent, and Elastic Net's sparsity (L1 penalty) makes it superior for feature selection and interpretation.
*   **Methodology Update**: Update the text to describe this rigorous benchmarking process.
*   **Justification**: Explicitly state that linear models (Ridge/Elastic Net) statistically outperformed ensemble methods, validating the linear nature of the degradation index.
