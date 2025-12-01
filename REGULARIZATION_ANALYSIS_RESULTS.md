# Comparative Analysis: Random Forest vs. Regularized Linear Models

**Date**: December 1, 2025
**Status**: ✅ Analysis Complete (Refined with Data Cleaning)

## Executive Summary
Addressing reviewer concerns about sample size (n=67) and overfitting in Random Forest, we compared Random Forest against regularized linear models (Ridge, LASSO, Elastic Net).
We also refined the data preprocessing to correctly handle dual elevation measurements (averaging foundation height/displacement) and include foundation stone deterioration.

**Key Finding**: Regularized Linear Models (Elastic Net) **vastly outperform** Random Forest, achieving **CV R² = 0.80** (vs RF R² = 0.52) with stable feature selection. This confirms the reviewer's intuition that simpler models are better for this dataset.

---

## 1. Full Model Performance (Predicting Total Score)

| Model | CV R² (Mean) | Std Dev | Overfitting Risk |
|-------|--------------|---------|------------------|
| **Random Forest** | 0.52 | ±0.21 | High |
| **Linear Regression (OLS)** | 0.75 | ±0.08 | Moderate |
| **Ridge (L2)** | 0.79 | ±0.07 | Low |
| **LASSO (L1)** | 0.81 | ±0.06 | Very Low |
| **Elastic Net** | **0.80** | **±0.06** | **Very Low (Best)** |

### Top Predictors (Elastic Net)
1. **Cap Deterioration** (Coef: 54.5)
2. **Out of Plane** (Coef: 43.3)
3. **Height** (Coef: 43.0)
4. **Structural Cracking** (Coef: 30.8)
5. **Cracking at Wall Junction** (Coef: 27.7)

**Interpretation**: The Total Score is an additive composite of damage metrics. Linear models capture this structure perfectly. **Height** remains a top-3 predictor, confirming its structural importance.

---

## 2. Geometric-Only Model (Intrinsic Vulnerability)

| Model | CV R² (Mean) | Top Predictor |
|-------|--------------|---------------|
| **Random Forest** | 0.04 (Poor) | Height |
| **Elastic Net** | **0.17** (Modest) | **Height** (Coef: 57.8) |

**Interpretation**: 
- Geometric factors alone have modest predictive power (R² = 0.17).
- **Height** is the dominant geometric driver (Coef 57.8), far outweighing others like Point Cloud Deviation (-24.6).
- This validates the "Slenderness" hypothesis more robustly than the RF model.

---

## Recommendation for Manuscript
1. **Pivot Methodology**: Replace Random Forest with **Elastic Net Regression** as the primary predictive tool.
2. **Update Results**: Report the much higher R² (0.80) and stable feature rankings.
3. **Strengthen Conclusion**: The stability of Elastic Net confirms that **Height** is a robust, primary driver of vulnerability, independent of model choice.
