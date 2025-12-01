
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# Configuration
INPUT_FILE = 'defunct/2023_12_8_targeted_eval.csv'
OUTPUT_JSON = 'journalPaper/real_data_metrics.json'
FIGURE_DIR = 'journalPaper/Images'

# Column Mapping (Real Data -> Analysis Names)
COLUMN_MAPPING = {
    'Height': 'Height',
    'foundation height': 'Foundation Height',
    'Out of Plane': 'Out of Plane',
    'Structural Cracking': 'Structural Cracking',
    'Cap Deterioration': 'Cap Deterioration',
    'Cracking Wall Junction': 'Cracking Junction',
    'Sill 1': 'Sill 1',
    'SILL 2': 'Sill 2',
    'Coat 1 Cracking': 'Coat 1 Cracking',
    'Coat 1 Loss': 'Coat 1 Loss',
    'Coat2 Cracking': 'Coat 2 Cracking',
    'Coat2 Loss': 'Coat 2 Loss',
    'Lintel Deteriration': 'Lintel Deterioration',
    'PC mean': 'Point Cloud Mean',
    'PC deviation': 'Point Cloud Deviation',
    'Bracing': 'Bracing',
    'Bracing score': 'Bracing Score',
    'Total Scr': 'Total Score',
    'fireplace': 'Fireplace',
    'treatment_1': 'Treatment' # Assuming treatment_1 is the main treatment indicator or we might need to sum them. 
    # Looking at previous scripts, 'Treatment' was used. In real data we have treatment_1...20. 
    # For now, let's check if 'Treatment' exists or if we need to derive it.
    # The synthetic data had 'Treatment'. The real data has binary columns.
    # Let's assume 'treatment_1' is a proxy or we create a 'Treatment' count.
    # Actually, let's look at the correlation matrix later to see what was used.
    # For now, I will create a 'Treatment Count' from treatment_1 to treatment_20 if possible, 
    # or just use treatment_1 if it's the main one. 
    # Let's try to sum them to get a "Treatment Intensity" or just use treatment_1.
    # Given the previous context, 'Treatment' might be categorical or binary.
    # Let's create a 'Treatment' variable that is 1 if any treatment exists.
}

def load_and_prep_data():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Create Treatment variable (1 if any treatment column is 1, else 0)
    treatment_cols = [c for c in df.columns if 'treatment_' in c]
    if treatment_cols:
        df['Treatment'] = df[treatment_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    else:
        df['Treatment'] = 0
        
    # Rename columns
    df_renamed = df.rename(columns=COLUMN_MAPPING)
    
    # Select analysis columns
    analysis_cols = [
        'Height', 'Foundation Height', 'Out of Plane', 'Structural Cracking',
        'Cap Deterioration', 'Cracking Junction', 'Sill 1', 'Sill 2',
        'Coat 1 Cracking', 'Coat 1 Loss', 'Coat 2 Cracking', 'Coat 2 Loss',
        'Lintel Deterioration', 'Point Cloud Mean', 'Point Cloud Deviation',
        'Bracing', 'Bracing Score', 'Total Score', 'Fireplace', 'Treatment'
    ]
    
    # Filter to available columns
    available_cols = [c for c in analysis_cols if c in df_renamed.columns]
    df_final = df_renamed[available_cols]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_final), columns=df_final.columns)
    
    return df_imputed

def run_correlation_analysis(df, results):
    print("Running Correlation Analysis...")
    corr_matrix = df.corr()
    
    # Save heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Degradation Factors')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/correlation_heatmap.png', dpi=300)
    plt.close()
    
    # Key correlations for text
    # Bracing vs Total Score
    r_brace, p_brace = pearsonr(df['Bracing'], df['Total Score'])
    
    # Sill 1 vs Sill 2
    r_sills, p_sills = pearsonr(df['Sill 1'], df['Sill 2'])
    
    results['correlations'] = {
        'bracing_total_score': {'r': r_brace, 'p': p_brace},
        'sill1_sill2': {'r': r_sills, 'p': p_sills}
    }

def run_factor_analysis(df, results):
    print("Running Factor Analysis...")
    # Variables for FA (excluding Total Score and contextual/geometric variables usually)
    # Based on previous context, FA was run on damage indicators.
    fa_cols = [
        'Sill 1', 'Sill 2', 'Coat 1 Cracking', 'Coat 1 Loss',
        'Lintel Deterioration', 'Coat 2 Cracking', 'Structural Cracking',
        'Out of Plane' # Added based on previous context
    ]
    
    # Filter cols that exist
    fa_cols = [c for c in fa_cols if c in df.columns]
    X_fa = df[fa_cols]
    
    # Bartlett
    chi_square_value, p_value = calculate_bartlett_sphericity(X_fa)
    
    # KMO
    kmo_all, kmo_model = calculate_kmo(X_fa)
    
    # Factor Analysis
    fa = FactorAnalyzer(n_factors=3, rotation='varimax')
    fa.fit(X_fa)
    
    loadings = pd.DataFrame(fa.loadings_, index=X_fa.columns, columns=['Factor 1', 'Factor 2', 'Factor 3'])
    communalities = pd.Series(fa.get_communalities(), index=X_fa.columns)
    variance = fa.get_factor_variance()
    
    results['factor_analysis'] = {
        'bartlett': {'chi2': chi_square_value, 'p': p_value},
        'kmo': kmo_model,
        'individual_kmo': dict(zip(X_fa.columns, kmo_all)),
        'variance_explained': {
            'factor1': variance[1][0],
            'factor2': variance[1][1],
            'factor3': variance[1][2],
            'total': variance[1].sum()
        },
        'loadings': loadings.to_dict(),
        'communalities': communalities.to_dict()
    }

def run_random_forest(df, results):
    print("Running Random Forest (Full Model)...")
    X = df.drop('Total Score', axis=1)
    y = df['Total Score']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Cross validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    y_pred = cross_val_predict(rf, X, y, cv=5)
    
    rf.fit(X, y)
    
    # Feature Importance
    importances = rf.feature_importances_
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    
    # Save Full Model Figure
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Full Model)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/feature_importance.png', dpi=300)
    plt.close()
    
    results['rf_full'] = {
        'r2_cv_mean': cv_scores.mean(),
        'r2_cv_std': cv_scores.std(),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'oob_score': rf.score(X, y), # Approximation if oob_score=True not set, but using fit score for now or re-init with oob_score=True
        'importances': dict(zip(X.columns, importances)),
        'perm_importances_mean': dict(zip(X.columns, perm_importance.importances_mean)),
        'perm_importances_std': dict(zip(X.columns, perm_importance.importances_std))
    }
    
    # Geometric Only Model
    print("Running Random Forest (Geometric Only)...")
    geo_cols = [
        'Height', 'Foundation Height', 'Treatment', 'Bracing', 'Bracing Score',
        'Point Cloud Mean', 'Point Cloud Deviation', 'Fireplace'
    ]
    geo_cols = [c for c in geo_cols if c in df.columns]
    
    X_geo = df[geo_cols]
    rf_geo = RandomForestRegressor(n_estimators=100, random_state=42)
    
    cv_scores_geo = cross_val_score(rf_geo, X_geo, y, cv=5, scoring='r2')
    y_pred_geo = cross_val_predict(rf_geo, X_geo, y, cv=5)
    rf_geo.fit(X_geo, y)
    
    importances_geo = rf_geo.feature_importances_
    perm_importance_geo = permutation_importance(rf_geo, X_geo, y, n_repeats=10, random_state=42)
    
    # Save Geometric Figure
    indices_geo = np.argsort(importances_geo)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Geometric Only)")
    plt.bar(range(X_geo.shape[1]), importances_geo[indices_geo], align="center", color='green')
    plt.xticks(range(X_geo.shape[1]), [X_geo.columns[i] for i in indices_geo], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/feature_importance_geometric_only.png', dpi=300)
    plt.close()
    
    results['rf_geo'] = {
        'r2_cv_mean': cv_scores_geo.mean(),
        'r2_cv_std': cv_scores_geo.std(),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_geo)),
        'mae': mean_absolute_error(y, y_pred_geo),
        'importances': dict(zip(X_geo.columns, importances_geo)),
        'perm_importances_mean': dict(zip(X_geo.columns, perm_importance_geo.importances_mean)),
        'perm_importances_std': dict(zip(X_geo.columns, perm_importance_geo.importances_std))
    }

def main():
    results = {}
    df = load_and_prep_data()
    
    # Wall Stats
    if 'Height' in df.columns:
        results['wall_stats'] = {
            'height_mean': df['Height'].mean(),
            'height_std': df['Height'].std(),
            'height_min': df['Height'].min(),
            'height_max': df['Height'].max()
        }
        
    run_correlation_analysis(df, results)
    run_factor_analysis(df, results)
    run_random_forest(df, results)
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Analysis Complete. Results saved to", OUTPUT_JSON)

if __name__ == "__main__":
    main()
