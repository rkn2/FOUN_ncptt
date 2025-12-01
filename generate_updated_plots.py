import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
import os

# Configuration
INPUT_FILE = 'defunct/2023_12_8_targeted_eval.csv'
FIGURE_DIR = 'journalPaper/Images'
RANDOM_STATE = 42

# Ensure directory exists
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

def load_and_prep_data():
    df = pd.read_csv(INPUT_FILE)
    
    # Handle Dual Elevation Measurements (Average them)
    if 'foundation height 2' in df.columns:
        df['Foundation Height'] = df[['foundation height', 'foundation height 2']].mean(axis=1)
    else:
        df['Foundation Height'] = df['foundation height']
        
    if 'ele 2 foundation displ' in df.columns:
        df['Foundation Displacement'] = df[['elev 1 foundation disp', 'ele 2 foundation displ']].mean(axis=1)
    else:
        df['Foundation Displacement'] = df['elev 1 foundation disp']
        
    if 'ele 2 foundation mortar condition' in df.columns:
        df['Foundation Mortar Condition'] = df[['ele 1 foundation mortar condition', 'ele 2 foundation mortar condition']].mean(axis=1)
    else:
        df['Foundation Mortar Condition'] = df['ele 1 foundation mortar condition']

    # Column Mapping
    column_mapping = {
        'Section ': 'WallID',
        'Orientation': 'Orientation',
        'Height': 'Height',
        'Length': 'Length',
        'Thickness': 'Thickness',
        'Material': 'Material',
        '# Courses': 'Courses',
        'Total Scr': 'Total Score',
        'out of plane': 'Out of Plane',
        'structural cracking': 'Structural Cracking',
        'cap deterioration': 'Cap Deterioration',
        'cracking at wall junction': 'Cracking Junction',
        'sill 1': 'Sill 1',
        'sill 2': 'Sill 2',
        'coat 1 cracking': 'Coat 1 Cracking',
        'coat 1 loss': 'Coat 1 Loss',
        'coat 2 cracking': 'Coat 2 Cracking',
        'coat 2 loss': 'Coat 2 Loss',
        'lintel deterioration': 'Lintel Deterioration',
        'surface loss top': 'Surface Loss Top',
        'surface loss mid': 'Surface Loss Mid',
        'surface loss low': 'Surface Loss Low',
        'Foundation Height': 'Foundation Height',
        'Foundation Displacement': 'Foundation Displacement',
        'foundation mortar 1': 'Foundation Mortar 1',
        'foundation mortar 2': 'Foundation Mortar 2',
        'Foundation Mortar Condition': 'Foundation Mortar Condition',
        'foundation stone det': 'Foundation Stone Det',
        'point cloud mean': 'Point Cloud Mean',
        'point cloud deviation': 'Point Cloud Deviation',
        'animal activity': 'Animal Activity',
        'fireplace': 'Fireplace',
        'treatment': 'Treatment',
        'bracing': 'Bracing',
        'bracing score': 'Bracing Score'
    }
    
    df_renamed = df.rename(columns=column_mapping)
    
    # Select columns
    selected_cols = list(column_mapping.values())
    # Remove ones that might not exist or are derived
    selected_cols = [c for c in selected_cols if c in df_renamed.columns]
    
    df_renamed = df_renamed[selected_cols]
    
    # Handle Treatment (Binary)
    if 'Treatment' in df_renamed.columns:
        df_renamed['Treatment'] = df_renamed['Treatment'].fillna(0).astype(int)
    else:
        df_renamed['Treatment'] = 0

    return df_renamed

def generate_plots():
    df = load_and_prep_data()
    
    # Prepare X and y
    drop_cols = ['WallID', 'Total Score', 'Material', 'Orientation', 'Length', 'Thickness', 'Courses']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Total Score']
    
    # Preprocessing Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    X_scaled = pipeline.fit_transform(X)
    feature_names = X.columns
    
    # --- 1. Full Model with Error Bars ---
    print("Generating Feature Importance Plot with Error Bars...")
    
    coefs = []
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    
    for train_index, test_index in rkf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], y.iloc[train_index]
        
        model = ElasticNet(random_state=RANDOM_STATE, l1_ratio=0.5, alpha=1.0)
        model.fit(X_train, X_test)
        coefs.append(model.coef_)
        
    coefs = np.array(coefs)
    mean_coefs = np.mean(coefs, axis=0)
    std_coefs = np.std(coefs, axis=0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_coefs,
        'Std': std_coefs
    })
    
    # Sort by absolute importance
    importance_df['AbsImportance'] = importance_df['Importance'].abs()
    importance_df = importance_df.sort_values('AbsImportance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    # Plot with error bars
    plt.errorbar(importance_df['Importance'], importance_df['Feature'], xerr=importance_df['Std']*1.96, 
                 fmt='o', color='teal', ecolor='gray', capsize=5, label='95% CI')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title('Top 15 Predictors of Degradation (Elastic Net Coefficients with 95% CI)')
    plt.xlabel('Standardized Coefficient (Impact on Total Score)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'feature_importance.png'), dpi=300)
    print("Saved feature_importance.png")
    
    # --- 2. Geometric Only Model ---
    print("Generating Geometric Only Plot...")
    
    # Filter for geometric/contextual features
    geo_features = [
        'Height', 'Foundation Height', 'Foundation Displacement', 
        'Point Cloud Mean', 'Point Cloud Deviation', 
        'Animal Activity', 'Fireplace', 'Treatment', 'Bracing'
    ]
    # Add any others that are strictly non-damage
    geo_features = [f for f in geo_features if f in X.columns]
    
    X_geo = X[geo_features]
    X_geo_scaled = pipeline.fit_transform(X_geo)
    
    model_geo = ElasticNet(random_state=RANDOM_STATE, l1_ratio=0.5, alpha=1.0)
    model_geo.fit(X_geo_scaled, y)
    
    geo_importance = pd.DataFrame({
        'Feature': geo_features,
        'Importance': model_geo.coef_
    })
    geo_importance = geo_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=geo_importance, palette='viridis')
    plt.title('Feature Importance: Geometric & Contextual Factors Only')
    plt.xlabel('Standardized Coefficient')
    plt.ylabel('Feature Name') # Added Y-axis label
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'feature_importance_geometric_only.png'), dpi=300)
    print("Saved feature_importance_geometric_only.png")

if __name__ == "__main__":
    generate_plots()
