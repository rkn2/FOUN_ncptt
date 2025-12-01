import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configuration
INPUT_FILE = 'defunct/2023_12_8_targeted_eval.csv'
FIGURE_DIR = 'journalPaper/Images'

def load_and_prep_data():
    df = pd.read_csv(INPUT_FILE)
    
    # Handle Dual Elevation Measurements (Average them)
    # Foundation Height
    if 'foundation height 2' in df.columns:
        df['Foundation Height'] = df[['foundation height', 'foundation height 2']].mean(axis=1)
    else:
        df['Foundation Height'] = df['foundation height']
        
    # Foundation Displacement
    if 'ele 2 foundation displ' in df.columns:
        df['Foundation Displacement'] = df[['elev 1 foundation disp', 'ele 2 foundation displ']].mean(axis=1)
    else:
        df['Foundation Displacement'] = df['elev 1 foundation disp']
        
    # Foundation Mortar Condition
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
        'Coat 1 Cracking': 'Coat 1 Cracking',
        'Coat 1 Loss': 'Coat 1 Loss',
        'Coat2 Cracking': 'Coat 2 Cracking',
        'Coat2 Loss': 'Coat 2 Loss',
        'Structural Cracking': 'Structural Cracking',
        'Cracking Wall Junction': 'Cracking at Wall Junction',
        'Lintel Deteriration': 'Lintel Deterioration',
        'Sill 1': 'Sill 1',
        'SILL 2': 'Sill 2',
        'Cap Deterioration': 'Cap Deterioration',
        'Out of Plane': 'Out of Plane',
        'Animal Activity': 'Animal Activity',
        'fireplace': 'Fireplace',
        'Bracing': 'Bracing',
        'Bracing score': 'Bracing Score',
        'PC mean': 'Point Cloud Mean',
        'PC deviation': 'Point Cloud Deviation',
        'Foundation Stone Deterioration': 'Foundation Stone Deterioration'
    }
    
    df_renamed = df.rename(columns=column_mapping)
    
    # Create Treatment variable
    treat_cols = [c for c in df.columns if 'treatment_' in c]
    if treat_cols:
        df_renamed['Treatment'] = df[treat_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    else:
        df_renamed['Treatment'] = 0

    return df_renamed

def plot_coefficients(coefs, names, title, filename, color='skyblue'):
    # Sort by absolute value
    indices = np.argsort(np.abs(coefs))[::-1]
    
    # Take top 10 for readability
    top_indices = indices[:10]
    top_coefs = coefs[top_indices]
    top_names = [names[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_coefs)), top_coefs, align="center", color=color)
    plt.xticks(range(len(top_coefs)), top_names, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Standardized Coefficient")
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/{filename}', dpi=300)
    plt.close()

def main():
    df = load_and_prep_data()
    
    # --- Full Model ---
    feature_cols = [
        'Height', 'Foundation Height', 'Treatment', 'Bracing', 'Bracing Score',
        'Animal Activity', 'Fireplace', 'Point Cloud Mean', 'Point Cloud Deviation',
        'Sill 1', 'Sill 2', 'Coat 1 Cracking', 'Coat 1 Loss',
        'Lintel Deterioration', 'Coat 2 Cracking', 'Coat 2 Loss',
        'Structural Cracking', 'Cracking at Wall Junction', 'Out of Plane',
        'Cap Deterioration', 'Foundation Stone Deterioration',
        'Foundation Displacement', 'Foundation Mortar Condition'
    ]
    existing_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[existing_cols]
    y = df['Total Score']
    
    # Impute
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    
    # Elastic Net
    enet = ElasticNetCV(cv=5, random_state=42, max_iter=10000)
    enet.fit(X_scaled, y)
    
    print(f"Full Model Training R2: {enet.score(X_scaled, y):.3f}")
    
    # Print Top Coefficients
    coefs = enet.coef_
    indices = np.argsort(np.abs(coefs))[::-1]
    print("\nTop Predictors (Full Model):")
    for i in range(5):
        idx = indices[i]
        print(f"{X.columns[idx]}: {coefs[idx]:.3f}")

    plot_coefficients(enet.coef_, X.columns, 
                     "Elastic Net Coefficients (Full Model)", 
                     "feature_importance.png", color='steelblue')
    
    # --- Geometric Only Model ---
    geo_cols = [
        'Height', 'Foundation Height', 'Treatment', 'Bracing', 'Bracing Score',
        'Point Cloud Mean', 'Point Cloud Deviation', 'Fireplace', 'Animal Activity'
    ]
    existing_geo = [c for c in geo_cols if c in df.columns]
    
    X_geo = df[existing_geo]
    
    # Impute
    X_geo_imputed = pd.DataFrame(imputer.fit_transform(X_geo), columns=X_geo.columns)
    
    # Scale
    X_geo_scaled = pd.DataFrame(scaler.fit_transform(X_geo_imputed), columns=X_geo.columns)
    
    # Elastic Net
    enet_geo = ElasticNetCV(cv=5, random_state=42, max_iter=10000)
    enet_geo.fit(X_geo_scaled, y)
    
    print(f"Geometric Model Training R2: {enet_geo.score(X_geo_scaled, y):.3f}")
    
    # Print Top Coefficients
    coefs_geo = enet_geo.coef_
    indices_geo = np.argsort(np.abs(coefs_geo))[::-1]
    print("\nTop Predictors (Geometric Only):")
    for i in range(min(5, len(indices_geo))):
        idx = indices_geo[i]
        print(f"{X_geo.columns[idx]}: {coefs_geo[idx]:.3f}")

    plot_coefficients(enet_geo.coef_, X_geo.columns, 
                     "Elastic Net Coefficients (Geometric Only)", 
                     "feature_importance_geometric_only.png", color='seagreen')

if __name__ == "__main__":
    main()
