import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
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

    # Map columns
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

    # Select features
    feature_cols = [
        'Height', 'Foundation Height', 'Treatment', 'Bracing', 'Bracing Score',
        'Animal Activity', 'Fireplace', 'Point Cloud Mean', 'Point Cloud Deviation',
        'Sill 1', 'Sill 2', 'Coat 1 Cracking', 'Coat 1 Loss',
        'Lintel Deterioration', 'Coat 2 Cracking', 'Coat 2 Loss',
        'Structural Cracking', 'Cracking at Wall Junction', 'Out of Plane',
        'Cap Deterioration', 'Foundation Stone Deterioration',
        'Foundation Displacement', 'Foundation Mortar Condition'
    ]
    
    # Ensure all cols exist
    existing_cols = [c for c in feature_cols if c in df_renamed.columns]
    
    X = df_renamed[existing_cols]
    y = df_renamed['Total Score']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X_imputed, y

def run_comparative_analysis():
    filepath = 'defunct/2023_12_8_targeted_eval.csv'
    X, y = load_and_preprocess_data(filepath)
    
    print(f"Data Shape: {X.shape}")
    
    # Standardize features for linear models
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Define models
    models = {
        "Random Forest (Original)": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge (L2)": RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
        "LASSO (L1)": LassoCV(cv=5, random_state=42, max_iter=10000),
        "Elastic Net": ElasticNetCV(cv=5, random_state=42, max_iter=10000, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
    }
    
    results = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- Model Performance Comparison (5-Fold CV) ---")
    for name, model in models.items():
        # For linear models use scaled data
        if "Random Forest" in name:
            X_curr = X
        else:
            X_curr = X_scaled
            
        scores = cross_val_score(model, X_curr, y, cv=kf, scoring='r2')
        mean_r2 = np.mean(scores)
        std_r2 = np.std(scores)
        
        # Fit on full data to get coefficients/importance
        model.fit(X_curr, y)
        
        # Get top features
        top_features = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in range(min(5, len(indices))):
                top_features.append(f"{X.columns[indices[i]]} ({importances[indices[i]]:.3f})")
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            # For linear models, magnitude of coefficient matters for importance (since scaled)
            indices = np.argsort(np.abs(coefs))[::-1]
            for i in range(min(5, len(indices))):
                top_features.append(f"{X.columns[indices[i]]} ({coefs[indices[i]]:.3f})")
        
        results[name] = {
            "R2 Mean": mean_r2,
            "R2 Std": std_r2,
            "Top Features": top_features
        }
        
        print(f"\n{name}:")
        print(f"  CV R²: {mean_r2:.3f} (+/- {std_r2:.3f})")
        print(f"  Top Predictors: {', '.join(top_features)}")
        if "LASSO" in name or "Elastic" in name:
             n_features = np.sum(model.coef_ != 0)
             print(f"  Features Selected: {n_features} / {X.shape[1]}")

    # Geometric Only Analysis
    print("\n--- Geometric Only Analysis (Regularized) ---")
    geo_cols = [
        'Height', 'Foundation Height', 'Treatment', 'Bracing', 'Bracing Score',
        'Point Cloud Mean', 'Point Cloud Deviation', 'Fireplace', 'Animal Activity'
    ]
    existing_geo = [c for c in geo_cols if c in X.columns]
    X_geo = X[existing_geo]
    X_geo_scaled = pd.DataFrame(scaler.fit_transform(X_geo), columns=X_geo.columns)
    
    models_geo = {
        "Random Forest (Geo)": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression (Geo)": LinearRegression(),
        "Elastic Net (Geo)": ElasticNetCV(cv=5, random_state=42, max_iter=10000)
    }
    
    for name, model in models_geo.items():
        if "Random Forest" in name:
            X_curr = X_geo
        else:
            X_curr = X_geo_scaled
            
        scores = cross_val_score(model, X_curr, y, cv=kf, scoring='r2')
        print(f"\n{name}:")
        print(f"  CV R²: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
        
        model.fit(X_curr, y)
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            indices = np.argsort(np.abs(coefs))[::-1]
            top = [f"{X_geo.columns[i]} ({coefs[i]:.3f})" for i in indices[:3]]
            print(f"  Top Predictors: {', '.join(top)}")
        elif hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            indices = np.argsort(imps)[::-1]
            top = [f"{X_geo.columns[i]} ({imps[i]:.3f})" for i in indices[:3]]
            print(f"  Top Predictors: {', '.join(top)}")

if __name__ == "__main__":
    run_comparative_analysis()
