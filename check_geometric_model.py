import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score

# Configuration
INPUT_FILE = 'defunct/2023_12_8_targeted_eval.csv'
RANDOM_STATE = 42

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
        # 'Height': 'Height', # EXCLUDED: Damage Score
        'Length': 'Length',
        'Thickness': 'Thickness',
        'Material': 'Material',
        '# Courses': 'Courses',
        'Total Scr': 'Total Score',
        'foundation height': 'Foundation Height',
        'elev 1 foundation disp': 'Foundation Displacement',
        'PC mean': 'Point Cloud Mean',
        'PC deviation': 'Point Cloud Deviation',
        'fireplace': 'Fireplace',
        'treatment': 'Treatment', # Derived or raw
        'Bracing': 'Bracing'
    }
    
    # Rename columns that exist
    rename_map = {}
    for old, new in column_mapping.items():
        if old in df.columns:
            rename_map[old] = new
            
    df_renamed = df.rename(columns=rename_map)
    
    # Handle Treatment (Binary)
    # Check for treatment columns
    treat_cols = [c for c in df.columns if 'treatment' in c.lower()]
    if treat_cols:
        df_renamed['Treatment'] = df[treat_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    else:
        df_renamed['Treatment'] = 0

    return df_renamed

def run_geo_analysis():
    df = load_and_prep_data()
    y = df['Total Score']
    
    # Select Geometric/Contextual Features
    # Note: Height is EXCLUDED
    potential_features = [
        'Foundation Height', 'Foundation Displacement', 
        'Point Cloud Mean', 'Point Cloud Deviation', 
        'Animal Activity', 'Fireplace', 'Treatment', 'Bracing',
        'Orientation' # If we can encode it
    ]
    
    # Filter to what exists
    features = [f for f in potential_features if f in df.columns]
    
    # Handle Orientation if present (One-Hot Encoding)
    if 'Orientation' in df.columns:
        df = pd.get_dummies(df, columns=['Orientation'], drop_first=True)
        # Update features list
        new_features = [c for c in df.columns if 'Orientation_' in c]
        features = [f for f in features if f != 'Orientation'] + new_features
        
    X = df[features]
    
    print(f"Features included in Geometric-Only Model (n={len(features)}):")
    print(features)
    
    from sklearn.dummy import DummyRegressor
    
    # CV
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    
    # 1. Naive Baseline
    dummy = DummyRegressor(strategy='mean')
    dummy_scores = cross_val_score(dummy, X, y, cv=rkf, scoring='r2')
    print(f"\nNaive Baseline (Mean) Performance:")
    print(f"Mean R2: {np.mean(dummy_scores):.3f} (+/- {np.std(dummy_scores):.3f})")

    # 2. Geometric Model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', ElasticNet(random_state=RANDOM_STATE, l1_ratio=0.5, alpha=1.0))
    ])
    
    # CV
    scores = cross_val_score(pipeline, X, y, cv=rkf, scoring='r2')
    
    print(f"\nGeometric-Only Model Performance (without Height):")
    print(f"Mean R2: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
    
    # Fit on full data for coefficients
    pipeline.fit(X, y)
    model = pipeline.named_steps['model']
    
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nTop Coefficients:")
    print(coef_df.head(10))

if __name__ == "__main__":
    run_geo_analysis()
