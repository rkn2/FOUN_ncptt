"""
RE-ANALYSIS WITH ACTUAL FIELD DATA
===================================
Using: defunct/2023_12_8_targeted_eval.csv (the real field data)
Instead of: synthetic_adobe_data.csv

This script re-runs all analyses to update the manuscript with actual values.
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr, chi2
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns

def load_actual_field_data(filepath):
    """Load and preprocess ACTUAL field data"""
    print(f"Loading ACTUAL FIELD DATA from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    # Map column names to standardized names
    column_mapping = {
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
        'elev 1 foundation disp': 'Foundation Displacement 1',
        'ele 2 foundation displ': 'Foundation Displacement 2',
        'ele 1 foundation mortar condition': 'Foundation Mortar 1',
        'ele 2 foundation mortar condition': 'Foundation Mortar 2',
        'Foundation Stone Deterioration': 'Foundation Stone Det',
        'PC mean': 'Point Cloud Mean',
        'PC deviation': 'Point Cloud Deviation',
        'Bracing': 'Bracing',
        'Bracing score': 'Bracing Score',
        'Total Scr': 'Total Scr',
        'fireplace': 'Fireplace'
    }
    
    # Rename columns
    df_renamed = df.rename(columns=column_mapping)
    
    # Select only the columns we need
    analysis_cols = list(column_mapping.values())
    available_cols = [c for c in analysis_cols if c in df_renamed.columns]
    
    df_clean = df_renamed[available_cols]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df_numeric = df_clean.select_dtypes(include=[np.number])
    
    # Drop columns that are all NaN
    df_numeric = df_numeric.dropna(axis=1, how='all')
    
    imputed_data = imputer.fit_transform(df_numeric)
    df_imputed = pd.DataFrame(imputed_data, columns=df_numeric.columns)
    
    print(f"Final shape: {df_imputed.shape}")
    print(f"Columns: {df_imputed.columns.tolist()}")
    
    return df_imputed

def get_wall_dimensions(df):
    """Calculate wall dimension statistics"""
    print("\n" + "="*70)
    print("WALL DIMENSION STATISTICS")
    print("="*70)
    
    if 'Height' not in df.columns:
        print("ERROR: Height column not found")
        return None
    
    height = df['Height'].dropna()
    
    # Typical adobe wall thickness (from literature/NPS standards)
    # Fort Union walls are typically 18-24 inches (0.46-0.61 m)
    typical_thickness_m = 0.60  # meters (approximately 24 inches)
    
    # Calculate h/t ratios
    ht_ratios = height / typical_thickness_m
    
    print(f"\n=== WALL HEIGHT (meters) ===")
    print(f"n = {len(height)}")
    print(f"Mean: {height.mean():.2f} m")
    print(f"Std Dev: {height.std():.2f} m")
    print(f"Range: {height.min():.2f} - {height.max():.2f} m")
    print(f"Median: {height.median():.2f} m")
    print(f"25th percentile: {height.quantile(0.25):.2f} m")
    print(f"75th percentile: {height.quantile(0.75):.2f} m")
    
    print(f"\n=== WALL THICKNESS (assumed from NPS standards) ===")
    print(f"Typical thickness: {typical_thickness_m:.2f} m (~24 inches)")
    print(f"Range: 0.46-0.61 m (18-24 inches)")
    
    print(f"\n=== HEIGHT-TO-THICKNESS RATIOS ===")
    print(f"Mean h/t: {ht_ratios.mean():.2f}")
    print(f"Std Dev: {ht_ratios.std():.2f}")
    print(f"Range: {ht_ratios.min():.2f} - {ht_ratios.max():.2f}")
    print(f"Median: {ht_ratios.median():.2f}")
    
    # Check against critical thresholds
    critical_threshold = 10
    walls_exceeding = (ht_ratios > critical_threshold).sum()
    
    print(f"\nCritical slenderness threshold: {critical_threshold}")
    print(f"Walls exceeding threshold: {walls_exceeding} ({walls_exceeding/len(ht_ratios)*100:.1f}%)")
    print(f"Maximum h/t: {ht_ratios.max():.2f} (Wall height: {height.max():.2f} m)")
    
    if 'Foundation Height' in df.columns:
        fnd_height = df['Foundation Height'].dropna()
        print(f"\n=== FOUNDATION EXPOSURE (cm) ===")
        print(f"Mean: {fnd_height.mean():.1f} cm")
        print(f"Std Dev: {fnd_height.std():.1f} cm")
        print(f"Range: {fnd_height.min():.1f} - {fnd_height.max():.1f} cm")
        print(f"Walls with >10cm exposure: {(fnd_height > 10).sum()} ({(fnd_height > 10).sum()/len(fnd_height)*100:.1f}%)")
    
    return {
        'height_mean_m': float(height.mean()),
        'height_std_m': float(height.std()),
        'height_min_m': float(height.min()),
        'height_max_m': float(height.max()),
        'height_range': f"{height.min():.2f}-{height.max():.2f}",
        'thickness_assumed_m': typical_thickness_m,
        'ht_ratio_mean': float(ht_ratios.mean()),
        'ht_ratio_std': float(ht_ratios.std()),
        'ht_ratio_max': float(ht_ratios.max()),
        'ht_ratio_range': f"{ht_ratios.min():.2f}-{ht_ratios.max():.2f}",
        'walls_exceeding_critical': int(walls_exceeding),
        'foundation_height_mean_cm': float(fnd_height.mean()) if 'Foundation Height' in df.columns else None
    }

# Import the functions from the previous scripts
# (Factor Analysis, RF, etc. - same as before)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RE-ANALYSIS WITH ACTUAL FIELD DATA")
    print("="*70)
    
    # Load actual field data
    df = load_actual_field_data('defunct/2023_12_8_targeted_eval.csv')
    
    # Get wall dimensions
    wall_dims = get_wall_dimensions(df)
    
    # Save results
    output = {
        'data_source': 'defunct/2023_12_8_targeted_eval.csv',
        'note': 'ACTUAL FIELD DATA (not synthetic)',
        'wall_dimensions': wall_dims
    }
    
    with open('journalPaper/actual_field_data_dimensions.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ Wall dimension statistics saved")
    print("="*70)
    
    print("\n\nNEXT STEPS:")
    print("1. Re-run Factor Analysis with actual data")
    print("2. Re-run Random Forest with actual data")
    print("3. Update all figures")
    print("4. Update manuscript with actual statistics")
