"""
Synthetic Data Generator for Adobe Structure Degradation Analysis (UPDATED)
==================================================================

This script generates synthetic data that preserves the statistical properties
and correlations observed in real adobe structure degradation data from Fort Union.

Updated to match actual correlation structure from Elastic Net analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_adobe_data(n_samples=67):
    """
    Generate synthetic adobe wall degradation data matching real FOUN patterns.
    """
    
    data = {}
    
    # Generate Wall IDs
    data['Wall ID'] = [f'Wall_{i:03d}' for i in range(1, n_samples + 1)]
    
    # --- GEOMETRIC FEATURES (Primary Drivers) ---
    # Wall Height: 1-5 scale (higher number = taller wall, more vulnerable)
    data['Height'] = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    
    # Foundation Height: Exposed foundation (continuous, inches)
    data['Foundation Height'] = np.random.gamma(shape=2.5, scale=5, size=n_samples)
    data['Foundation Height'] = np.clip(data['Foundation Height'], 0, 40)
    
    # --- CAP DETERIORATION (Strongest predictor, coef=54.5) ---
    # Highly correlated with Total Score (r=0.61 in real data)
    cap_base = 0.8 * data['Height'] + np.random.normal(0, 1.2, n_samples)
    data['Cap Deterioration'] = np.clip(cap_base, 0, 5).astype(int)
    
    # --- OUT OF PLANE (Second strongest, coef=43.3) ---
    # Correlated with Height and Cap
    oop_base = 0.6 * data['Height'] + 0.3 * data['Cap Deterioration'] + np.random.normal(0, 1.0, n_samples)
    data['Out of Plane'] = np.clip(oop_base, 0, 5).astype(int)
    
    # --- STRUCTURAL CRACKING (coef=30.8) ---
    # Correlated with Out of Plane and Cap
    crack_base = 0.4 * data['Out of Plane'] + 0.3 * data['Cap Deterioration'] + np.random.normal(0, 1.0, n_samples)
    data['Structural Cracking'] = np.clip(crack_base, 0, 5).astype(int)
    
    # --- CRACKING AT WALL JUNCTION (coef=27.7) ---
    junction_base = 0.3 * data['Structural Cracking'] + np.random.poisson(1.5, n_samples)
    data['Cracking Junction'] = np.clip(junction_base, 0, 5).astype(int)
    
    # --- SILL FEATURES (Factor 1 - Independent) ---
    # High correlation between Sill 1 and Sill 2
    sill_base = np.random.poisson(lam=2, size=n_samples)
    data['Sill 1'] = np.clip(sill_base, 0, 5).astype(int)
    data['Sill 2'] = np.clip(sill_base + np.random.normal(0, 0.3, n_samples), 0, 5).astype(int)
    
    # --- SURFACE COAT FEATURES (Factor 2) ---
    coat_base = np.random.poisson(lam=2, size=n_samples)
    
    data['Coat 1 Cracking'] = np.clip(coat_base + np.random.normal(0, 1, n_samples), 0, 5).astype(int)
    data['Coat 1 Loss'] = np.clip(coat_base * 0.6 + np.random.poisson(1, n_samples), 0, 5).astype(int)
    data['Coat 2 Cracking'] = np.clip(coat_base + np.random.normal(0, 1.2, n_samples), 0, 5).astype(int)
    data['Coat 2 Loss'] = np.clip(coat_base * 0.5 + np.random.poisson(1, n_samples), 0, 5).astype(int)
    
    # Lintel Deterioration (coupled with coat cracking)
    data['Lintel Deterioration'] = np.clip(
        0.6 * data['Coat 1 Cracking'] + np.random.poisson(1, n_samples), 
        0, 5
    ).astype(int)
    
    # --- SURFACE LOSS (removed from final model but kept for completeness) ---
    data['Surface Loss Top'] = np.random.poisson(lam=2.5, size=n_samples)
    data['Surface Loss Top'] = np.clip(data['Surface Loss Top'], 0, 5)
    
    data['Surface Loss Mid'] = np.random.poisson(lam=1.5, size=n_samples)
    data['Surface Loss Mid'] = np.clip(data['Surface Loss Mid'], 0, 5)
    
    data['Surface Loss Low'] = np.random.poisson(lam=2, size=n_samples)
    data['Surface Loss Low'] = np.clip(data['Surface Loss Low'], 0, 5)
    
    # --- FOUNDATION FEATURES ---
    foundation_effect = 0.05 * data['Foundation Height']
    
    data['Foundation Displacement 1'] = np.clip(
        np.random.poisson(1, n_samples) + foundation_effect, 0, 5
    ).astype(int)
    data['Foundation Displacement 2'] = np.clip(
        np.random.poisson(1, n_samples) + foundation_effect * 0.9, 0, 5
    ).astype(int)
    
    data['Foundation Mortar 1'] = np.random.poisson(lam=2, size=n_samples)
    data['Foundation Mortar 1'] = np.clip(data['Foundation Mortar 1'], 0, 5)
    data['Foundation Mortar 2'] = np.clip(
        data['Foundation Mortar 1'] + np.random.normal(0, 0.5, n_samples), 0, 5
    ).astype(int)
    
    data['Foundation Stone Det'] = np.clip(
        np.random.poisson(2, n_samples) + 0.03 * data['Foundation Height'], 0, 5
    ).astype(int)
    
    # --- TREATMENT HISTORY (Binary) ---
    treatment_prob = 0.3 + 0.01 * (data['Structural Cracking'] + data['Out of Plane'])
    data['Treatment'] = (np.random.random(n_samples) < treatment_prob).astype(int)
    
    bracing_prob = 0.2 + 0.05 * (data['Out of Plane'] + data['Structural Cracking'])
    data['Bracing'] = (np.random.random(n_samples) < bracing_prob).astype(int)
    
    data['Bracing Score'] = np.where(
        data['Bracing'] == 1,
        np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        0
    )
    
    # --- OTHER FEATURES ---
    data['Animal Activity'] = np.random.poisson(lam=0.5, size=n_samples)
    data['Animal Activity'] = np.clip(data['Animal Activity'], 0, 3)
    
    data['Fireplace'] = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    
    # Point Cloud Metrics
    data['Point Cloud Mean'] = np.random.normal(0, 2.5, n_samples)
    data['Point Cloud Deviation'] = np.abs(np.random.exponential(1.5, n_samples))
    
    # --- CALCULATE TOTAL DEGRADATION SCORE ---
    # Based on Elastic Net coefficients but with added noise to match real correlations (r~0.5-0.6)
    total_score = (
        data['Cap Deterioration'] * 3.5 +      # Coef 54.5 (strongest)
        data['Out of Plane'] * 3.0 +           # Coef 43.3
        data['Height'] * 2.5 +                 # Coef 43.0
        data['Structural Cracking'] * 2.0 +    # Coef 30.8
        data['Cracking Junction'] * 1.5 +      # Coef 27.7
        data['Coat 1 Cracking'] * 1.0 +
        data['Coat 2 Cracking'] * 1.0 +
        data['Sill 1'] * 0.8 +
        data['Sill 2'] * 0.8 +
        data['Lintel Deterioration'] * 1.2 +
        data['Foundation Height'] * 0.08 +
        data['Foundation Stone Det'] * 1.0 +
        np.random.normal(0, 12, n_samples)  # Increased noise to match real correlations (r~0.55)
    )
    
    # Scale to 0-100 range
    data['Total Scr'] = np.clip(total_score, 0, 100)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate synthetic dataset
    synthetic_df = generate_synthetic_adobe_data(n_samples=67)
    
    # Save to file
    synthetic_df.to_csv('synthetic_adobe_data.csv', index=False)
    print(f"Synthetic data saved to synthetic_adobe_data.csv")
    print(f"Shape: {synthetic_df.shape}")
    
    # Verify key correlations
    print("\n" + "="*50)
    print("VALIDATION: Key Correlations")
    print("="*50)
    print(f"Total Scr vs Cap Deterioration: {synthetic_df['Total Scr'].corr(synthetic_df['Cap Deterioration']):.3f}")
    print(f"Total Scr vs Out of Plane: {synthetic_df['Total Scr'].corr(synthetic_df['Out of Plane']):.3f}")
    print(f"Total Scr vs Height: {synthetic_df['Total Scr'].corr(synthetic_df['Height']):.3f}")
    print(f"Total Scr vs Structural Cracking: {synthetic_df['Total Scr'].corr(synthetic_df['Structural Cracking']):.3f}")
    print(f"Sill 1 vs Sill 2: {synthetic_df['Sill 1'].corr(synthetic_df['Sill 2']):.3f}")
