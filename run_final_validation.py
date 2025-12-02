import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# Configuration
INPUT_FILE = 'defunct/2023_12_8_targeted_eval.csv'
GEOM_FILE = 'wall_geometry_from_dxf.csv'
OUTPUT_DIR = 'journalPaper/Images'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_validation():
    # 1. Load and Clean Original Data
    print("Loading and cleaning data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Drop Point Cloud columns
    drop_cols = ['PC mean', 'PC deviation']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Rename Height -> Height Loss Score
    if 'Height' in df_clean.columns:
        df_clean = df_clean.rename(columns={'Height': 'Height Loss Score'})
        
    # Also handle the ID column (first column)
    df_clean = df_clean.rename(columns={df_clean.columns[0]: 'WallID'})
    df_clean['WallID'] = df_clean['WallID'].astype(str).str.strip()
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    
    # 2. Load Geometric Data
    print("\nLoading geometric data...")
    df_geom = pd.read_csv(GEOM_FILE)
    df_geom['WallID'] = df_geom['WallID'].astype(str).str.strip()
    
    # 3. Merge
    # Inner join to get only the 36 overlapping walls
    df_merged = pd.merge(df_clean, df_geom, on='WallID', how='inner')
    print(f"Merged dataset shape (N): {df_merged.shape[0]}")
    
    if df_merged.empty:
        print("Error: No overlapping walls found!")
        return

    # 4. Calculate Slenderness
    # Slenderness = Height / Width
    # We'll use Avg_Width_in for a representative measure
    df_merged['Slenderness Ratio'] = df_merged['Height_in'] / df_merged['Avg_Width_in']
    
    # 5. Correlation Analysis
    # Correlate Slenderness with Total Score
    # And maybe with "Structural Cracking" or "Out of Plane"
    
    targets = ['Total Scr', 'Structural Cracking', 'Out of Plane', 'Height Loss Score']
    
    print("\n--- Geometric Validation Results (N=36) ---")
    print(f"{'Metric':<25} | {'Correlation (r)':<15} | {'P-Value':<10}")
    print("-" * 60)
    
    stats = []
    for target in targets:
        if target in df_merged.columns:
            r, p = pearsonr(df_merged['Slenderness Ratio'], df_merged[target])
            print(f"{target:<25} | {r:.3f}           | {p:.3f}")
            stats.append((target, r, p))
            
    # 6. Generate Plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x='Slenderness Ratio', y='Total Scr', data=df_merged, 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    
    r_val = stats[0][1] # Total Scr r
    p_val = stats[0][2]
    
    plt.title(f'Geometric Validation (N={len(df_merged)})\nSlenderness vs. Total Degradation (r={r_val:.2f}, p={p_val:.2f})')
    plt.xlabel('Slenderness Ratio (Height / Avg Width)')
    plt.ylabel('Total Degradation Score')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, 'geometric_validation_subset.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved validation plot to {plot_path}")
    
    # 7. Save Cleaned Data for future use
    df_clean.to_csv('cleaned_data_final.csv', index=False)
    print("Saved cleaned dataset to 'cleaned_data_final.csv'")

if __name__ == "__main__":
    run_validation()
