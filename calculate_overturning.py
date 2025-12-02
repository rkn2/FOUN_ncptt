import pandas as pd
import numpy as np

def calculate_overturning():
    """
    Calculate overturning stability for adobe walls under wind loading.
    Uses simplified rigid body mechanics for freestanding walls.
    """
    # 1. Load Geometry
    try:
        df = pd.read_csv('wall_geometry_from_dxf.csv')
    except FileNotFoundError:
        print("Error: wall_geometry_from_dxf.csv not found.")
        return

    print(f"Loaded {len(df)} walls with geometry.")
    
    # 2. Parameters
    wind_psf = 25.0         # Wind pressure (psf) - typical for 50 mph wind per ASCE 7
    rho_pcf = 100.0         # Adobe density (lb/ft^3)
    cap_loss_pct = 0.30     # Assumed cap deterioration (30% width loss in severe cases)
    
    # 3. Calculations
    results = []
    
    for _, row in df.iterrows():
        wall_id = row['WallID']
        h_in = row['Height_in']
        w_in = row['Max_Width_in']  # Use max width (base thickness)
        
        if h_in <= 0 or w_in <= 0:
            continue
            
        # Convert to feet
        H = h_in / 12.0  # Wall height (ft)
        W = w_in / 12.0  # Wall thickness/width (ft)
        L = 1.0          # Analyze a 1-ft length strip
        
        # Self-weight
        Volume_ft3 = H * W * L
        Weight_lbs = Volume_ft3 * rho_pcf
        
        # Overturning Moment (wind pressure on wall face)
        # M_o = 0.5 * q * H^2 * L (triangular pressure distribution)
        M_overturn = 0.5 * wind_psf * (H**2) * L
        
        # Restoring Moment (self-weight about toe)
        # M_r = Weight * (W/2)
        M_restoring = Weight_lbs * (W / 2.0)
        
        # Factor of Safety (intact condition)
        FS_intact = M_restoring / M_overturn if M_overturn > 0 else 999
        
        # Factor of Safety with cap deterioration (30% width loss)
        W_degraded = W * (1 - cap_loss_pct)
        M_restoring_degraded = Weight_lbs * (W_degraded / 2.0)
        FS_degraded = M_restoring_degraded / M_overturn if M_overturn > 0 else 999
        
        # Percent reduction in FS
        FS_reduction_pct = ((FS_intact - FS_degraded) / FS_intact) * 100 if FS_intact > 0 else 0
        
        results.append({
            'WallID': wall_id,
            'Height_ft': H,
            'Width_ft': W,
            'Slenderness': H/W,
            'Weight_lbs': Weight_lbs,
            'M_overturn': M_overturn,
            'M_restoring': M_restoring,
            'FS_intact': FS_intact,
            'FS_degraded': FS_degraded,
            'FS_reduction_pct': FS_reduction_pct
        })
        
    res_df = pd.DataFrame(results)
    
    # 4. Summary Report
    print("\n--- Overturning Stability Analysis ---")
    print(f"Wind Pressure: {wind_psf} psf (typical 50 mph design wind)")
    print(f"Cap Deterioration Assumption: {cap_loss_pct*100:.0f}% width loss")
    
    print("\n--- Intact Condition ---")
    print(f"Min FS: {res_df['FS_intact'].min():.2f}")
    print(f"Max FS: {res_df['FS_intact'].max():.2f}")
    print(f"Mean FS: {res_df['FS_intact'].mean():.2f}")
    print(f"Walls with FS < 1.5 (Marginal): {len(res_df[res_df['FS_intact'] < 1.5])}")
    print(f"Walls with FS < 1.0 (Failure): {len(res_df[res_df['FS_intact'] < 1.0])}")
    print(f"Percent with FS > 1.5: {(len(res_df[res_df['FS_intact'] >= 1.5]) / len(res_df) * 100):.1f}%")
    
    print("\n--- Degraded Condition (30% Cap Loss) ---")
    print(f"Min FS: {res_df['FS_degraded'].min():.2f}")
    print(f"Mean FS: {res_df['FS_degraded'].mean():.2f}")
    print(f"Walls with FS < 1.5: {len(res_df[res_df['FS_degraded'] < 1.5])}")
    print(f"Mean FS Reduction: {res_df['FS_reduction_pct'].mean():.1f}%")
    
    # Save results
    res_df.to_csv('overturning_analysis_results.csv', index=False)
    print("\nDetailed results saved to overturning_analysis_results.csv")
    
    return res_df

if __name__ == "__main__":
    calculate_overturning()
