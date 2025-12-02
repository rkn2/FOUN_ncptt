import pandas as pd

def compare():
    # 1. Expected (Original CSV)
    df_orig = pd.read_csv('defunct/2023_12_8_targeted_eval.csv')
    expected = set(df_orig.iloc[:, 0].astype(str).str.strip())
    
    # 2. Found (DXF CSV)
    df_found = pd.read_csv('wall_geometry_from_dxf.csv')
    found = set(df_found['WallID'].astype(str).str.strip())
    
    # 3. Compare
    missing = sorted(list(expected - found))
    extra = sorted(list(found - expected))
    common = sorted(list(expected.intersection(found)))
    
    print(f"Expected (CSV): {len(expected)}")
    print(f"Found (DXF): {len(found)}")
    print(f"Common: {len(common)}")
    print(f"Missing from DXF ({len(missing)}): {missing}")
    print(f"Extra in DXF ({len(extra)}): {extra}")

if __name__ == "__main__":
    compare()
