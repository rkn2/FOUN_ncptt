import pandas as pd
try:
    df = pd.read_csv('defunct/2023_12_8_targeted_eval.csv')
    # First column is likely the ID
    ids = df.iloc[:, 0].unique()
    print(f"Total IDs in CSV: {len(ids)}")
    print("First 10 IDs:")
    print(ids[:10])
    
    # Check for 36xx IDs
    ids_36 = [i for i in ids if str(i).startswith('36')]
    print(f"IDs starting with '36': {len(ids_36)}")
    if ids_36:
        print(ids_36[:10])
except Exception as e:
    print(e)
