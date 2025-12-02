import pandas as pd
import os

def check_missing():
    # 1. Get Expected IDs from CSV
    try:
        df = pd.read_csv('defunct/2023_12_8_targeted_eval.csv')
        expected_ids = set(df.iloc[:, 0].astype(str).str.strip())
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Get Found IDs from wesResults (and map 36->57)
    found_ids = set()
    try:
        files = os.listdir('wesResults')
        for f in files:
            if f.endswith('.pdf'):
                # Remove extension
                name = os.path.splitext(f)[0]
                # Handle (2), (3) etc if present? User has 3610A(2).pdf
                # Assuming these are duplicates or versions, we just take the base ID?
                # Or maybe they are different walls?
                # Let's clean the name: remove (x)
                if '(' in name:
                    name = name.split('(')[0]
                
                # Map 36 -> 57
                if name.startswith('36'):
                    mapped_name = '57' + name[2:]
                    found_ids.add(mapped_name)
                else:
                    found_ids.add(name) # Keep as is if not 36
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    # 3. Compare
    missing = sorted(list(expected_ids - found_ids))
    
    print(f"Total Expected Walls: {len(expected_ids)}")
    print(f"Total Found Walls (Mapped): {len(found_ids)}")
    print(f"Missing Walls ({len(missing)}):")
    for m in missing:
        print(m)

if __name__ == "__main__":
    check_missing()
