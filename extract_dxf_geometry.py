import ezdxf
import pandas as pd
import numpy as np

def get_intersections(poly_points, y):
    # poly_points is list of (x, y)
    intersections = []
    for i in range(len(poly_points)):
        p1 = poly_points[i]
        p2 = poly_points[(i + 1) % len(poly_points)]
        
        if (p1[1] > y) != (p2[1] > y):
            x = (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1]) + p1[0]
            intersections.append(x)
    return sorted(intersections)

def extract_geometry():
    filename = "wesResults/XYZBLKS-1210.dxf"
    try:
        doc = ezdxf.readfile(filename)
        msp = doc.modelspace()
    except Exception as e:
        print(f"Error reading DXF: {e}")
        return

    # 1. Get MTEXTs (IDs)
    texts = []
    for e in msp.query('MTEXT'):
        if e.dxf.layer == '01_Code':
            text = e.text
            # Clean text (remove formatting like \P)
            if '\\P' in text:
                text = text.split('\\P')[0]
            # Filter for valid IDs (e.g., 36xx)
            if text.startswith('36'):
                texts.append({'id': text, 'x': e.dxf.insert.x, 'y': e.dxf.insert.y})
    
    print(f"Found {len(texts)} Wall IDs in DXF.")

    # 2. Get Polylines
    polys = []
    for e in msp.query('LWPOLYLINE'):
        if e.dxf.layer == '01_Code':
            points = list(e.get_points()) # (x, y, ...)
            # Convert to (x,y)
            pts = [(p[0], p[1]) for p in points]
            if not pts: continue
            
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            center_x = (min_x + max_x) / 2
            
            polys.append({
                'pts': pts,
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'center_x': center_x,
                'width': max_x - min_x,
                'height': max_y - min_y
            })
            
    print(f"Found {len(polys)} Polylines.")

    # 3. Match Polys to Texts
    results = []
    
    for t in texts:
        # Find polys close in X (e.g., within +/- 40 units)
        # And above in Y (poly Y > text Y)
        candidates = []
        for p in polys:
            if abs(p['center_x'] - t['x']) < 50 and p['min_y'] > t['y']:
                candidates.append(p)
        
        if t['id'] == '3601A':
            print(f"Candidates for 3601A: {len(candidates)}")
            for c in candidates:
                print(f"  Poly: W={c['width']:.2f}, H={c['height']:.2f}, CenterX={c['center_x']:.2f}")
        
        if not candidates:
            # print(f"No poly found for {t['id']}")
            continue
            
        # Pick the WIDEST candidate (assuming it's the        # Filter out small/short objects (e.g. labels)
        valid_candidates = [c for c in candidates if c['height'] > 50]
        
        if not valid_candidates:
            # Fallback if no tall objects found
            valid_candidates = candidates
            
        # Pick the WIDEST candidate
        best_poly = max(valid_candidates, key=lambda x: x['width'])
        
        # Calculate Min/Max Width
        # Scan Y from min_y + epsilon to max_y - epsilon
        y_steps = np.linspace(best_poly['min_y'] + 1, best_poly['max_y'] - 1, 100)
        widths = []
        for y in y_steps:
            x_ints = get_intersections(best_poly['pts'], y)
            if len(x_ints) >= 2:
                w = x_ints[-1] - x_ints[0]
                widths.append(w)
        
        if widths:
            avg_width = np.mean(widths)
            min_width = np.min(widths)
            max_width = best_poly['width'] # Use bbox width for robustness
        else:
            avg_width = best_poly['width']
            min_width = best_poly['width']
            max_width = best_poly['width']

        # Map ID 36xx -> 57xx
        mapped_id = '57' + t['id'][2:]
        
        results.append({
            'WallID': mapped_id,
            'OriginalID': t['id'],
            'Height_in': best_poly['height'],
            'Max_Width_in': max_width,
            'Min_Width_in': min_width,
            'Avg_Width_in': avg_width
        })

    # 4. Save to CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv('wall_geometry_from_dxf.csv', index=False)
    print(f"Saved geometry for {len(df_res)} walls to wall_geometry_from_dxf.csv")
    
    # 5. Check Missing
    try:
        orig_df = pd.read_csv('defunct/2023_12_8_targeted_eval.csv')
        expected_ids = set(orig_df.iloc[:, 0].astype(str).str.strip())
        found_ids = set(df_res['WallID'])
        
        missing = sorted(list(expected_ids - found_ids))
        print(f"\nMissing Walls ({len(missing)}):")
        for m in missing:
            print(m)
            
    except Exception as e:
        print(f"Error checking missing: {e}")

if __name__ == "__main__":
    extract_geometry()
