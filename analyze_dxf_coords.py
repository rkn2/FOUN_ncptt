import ezdxf

def analyze_coords(filename):
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    
    print("--- MTEXT (ID, Location) ---")
    texts = []
    for e in msp.query('MTEXT'):
        # MTEXT location is insert point
        loc = e.dxf.insert
        text = e.text
        texts.append((text, loc))
        if len(texts) <= 10:
            print(f"ID: {text}, Loc: {loc}")
            
    print("\n--- LWPOLYLINE (Bounding Box) ---")
    polys = []
    for i, e in enumerate(msp.query('LWPOLYLINE')):
        # Get bounding box
        layer = e.dxf.layer
            
        points = e.get_points() # list of (x, y, start_width, end_width, bulge)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        if not xs: continue
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center = ((min_x + max_x)/2, (min_y + max_y)/2)
        
        if layer != '01_Elevation': continue
        
        polys.append({'id': i, 'layer': layer, 'bbox': (min_x, min_y, max_x, max_y), 'center': center})
        
        if len(polys) <= 10:
            print(f"Poly {i} (Layer: {layer}): Center: {center}, Size: ({max_x-min_x:.2f}, {max_y-min_y:.2f})")

if __name__ == "__main__":
    analyze_coords("wesResults/XYZBLKS-1210.dxf")
