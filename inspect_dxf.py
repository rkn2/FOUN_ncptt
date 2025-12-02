import ezdxf
import sys

def inspect_dxf(filename):
    try:
        doc = ezdxf.readfile(filename)
        msp = doc.modelspace()
        
        print(f"DXF Version: {doc.dxfversion}")
        
        # 1. List Layers
        layers = [layer.dxf.name for layer in doc.layers]
        print(f"\nTotal Layers: {len(layers)}")
        print("Layers:", layers)
        
        # 2. Count Entities by Type
        entity_counts = {}
        for e in msp:
            etype = e.dxftype()
            entity_counts[etype] = entity_counts.get(etype, 0) + 1
            
        print("\nEntity Counts:")
        for etype, count in entity_counts.items():
            print(f"{etype}: {count}")
            
        # 3. Check for Text (might contain Wall IDs)
        print("\nText Entities (First 20):")
        text_count = 0
        for e in msp.query('TEXT MTEXT'):
            if text_count < 20:
                text = e.dxf.text if e.dxftype() == 'TEXT' else e.text
                print(f"  {e.dxftype()}: {text}")
                text_count += 1
            else:
                break
                
        # 4. Check for Block References (INSERT)
        inserts = msp.query('INSERT')
        print(f"\nBlock References (INSERT): {len(inserts)}")
        if len(inserts) > 0:
            block_names = set(e.dxf.name for e in inserts)
            print(f"Unique Block Names: {block_names}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_dxf("wesResults/XYZBLKS-1210.dxf")
