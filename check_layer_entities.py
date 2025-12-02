import ezdxf

def check_layer(filename, layer_name):
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    
    layer_counts = {}
    for e in msp:
        layer = e.dxf.layer
        etype = e.dxftype()
        if layer not in layer_counts: layer_counts[layer] = {}
        layer_counts[layer][etype] = layer_counts[layer].get(etype, 0) + 1
            
    for layer, counts in layer_counts.items():
        print(f"\nLayer '{layer}':")
        for etype, count in counts.items():
            print(f"  {etype}: {count}")

if __name__ == "__main__":
    check_layer("wesResults/XYZBLKS-1210.dxf", "")
