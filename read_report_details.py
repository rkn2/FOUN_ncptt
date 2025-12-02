from pypdf import PdfReader

def read_report_details():
    try:
        reader = PdfReader("FOUN phase 4 Final Report 03242020.pdf")
        
        # 1. Recommendations (Page 31-33)
        print("--- RECOMMENDATIONS (Pages 31-33) ---")
        for i in range(31, 34):
            if i < len(reader.pages):
                print(f"Page {i+1}:")
                print(reader.pages[i].extract_text())

        # 2. Embedded Monitoring Results (Page 31 is listed in TOC, let's check 26-31)
        print("\n--- EMBEDDED MONITORING (Pages 26-28) ---")
        for i in range(26, 29):
            if i < len(reader.pages):
                print(f"Page {i+1}:")
                print(reader.pages[i].extract_text())

        # 3. Glossary for "Height" (Appendix B starts Page 39)
        print("\n--- GLOSSARY (Pages 39-42) ---")
        for i in range(39, 43):
            if i < len(reader.pages):
                print(f"Page {i+1}:")
                print(reader.pages[i].extract_text())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_report_details()
