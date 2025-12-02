from pypdf import PdfReader

def read_pdf_sections():
    try:
        reader = PdfReader("FOUN phase 4 Final Report 03242020.pdf")
        
        # Glossary is around page 35 (0-indexed: 34)
        print("--- GLOSSARY (Pages 34-36) ---")
        for i in range(34, 37):
            if i < len(reader.pages):
                print(f"Page {i+1}:")
                print(reader.pages[i].extract_text())
                
        # Appendix G/H starts around page 59 (0-indexed: 58)
        print("\n--- APPENDIX G/H (Pages 58-65) ---")
        for i in range(58, 66):
            if i < len(reader.pages):
                print(f"Page {i+1}:")
                print(reader.pages[i].extract_text())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_pdf_sections()
