from pypdf import PdfReader

def read_pdf_intro():
    try:
        reader = PdfReader("FOUN phase 4 Final Report 03242020.pdf")
        print(f"Total Pages: {len(reader.pages)}")
        
        # Read first 10 pages for definitions
        text = ""
        for i in range(min(10, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"
            
        print("--- EXTRACTED TEXT (First 10 Pages) ---")
        print(text[:2000]) # Print first 2000 chars
        
        # Search for specific keywords
        keywords = ["wall section", "naming convention", "identifier", " A ", " B ", "segment"]
        print("\n--- KEYWORD SEARCH ---")
        for k in keywords:
            if k in text:
                print(f"Found '{k}' in text.")
                # Print context
                idx = text.find(k)
                print(text[max(0, idx-100):min(len(text), idx+100)])
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_pdf_intro()
