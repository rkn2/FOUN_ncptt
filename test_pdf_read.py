import os
try:
    from pypdf import PdfReader
    reader = PdfReader("wesResults/3610A.pdf")
    page = reader.pages[0]
    text = page.extract_text()
    print("--- TEXT CONTENT ---")
    print(text)
    print("--------------------")
except ImportError:
    print("pypdf not installed.")
except Exception as e:
    print(f"Error: {e}")
