import os
import sys
from typing import List
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
   
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")
        
    reader = PdfReader(file_path)
    extracted_pages = []
    
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text.strip())
            
    return "\n\n".join(extracted_pages)


