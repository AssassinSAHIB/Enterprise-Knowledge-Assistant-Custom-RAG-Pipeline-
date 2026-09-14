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


def custom_text_splitter(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
   
    if not text:
        return []
        
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
        
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size.")
        
    chunks: List[str] = []
    start: int = 0
    text_length: int = len(text)
    step: int = chunk_size - chunk_overlap
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_length:
            break
            
        start += step
        
    return chunks
