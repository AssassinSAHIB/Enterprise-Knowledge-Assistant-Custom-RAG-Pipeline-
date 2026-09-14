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
        raise ValueError("Chunk size must be a positive integer.")
        
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non negative and strictly less than chunk size.")
        
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

def run_ingestion_pipeline(
    pdf_path: str = "sample.pdf",
    output_debug_path: str = "debug_chunks.txt",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    print("=" * 60)
    print("WEEK 1: INGESTION PIPELINE EXECUTION")
    print("=" * 60)
    
    # 1. Extract raw text 
    try:
        raw_text = extract_text_from_pdf(pdf_path)
    except FileNotFoundError:
        print(f"[ERROR] Target file '{pdf_path}' does not exist.")
        print("Tip: Run create_sample_pdf.py to create the benchmark fixture.")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to extract text from '{pdf_path}': {e}")
        return []

    raw_char_count = len(raw_text)

    # 2. Chunk text 
    chunks = custom_text_splitter(
        raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    total_chunks = len(chunks)

    # 3. Print metric to console
    print(f"File Processed:                   {pdf_path}")
    print(f"Total Raw Characters Extracted:   {raw_char_count:,}")
    print(f"Chunk Size Configuration:         {chunk_size} characters")
    print(f"Chunk Overlap Configuration:      {chunk_overlap} characters")
    print(f"Total Chunks Generated:           {total_chunks}")
    print("-" * 60)

    # 4. Write all generated chunks
    try:
        with open(output_debug_path, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks):
                f.write("==================================================\n")
                f.write(f"CHUNK {idx} | Length: {len(chunk)}\n")
                f.write("==================================================\n")
                f.write(f"{chunk}\n\n")
        print(f"[SUCCESS] Wrote {total_chunks} chunks to '{output_debug_path}'.")
    except Exception as e:
        print(f"[ERROR] Failed writing to '{output_debug_path}': {e}")

    return chunks


if __name__ == "__main__":
    test_pdf = "sample.pdf"
    run_ingestion_pipeline(pdf_path=test_pdf)
