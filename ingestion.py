import csv
import os
import sys
import zipfile
from typing import List

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".csv")


class DocumentParseError(ValueError):
    """Raised when a document is corrupt, encrypted, or otherwise unreadable."""


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from every page of a PDF. Returns "" for image-only PDFs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            # Many "encrypted" PDFs only carry an owner password; try empty user password.
            try:
                reader.decrypt("")
            except Exception as e:
                raise DocumentParseError("PDF is password-protected.") from e
        pages = reader.pages
    except DocumentParseError:
        raise
    except (PdfReadError, OSError, ValueError) as e:
        raise DocumentParseError(f"PDF is corrupt or unreadable: {e}") from e

    extracted_pages = []
    for page in pages:
        try:
            page_text = page.extract_text()
        except Exception:
            # A single malformed page should not abort the whole document.
            continue
        if page_text and page_text.strip():
            extracted_pages.append(page_text.strip())

    return "\n\n".join(extracted_pages)


def extract_text_from_docx(file_path: str) -> str:
    """Extract paragraph and table text from a .docx file, in document order."""
    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DOCX file not found at path: {file_path}")

    try:
        document = Document(file_path)
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as e:
        raise DocumentParseError(f"DOCX is corrupt or not a Word document: {e}") from e

    blocks: List[str] = []
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(element, document).text.strip()
            if text:
                blocks.append(text)
        elif tag == "tbl":
            for row in Table(element, document).rows:
                cells: List[str] = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    # Merged cells repeat the same object across the row.
                    if cell_text and (not cells or cells[-1] != cell_text):
                        cells.append(cell_text)
                if cells:
                    blocks.append(" | ".join(cells))

    return "\n".join(blocks)


def _dataframe_to_text(df, title: str = "") -> str:
    """
    Serialise a DataFrame row-by-row as "column: value" pairs.

    DataFrame.to_string() pads every column to a fixed width (wasting chunk
    space on whitespace) and its repr is truncated by pandas display options,
    so rows are written out explicitly instead. Empty / NaN cells are dropped.
    """
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return ""

    columns = [str(c).strip() for c in df.columns]
    lines: List[str] = [f"## {title}"] if title else []
    for row in df.itertuples(index=False, name=None):
        pairs = []
        for col, value in zip(columns, row):
            if value is None:
                continue
            text = str(value).strip()
            if not text or text.lower() in ("nan", "nat", "none"):
                continue
            pairs.append(f"{col}: {text}" if col and not col.startswith("Unnamed") else text)
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def extract_text_from_excel(file_path: str) -> str:
    """Extract all sheets of an .xlsx workbook as text."""
    import pandas as pd

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found at path: {file_path}")

    try:
        # dtype=object keeps values as-is (no float coercion of ID columns).
        sheets = pd.read_excel(file_path, sheet_name=None, dtype=object, engine="openpyxl")
    except (zipfile.BadZipFile, ValueError, KeyError, OSError) as e:
        raise DocumentParseError(f"Excel file is corrupt or unreadable: {e}") from e

    parts = [_dataframe_to_text(df, title=f"Sheet: {name}") for name, df in sheets.items()]
    return "\n\n".join(p for p in parts if p)


def extract_text_from_csv(file_path: str) -> str:
    """Extract text from a CSV file, tolerating common encodings and bad rows."""
    import pandas as pd

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at path: {file_path}")

    last_error = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding, newline="") as f:
                sample = f.read(64 * 1024)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                # Single-column files (or ambiguous samples) have no detectable delimiter.
                delimiter = ","
            df = pd.read_csv(
                file_path,
                dtype=object,
                encoding=encoding,
                sep=delimiter,
                on_bad_lines="skip",
            )
            return _dataframe_to_text(df)
        except UnicodeDecodeError as e:
            last_error = e
        except pd.errors.EmptyDataError:
            return ""
        except (pd.errors.ParserError, ValueError, OSError) as e:
            raise DocumentParseError(f"CSV is malformed or unreadable: {e}") from e

    raise DocumentParseError(f"Could not decode CSV file: {last_error}")


def extract_text(file_path: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    if ext == ".xlsx":
        return extract_text_from_excel(file_path)
    if ext == ".csv":
        return extract_text_from_csv(file_path)
    raise DocumentParseError(
        f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


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
        # Skip whitespace-only chunks: they carry no meaning and embed to
        # near-degenerate vectors that pollute retrieval.
        if chunk.strip():
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
        raw_text = extract_text(pdf_path)
    except FileNotFoundError:
        print(f"[ERROR] Target file '{pdf_path}' does not exist.")
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
