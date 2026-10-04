# Enterprise Knowledge Assistant (Custom RAG Pipeline)

A retrieval-augmented question-answering app built from scratch: custom chunking, NumPy cosine-similarity search over `sentence-transformers` embeddings, and direct Groq API calls, with no LangChain. It has a Streamlit UI with Retro Dark and Retro Bright themes.

## Supported documents

| Format | Notes |
|--------|-------|
| `.pdf`  | Text-based PDFs. Scanned or image-only PDFs have no extractable text. |
| `.docx` | Paragraphs and tables, in document order. |
| `.xlsx` | Every sheet, serialised row by row as `column: value` pairs. Empty and NaN cells are skipped. |
| `.csv`  | Encoding and delimiter are auto-detected. Malformed rows are skipped. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your Groq key in .env
streamlit run app.py
```

The API key is read from `GROQ_API_KEY` in `.env` or the environment. You can also paste it into the sidebar, which takes precedence. Never commit a real key.

## Modules

- `ingestion.py`: multi-format text extraction and the character-window text splitter
- `retrieval.py`: embeddings and vectorised cosine-similarity top-k search
- `llm_generator.py`: grounded prompt construction and Groq API calls, with retry on 429 and 5xx errors
- `app.py`: the Streamlit interface
