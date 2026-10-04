"""
Week 3 - Direct API LLM Integration & Grounding Validation
Module: llm_generator.py
Enterprise Knowledge Assistant (Custom RAG Pipeline)

Connects the manual retrieval engine to an LLM provider (Groq API) via
direct HTTP POST requests - no LangChain or orchestration framework used.
"""

import os
import sys
import re
import time
import datetime
import requests
from typing import List, Optional, Tuple

try:
# pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional; fall back to real env vars
    load_dotenv = None

# Load variables from a local .env file (if present) without overriding
# variables already exported in the shell.
if load_dotenv is not None:
    load_dotenv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        override=False,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
REQUEST_TIMEOUT_SECONDS = 60  # Hard timeout per API call
MAX_RETRIES = 2               # Extra attempts on 429 / 5xx / timeout
MAX_RETRY_WAIT_SECONDS = 20   # Cap on a single backoff sleep


# Known placeholder values shipped in .env templates — treated as "not set".
_PLACEHOLDER_KEYS = {
    "gsk_your_key_here",
    "your_groq_api_key_here",
    "<your_api_key>",
    "YOUR_KEY_HERE",
}


def get_api_key(explicit_key: Optional[str] = None) -> str:
    """
    Resolve the Groq API key: an explicitly supplied key wins, otherwise
    GROQ_API_KEY from the environment / .env file.
    Returns "" if the key is missing, empty, or still set to a known placeholder.
    """
    key = ""
    if explicit_key and explicit_key.strip():
        key = explicit_key.strip()
    else:
        key = os.environ.get("GROQ_API_KEY", "").strip()

    if key in _PLACEHOLDER_KEYS:
        return ""   # Treat placeholders the same as a missing key
    return key


def _retry_wait(response: Optional[requests.Response], attempt: int) -> float:
    """Seconds to wait before retrying: honour Retry-After, else exponential backoff."""
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), MAX_RETRY_WAIT_SECONDS)
            except ValueError:
                pass
    return min(2.0 ** attempt, MAX_RETRY_WAIT_SECONDS)


def _api_error_detail(response: requests.Response) -> str:
    """Extract Groq's human-readable error message from a failed response."""
    try:
        return response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return response.text[:300].strip()


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_grounded_prompt(user_query: str, retrieved_chunks: List[str]) -> str:
    """
    Constructs a fact-grounded system + user prompt that instructs the LLM to
    answer strictly from the supplied context snippets with inline citations.

    Args:
        user_query (str): The natural-language question posed by the user.
        retrieved_chunks (List[str]): Ordered list of retrieved document chunks
            (most relevant first). Each item will be labelled [Chunk N].

    Returns:
        str: The fully assembled prompt string sent to the LLM.
    """
    if not retrieved_chunks:
        raise ValueError("retrieved_chunks must contain at least one chunk.")

    # Build numbered context block
    context_block_lines = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_block_lines.append(f"[Chunk {idx}]\n{chunk.strip()}")
    context_block = "\n\n".join(context_block_lines)

    prompt = (
        "You are an enterprise AI assistant with strict grounding rules.\n\n"
        "## Context Snippets (Retrieved from internal knowledge base)\n"
        f"{context_block}\n\n"
        "## Instructions\n"
        "- Answer the user's question using ONLY the information present in the "
        "Context Snippets above.\n"
        "- Every factual claim in your answer MUST include an inline citation "
        "referencing the chunk it originates from, e.g. [Chunk 1].\n"
        "- Do NOT use any external knowledge or make assumptions beyond what is "
        "explicitly stated in the context.\n"
        "- If the context does not contain sufficient information to answer the "
        'question, output exactly: "NOT FOUND: The requested information is not '
        'present in the provided context."\n'
        "- Keep the response professional, concise, and enterprise-appropriate.\n\n"
        f"## User Question\n{user_query}"
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM API Call
# ---------------------------------------------------------------------------

def query_llm_api(
    prompt: str,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[str, float]:
    """
    Dispatches a direct HTTP POST request to the Groq Chat Completions endpoint
    and returns the generated response text along with measured latency.

    Args:
        prompt (str): The fully assembled grounded prompt.
        api_key (str): Groq API key for Bearer authentication.
        model_name (str): Groq model identifier. Default: llama-3.3-70b-versatile.

    Returns:
        Tuple[str, float]: (response_text, latency_seconds)

    Raises:
        ValueError: On invalid API key or malformed response.
        requests.exceptions.Timeout: When the API does not respond in time.
        requests.exceptions.HTTPError: On 4xx / 5xx HTTP status codes.
        requests.exceptions.ConnectionError: On network connectivity failure.
    """
    api_key = get_api_key(api_key)
    if not api_key:
        raise ValueError(
            "API key is empty or None. "
            "Set GROQ_API_KEY in your environment / .env file or pass the key explicitly."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 0.9,
        "stream": False,
    }

    t_start = time.perf_counter()
    response: Optional[requests.Response] = None

    for attempt in range(MAX_RETRIES + 1):
        is_last = attempt == MAX_RETRIES
        try:
            response = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.Timeout:
            if not is_last:
                time.sleep(_retry_wait(None, attempt))
                continue
            raise requests.exceptions.Timeout(
                f"[TIMEOUT] Groq API did not respond within {REQUEST_TIMEOUT_SECONDS}s "
                f"after {MAX_RETRIES + 1} attempts. Check your network connection or retry later."
            )
        except requests.exceptions.ConnectionError as conn_err:
            raise requests.exceptions.ConnectionError(
                f"[CONNECTION ERROR] Failed to reach Groq API endpoint: {conn_err}"
            )

        # Transient failures: back off and retry.
        if (response.status_code == 429 or response.status_code >= 500) and not is_last:
            time.sleep(_retry_wait(response, attempt))
            continue
        break

    latency = time.perf_counter() - t_start

    if response.status_code == 401:
        raise ValueError(
            "[AUTH ERROR] Invalid Groq API key (HTTP 401). "
            "Verify your key at https://console.groq.com/keys"
        )
    if response.status_code == 403:
        raise ValueError(
            f"[FORBIDDEN] Groq API refused the request (HTTP 403): {_api_error_detail(response)}"
        )
    if response.status_code == 429:
        raise requests.exceptions.HTTPError(
            "[RATE LIMIT] Groq API rate limit exceeded (HTTP 429) after "
            f"{MAX_RETRIES + 1} attempts. Wait a moment and retry, or upgrade your plan."
        )
    if response.status_code >= 500:
        raise requests.exceptions.HTTPError(
            f"[SERVER ERROR] Groq API returned HTTP {response.status_code}. "
            "The service may be temporarily unavailable."
        )
    if response.status_code >= 400:
        raise requests.exceptions.HTTPError(
            f"[HTTP ERROR] Groq API responded with HTTP {response.status_code}: "
            f"{_api_error_detail(response)}"
        )

    try:
        data = response.json()
        response_text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, ValueError) as parse_err:
        raise ValueError(
            f"[PARSE ERROR] Unexpected API response structure: {parse_err}\n"
            f"Raw response body: {response.text[:500]}"
        )

    return response_text, latency


# ---------------------------------------------------------------------------
# Debug File Parser
# ---------------------------------------------------------------------------

def load_chunks_from_retrieval_debug(
    file_path: str = "debug_retrieval_results.txt",
) -> Tuple[str, List[Tuple[int, float, str]]]:
    """
    Parses debug_retrieval_results.txt produced by retrieval.py and extracts:
      - The original query string
      - An ordered list of (chunk_index, similarity_score, chunk_text) tuples

    Args:
        file_path (str): Path to the retrieval debug output file.

    Returns:
        Tuple[str, List[Tuple[int, float, str]]]:
            (query, [(chunk_idx, score, text), ...])

    Raises:
        FileNotFoundError: If the debug file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[ERROR] Retrieval debug file not found: '{file_path}'\n"
            "Run retrieval.py first to generate this file."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        raise ValueError(f"[ERROR] '{file_path}' is empty.")

    query_match = re.search(r"^QUERY:\s*(.+)$", content, re.MULTILINE)
    query = query_match.group(1).strip() if query_match else "Unknown Query"

    rank_pattern = re.compile(
        r"RANK \d+ \| SIMILARITY SCORE: ([\d.]+) \| CHUNK INDEX: (\d+)\n(.*?)(?=\n-{50}|\Z)",
        re.DOTALL,
    )

    results: List[Tuple[int, float, str]] = []
    for match in rank_pattern.finditer(content):
        score = float(match.group(1))
        chunk_idx = int(match.group(2))
        chunk_text = match.group(3).strip()
        results.append((chunk_idx, score, chunk_text))

    if not results:
        raise ValueError(
            f"[ERROR] No ranked chunks found in '{file_path}'. "
            "Ensure retrieval.py ran successfully and the file format is intact."
        )

    return query, results


# ---------------------------------------------------------------------------
# Debug Output Writer
# ---------------------------------------------------------------------------

def write_debug_output(
    prompt: str,
    response_text: str,
    model_name: str,
    latency: float,
    output_path: str = "debug_llm_response.txt",
) -> None:
    """
    Writes the complete LLM interaction to a structured debug file.

    Args:
        prompt (str): The augmented grounded prompt sent to the API.
        response_text (str): The raw text returned by the LLM.
        model_name (str): The model identifier used for the request.
        latency (float): Measured round-trip latency in seconds.
        output_path (str): Destination path for the debug file.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    separator = "=" * 50
    thin_sep = "-" * 50

    content = (
        f"{separator}\n"
        f"TIMESTAMP: {timestamp} | LATENCY: {latency:.2f}s\n"
        f"MODEL: {model_name}\n"
        f"{separator}\n"
        f"AUGMENTED PROMPT SENT TO API:\n"
        f"{thin_sep}\n"
        f"{prompt}\n"
        f"{thin_sep}\n"
        f"RAW LLM RESPONSE:\n"
        f"{thin_sep}\n"
        f"{response_text}\n"
        f"{separator}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[SUCCESS] Debug output written to '{output_path}'")


# ---------------------------------------------------------------------------
# Main Validation Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("WEEK 3: LLM INTEGRATION & GROUNDING VALIDATION")
    print("=" * 60)

    # Step 1: Resolve API key
    api_key = get_api_key()
    if not api_key:
        print(
            "\n[ERROR] GROQ_API_KEY is not set.\n"
            "Add it to a .env file next to this script (GROQ_API_KEY=gsk_...) or export it:\n"
            "  Windows (PowerShell): $env:GROQ_API_KEY = 'gsk_...'\n"
            "  Linux/macOS:          export GROQ_API_KEY='gsk_...'\n"
        )
        raise SystemExit(1)

    model_name = DEFAULT_MODEL

    # Step 2: Load top retrieved chunks from debug_retrieval_results.txt
    retrieval_debug_path = "debug_retrieval_results.txt"
    print(f"\n[1/4] Loading retrieval results from '{retrieval_debug_path}'...")

    try:
        query, ranked_results = load_chunks_from_retrieval_debug(retrieval_debug_path)
    except (FileNotFoundError, ValueError) as load_err:
        print(str(load_err))
        raise SystemExit(1)

    print(f"      Query  : {query}")
    print(f"      Chunks : {len(ranked_results)}")
    for rank, (chunk_idx, score, text) in enumerate(ranked_results, start=1):
        preview = text[:80].replace("\n", " ")
        print(f"      Rank {rank} [Chunk #{chunk_idx}, Score={score:.4f}]: {preview}...")

    # Step 3: Build grounded prompt
    print("\n[2/4] Building grounded prompt...")
    chunk_texts = [text for (_, _, text) in ranked_results]
    prompt = build_grounded_prompt(query, chunk_texts)
    print(f"      Prompt length: {len(prompt)} characters")

    # Step 4: Query the LLM API
    print(f"\n[3/4] Querying Groq API (model: {model_name})...")
    print("      Please wait - dispatching HTTP POST request...")

    try:
        response_text, latency = query_llm_api(prompt, api_key, model_name)
    except (ValueError, requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError) as api_err:
        print(f"\n{str(api_err)}")
        raise SystemExit(1)

    print(f"      [OK] Response received in {latency:.2f}s")

    # Step 5: Write debug output
    output_debug_path = "debug_llm_response.txt"
    print(f"\n[4/4] Writing debug output to '{output_debug_path}'...")

    write_debug_output(
        prompt=prompt,
        response_text=response_text,
        model_name=model_name,
        latency=latency,
        output_path=output_debug_path,
    )

    # Console Summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  Query   : {query}")
    print(f"  Model   : {model_name}")
    print(f"  Latency : {latency:.2f}s")
    print(f"  Output  : {output_debug_path}")
    print("-" * 60)
    print("LLM RESPONSE PREVIEW:")
    print("-" * 60)
    preview_lines = response_text.strip().splitlines()[:10]
    safe_preview = "\n".join(preview_lines).encode(sys.stdout.encoding or 'utf-8', 'replace').decode(sys.stdout.encoding or 'utf-8')
    print(safe_preview)
    if len(response_text.strip().splitlines()) > 10:
        print("  [...truncated - see debug_llm_response.txt for full output]")
    print("=" * 60)
