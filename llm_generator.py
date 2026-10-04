"""
Week 3 - Direct API LLM Integration & Grounding Validation
Module: llm_generator.py
Enterprise Knowledge Assistant (Custom RAG Pipeline)

Connects the manual retrieval engine to an LLM provider (Groq API) via
direct HTTP POST requests - no LangChain or orchestration framework used.
"""

import os
import re
import time
import datetime
import requests
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 60  # Hard timeout per API call


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
    if not api_key or not api_key.strip():
        raise ValueError(
            "API key is empty or None. "
            "Set GROQ_API_KEY environment variable or pass the key explicitly."
        )

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
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

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout(
            f"[TIMEOUT] Groq API did not respond within {REQUEST_TIMEOUT_SECONDS}s. "
            "Check your network connection or retry later."
        )
    except requests.exceptions.ConnectionError as conn_err:
        raise requests.exceptions.ConnectionError(
            f"[CONNECTION ERROR] Failed to reach Groq API endpoint: {conn_err}"
        )

    latency = time.perf_counter() - t_start

    if response.status_code == 401:
        raise ValueError(
            "[AUTH ERROR] Invalid Groq API key (HTTP 401). "
            "Verify your key at https://console.groq.com/keys"
        )
    if response.status_code == 429:
        raise requests.exceptions.HTTPError(
            "[RATE LIMIT] Groq API rate limit exceeded (HTTP 429). "
            "Wait a moment and retry, or upgrade your plan."
        )
    if response.status_code >= 500:
        raise requests.exceptions.HTTPError(
            f"[SERVER ERROR] Groq API returned HTTP {response.status_code}. "
            "The service may be temporarily unavailable."
        )

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise requests.exceptions.HTTPError(
            f"[HTTP ERROR] Groq API responded with HTTP {response.status_code}: {http_err}"
        )

    try:
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as parse_err:
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
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print(
            "\n[ERROR] GROQ_API_KEY environment variable is not set.\n"
            "Export your key before running:\n"
            "  Windows (PowerShell): $env:GROQ_API_KEY = 'gsk_...'\n"
            "  Linux/macOS:          export GROQ_API_KEY='gsk_...'\n"
        )
        raise SystemExit(1)

    model_name = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

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
    print("\n".join(preview_lines))
    if len(response_text.strip().splitlines()) > 10:
        print("  [...truncated - see debug_llm_response.txt for full output]")
    print("=" * 60)
