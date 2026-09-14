import os
import re
import time
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Global model cache to prevent reloading during subsequent calls
_MODEL_INSTANCE = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Retrieves or initializes the cached SentenceTransformer instance.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = SentenceTransformer(model_name)
    return _MODEL_INSTANCE


def generate_embeddings(text_chunks: List[str]) -> np.ndarray:
    """
    Converts a list of strings into a NumPy array of 384-dimensional vectors
    using sentence-transformers (all-MiniLM-L6-v2).
    
    Args:
        text_chunks (List[str]): Input text strings to embed.
        
    Returns:
        np.ndarray: Matrix of shape (N, 384) with float32 values.
    """
    if not text_chunks:
        return np.empty((0, 384), dtype=np.float32)
        
    model = get_embedding_model("all-MiniLM-L6-v2")
    embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False  # Keep raw vectors to test manual cosine norm
    )
    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes mathematical cosine similarity between two 1D vectors:
    cos_sim(a, b) = (a . b) / (||a|| * ||b||)
    
    Args:
        vec_a (np.ndarray): Vector A (1D array).
        vec_b (np.ndarray): Vector B (1D array).
        
    Returns:
        float: Cosine similarity score bounded in [-1.0, 1.0].
    """
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    
    # Epsilon / zero-norm protection to prevent divide-by-zero
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    dot_product = float(np.dot(vec_a, vec_b))
    return dot_product / (norm_a * norm_b)


def search_top_k(
    query_vector: np.ndarray,
    document_vectors: np.ndarray,
    top_k: int = 3
) -> List[Tuple[int, float]]:
    """
    Calculates cosine similarity across all document vectors and returns
    the indices and similarity scores of the top K matches in descending order.
    
    Args:
        query_vector (np.ndarray): 1D array representing the encoded query.
        document_vectors (np.ndarray): 2D array of shape (N, D) containing document embeddings.
        top_k (int): Number of top matches to retrieve. Default is 3.
        
    Returns:
        List[Tuple[int, float]]: List of (chunk_index, similarity_score) sorted descending.
    """
    if len(document_vectors) == 0:
        return []
        
    scores: List[Tuple[int, float]] = []
    
    for idx, doc_vec in enumerate(document_vectors):
        sim = cosine_similarity(query_vector, doc_vec)
        scores.append((idx, sim))
        
    # Sort by similarity score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_k]


def load_chunks_from_debug_file(file_path: str = "debug_chunks.txt") -> List[str]:
    """
    Parses debug_chunks.txt file and returns the list of extracted chunks.
    """
    if not os.path.exists(file_path):
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern matches chunk blocks produced by ingestion.py
    pattern = r"={50}\nCHUNK \d+ \| Length: \d+\n={50}\n(.*?)(?=\n={50}|$)"
    matches = re.findall(pattern, content, flags=re.DOTALL)
    
    chunks = [m.strip() for m in matches if m.strip()]
    return chunks


def run_retrieval_validation(
    query: str = "What is the policy guidelines mentioned?",
    chunks_file: str = "debug_chunks.txt",
    output_debug_path: str = "debug_retrieval_results.txt",
    top_k: int = 3
) -> None:
    """
    Executes the vector storage, manual search, and retrieval validation pipeline.
    """
    print("=" * 60)
    print("WEEK 2: RETRIEVAL & VECTOR SEARCH EXECUTION")
    print("=" * 60)
    
    # 1. Load chunks from debug_chunks.txt or generate via ingestion.py
    chunks = load_chunks_from_debug_file(chunks_file)
    if not chunks:
        print(f"'{chunks_file}' not found or empty. Running ingestion pipeline...")
        from ingestion import run_ingestion_pipeline
        chunks = run_ingestion_pipeline()
        
    if not chunks:
        print("[ERROR] No chunks available for vectorization.")
        return

    print(f"Total Chunks to Vectorize:        {len(chunks)}")
# 2. Convert all chunks into a NumPy embedding matrix & benchmark time
    start_time = time.perf_counter()
    document_vectors = generate_embeddings(chunks)
    embedding_duration = time.perf_counter() - start_time
    
    print(f"Embedding Execution Time:         {embedding_duration:.3f} seconds")
    print(f"NumPy Matrix Shape:               {document_vectors.shape}")
    print("-" * 60)

    # 3. Vectorize sample query
    query_vector = generate_embeddings([query])[0]
    
    # 4. Execute search_top_k
    top_results = search_top_k(query_vector, document_vectors, top_k=top_k)

    # 5. Format and write search results
    with open(output_debug_path, "w", encoding="utf-8") as f:
        f.write(f"QUERY: {query}\n")
        f.write("-" * 50 + "\n")
        
        for rank, (chunk_idx, score) in enumerate(top_results, start=1):
            f.write(f"RANK {rank} | SIMILARITY SCORE: {score:.4f} | CHUNK INDEX: {chunk_idx}\n")
            f.write(f"{chunks[chunk_idx]}\n")
            f.write("-" * 50 + "\n")
            
    print(f"[SUCCESS] Top-{top_k} results written to '{output_debug_path}'")
    print("\nRetrieval Preview:")
    for rank, (chunk_idx, score) in enumerate(top_results, start=1):
        preview = chunks[chunk_idx][:100].replace("\n", " ")
        print(f"  Rank {rank} [Score: {score:.4f}, Chunk #{chunk_idx}]: {preview}...")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_query = "What is the policy guidelines mentioned?"
    run_retrieval_validation(query=benchmark_query)
