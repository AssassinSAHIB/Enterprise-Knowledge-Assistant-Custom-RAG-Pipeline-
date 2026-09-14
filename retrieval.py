"""
Week 2 - Vector Storage, Manual Search & Retrieval Validation
Module: retrieval.py
Enterprise Knowledge Assistant (Custom RAG Pipeline)
"""

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