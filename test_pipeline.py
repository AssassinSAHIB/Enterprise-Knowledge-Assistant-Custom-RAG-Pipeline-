"""
Unit and Integration Tests for the Custom RAG Pipeline.
Verifies mathematical correctness, sliding-window chunking, and retrieval precision.
"""

import math
import numpy as np
from ingestion import custom_text_splitter
from retrieval import cosine_similarity, search_top_k


def test_custom_text_splitter_basic():
    text = "A" * 1000
    chunks = custom_text_splitter(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500
    assert len(chunks[2]) == 100  # remainder from index 900 to 1000


def test_custom_text_splitter_overlap_continuity():
    text = "0123456789" * 10  # 100 characters
    chunks = custom_text_splitter(text, chunk_size=40, chunk_overlap=10)
    
    # Check that consecutive chunks have exact overlap
    # Chunk 0: text[0:40]
    # Chunk 1: text[30:70] -> overlap of 10 chars should match
    assert chunks[0][30:40] == chunks[1][0:10]
    assert chunks[1][30:40] == chunks[2][0:10]


def test_custom_text_splitter_edge_cases():
    # Empty string
    assert custom_text_splitter("", chunk_size=500, chunk_overlap=50) == []
    
    # Text smaller than chunk size
    short_text = "Small enterprise note"
    chunks = custom_text_splitter(short_text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == short_text

    # Invalid configurations
    try:
        custom_text_splitter("test", chunk_size=50, chunk_overlap=50)
        assert False, "Should raise ValueError when overlap >= chunk_size"
    except ValueError:
        pass


def test_cosine_similarity_mathematical_properties():
    # Identical vectors -> cosine similarity = 1.0
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert math.isclose(cosine_similarity(v1, v1), 1.0, rel_tol=1e-5)

    # Orthogonal vectors -> cosine similarity = 0.0
    v_ortho1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_ortho2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert math.isclose(cosine_similarity(v_ortho1, v_ortho2), 0.0, abs_tol=1e-6)

    # Opposite vectors -> cosine similarity = -1.0
    v_opp = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    assert math.isclose(cosine_similarity(v1, v_opp), -1.0, rel_tol=1e-5)

    # Zero vector handling (must not crash, return 0.0)
    v_zero = np.zeros(3, dtype=np.float32)
    assert cosine_similarity(v1, v_zero) == 0.0


def test_search_top_k():
    doc_vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    results = search_top_k(query_vector, doc_vectors, top_k=2)
    assert len(results) == 2
    # Rank 1 must be doc 0 (similarity 1.0)
    assert results[0][0] == 0
    assert math.isclose(results[0][1], 1.0, rel_tol=1e-5)
    # Rank 2 must be doc 1 (similarity 0.8)
    assert results[1][0] == 1
    assert math.isclose(results[1][1], 0.8, rel_tol=1e-5)


if __name__ == "__main__":
    test_custom_text_splitter_basic()
    test_custom_text_splitter_overlap_continuity()
    test_custom_text_splitter_edge_cases()
    test_cosine_similarity_mathematical_properties()
    test_search_top_k()
    print("[ALL UNIT TESTS PASSED]")
