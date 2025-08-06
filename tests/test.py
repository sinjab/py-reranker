"""
Additional tests to improve code coverage
"""

import os
import pytest
from utils.common import load_test_data, initialize_rerankers, get_device, benchmark_reranker


def test_benchmark_reranker(test_data_dir):
    """Test the benchmark_reranker function"""
    test_file = os.path.join(test_data_dir, 'test_simple.json')
    query, documents = load_test_data(test_file)
    
    device = get_device()
    rerankers = initialize_rerankers(device)
    
    # Test benchmark with the first available reranker
    for name, reranker in rerankers.items():
        if reranker is not None:
            execution_time = benchmark_reranker(reranker, name, query, documents)
            assert execution_time is not None
            assert execution_time >= 0
            break


def test_reranker_edge_cases():
    """Test edge cases for rerankers"""
    from rerankers.mxbai_reranker import MxbaiReranker
    
    reranker = MxbaiReranker()
    query = "Test query"
    
    # Test with single document
    documents = ["Single document"]
    results = reranker.rank(query, documents, top_k=3)
    assert isinstance(results, list)
    assert len(results) == 1
    assert 'score' in results[0]
    assert 'text' in results[0]
    
    # Test without returning documents
    results = reranker.rank(query, documents, top_k=3, return_documents=False)
    assert isinstance(results, list)
    assert len(results) == 1
    assert 'score' in results[0]
    assert 'text' not in results[0]


def test_reranker_top_k_limiting():
    """Test that top_k parameter correctly limits results"""
    from rerankers.mxbai_reranker import MxbaiReranker
    
    reranker = MxbaiReranker()
    query = "Test query"
    documents = [f"Document {i}" for i in range(10)]
    
    # Test with top_k=5
    results = reranker.rank(query, documents, top_k=5)
    assert isinstance(results, list)
    assert len(results) == 5
    
    # Test without top_k (should return all)
    results = reranker.rank(query, documents)
    assert isinstance(results, list)
    assert len(results) == 10
