"""
Pytest tests for all rerankers
"""

import os
import pytest
from utils.common import load_test_data, initialize_rerankers, get_device


def test_load_test_data(test_data_dir):
    """Test that test data can be loaded successfully"""
    test_file = os.path.join(test_data_dir, 'test_simple.json')
    query, documents = load_test_data(test_file)
    
    assert isinstance(query, str)
    assert isinstance(documents, list)
    assert len(documents) > 0


def test_initialize_rerankers():
    """Test that all rerankers can be initialized"""
    device = get_device()
    rerankers = initialize_rerankers(device)
    
    # Check that we have at least one reranker
    assert len(rerankers) > 0
    
    # Check that each reranker has a rank method
    for name, reranker in rerankers.items():
        assert hasattr(reranker, 'rank')


def test_mxbai_reranker_basic():
    """Test basic functionality of MxbaiReranker"""
    from rerankers.mxbai_reranker import MxbaiReranker
    
    reranker = MxbaiReranker()
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Madrid is the capital of Spain."
    ]
    
    results = reranker.rank(query, documents, top_k=3)
    
    assert isinstance(results, list)
    assert len(results) == 3
    assert all('score' in result for result in results)
    assert all('text' in result for result in results)
    
    # The correct answer should have the highest score
    assert results[0]['text'] == "Paris is the capital of France."


def test_reranker_consistency(test_data_dir):
    """Test that rerankers produce consistent results"""
    test_file = os.path.join(test_data_dir, 'test_multilingual.json')
    query, documents = load_test_data(test_file)
    
    device = get_device()
    rerankers = initialize_rerankers(device)
    
    for name, reranker in rerankers.items():
        if reranker is not None:
            # Test with top_k=3
            if name == "Mixedbread AI Reranker":
                results = reranker.rank(query, documents, top_k=3)
                assert len(results) == 3
            elif name == "Mixedbread AI Reranker V2":
                results = reranker.rank(query, documents, top_k=3)
                assert len(results) == 3
            else:
                results = reranker.rank(query, documents, top_n=3)
                assert len(results) == 3
