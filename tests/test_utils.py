"""
Pytest tests for utility functions
"""

import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from utils.common import (
    get_device, 
    load_test_data, 
    run_reranker_test, 
    initialize_rerankers,
    benchmark_reranker
)


def test_get_device():
    """Test that get_device returns a valid device"""
    device = get_device()
    
    # Device should be either 'cpu' or 'cuda'
    assert device in ['cpu', 'cuda']
    
    # If cuda is available, it should return 'cuda', otherwise 'cpu'
    import torch
    if torch.cuda.is_available():
        assert device == 'cuda'
    else:
        assert device == 'cpu'


def test_load_test_data():
    """Test load_test_data function with various scenarios"""
    # Create a temporary test file
    test_data = {
        "query": "What is machine learning?",
        "documents": [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Python is a programming language"
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        # Test successful loading
        query, documents = load_test_data(temp_file)
        assert query == test_data["query"]
        assert documents == test_data["documents"]
        assert isinstance(query, str)
        assert isinstance(documents, list)
        assert len(documents) == 3
    finally:
        os.unlink(temp_file)


def test_load_test_data_file_not_found():
    """Test load_test_data with non-existent file"""
    with pytest.raises(FileNotFoundError):
        load_test_data("non_existent_file.json")


def test_load_test_data_invalid_json():
    """Test load_test_data with invalid JSON"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json content")
        temp_file = f.name
    
    try:
        with pytest.raises(json.JSONDecodeError):
            load_test_data(temp_file)
    finally:
        os.unlink(temp_file)


@patch('builtins.print')
def test_run_reranker_test_mxbai(mock_print):
    """Test run_reranker_test function with MxbaiReranker"""
    # Mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rank.return_value = [
        {'score': 0.95, 'text': 'Document 1 content'},
        {'score': 0.85, 'text': 'Document 2 content'}
    ]
    
    query = "test query"
    documents = ["doc1", "doc2"]
    
    run_reranker_test(mock_reranker, "Mixedbread AI Reranker", query, documents, top_k=2)
    
    # Verify reranker was called correctly
    mock_reranker.rank.assert_called_once_with(query, documents, top_k=2)
    
    # Verify print was called (output was generated)
    assert mock_print.called


@patch('builtins.print')
def test_run_reranker_test_mxbai_v2(mock_print):
    """Test run_reranker_test function with MxbaiRerankV2"""
    # Mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rank.return_value = [
        ('Document 1 content', 0.95),
        ('Document 2 content', 0.85)
    ]
    
    query = "test query"
    documents = ["doc1", "doc2"]
    
    run_reranker_test(mock_reranker, "Mixedbread AI Reranker V2", query, documents, top_k=2)
    
    # Verify reranker was called correctly
    mock_reranker.rank.assert_called_once_with(query, documents, top_k=2)
    
    # Verify print was called (output was generated)
    assert mock_print.called


@patch('builtins.print')
def test_run_reranker_test_other(mock_print):
    """Test run_reranker_test function with other rerankers"""
    # Mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rank.return_value = [
        ('Document 1 content', 0.95),
        ('Document 2 content', 0.85)
    ]
    
    query = "test query"
    documents = ["doc1", "doc2"]
    
    run_reranker_test(mock_reranker, "Jina Reranker", query, documents, top_k=2)
    
    # Verify reranker was called correctly
    mock_reranker.rank.assert_called_once_with(query, documents, top_n=2)
    
    # Verify print was called (output was generated)
    assert mock_print.called


@patch('builtins.print')
def test_run_reranker_test_exception(mock_print):
    """Test run_reranker_test function when reranker raises exception"""
    # Mock reranker that raises exception
    mock_reranker = MagicMock()
    mock_reranker.rank.side_effect = Exception("Test error")
    
    query = "test query"
    documents = ["doc1", "doc2"]
    
    run_reranker_test(mock_reranker, "Test Reranker", query, documents)
    
    # Verify error message was printed
    mock_print.assert_called()
    error_calls = [call for call in mock_print.call_args_list if 'Error testing' in str(call)]
    assert len(error_calls) > 0


@patch('builtins.print')
@patch('utils.common.JinaReranker')
@patch('utils.common.MxbaiReranker')
@patch('utils.common.MxbaiRerankV2')
@patch('utils.common.QwenReranker')
@patch('utils.common.QwenReranker0_6B')
@patch('utils.common.QwenReranker4B')
@patch('utils.common.QwenReranker8B')
@patch('utils.common.MSMarcoReranker')
@patch('utils.common.MSMarcoRerankerV2')
@patch('utils.common.BGERerankerV25Gemma2Lightweight')
@patch('utils.common.BGERerankerV2MiniCPMLayerwise')
@patch('utils.common.BGERerankerV2Gemma')
@patch('utils.common.BGERerankerV2M3')
@patch('utils.common.BGERerankerLarge')
@patch('utils.common.BGERerankerBase')
@patch('utils.common.BGEReranker')
def test_initialize_rerankers_success(mock_bge, mock_bge_base, mock_bge_large, mock_bge_v2m3, mock_bge_v2gemma, mock_bge_v2minicpm, mock_bge_v25lightweight, mock_msmarco_v2, mock_msmarco, mock_qwen_8b, mock_qwen_4b, mock_qwen_0_6b, mock_qwen, mock_mxbai_v2, mock_mxbai, mock_jina, mock_print):
    """Test initialize_rerankers function with successful initialization"""
    # Mock all rerankers to return mock instances
    mock_instances = {
        'jina': MagicMock(),
        'mxbai': MagicMock(),
        'mxbai_v2': MagicMock(),
        'qwen': MagicMock(),
        'qwen_0_6b': MagicMock(),
        'qwen_4b': MagicMock(),
        'qwen_8b': MagicMock(),
        'msmarco': MagicMock(),
        'msmarco_v2': MagicMock(),
        'bge': MagicMock(),
        'bge_base': MagicMock(),
        'bge_large': MagicMock(),
        'bge_v2m3': MagicMock(),
        'bge_v2gemma': MagicMock(),
        'bge_v2minicpm': MagicMock(),
        'bge_v25lightweight': MagicMock()
    }
    
    mock_jina.return_value = mock_instances['jina']
    mock_mxbai.return_value = mock_instances['mxbai']
    mock_mxbai_v2.return_value = mock_instances['mxbai_v2']
    mock_qwen.return_value = mock_instances['qwen']
    mock_qwen_0_6b.return_value = mock_instances['qwen_0_6b']
    mock_qwen_4b.return_value = mock_instances['qwen_4b']
    mock_qwen_8b.return_value = mock_instances['qwen_8b']
    mock_msmarco.return_value = mock_instances['msmarco']
    mock_msmarco_v2.return_value = mock_instances['msmarco_v2']
    mock_bge.return_value = mock_instances['bge']
    mock_bge_base.return_value = mock_instances['bge_base']
    mock_bge_large.return_value = mock_instances['bge_large']
    mock_bge_v2m3.return_value = mock_instances['bge_v2m3']
    mock_bge_v2gemma.return_value = mock_instances['bge_v2gemma']
    mock_bge_v2minicpm.return_value = mock_instances['bge_v2minicpm']
    mock_bge_v25lightweight.return_value = mock_instances['bge_v25lightweight']
    
    device = 'cpu'
    rerankers = initialize_rerankers(device)
    
    # Verify all rerankers were initialized
    assert len(rerankers) == 16
    assert "Jina Reranker" in rerankers
    assert "Mixedbread AI Reranker" in rerankers
    assert "Mixedbread AI Reranker V2" in rerankers
    assert "Qwen Reranker 4B" in rerankers
    assert "Qwen Reranker 4B (explicit)" in rerankers
    assert "Qwen Reranker 0.6B" in rerankers
    assert "Qwen Reranker 8B" in rerankers
    assert "MS MARCO Reranker" in rerankers
    assert "MS MARCO Reranker V2" in rerankers
    assert "BGE Reranker V2-M3" in rerankers
    assert "BGE Reranker Base" in rerankers
    assert "BGE Reranker Large" in rerankers
    assert "BGE Reranker V2-M3 (Class)" in rerankers
    assert "BGE Reranker V2-Gemma" in rerankers
    assert "BGE Reranker V2-MiniCPM-Layerwise" in rerankers
    assert "BGE Reranker V2.5-Gemma2-Lightweight" in rerankers
    
    # Verify device-dependent rerankers were called with device
    mock_jina.assert_called_once_with(device=device)
    mock_qwen.assert_called_once_with(device=device)
    mock_qwen_0_6b.assert_called_once_with(device=device)
    mock_qwen_4b.assert_called_once_with(device=device)
    mock_qwen_8b.assert_called_once_with(device=device)
    mock_msmarco.assert_called_once_with(device=device)
    mock_msmarco_v2.assert_called_once_with(device=device)
    mock_bge.assert_called_once_with(device=device)
    mock_bge_base.assert_called_once_with(device=device)
    mock_bge_large.assert_called_once_with(device=device)
    mock_bge_v2m3.assert_called_once_with(device=device)
    mock_bge_v2gemma.assert_called_once_with(device=device)
    mock_bge_v2minicpm.assert_called_once_with(device=device)
    mock_bge_v25lightweight.assert_called_once_with(device=device)
    
    # Verify device-independent rerankers were called without device
    mock_mxbai.assert_called_once_with()
    mock_mxbai_v2.assert_called_once_with()


@patch('builtins.print')
@patch('utils.common.JinaReranker')
def test_initialize_rerankers_with_exceptions(mock_jina, mock_print):
    """Test initialize_rerankers function when some rerankers fail to initialize"""
    # Mock JinaReranker to raise exception
    mock_jina.side_effect = Exception("Failed to initialize")
    
    rerankers = initialize_rerankers('cpu')
    
    # Should still return a dictionary, but without the failed reranker
    assert isinstance(rerankers, dict)
    
    # Verify error message was printed
    error_calls = [call for call in mock_print.call_args_list if 'Error initializing' in str(call)]
    assert len(error_calls) > 0


@patch('builtins.print')
@patch('time.time')
def test_benchmark_reranker_mxbai(mock_time, mock_print):
    """Test benchmark_reranker function with MxbaiReranker"""
    # Mock time to control execution time measurement
    mock_time.side_effect = [0.0, 1.5]  # start_time, end_time
    
    # Mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rank.return_value = [
        {'score': 0.95, 'text': 'Document 1 content'},
        {'score': 0.85, 'text': 'Document 2 content'}
    ]
    
    query = "test query"
    documents = ["doc1", "doc2"]
    
    execution_time = benchmark_reranker(mock_reranker, "Mixedbread AI Reranker", query, documents)
    
    # Verify execution time was calculated correctly
    assert execution_time == 1.5
    
    # Verify reranker was called correctly
    mock_reranker.rank.assert_called_once_with(query, documents, top_k=3)
    
    # Verify print was called with timing information
    assert mock_print.called


@patch('builtins.print')
@patch('time.time')
def test_benchmark_reranker_other(mock_time, mock_print):
    """Test benchmark_reranker function with other rerankers"""
    # Mock time to control execution time measurement
    mock_time.side_effect = [0.0, 2.0]  # start_time, end_time
    
    # Mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rank.return_value = [
        ('Document 1 content', 0.95),
        ('Document 2 content', 0.85)
    ]
    
    query = "test query"
    documents = ["doc1", "doc2"]
    
    execution_time = benchmark_reranker(mock_reranker, "Jina Reranker", query, documents)
    
    # Verify execution time was calculated correctly
    assert execution_time == 2.0
    
    # Verify reranker was called correctly
    mock_reranker.rank.assert_called_once_with(query, documents, top_n=3)


@patch('builtins.print')
def test_benchmark_reranker_exception(mock_print):
    """Test benchmark_reranker function when reranker raises exception"""
    # Mock reranker that raises exception
    mock_reranker = MagicMock()
    mock_reranker.rank.side_effect = Exception("Test error")
    
    query = "test query"
    documents = ["doc1", "doc2"]
    
    execution_time = benchmark_reranker(mock_reranker, "Test Reranker", query, documents)
    
    # Should return None on error
    assert execution_time is None
    
    # Verify error message was printed
    error_calls = [call for call in mock_print.call_args_list if 'Error:' in str(call)]
    assert len(error_calls) > 0
