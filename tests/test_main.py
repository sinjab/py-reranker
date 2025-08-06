"""
Pytest tests for main functionality
"""

import pytest
import os
import sys
import tempfile
import json
from unittest.mock import patch, MagicMock, call
from utils.common import load_test_data

# Add the project root to the path so we can import main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import main


def test_main_with_test_file(test_data_dir):
    """Test main functionality with test file"""
    test_file = os.path.join(test_data_dir, 'test_simple.json')
    
    # Check that the test file exists
    assert os.path.exists(test_file)
    
    # Load test data
    query, documents = load_test_data(test_file)
    
    # Verify data structure
    assert isinstance(query, str)
    assert isinstance(documents, list)
    assert len(documents) > 0


def test_main_with_multilingual_data(test_data_dir):
    """Test main functionality with multilingual test data"""
    test_file = os.path.join(test_data_dir, 'test_multilingual.json')
    
    # Check that the test file exists
    assert os.path.exists(test_file)
    
    # Load test data
    query, documents = load_test_data(test_file)
    
    # Verify data structure
    assert isinstance(query, str)
    assert isinstance(documents, list)
    assert len(documents) > 0


@patch('main.get_device')
@patch('main.load_test_data')
@patch('main.run_reranker_test')
@patch('main.initialize_rerankers')
@patch('builtins.print')
def test_main_with_test_file_all_rerankers(mock_print, mock_init_rerankers, mock_test_reranker, mock_load_test_data, mock_get_device):
    """Test main function with test file and all rerankers"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_load_test_data.return_value = ('test query', ['doc1', 'doc2', 'doc3'])
    
    mock_reranker1 = MagicMock()
    mock_reranker2 = MagicMock()
    mock_init_rerankers.return_value = {
        'Jina Reranker': mock_reranker1,
        'Mixedbread AI Reranker': mock_reranker2
    }
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'query': 'test', 'documents': ['doc1']}, f)
        temp_file = f.name
    
    try:
        # Mock sys.argv
        with patch('sys.argv', ['main.py', '--test-file', temp_file, '--top-k', '5']):
            main.main()
        
        # Verify function calls
        mock_get_device.assert_called_once()
        mock_load_test_data.assert_called_once_with(temp_file)
        mock_init_rerankers.assert_called_once_with('cpu')
        
        # Verify test_reranker was called for each reranker
        expected_calls = [
            call(mock_reranker1, 'Jina Reranker', 'test query', ['doc1', 'doc2', 'doc3'], 5),
            call(mock_reranker2, 'Mixedbread AI Reranker', 'test query', ['doc1', 'doc2', 'doc3'], 5)
        ]
        mock_test_reranker.assert_has_calls(expected_calls, any_order=True)
    finally:
        os.unlink(temp_file)


@patch('main.get_device')
@patch('main.run_reranker_test')
@patch('main.JinaReranker')
@patch('builtins.print')
def test_main_with_specific_reranker(mock_print, mock_jina_class, mock_test_reranker, mock_get_device):
    """Test main function with specific reranker"""
    # Setup mocks
    mock_get_device.return_value = 'cuda'
    mock_jina_instance = MagicMock()
    mock_jina_class.return_value = mock_jina_instance
    
    # Mock sys.argv for specific reranker
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', 'doc2', '--reranker', 'jina']):
        main.main()
    
    # Verify function calls
    mock_get_device.assert_called_once()
    mock_jina_class.assert_called_once_with(device='cuda')
    mock_test_reranker.assert_called_once_with(mock_jina_instance, 'Jina Reranker', 'test query', ['doc1', 'doc2'], 3)


@patch('main.get_device')
@patch('main.run_reranker_test')
@patch('main.MxbaiReranker')
@patch('builtins.print')
def test_main_with_mxbai_reranker(mock_print, mock_mxbai_class, mock_test_reranker, mock_get_device):
    """Test main function with MxbaiReranker"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_mxbai_instance = MagicMock()
    mock_mxbai_class.return_value = mock_mxbai_instance
    
    # Mock sys.argv for mxbai reranker
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', '--reranker', 'mxbai', '--top-k', '1']):
        main.main()
    
    # Verify function calls
    mock_get_device.assert_called_once()
    mock_mxbai_class.assert_called_once_with()
    mock_test_reranker.assert_called_once_with(mock_mxbai_instance, 'Mixedbread AI Reranker', 'test query', ['doc1'], 1)


@patch('main.get_device')
@patch('main.run_reranker_test')
@patch('main.MxbaiRerankV2')
@patch('builtins.print')
def test_main_with_mxbai_v2_reranker(mock_print, mock_mxbai_v2_class, mock_test_reranker, mock_get_device):
    """Test main function with MxbaiRerankV2"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_mxbai_v2_instance = MagicMock()
    mock_mxbai_v2_class.return_value = mock_mxbai_v2_instance
    
    # Mock sys.argv for mxbai-v2 reranker
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', 'doc2', '--reranker', 'mxbai-v2']):
        main.main()
    
    # Verify function calls
    mock_get_device.assert_called_once()
    mock_mxbai_v2_class.assert_called_once_with()
    mock_test_reranker.assert_called_once_with(mock_mxbai_v2_instance, 'Mixedbread AI Reranker V2', 'test query', ['doc1', 'doc2'], 3)


@patch('main.get_device')
@patch('main.run_reranker_test')
@patch('main.QwenReranker')
@patch('builtins.print')
def test_main_with_qwen_reranker(mock_print, mock_qwen_class, mock_test_reranker, mock_get_device):
    """Test main function with QwenReranker"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_qwen_instance = MagicMock()
    mock_qwen_class.return_value = mock_qwen_instance
    
    # Mock sys.argv for qwen reranker
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', '--reranker', 'qwen']):
        main.main()
    
    # Verify function calls
    mock_get_device.assert_called_once()
    mock_qwen_class.assert_called_once_with(device='cpu')
    mock_test_reranker.assert_called_once_with(mock_qwen_instance, 'Qwen Reranker', 'test query', ['doc1'], 3)


@patch('main.get_device')
@patch('main.run_reranker_test')
@patch('main.MSMarcoReranker')
@patch('builtins.print')
def test_main_with_msmarco_reranker(mock_print, mock_msmarco_class, mock_test_reranker, mock_get_device):
    """Test main function with MSMarcoReranker"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_msmarco_instance = MagicMock()
    mock_msmarco_class.return_value = mock_msmarco_instance
    
    # Mock sys.argv for msmarco reranker
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', '--reranker', 'msmarco']):
        main.main()
    
    # Verify function calls
    mock_get_device.assert_called_once()
    mock_msmarco_class.assert_called_once_with(device='cpu')
    mock_test_reranker.assert_called_once_with(mock_msmarco_instance, 'MS MARCO Reranker', 'test query', ['doc1'], 3)


@patch('main.get_device')
@patch('main.run_reranker_test')
@patch('main.BGEReranker')
@patch('builtins.print')
def test_main_with_bge_reranker(mock_print, mock_bge_class, mock_test_reranker, mock_get_device):
    """Test main function with BGEReranker"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_bge_instance = MagicMock()
    mock_bge_class.return_value = mock_bge_instance
    
    # Mock sys.argv for bge reranker
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', '--reranker', 'bge']):
        main.main()
    
    # Verify function calls
    mock_get_device.assert_called_once()
    mock_bge_class.assert_called_once_with(device='cpu')
    mock_test_reranker.assert_called_once_with(mock_bge_instance, 'BGE Reranker', 'test query', ['doc1'], 3)


@patch('main.get_device')
@patch('main.JinaReranker')
@patch('builtins.print')
def test_main_reranker_initialization_error(mock_print, mock_jina_class, mock_get_device):
    """Test main function when reranker initialization fails"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_jina_class.side_effect = Exception('Initialization failed')
    
    # Mock sys.argv for specific reranker that will fail
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', '--reranker', 'jina']):
        main.main()
    
    # Verify error message was printed
    error_calls = [call for call in mock_print.call_args_list if 'Error initializing' in str(call)]
    assert len(error_calls) > 0


@patch('main.get_device')
@patch('builtins.print')
@patch('sys.exit')
def test_main_missing_test_file(mock_exit, mock_print, mock_get_device):
    """Test main function with non-existent test file"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_exit.side_effect = SystemExit(1)  # Make sys.exit actually exit
    
    # Mock sys.argv with non-existent test file
    with patch('sys.argv', ['main.py', '--test-file', 'non_existent_file.json']):
        with pytest.raises(SystemExit):
            main.main()
    
    # Verify error message was printed and sys.exit was called
    error_calls = [call for call in mock_print.call_args_list if 'Error: Test file' in str(call)]
    assert len(error_calls) > 0
    mock_exit.assert_called_once_with(1)


@patch('main.get_device')
@patch('builtins.print')
@patch('sys.exit')
def test_main_missing_arguments(mock_exit, mock_print, mock_get_device):
    """Test main function with missing required arguments"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_exit.side_effect = SystemExit(1)  # Make sys.exit actually exit
    
    # Mock sys.argv with missing arguments
    with patch('sys.argv', ['main.py']):
        with pytest.raises(SystemExit):
            main.main()
    
    # Verify error message was printed and sys.exit was called
    error_calls = [call for call in mock_print.call_args_list if 'Error: Either --test-file or both --query and --documents must be provided' in str(call)]
    assert len(error_calls) > 0
    mock_exit.assert_called_once_with(1)


@patch('main.get_device')
@patch('builtins.print')
@patch('sys.exit')
def test_main_partial_arguments(mock_exit, mock_print, mock_get_device):
    """Test main function with partial arguments (query without documents)"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_exit.side_effect = SystemExit(1)  # Make sys.exit actually exit
    
    # Mock sys.argv with only query but no documents
    with patch('sys.argv', ['main.py', '--query', 'test query']):
        with pytest.raises(SystemExit):
            main.main()
    
    # Verify error message was printed and sys.exit was called
    error_calls = [call for call in mock_print.call_args_list if 'Error: Either --test-file or both --query and --documents must be provided' in str(call)]
    assert len(error_calls) > 0
    mock_exit.assert_called_once_with(1)


@patch('main.get_device')
@patch('main.initialize_rerankers')
@patch('main.run_reranker_test')
@patch('builtins.print')
def test_main_with_none_rerankers(mock_print, mock_test_reranker, mock_init_rerankers, mock_get_device):
    """Test main function when initialize_rerankers returns rerankers with None values"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_init_rerankers.return_value = {
        'Jina Reranker': None,  # Failed to initialize
        'Mixedbread AI Reranker': MagicMock()  # Successfully initialized
    }
    
    # Mock sys.argv
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1']):
        main.main()
    
    # Verify test_reranker was only called for the successful reranker
    assert mock_test_reranker.call_count == 1
    
    # Verify the call was made with the non-None reranker
    call_args = mock_test_reranker.call_args[0]
    assert call_args[1] == 'Mixedbread AI Reranker'  # name parameter


@patch('main.get_device')
@patch('main.benchmark_reranker')
@patch('main.JinaReranker')
@patch('builtins.print')
def test_main_benchmark_specific_reranker(mock_print, mock_jina_class, mock_benchmark_reranker, mock_get_device):
    """Test main function with benchmark flag and specific reranker"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_jina_instance = MagicMock()
    mock_jina_class.return_value = mock_jina_instance
    mock_benchmark_reranker.return_value = 1.5  # Mock execution time
    
    # Mock sys.argv for benchmark with specific reranker
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', '--benchmark', '--reranker', 'jina']):
        main.main()
    
    # Verify function calls
    mock_get_device.assert_called_once()
    mock_jina_class.assert_called_once_with(device='cpu')
    mock_benchmark_reranker.assert_called_once_with(mock_jina_instance, 'Jina Reranker', 'test query', ['doc1'])
    
    # Verify benchmark summary was printed
    summary_calls = [call for call in mock_print.call_args_list if 'BENCHMARK SUMMARY' in str(call)]
    assert len(summary_calls) > 0


@patch('main.get_device')
@patch('main.benchmark_reranker')
@patch('main.initialize_rerankers')
@patch('builtins.print')
def test_main_benchmark_all_rerankers(mock_print, mock_init_rerankers, mock_benchmark_reranker, mock_get_device):
    """Test main function with benchmark flag for all rerankers"""
    # Setup mocks
    mock_get_device.return_value = 'cpu'
    mock_reranker1 = MagicMock()
    mock_reranker2 = MagicMock()
    mock_init_rerankers.return_value = {
        'Jina Reranker': mock_reranker1,
        'Mixedbread AI Reranker': mock_reranker2
    }
    mock_benchmark_reranker.side_effect = [2.1, 1.5]  # Mock execution times
    
    # Mock sys.argv for benchmark all
    with patch('sys.argv', ['main.py', '--query', 'test query', '--documents', 'doc1', '--benchmark']):
        main.main()
    
    # Verify function calls
    mock_get_device.assert_called_once()
    mock_init_rerankers.assert_called_once_with('cpu')
    
    # Verify benchmark_reranker was called for each reranker
    expected_calls = [
        call(mock_reranker1, 'Jina Reranker', 'test query', ['doc1']),
        call(mock_reranker2, 'Mixedbread AI Reranker', 'test query', ['doc1'])
    ]
    mock_benchmark_reranker.assert_has_calls(expected_calls, any_order=True)
    
    # Verify benchmark summary was printed
    summary_calls = [call for call in mock_print.call_args_list if 'BENCHMARK SUMMARY' in str(call)]
    assert len(summary_calls) > 0
