"""
Test script for all rerankers
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import load_test_data, test_reranker, initialize_rerankers, get_device

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test all rerankers')
    parser.add_argument('--test-file', type=str, default=os.path.join('tests', 'test_multilingual.json'),
                        help='Path to JSON test file (default: tests/test_multilingual.json)')
    args = parser.parse_args()
    
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load test data
    test_file = args.test_file
    query, documents = load_test_data(test_file)
    
    print(f"Query: {query}")
    print(f"Number of documents: {len(documents)}")
    
    # Initialize all rerankers
    rerankers = initialize_rerankers(device)
    
    # Test all rerankers
    for name, reranker in rerankers.items():
        if reranker is not None:  # Only test if initialization was successful
            test_reranker(reranker, name, query, documents)

if __name__ == "__main__":
    main()
