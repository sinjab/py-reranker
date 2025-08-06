"""
Test script for Mixedbread AI Reranker V2
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_test_data, test_reranker, get_device
from rerankers.mxbai_v2_reranker import MxbaiRerankV2

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test Mixedbread AI Reranker V2')
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
    
    # Initialize the reranker
    try:
        reranker = MxbaiRerankV2()
        test_reranker(reranker, "Mixedbread AI Reranker V2", query, documents)
    except Exception as e:
        print(f"Error initializing Mixedbread AI Reranker V2: {str(e)}")

if __name__ == "__main__":
    main()
