"""
Test all rerankers with multiple test files
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import load_test_data, test_reranker, initialize_rerankers, get_device

def main():
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get all test files
    test_dir = 'tests'
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
    
    # Initialize all rerankers
    rerankers = initialize_rerankers(device)
    
    # Test each file
    for test_file in test_files:
        print("\n" + "="*60)
        print(f"TESTING WITH FILE: {test_file}")
        print("="*60)
        
        # Load test data
        full_path = os.path.join(test_dir, test_file)
        query, documents = load_test_data(full_path)
        
        print(f"Query: {query}")
        print(f"Number of documents: {len(documents)}")
        
        # Test all rerankers
        for name, reranker in rerankers.items():
            if reranker is not None:  # Only test if initialization was successful
                test_reranker(reranker, name, query, documents)

if __name__ == "__main__":
    main()
