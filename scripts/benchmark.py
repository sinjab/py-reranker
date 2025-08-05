"""
Benchmark script for comparing reranker performance
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import load_test_data, benchmark_reranker, initialize_rerankers, get_device

# For backward compatibility, we still need these imports for the reranker_map
from rerankers.jina_reranker import JinaReranker
from rerankers.mxbai_reranker import MxbaiReranker
from rerankers.qwen_reranker import QwenReranker
from rerankers.msmarco_reranker import MSMarcoReranker
from rerankers.bge_reranker import BGEReranker

def main():
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark reranker performance')
    parser.add_argument('--test-file', type=str, default=os.path.join('tests', 'test_multilingual.json'),
                        help='Path to JSON test file (default: tests/test_multilingual.json)')
    args = parser.parse_args()
    
    # Load test data
    test_file = args.test_file
    query, documents = load_test_data(test_file)
    
    print(f"Query: {query}")
    print(f"Number of documents: {len(documents)}")
    
    # Initialize all rerankers
    rerankers = initialize_rerankers(device)
    
    # Store benchmark results
    benchmark_results = {}
    
    # Benchmark all rerankers
    for name, reranker in rerankers.items():
        if reranker is not None:  # Only benchmark if initialization was successful
            time_taken = benchmark_reranker(reranker, name, query, documents)
            if time_taken:
                benchmark_results[name] = time_taken
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    # Sort by execution time
    sorted_results = sorted(benchmark_results.items(), key=lambda x: x[1])
    
    print("\nReranker Performance (fastest to slowest):")
    for i, (name, time_taken) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {time_taken:.4f} seconds")

if __name__ == "__main__":
    main()
