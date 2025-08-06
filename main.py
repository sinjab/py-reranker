#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.common import load_test_data, run_reranker_test, initialize_rerankers, get_device, benchmark_reranker

# Import rerankers
from rerankers import (
    JinaReranker,
    MxbaiReranker,
    MxbaiRerankV2,
    QwenReranker,
    MSMarcoReranker,
    MSMarcoRerankerV2,
    BGEReranker
)

def main():
    parser = argparse.ArgumentParser(description='Test various reranker models')
    parser.add_argument('--test-file', type=str, help='Path to JSON test file')
    parser.add_argument('--query', type=str, help='Query string (if not using test file)')
    parser.add_argument('--documents', type=str, nargs='+', help='Document strings (if not using test file)')
    parser.add_argument('--reranker', type=str, choices=['jina', 'mxbai', 'mxbai-v2', 'qwen', 'msmarco', 'msmarco-v2', 'bge'], 
                        help='Specific reranker to use (default: all)')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top results to return')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark instead of normal ranking')
    
    args = parser.parse_args()
    
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get query and documents
    if args.test_file:
        if not os.path.exists(args.test_file):
            print(f"Error: Test file {args.test_file} not found")
            sys.exit(1)
        query, documents = load_test_data(args.test_file)
    elif args.query and args.documents:
        query = args.query
        documents = args.documents
    else:
        print("Error: Either --test-file or both --query and --documents must be provided")
        sys.exit(1)
    
    print(f"Query: {query}")
    print(f"Number of documents: {len(documents)}")
    
    # Select rerankers to test
    reranker_map = {
        'jina': ('Jina Reranker', lambda: JinaReranker(device=device)),
        'mxbai': ('Mixedbread AI Reranker', lambda: MxbaiReranker()),
        'mxbai-v2': ('Mixedbread AI Reranker V2', lambda: MxbaiRerankV2()),
        'qwen': ('Qwen Reranker', lambda: QwenReranker(device=device)),
        'msmarco': ('MS MARCO Reranker', lambda: MSMarcoReranker(device=device)),
        'msmarco-v2': ('MS MARCO Reranker V2', lambda: MSMarcoRerankerV2(device=device)),
        'bge': ('BGE Reranker', lambda: BGEReranker(device=device))
    }
    
    if args.benchmark:
        # Run benchmark mode
        benchmark_results = {}
        
        if args.reranker:
            # Benchmark only specified reranker
            name, factory = reranker_map[args.reranker]
            try:
                reranker = factory()
                time_taken = benchmark_reranker(reranker, name, query, documents)
                if time_taken:
                    benchmark_results[name] = time_taken
            except Exception as e:
                print(f"Error initializing {name}: {str(e)}")
        else:
            # Benchmark all rerankers
            rerankers = initialize_rerankers(device)
            for name, reranker in rerankers.items():
                if reranker is not None:
                    time_taken = benchmark_reranker(reranker, name, query, documents)
                    if time_taken:
                        benchmark_results[name] = time_taken
        
        # Print benchmark summary
        if benchmark_results:
            print("\n" + "="*50)
            print("BENCHMARK SUMMARY")
            print("="*50)
            
            # Sort by execution time
            sorted_results = sorted(benchmark_results.items(), key=lambda x: x[1])
            
            print("\nReranker Performance (fastest to slowest):")
            for i, (name, time_taken) in enumerate(sorted_results, 1):
                print(f"  {i}. {name}: {time_taken:.4f} seconds")
    
    elif args.reranker:
        # Test only specified reranker
        name, factory = reranker_map[args.reranker]
        try:
            reranker = factory()
            run_reranker_test(reranker, name, query, documents, args.top_k)
        except Exception as e:
            print(f"Error initializing {name}: {str(e)}")
    else:
        # Test all rerankers
        rerankers = initialize_rerankers(device)
        for name, reranker in rerankers.items():
            if reranker is not None:  # Only test if initialization was successful
                run_reranker_test(reranker, name, query, documents, args.top_k)

if __name__ == "__main__":
    main()
