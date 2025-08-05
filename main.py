#!/usr/bin/env python3

import argparse
import json
import os
import sys
import torch
from rerankers.jina_reranker import JinaReranker
from rerankers.mxbai_reranker import MxbaiReranker
from rerankers.qwen_reranker import QwenReranker
from rerankers.msmarco_reranker import MSMarcoReranker
from rerankers.bge_reranker import BGEReranker

def load_test_data(test_file):
    """Load test data from JSON file"""
    with open(test_file, 'r') as f:
        data = json.load(f)
    return data['query'], data['documents']

def test_reranker(reranker, name, query, documents, top_k=3):
    """Test a reranker and print results"""
    print(f"\n=== {name} Results ===")
    try:
        if name == "Mixedbread AI Reranker":
            results = reranker.rank(query, documents, top_k=top_k)
            for i, result in enumerate(results):
                print(f"{i+1}. Score: {result['score']:.4f}")
                print(f"   Document: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
        else:
            results = reranker.rank(query, documents, top_n=top_k)
            for i, (doc, score) in enumerate(results):
                print(f"{i+1}. Score: {score:.4f}")
                print(f"   Document: {doc[:100]}{'...' if len(doc) > 100 else ''}")
    except Exception as e:
        print(f"Error testing {name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Test various reranker models')
    parser.add_argument('--test-file', type=str, help='Path to JSON test file')
    parser.add_argument('--query', type=str, help='Query string (if not using test file)')
    parser.add_argument('--documents', type=str, nargs='+', help='Document strings (if not using test file)')
    parser.add_argument('--reranker', type=str, choices=['jina', 'mxbai', 'qwen', 'msmarco', 'bge'], 
                        help='Specific reranker to use (default: all)')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top results to return')
    
    args = parser.parse_args()
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        'qwen': ('Qwen Reranker', lambda: QwenReranker(device=device)),
        'msmarco': ('MS MARCO Reranker', lambda: MSMarcoReranker(device=device)),
        'bge': ('BGE Reranker', lambda: BGEReranker(device=device))
    }
    
    if args.reranker:
        # Test only specified reranker
        name, factory = reranker_map[args.reranker]
        try:
            reranker = factory()
            test_reranker(reranker, name, query, documents, args.top_k)
        except Exception as e:
            print(f"Error initializing {name}: {str(e)}")
    else:
        # Test all rerankers
        for name, factory in reranker_map.values():
            try:
                reranker = factory()
                test_reranker(reranker, name, query, documents, args.top_k)
            except Exception as e:
                print(f"Error initializing {name}: {str(e)}")

if __name__ == "__main__":
    main()
