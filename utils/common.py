"""
Common utility functions for the py-reranker project
"""

import json
import torch
from rerankers import (
    JinaReranker,
    MxbaiReranker,
    MxbaiRerankV2,
    QwenReranker,
    QwenReranker0_6B,
    QwenReranker4B,
    QwenReranker8B,
    MSMarcoReranker,
    MSMarcoRerankerV2,
    BGEReranker,
    BGERerankerBase,
    BGERerankerLarge,
    BGERerankerV2M3,
    BGERerankerV2Gemma,
    BGERerankerV2MiniCPMLayerwise,
    BGERerankerV25Gemma2Lightweight
)

def load_test_data(test_file):
    """Load test data from JSON file"""
    with open(test_file, 'r') as f:
        data = json.load(f)
    return data['query'], data['documents']

def run_reranker_test(reranker, name, query, documents, top_k=3):
    """Test a reranker and print results"""
    print(f"\n=== {name} Results ===")
    try:
        if name == "Mixedbread AI Reranker":
            results = reranker.rank(query, documents, top_k=top_k)
            for i, result in enumerate(results):
                print(f"{i+1}. Score: {result['score']:.4f}")
                print(f"   Document: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
        elif name == "Mixedbread AI Reranker V2":
            results = reranker.rank(query, documents, top_k=top_k)
            for i, (doc, score) in enumerate(results):
                print(f"{i+1}. Score: {score:.4f}")
                print(f"   Document: {doc[:100]}{'...' if len(doc) > 100 else ''}")
        else:
            results = reranker.rank(query, documents, top_n=top_k)
            for i, (doc, score) in enumerate(results):
                print(f"{i+1}. Score: {score:.4f}")
                print(f"   Document: {doc[:100]}{'...' if len(doc) > 100 else ''}")
    except Exception as e:
        print(f"Error testing {name}: {str(e)}")

def initialize_rerankers(device='cpu'):
    """Initialize all rerankers"""
    rerankers = {}
    
    try:
        rerankers["Jina Reranker"] = JinaReranker(device=device)
    except Exception as e:
        print(f"Error initializing Jina Reranker: {str(e)}")
    
    try:
        rerankers["Mixedbread AI Reranker"] = MxbaiReranker()
    except Exception as e:
        print(f"Error initializing Mixedbread AI Reranker: {str(e)}")
    
    try:
        rerankers["Mixedbread AI Reranker V2"] = MxbaiRerankV2()
    except Exception as e:
        print(f"Error initializing Mixedbread AI Reranker V2: {str(e)}")
    
    try:
        rerankers["Qwen Reranker 4B"] = QwenReranker(device=device)
    except Exception as e:
        print(f"Error initializing Qwen Reranker 4B: {str(e)}")
    
    try:
        rerankers["Qwen Reranker 4B (explicit)"] = QwenReranker4B(device=device)
    except Exception as e:
        print(f"Error initializing Qwen Reranker 4B (explicit): {str(e)}")
    
    try:
        rerankers["Qwen Reranker 0.6B"] = QwenReranker0_6B(device=device)
    except Exception as e:
        print(f"Error initializing Qwen Reranker 0.6B: {str(e)}")
    
    try:
        rerankers["Qwen Reranker 8B"] = QwenReranker8B(device=device)
    except Exception as e:
        print(f"Error initializing Qwen Reranker 8B: {str(e)}")
    
    try:
        rerankers["MS MARCO Reranker"] = MSMarcoReranker(device=device)
    except Exception as e:
        print(f"Error initializing MS MARCO Reranker: {str(e)}")
    
    try:
        rerankers["MS MARCO Reranker V2"] = MSMarcoRerankerV2(device=device)
    except Exception as e:
        print(f"Error initializing MS MARCO Reranker V2: {str(e)}")
    
    try:
        rerankers["BGE Reranker V2-M3"] = BGEReranker(device=device)
    except Exception as e:
        print(f"Error initializing BGE Reranker V2-M3: {str(e)}")
    
    try:
        rerankers["BGE Reranker Base"] = BGERerankerBase(device=device)
    except Exception as e:
        print(f"Error initializing BGE Reranker Base: {str(e)}")
    
    try:
        rerankers["BGE Reranker Large"] = BGERerankerLarge(device=device)
    except Exception as e:
        print(f"Error initializing BGE Reranker Large: {str(e)}")
    
    try:
        rerankers["BGE Reranker V2-M3 (Class)"] = BGERerankerV2M3(device=device)
    except Exception as e:
        print(f"Error initializing BGE Reranker V2-M3 (Class): {str(e)}")
    
    try:
        rerankers["BGE Reranker V2-Gemma"] = BGERerankerV2Gemma(device=device)
    except Exception as e:
        print(f"Error initializing BGE Reranker V2-Gemma: {str(e)}")
    
    try:
        rerankers["BGE Reranker V2-MiniCPM-Layerwise"] = BGERerankerV2MiniCPMLayerwise(device=device)
    except Exception as e:
        print(f"Error initializing BGE Reranker V2-MiniCPM-Layerwise: {str(e)}")
    
    try:
        rerankers["BGE Reranker V2.5-Gemma2-Lightweight"] = BGERerankerV25Gemma2Lightweight(device=device)
    except Exception as e:
        print(f"Error initializing BGE Reranker V2.5-Gemma2-Lightweight: {str(e)}")
    
    return rerankers

def get_device():
    """Determine device to use for computation"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def benchmark_reranker(reranker, name, query, documents):
    """Benchmark a reranker and return timing and results"""
    import time
    print(f"\nBenchmarking {name}...")
    start_time = time.time()
    
    try:
        if name == "Mixedbread AI Reranker":
            results = reranker.rank(query, documents, top_k=3)
        elif name == "Mixedbread AI Reranker V2":
            results = reranker.rank(query, documents, top_k=3)
        else:
            results = reranker.rank(query, documents, top_n=3)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"  Execution time: {execution_time:.4f} seconds")
        print(f"  Top result score: {results[0][1] if name != 'Mixedbread AI Reranker' else results[0]['score']:.4f}")
        
        return execution_time
    except Exception as e:
        print(f"  Error: {str(e)}")
        return None
