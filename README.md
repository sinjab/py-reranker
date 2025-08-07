# 🔍 py-reranker

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-97%25%20coverage-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

A comprehensive Python library for testing and comparing state-of-the-art reranker models. This project provides unified interfaces to multiple reranking models, making it easy to evaluate and compare their performance on your specific use cases.

## ✨ Features

- 🚀 **14 State-of-the-art Rerankers** - Compare leading models in one place
- 🎯 **Unified API** - Consistent interface across all models
- 📊 **Comprehensive Testing** - 97% test coverage with robust validation
- 🌍 **Multilingual Support** - Test with multiple languages
- ⚡ **GPU/CPU Optimization** - Automatic device detection and optimization
- 📈 **Benchmarking Tools** - Performance analysis and comparison utilities
- 🛠️ **CLI Interface** - Easy-to-use command-line tools

## 🤖 Supported Reranker Models

| Model | Provider | Model ID | Strengths |
|-------|----------|----------|----------|
| **Jina Reranker** | Jina AI | `jinaai/jina-reranker-v2-base-multilingual` | Fast inference, multilingual |
| **MixedBread AI v1** | MixedBread AI | `mixedbread-ai/mxbai-rerank-large-v1` | Balanced performance |
| **MixedBread AI v2** | MixedBread AI | `mixedbread-ai/mxbai-rerank-large-v2` | Latest generation, high accuracy |
| **Qwen Reranker 0.6B** | Alibaba | `Qwen/Qwen3-Reranker-0.6B` | Fastest, smallest model |
| **Qwen Reranker 4B** | Alibaba | `Qwen/Qwen3-Reranker-4B` | Balanced size and quality |
| **Qwen Reranker 8B** | Alibaba | `Qwen/Qwen3-Reranker-8B` | Largest, highest accuracy |
| **MS MARCO** | Microsoft | `cross-encoder/ms-marco-MiniLM-L12-v2` | Fast, well-established |
| **BGE Base** | BAAI | `BAAI/bge-reranker-base` | Fast, lightweight baseline |
| **BGE Large** | BAAI | `BAAI/bge-reranker-large` | Larger, more accurate |
| **BGE V2-M3** | BAAI | `BAAI/bge-reranker-v2-m3` | Latest multilingual model |
| **BGE V2-Gemma** | BAAI | `BAAI/bge-reranker-v2-gemma` | LLM-based reranker |
| **BGE V2-MiniCPM-Layerwise** | BAAI | `BAAI/bge-reranker-v2-minicpm-layerwise` | Advanced layerwise model |
| **BGE V2.5-Gemma2-Lightweight** | BAAI | `BAAI/bge-reranker-v2.5-gemma2-lightweight` | Lightweight LLM reranker* |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/py-reranker.git
cd py-reranker

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Basic Usage

#### 1. Test All Rerankers with Sample Data

```bash
# Run all rerankers on machine learning test data
uv run python main.py --test-file tests/data/test_ml.json --top-k 3
```

#### 2. Test Specific Reranker

```bash
# Test only MixedBread AI v2 reranker
uv run python main.py --test-file tests/data/test_qa.json --reranker mxbai-v2
```

#### 3. Custom Query and Documents

```bash
# Test with your own query and documents
uv run python main.py \
  --query "What is artificial intelligence?" \
  --documents \
    "AI is machine intelligence" \
    "Cooking is an art" \
    "Neural networks are powerful" \
  --reranker mxbai-v2 \
  --top-k 2

# Test different Qwen model sizes
uv run python main.py \
  --query "What is machine learning?" \
  --documents \
    "AI is machine intelligence" \
    "Cooking is an art" \
    "Neural networks are powerful" \
  --reranker qwen-0.6b  # Fastest Qwen model

uv run python main.py \
  --query "What is machine learning?" \
  --documents \
    "AI is machine intelligence" \
    "Cooking is an art" \
    "Neural networks are powerful" \
  --reranker qwen-8b    # Most accurate Qwen model
```

### Example Output

```
Using device: cpu
Query: What is machine learning?
Number of documents: 3

=== MixedBread AI Reranker V2 Results ===
1. Score: 9.8525
   Document: Machine learning is a subset of artificial intelligence.
2. Score: 1.3604
   Document: Deep learning uses neural networks.
3. Score: -4.7962
   Document: The weather today is sunny.
```

### 🎯 BGE Model Variants

The BGE (BAAI General Embedding) rerankers offer multiple model sizes and architectures:

| Model | CLI Option | Model Size | Type | Best For |
|-------|------------|------------|------|----------|
| **BGE Base** | `bge-base` | ~110M params | Standard | Fast inference, baseline |
| **BGE Large** | `bge-large` | ~340M params | Standard | Better accuracy |
| **BGE V2-M3** | `bge-v2-m3` | 568M params | Standard | Multilingual, lightweight, fast inference |
| **BGE V2-Gemma** | `bge-v2-gemma` | 2.51B params | LLM-based | Multilingual, strong performance |
| **BGE V2-MiniCPM-Layerwise** | `bge-v2-minicpm-layerwise` | 2.72B params | Layerwise | Layer selection, accelerated inference |
| **BGE V2.5-Gemma2-Lightweight** | `bge-v2.5-gemma2-lightweight` | 2.72B params | Lightweight LLM | Layer selection, compression, efficiency* |

#### Model Selection Guide

**Choose your BGE model based on your requirements:**

- **For multilingual contexts**: Use `bge-v2-m3`, `bge-v2-gemma`, or `bge-v2.5-gemma2-lightweight`
- **For Chinese or English**: Use `bge-v2-m3` or `bge-v2-minicpm-layerwise`
- **For efficiency**: Use `bge-v2-m3` or low layers of `bge-v2-minicpm-layerwise`
- **For best performance**: Use `bge-v2-minicpm-layerwise` or `bge-v2-gemma`

> 💡 **Tip**: Always test on your real use case and choose the model with the best speed-quality balance!

#### BGE Usage Examples

```bash
# Fast baseline model
uv run python main.py \
  --query "What is machine learning?" \
  --documents "ML is AI subset" "Deep learning uses neural networks" \
  --reranker bge-base

# High accuracy model
uv run python main.py \
  --query "What is machine learning?" \
  --documents "ML is AI subset" "Deep learning uses neural networks" \
  --reranker bge-large

# Latest multilingual model (default BGE)
uv run python main.py \
  --query "What is machine learning?" \
  --documents "ML is AI subset" "Deep learning uses neural networks" \
  --reranker bge-v2-m3

# LLM-based reranker for complex queries
uv run python main.py \
  --query "Explain the relationship between neural networks and deep learning" \
  --documents "Neural networks are the foundation" "Deep learning uses multiple layers" \
  --reranker bge-v2-gemma

# Lightweight LLM reranker (requires newer transformers)
uv run python main.py \
  --query "What is machine learning?" \
  --documents "ML is AI subset" "Deep learning uses neural networks" \
  --reranker bge-v2.5-gemma2-lightweight
```

#### Programmatic BGE Usage

```python
from rerankers import (
    BGEReranker,           # Generic class with model_size parameter
    BGERerankerBase,       # Convenience class for base model
    BGERerankerLarge,      # Convenience class for large model
    BGERerankerV2M3,       # Convenience class for V2-M3 model
    BGERerankerV2Gemma,    # Convenience class for V2-Gemma model
    BGERerankerV2MiniCPMLayerwise,  # Convenience class for layerwise model
    BGERerankerV25Gemma2Lightweight  # Convenience class for lightweight model
)

# Method 1: Using generic class with model_size parameter
reranker = BGEReranker(model_size='base')  # or 'large', 'v2-m3', 'v2-gemma', 'v2-minicpm-layerwise', 'v2.5-gemma2-lightweight'

# Method 2: Using convenience classes
base_reranker = BGERerankerBase()
large_reranker = BGERerankerLarge()
v2m3_reranker = BGERerankerV2M3()
gemma_reranker = BGERerankerV2Gemma(use_bf16=True)  # Enable bf16 for faster inference
layerwise_reranker = BGERerankerV2MiniCPMLayerwise()  # Uses bf16 by default
lightweight_reranker = BGERerankerV25Gemma2Lightweight()  # Requires newer transformers

# Compute scores
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "The weather today is sunny"
]

scores = base_reranker.compute_score(query, documents)
print(f"Base model scores: {scores}")

# Rank documents
ranked = large_reranker.rank(query, documents, top_n=2)
for i, (doc, score) in enumerate(ranked, 1):
    print(f"{i}. Score: {score:.4f} - {doc[:50]}...")

# For layerwise model, you can specify which layers to use
layerwise_scores = layerwise_reranker.compute_score(
    query, documents, cutoff_layers=[28]  # Use layer 28 for scoring
)

# For lightweight model, you can specify compression parameters
try:
    lightweight_scores = lightweight_reranker.compute_score(
        query, documents, 
        cutoff_layers=[28], 
        compress_ratio=2, 
        compress_layer=[24, 40]
    )
except ImportError as e:
    print(f"Lightweight model requires newer transformers: {e}")
```

#### Alternative: Official FlagEmbedding Usage

If you prefer to use the official FlagEmbedding library directly (requires `pip install -U FlagEmbedding`):

```python
from FlagEmbedding import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker, LightWeightFlagLLMReranker

# Standard models (bge-reranker-v2-m3, bge-reranker-base, bge-reranker-large)
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
score = reranker.compute_score(['query', 'passage'])

# LLM-based model (bge-reranker-v2-gemma)
llm_reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)
score = llm_reranker.compute_score(['query', 'passage'])

# Layerwise model (bge-reranker-v2-minicpm-layerwise)
layerwise_reranker = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True)
score = layerwise_reranker.compute_score(['query', 'passage'], cutoff_layers=[28])

# Lightweight model (bge-reranker-v2.5-gemma2-lightweight)
lightweight_reranker = LightWeightFlagLLMReranker('BAAI/bge-reranker-v2.5-gemma2-lightweight', use_fp16=True)
score = lightweight_reranker.compute_score(['query', 'passage'], cutoff_layers=[28], compress_ratio=2, compress_layer=[24, 40])
```

> **Note**: Our implementation uses transformers directly for better compatibility and unified API, but both approaches produce equivalent results.

## 📖 Detailed Usage

### Command Line Interface

The main CLI tool supports various options:

```bash
uv run python main.py [OPTIONS]

Options:
  --test-file PATH              Path to JSON test file
  --query TEXT                  Query string (alternative to test file)
  --documents TEXT [TEXT ...]   Document strings to rank
  --reranker {jina,mxbai,mxbai-v2,qwen,qwen-0.6b,qwen-4b,qwen-8b,msmarco,msmarco-v2,bge,bge-base,bge-large,bge-v2-m3,bge-v2-gemma,bge-v2-minicpm-layerwise,bge-v2.5-gemma2-lightweight}
                               Specific reranker to use (default: all)
  --top-k INTEGER              Number of top results to return (default: 3)
  --benchmark                  Run performance benchmark instead of normal ranking
  --help                       Show help message
```

### Test Data Format

Create JSON files with the following structure:

```json
{
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny.",
    "Deep learning uses neural networks."
  ]
}
```

### Available Scripts

#### Test All Models with All Data Files
```bash
# Comprehensive testing across all models and test files
./test-all.sh
```

#### Benchmark Performance
```bash
# Performance benchmarking with timing analysis
uv run python main.py --benchmark --test-file tests/data/test_qa.json

# Benchmark specific reranker
uv run python main.py --benchmark --reranker mxbai-v2 --test-file tests/data/test_ml.json

# Benchmark all rerankers with inline query/documents
uv run python main.py --benchmark --query "What is machine learning?" --documents "ML is AI" "Deep learning uses neural networks"
```

## 🧪 Testing

This project maintains high code quality with comprehensive testing:

```bash
# Run all tests with coverage report
uv run pytest --cov=. --cov-report=term-missing

# Run specific test categories
uv run pytest tests/test_main.py -v
uv run pytest tests/test_utils.py -v
uv run pytest tests/test_rerankers.py -v
```

**Test Coverage: 97%** ✅
- `main.py`: 98% coverage
- `utils/common.py`: 86% coverage
- All reranker modules: 89-100% coverage

## 📁 Project Structure

```
py-reranker/
├── 📁 rerankers/           # Reranker model implementations
│   ├── jina_reranker.py
│   ├── mxbai_reranker.py
│   ├── mxbai_v2_reranker.py
│   ├── qwen_reranker.py
│   ├── msmarco_reranker.py
│   └── bge_reranker.py
├── 📁 tests/               # Test suite
│   ├── 📁 data/            # JSON test files
│   ├── conftest.py
│   ├── test_main.py
│   ├── test_utils.py
│   └── test_rerankers.py
├── test-all.sh             # Shell script for batch testing
├── 📁 utils/               # Common utilities
│   └── common.py
├── main.py                 # Main CLI interface
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## 📊 Performance Comparison

### Speed Benchmark (CPU)

Based on our benchmarking tests:

| Rank | Model | Relative Speed | Model Size | Best Use Case |
|------|-------|----------------|------------|---------------|
| 🥇 | Jina Reranker | Fastest | Small | Real-time applications |
| 🥈 | MS MARCO | Fast | Small | Production systems |
| 🥉 | MixedBread AI v1 | Moderate | Medium | Balanced performance |
| 4️⃣ | MixedBread AI v2 | Moderate | Medium | Latest accuracy |
| 5️⃣ | BGE Reranker | Slower | Large | Research/Quality focus |
| 6️⃣ | Qwen Reranker | Slowest | Largest | Maximum accuracy |

### Quality vs Speed Trade-off

- **Speed Priority**: Jina Reranker, MS MARCO
- **Balanced**: MixedBread AI models
- **Quality Priority**: Qwen Reranker, BGE Reranker

## 🛠️ Development

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure tests pass: `uv run pytest`
5. Submit a pull request

### Adding New Rerankers

1. Create a new file in `rerankers/` following the existing pattern
2. Implement the required interface methods
3. Add tests in `tests/test_rerankers.py`
4. Update the main CLI to include the new model

## 📋 Requirements

- **Python**: 3.8+
- **PyTorch**: Latest stable version
- **Transformers**: Latest stable version
- **Additional**: See `pyproject.toml` for complete dependencies

### Model-Specific Requirements

- **BGE V2.5-Gemma2-Lightweight**: Requires the latest transformers version (>= 4.45.0) for compatibility with Gemma2 architecture. If you encounter import errors, upgrade transformers:
  ```bash
  pip install --upgrade transformers
  ```

## 🌟 Use Cases

- **Search Systems**: Improve search result ranking
- **RAG Applications**: Enhance retrieval quality in RAG pipelines
- **Question Answering**: Rank candidate answers by relevance
- **Document Retrieval**: Find most relevant documents for queries
- **Model Comparison**: Evaluate different reranking approaches
- **Research**: Academic research on information retrieval

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- [Jina AI](https://jina.ai/) for the multilingual reranker
- [MixedBread AI](https://mixedbread.ai/) for the high-performance rerankers
- [Alibaba](https://github.com/QwenLM) for the Qwen reranker
- [Microsoft](https://microsoft.com/) for the MS MARCO model
- [BAAI](https://www.baai.ac.cn/) for the BGE reranker

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/py-reranker/issues) page
2. Create a new issue with detailed information
3. Join our community discussions

---

⭐ **Star this repository if you find it helpful!** ⭐