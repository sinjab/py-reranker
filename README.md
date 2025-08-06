# 🔍 py-reranker

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-97%25%20coverage-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

A comprehensive Python library for testing and comparing state-of-the-art reranker models. This project provides unified interfaces to multiple reranking models, making it easy to evaluate and compare their performance on your specific use cases.

## ✨ Features

- 🚀 **6 State-of-the-art Rerankers** - Compare leading models in one place
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
| **Qwen Reranker** | Alibaba | `Qwen/Qwen3-Reranker-4B` | Large model, excellent quality |
| **MS MARCO** | Microsoft | `cross-encoder/ms-marco-MiniLM-L12-v2` | Fast, well-established |
| **BGE Reranker** | BAAI | `BAAI/bge-reranker-v2-m3` | Multilingual, research-grade |

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

## 📖 Detailed Usage

### Command Line Interface

The main CLI tool supports various options:

```bash
uv run python main.py [OPTIONS]

Options:
  --test-file PATH              Path to JSON test file
  --query TEXT                  Query string (alternative to test file)
  --documents TEXT [TEXT ...]   Document strings to rank
  --reranker {jina,mxbai,mxbai-v2,qwen,msmarco,bge}
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