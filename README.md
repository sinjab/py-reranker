# py-reranker

This project provides implementations and comparisons of various reranker models.

## Reranker Implementations

1. **Jina Reranker** - `jinaai/jina-reranker-v2-base-multilingual`
2. **Mixedbread AI Reranker** - `mixedbread-ai/mxbai-rerank-large-v1`
3. **Mixedbread AI Reranker V2** - `mixedbread-ai/mxbai-rerank-large-v2`
3. **Qwen Reranker** - `Qwen/Qwen3-Reranker-4B`
4. **MS MARCO Reranker** - `cross-encoder/ms-marco-MiniLM-L12-v2`
5. **BGE Reranker** - `BAAI/bge-reranker-v2-m3`

## Installation

```bash
uv sync
```

## Usage

### CLI Interface

The project provides a CLI interface for testing rerankers:

```bash
# Test all rerankers with a JSON test file
uv run python main.py --test-file tests/test_qa.json

# Test a specific reranker
uv run python main.py --test-file tests/test_qa.json --reranker mxbai

# Test the new Mixedbread AI Reranker V2
uv run python main.py --test-file tests/test_qa.json --reranker mxbai-v2

# Test with inline query and documents
uv run python main.py --query "What is the capital of France?" --documents "Paris is the capital of France." "London is the capital of England." "Berlin is the capital of Germany."
```

### Scripts

To test all rerankers with all test files:

```bash
uv run python scripts/test_all.py
```

To test all rerankers with a specific test file:

```bash
uv run python scripts/test_rerankers.py --test-file tests/test_qa.json
```

To benchmark the performance of all rerankers with a specific test file:

```bash
uv run python scripts/benchmark.py --test-file tests/test_qa.json
```

## Project Structure

- `rerankers/` - Contains implementations for each reranker model
- `tests/` - Contains test data in JSON format
- `main.py` - CLI interface for testing rerankers
- `scripts/` - Contains test and benchmark scripts
  - `benchmark.py` - Performance benchmarking script
  - `test_all.py` - Comprehensive test script
  - `test_rerankers.py` - Simple test script
- `utils/` - Common utility functions
- `RESULTS.md` - Detailed results summary

## Dependencies

- torch
- transformers
- sentence-transformers
- einops

## Performance Benchmark (CPU)

Based on our benchmarking tests, here's the relative performance of each reranker:

1. **Jina Reranker** - Fastest inference time
2. **MS MARCO Reranker** - Second fastest
3. **Mixedbread AI Reranker** - Moderate speed
4. **BGE Reranker** - Slower inference
5. **Qwen Reranker** - Slowest inference (largest model)

## Notes

- All rerankers are tested on the same multilingual dataset
- Results are displayed with scores and top documents
- GPU acceleration is automatically used if available
- See `RESULTS.md` for detailed comparison of ranking quality
- Performance may vary significantly when using GPU vs CPU