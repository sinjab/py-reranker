# Reranker Comparison Results

This document shows the results of testing various reranker models with different test datasets. The project now supports using different JSON test files for evaluation rather than hard-coded data. The scripts have been reorganized into a `scripts/` directory for better structure.

## Test Results by JSON File

### 1. test_multilingual.json
**Query**: "Organic skincare products for sensitive skin"
**Documents**: 10 multilingual documents

| Rank | Jina Reranker | Mixedbread AI | Qwen Reranker | MS MARCO | BGE Reranker |
|------|---------------|---------------|---------------|----------|--------------|
| 1 | 针对敏感肌专门设计的天然有机护肤产品 (2.5156) | Organic skincare for sensitive skin with aloe vera and chamomile. (0.9625) | Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla (0.9737) | Organic skincare for sensitive skin with aloe vera and chamomile. (9.5712) | 敏感肌のために特別に設計された天然有機スキンケア製品 (6.7607) |
| 2 | 敏感肌のために特別に設計された天然有機スキンケア製品 (1.6719) | 敏感肌のために特別に設計された天然有機スキンケア製品 (0.8244) | Organic skincare for sensitive skin with aloe vera and chamomile. (0.9629) | Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla (-8.0715) | 针对敏感肌专门设计的天然有机护肤产品 (5.8228) |
| 3 | Organic skincare for sensitive skin with aloe vera and chamomile. (1.5938) | Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla (0.7566) | 敏感肌のために特別に設計された天然有機スキンケア製品 (0.9549) | Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille (-9.4963) | Organic skincare for sensitive skin with aloe vera and chamomile. (4.4280) |

### 2. test_capital.json
**Query**: "What is the capital of China?"
**Documents**: 3 documents

| Rank | Jina Reranker | Mixedbread AI | Qwen Reranker | MS MARCO | BGE Reranker |
|------|---------------|---------------|---------------|----------|--------------|
| 1 | The capital of China is Beijing. (1.7734) | The capital of China is Beijing. (0.9993) | The capital of China is Beijing. (0.9402) | The capital of China is Beijing. (9.5041) | The capital of China is Beijing. (10.3537) |
| 2 | China is a large country in Asia. (-1.5156) | China is a large country in Asia. (0.1517) | China is a large country in Asia. (0.6552) | China is a large country in Asia. (-1.8919) | China is a large country in Asia. (-2.5243) |
| 3 | Paris is the capital of France. (-1.6406) | Paris is the capital of France. (0.0043) | Paris is the capital of France. (0.2435) | Paris is the capital of France. (-5.1392) | Paris is the capital of France. (-7.9853) |

### 3. test_ml.json
**Query**: "What is machine learning?"
**Documents**: 3 documents

| Rank | Jina Reranker | Mixedbread AI | Qwen Reranker | MS MARCO | BGE Reranker |
|------|---------------|---------------|---------------|----------|--------------|
| 1 | Machine learning is a subset of artificial intelligence. (2.1875) | Machine learning is a subset of artificial intelligence. (0.9819) | Machine learning is a subset of artificial intelligence. (0.9620) | Machine learning is a subset of artificial intelligence. (10.0738) | Machine learning is a subset of artificial intelligence. (7.4042) |
| 2 | Deep learning uses neural networks. (-1.1328) | Deep learning uses neural networks. (0.9017) | Deep learning uses neural networks. (0.5750) | Deep learning uses neural networks. (-6.1697) | Deep learning uses neural networks. (-6.3878) |
| 3 | The weather today is sunny. (-3.0938) | The weather today is sunny. (0.0015) | The weather today is sunny. (0.2650) | The weather today is sunny. (-11.3155) | The weather today is sunny. (-11.0403) |

### 4. test_cooking.json
**Query**: "How to cook pasta?"
**Documents**: 3 documents

| Rank | Jina Reranker | Mixedbread AI | Qwen Reranker | MS MARCO | BGE Reranker |
|------|---------------|---------------|---------------|----------|--------------|
| 1 | Boil water, add pasta, cook 8-12 minutes. (0.4297) | Boil water, add pasta, cook 8-12 minutes. (0.9726) | Boil water, add pasta, cook 8-12 minutes. (0.9677) | Boil water, add pasta, cook 8-12 minutes. (4.2800) | Boil water, add pasta, cook 8-12 minutes. (1.1716) |
| 2 | Pasta is made from wheat flour. (-1.0469) | Pasta is made from wheat flour. (0.4133) | Italy is famous for pasta. (0.7941) | Pasta is made from wheat flour. (2.5001) | Pasta is made from wheat flour. (-2.8848) |
| 3 | Italy is famous for pasta. (-2.3906) | Italy is famous for pasta. (0.2572) | Pasta is made from wheat flour. (0.7148) | Italy is famous for pasta. (-3.2866) | Italy is famous for pasta. (-8.0610) |

### 5. test_simple.json
**Query**: "simple test"
**Documents**: 1 document

| Rank | Jina Reranker | Mixedbread AI | Qwen Reranker | MS MARCO | BGE Reranker |
|------|---------------|---------------|---------------|----------|--------------|
| 1 | This is a test document (-0.7578) | This is a test document (0.0281) | This is a test document (0.4917) | This is a test document (-6.5597) | This is a test document (2.1532) |

### 6. test_invalid.json
**Query**: "test"
**Documents**: 1 document (minimal content)

| Rank | Jina Reranker | Mixedbread AI | Qwen Reranker | MS MARCO | BGE Reranker |
|------|---------------|---------------|---------------|----------|--------------|
| 1 | doc (-3.0156) | doc (0.1144) | doc (0.4210) | doc (-10.1967) | doc (-3.6468) |

### 7. test_qa.json
**Query**: "Who wrote 'To Kill a Mockingbird'?"
**Documents**: 3 documents

| Rank | Jina Reranker | Mixedbread AI | Qwen Reranker | MS MARCO | BGE Reranker |
|------|---------------|---------------|---------------|----------|--------------|
| 1 | Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926. (0.7188) | Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926. (0.9692) | Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926. (0.9674) | Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926. (10.3088) | Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926. (6.9600) |
| 2 | Nelle Harper Lee was awarded the Pulitzer Prize for Fiction in 1961 for 'To Kill a Mockingbird'. (0.6797) | Nelle Harper Lee was awarded the Pulitzer Prize for Fiction in 1961 for 'To Kill a Mockingbird'. (0.9530) | Nelle Harper Lee was awarded the Pulitzer Prize for Fiction in 1961 for 'To Kill a Mockingbird'. (0.9590) | Nelle Harper Lee was awarded the Pulitzer Prize for Fiction in 1961 for 'To Kill a Mockingbird'. (9.1400) | Nelle Harper Lee was awarded the Pulitzer Prize for Fiction in 1961 for 'To Kill a Mockingbird'. (6.1122) |
| 3 | The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, has sold over 500 million copies worldwide. (-2.6406) | The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, has sold over 500 million copies worldwide. (0.0085) | The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, has sold over 500 million copies worldwide. (0.5819) | The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, has sold over 500 million copies worldwide. (-2.5135) | The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, has sold over 500 million copies worldwide. (-2.5135) |

### 8. test_empty.json
**Query**: "test"
**Documents**: 0 documents

All models failed with "list index out of range" error due to empty document array.

## Analysis

All rerankers successfully identified the most relevant documents for queries when documents were provided. The models showed different scoring patterns:

- **Jina Reranker**: Gave highest scores to Chinese and Japanese translations of the query
- **Mixedbread AI**: Ranked the exact English match highest
- **Qwen Reranker**: Performed well with Spanish text
- **MS MARCO**: Showed extreme differences between relevant and irrelevant documents
- **BGE Reranker**: Ranked Japanese and Chinese translations highly

The differences in scoring approaches highlight the importance of understanding each model's characteristics when selecting a reranker for a specific use case. The empty document test case shows that all models need to handle edge cases better.
