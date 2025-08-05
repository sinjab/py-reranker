"""
Mixedbread AI Reranker Implementation
"""

from sentence_transformers import CrossEncoder

class MxbaiReranker:
    def __init__(self, model_name='mixedbread-ai/mxbai-rerank-large-v1'):
        """
        Initialize the Mixedbread AI Reranker
        
        Args:
            model_name (str): The name of the model to use
        """
        self.model = CrossEncoder(model_name)
    
    def rank(self, query, documents, top_k=None, return_documents=True):
        """
        Rank documents based on relevance to the query
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            top_k (int): Number of top documents to return (None for all)
            return_documents (bool): Whether to return the documents with scores
            
        Returns:
            list: List of results with scores and optionally documents
        """
        # Prepare the input
        sentence_pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self.model.predict(sentence_pairs)
        
        # Create results
        results = []
        for i, doc in enumerate(documents):
            result = {'index': i, 'score': float(scores[i])}
            if return_documents:
                result['text'] = doc
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k if specified
        if top_k:
            return results[:top_k]
        return results
