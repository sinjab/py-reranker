"""
Mixedbread AI Reranker V2 Implementation
"""

from mxbai_rerank import MxbaiRerankV2 as MxbaiRerankV2Lib

class MxbaiRerankV2:
    def __init__(self, model_name='mixedbread-ai/mxbai-rerank-large-v2'):
        """
        Initialize the Mixedbread AI Reranker V2
        
        Args:
            model_name (str): The name of the model to use
        """
        self.model = MxbaiRerankV2Lib(model_name)
        self.model_name = model_name
    
    def rank(self, query, documents, top_k=None, return_documents=True):
        """
        Rank documents based on relevance to the query using V2 model
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            top_k (int): Number of top documents to return (None for all)
            return_documents (bool): Whether to return the documents with scores
            
        Returns:
            list: List of results with scores and optionally documents
        """
        # Get scores
        results = self.model.rank(query, documents, return_documents=return_documents, top_k=top_k)
        
        # Convert to the format expected by the rest of the project
        formatted_results = []
        for result in results:
            formatted_result = (result.document, result.score)
            formatted_results.append(formatted_result)
        
        # Return results
        return formatted_results
