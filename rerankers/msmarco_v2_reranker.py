"""
MS MARCO Reranker Implementation using cross-encoder/ms-marco-MiniLM-L12-v2
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class MSMarcoReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L12-v2', device='cpu'):
        """
        Initialize the MS MARCO Reranker
        
        Args:
            model_name (str): The name of the model to use
            device (str): The device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
    
    def compute_score(self, query, documents):
        """
        Compute relevance scores for the given query and documents
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            
        Returns:
            list: List of relevance scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        queries = [query] * len(documents)
        
        # Tokenize the pairs
        features = self.tokenizer(
            queries, 
            documents,
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Move features to device
        features = {k: v.to(self.device) for k, v in features.items()}
        
        # Compute scores
        self.model.eval()
        with torch.no_grad():
            scores = self.model(**features).logits
        
        # Return scores as a list
        return scores.squeeze().cpu().tolist() if len(documents) > 1 else [scores.squeeze().cpu().item()]
    
    def rank(self, query, documents, top_n=None):
        """
        Rank documents based on relevance to the query
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            top_n (int): Number of top documents to return (None for all)
            
        Returns:
            list: List of (document, score) tuples sorted by score
        """
        if not documents:
            return []
        
        # Compute scores
        scores = self.compute_score(query, documents)
        
        # Create ranked list of (document, score) tuples
        ranked = list(zip(documents, scores))
        
        # Sort by score in descending order
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n results if specified
        if top_n is not None:
            return ranked[:top_n]
        
        return ranked
