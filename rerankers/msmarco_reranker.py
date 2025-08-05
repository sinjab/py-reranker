"""
MS MARCO Reranker Implementation
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
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
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
        # Prepare sentence pairs
        sentence_pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        with torch.no_grad():
            features = self.tokenizer(
                [pair[0] for pair in sentence_pairs], 
                [pair[1] for pair in sentence_pairs],
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}
            
            # Compute scores
            scores = self.model(**features).logits
        
        return scores.cpu().tolist()
    
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
        scores = self.compute_score(query, documents)
        # Flatten the scores list
        scores = [score[0] for score in scores]
        ranked = list(zip(documents, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            return ranked[:top_n]
        return ranked
