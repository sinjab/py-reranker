"""
BGE Reranker Implementation
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class BGEReranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', device='cpu', use_fp16=False):
        """
        Initialize the BGE Reranker
        
        Args:
            model_name (str): The name of the model to use
            device (str): The device to run the model on ('cpu' or 'cuda')
            use_fp16 (bool): Whether to use fp16 for faster computation
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device
        
        if use_fp16:
            self.model = self.model.half()
            
        self.model.to(self.device)
        self.model.eval()
    
    def compute_score(self, query, documents, normalize=False, max_length=512):
        """
        Compute relevance scores for the given query and documents
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            normalize (bool): Whether to normalize scores to 0-1 range using sigmoid
            max_length (int): Maximum sequence length
            
        Returns:
            list: List of relevance scores
        """
        # Prepare sentence pairs
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Compute scores
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu()
            
            # Normalize if requested
            if normalize:
                scores = torch.sigmoid(scores)
        
        return scores.tolist()
    
    def rank(self, query, documents, top_n=None, normalize=False):
        """
        Rank documents based on relevance to the query
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            top_n (int): Number of top documents to return (None for all)
            normalize (bool): Whether to normalize scores to 0-1 range using sigmoid
            
        Returns:
            list: List of (document, score) tuples sorted by score
        """
        scores = self.compute_score(query, documents, normalize)
        ranked = list(zip(documents, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            return ranked[:top_n]
        return ranked
