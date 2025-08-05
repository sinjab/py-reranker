"""
Jina Reranker Implementation
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class JinaReranker:
    def __init__(self, model_name='jinaai/jina-reranker-v2-base-multilingual', device='cpu'):
        """
        Initialize the Jina Reranker
        
        Args:
            model_name (str): The name of the model to use
            device (str): The device to run the model on ('cpu' or 'cuda')
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def compute_score(self, query, documents, max_length=1024):
        """
        Compute relevance scores for the given query and documents
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            max_length (int): Maximum sequence length
            
        Returns:
            list: List of relevance scores
        """
        # Construct sentence pairs
        sentence_pairs = [[query, doc] for doc in documents]
        
        # Compute scores
        with torch.no_grad():
            inputs = self.tokenizer(
                sentence_pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            scores = self.model(**inputs).logits.view(-1, ).float()
        
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
        ranked = list(zip(documents, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            return ranked[:top_n]
        return ranked
