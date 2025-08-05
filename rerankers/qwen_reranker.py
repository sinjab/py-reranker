"""
Qwen Reranker Implementation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenReranker:
    def __init__(self, model_name='Qwen/Qwen3-Reranker-4B', device='cpu'):
        """
        Initialize the Qwen Reranker
        
        Args:
            model_name (str): The name of the model to use
            device (str): The device to run the model on ('cpu' or 'cuda')
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self.device = device
        self.model.to(self.device)
        
        # Token IDs for yes/no
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        # Prefix and suffix tokens
        prefix = "\n\n\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".\n\n\n\n"
        suffix = "\n\n\n\n\n\n\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
    
    def format_instruction(self, instruction, query, doc):
        """
        Format the instruction for the reranker
        """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)
        return output
    
    def process_inputs(self, pairs):
        """
        Process input pairs
        """
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs
    
    def compute_logits(self, inputs):
        """
        Compute logits from model outputs
        """
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def compute_score(self, query, documents, task=None):
        """
        Compute relevance scores for the given query and documents
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            task (str): The task instruction
            
        Returns:
            list: List of relevance scores
        """
        if task is None:
            task = 'Given a web search query, retrieve relevant passages that answer the query'
        
        # Create pairs
        pairs = [self.format_instruction(task, query, doc) for doc in documents]
        
        # Process inputs
        inputs = self.process_inputs(pairs)
        
        # Compute scores
        scores = self.compute_logits(inputs)
        
        return scores
    
    def rank(self, query, documents, top_n=None, task=None):
        """
        Rank documents based on relevance to the query
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            top_n (int): Number of top documents to return (None for all)
            task (str): The task instruction
            
        Returns:
            list: List of (document, score) tuples sorted by score
        """
        scores = self.compute_score(query, documents, task)
        ranked = list(zip(documents, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            return ranked[:top_n]
        return ranked
