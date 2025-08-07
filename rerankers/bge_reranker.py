"""
BGE Reranker Implementation with Multiple Model Support

Supported models:
- bge-reranker-base: Base model (BAAI/bge-reranker-base)
- bge-reranker-large: Large model (BAAI/bge-reranker-large) 
- bge-reranker-v2-m3: V2 M3 model (BAAI/bge-reranker-v2-m3) - Default
- bge-reranker-v2-gemma: LLM-based reranker (BAAI/bge-reranker-v2-gemma)
- bge-reranker-v2-minicpm-layerwise: Layerwise reranker (BAAI/bge-reranker-v2-minicpm-layerwise)
- bge-reranker-v2.5-gemma2-lightweight: Lightweight LLM reranker (BAAI/bge-reranker-v2.5-gemma2-lightweight)
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch

# Note: FlagEmbedding has compatibility issues, so we use transformers directly

class BGEReranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', model_size=None, device='cpu', use_fp16=False, use_bf16=False):
        """
        Initialize the BGE Reranker
        
        Args:
            model_name (str): The name of the model to use
            model_size (str): Shorthand for model size ('base', 'large', 'v2-m3', 'v2-gemma', 'v2-minicpm-layerwise')
            device (str): The device to run the model on ('cpu' or 'cuda')
            use_fp16 (bool): Whether to use fp16 for faster computation
            use_bf16 (bool): Whether to use bf16 for faster computation (LLM models)
        """
        # Handle shorthand model sizes
        if model_size:
            model_map = {
                'base': 'BAAI/bge-reranker-base',
                'large': 'BAAI/bge-reranker-large',
                'v2-m3': 'BAAI/bge-reranker-v2-m3',
                'v2-gemma': 'BAAI/bge-reranker-v2-gemma',
                'v2-minicpm-layerwise': 'BAAI/bge-reranker-v2-minicpm-layerwise',
                'v2.5-gemma2-lightweight': 'BAAI/bge-reranker-v2.5-gemma2-lightweight'
            }
            if model_size in model_map:
                model_name = model_map[model_size]
            else:
                raise ValueError(f"Unsupported model size: {model_size}. Choose from: {list(model_map.keys())}")
        
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        
        # Determine model type based on model name
        self.is_llm_based = 'gemma' in model_name.lower() and 'lightweight' not in model_name.lower()
        self.is_layerwise = 'layerwise' in model_name.lower()
        self.is_lightweight = 'lightweight' in model_name.lower()
        
        # Initialize tokenizer
        if self.is_layerwise:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model based on type
        if self.is_lightweight:
            # Use transformers directly for lightweight models
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.tokenizer.padding_side = 'right'
                self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
                
                if use_fp16:
                    self.model = self.model.half()
                elif use_bf16:
                    self.model = self.model.to(torch.bfloat16)
                    
                self.model.to(self.device)
                self.model.eval()
                return
            except ImportError as e:
                if 'Gemma2FlashAttention2' in str(e) or 'GEMMA2_START_DOCSTRING' in str(e):
                    raise ImportError(
                        f"BGE V2.5 Gemma2 Lightweight model requires a newer version of transformers. "
                        f"Current version: {torch.__version__ if 'torch' in globals() else 'unknown'}. "
                        f"Please upgrade transformers to the latest version: pip install --upgrade transformers"
                    ) from e
                else:
                    raise
        elif self.is_llm_based or self.is_layerwise:
            if self.is_layerwise:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if use_bf16 else None
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # For LLM-based models, get the 'Yes' token location
            if self.is_llm_based:
                self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        else:
            # Normal reranker models
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Apply precision settings
        if use_fp16 and not use_bf16:
            self.model = self.model.half()
        elif use_bf16 and not self.is_layerwise:  # layerwise already handles bf16 in from_pretrained
            self.model = self.model.to(torch.bfloat16)
            
        self.model.to(self.device)
        self.model.eval()
    
    def compute_score(self, query, documents, normalize=False, max_length=512, cutoff_layers=None, compress_ratio=None, compress_layer=None):
        """
        Compute relevance scores for the given query and documents
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            normalize (bool): Whether to normalize scores to 0-1 range using sigmoid
            max_length (int): Maximum sequence length
            cutoff_layers (list): For layerwise models, which layers to use for scoring
            compress_ratio (int): For lightweight models, compression ratio
            compress_layer (list): For lightweight models, which layers to compress
            
        Returns:
            list: List of relevance scores
        """
        if self.is_lightweight:
            return self._compute_lightweight_score(query, documents, normalize, cutoff_layers, compress_ratio, compress_layer)
        elif self.is_llm_based:
            return self._compute_llm_score(query, documents, normalize, max_length)
        elif self.is_layerwise:
            return self._compute_layerwise_score(query, documents, normalize, max_length, cutoff_layers)
        else:
            return self._compute_normal_score(query, documents, normalize, max_length)
    
    def _compute_normal_score(self, query, documents, normalize=False, max_length=512):
        """Compute scores for normal reranker models (base, large, v2-m3)"""
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
    
    def _get_llm_inputs(self, pairs, prompt=None, max_length=1024):
        """Prepare inputs for LLM-based rerankers"""
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        
        sep = "\n"
        prompt_inputs = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep, return_tensors=None, add_special_tokens=False)['input_ids']
        
        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(
                f'A: {query}',
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True
            )
            passage_inputs = self.tokenizer(
                f'B: {passage}',
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True
            )
            
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        
        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )
    
    def _compute_llm_score(self, query, documents, normalize=False, max_length=1024):
        """Compute scores for LLM-based rerankers (v2-gemma)"""
        pairs = [[query, doc] for doc in documents]
        
        with torch.no_grad():
            inputs = self._get_llm_inputs(pairs, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            scores = self.model(**inputs, return_dict=True).logits[:, -1, self.yes_loc].view(-1, ).float()
            scores = scores.cpu()
            
            # Normalize if requested
            if normalize:
                scores = torch.sigmoid(scores)
        
        return scores.tolist()
    
    def _compute_layerwise_score(self, query, documents, normalize=False, max_length=1024, cutoff_layers=None):
        """Compute scores for layerwise rerankers (v2-minicpm-layerwise)"""
        if cutoff_layers is None:
            cutoff_layers = [28]  # Default layer
        
        pairs = [[query, doc] for doc in documents]
        
        with torch.no_grad():
            inputs = self._get_llm_inputs(pairs, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            all_scores = self.model(**inputs, return_dict=True, cutoff_layers=cutoff_layers)
            # Extract scores from the specified layers
            layer_scores = [scores[:, -1].view(-1, ).float() for scores in all_scores[0]]
            
            # Use the first (or only) layer's scores
            scores = layer_scores[0].cpu()
            
            # Normalize if requested
            if normalize:
                scores = torch.sigmoid(scores)
        
        return scores.tolist()
    
    def _last_logit_pool(self, logits, attention_mask):
        """Pool the last logits based on attention mask"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return logits[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = logits.shape[0]
            return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)
    
    def _get_lightweight_inputs(self, pairs, prompt=None, max_length=1024):
        """Prepare inputs for lightweight rerankers"""
        if prompt is None:
            prompt = "Predict whether passage B contains an answer to query A."
        
        sep = "\n"
        prompt_inputs = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep, return_tensors=None, add_special_tokens=False)['input_ids']
        
        inputs = []
        query_lengths = []
        prompt_lengths = []
        
        for query, passage in pairs:
            query_inputs = self.tokenizer(
                f'A: {query}',
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True
            )
            passage_inputs = self.tokenizer(
                f'B: {passage}',
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True
            )
            
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
            
            query_lengths.append(len([self.tokenizer.bos_token_id] + query_inputs['input_ids'] + sep_inputs))
            prompt_lengths.append(len(sep_inputs + prompt_inputs))
        
        padded_inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )
        
        return padded_inputs, query_lengths, prompt_lengths
    
    def _compute_lightweight_score(self, query, documents, normalize=False, cutoff_layers=None, compress_ratio=None, compress_layer=None):
        """Compute scores for lightweight rerankers (v2.5-gemma2-lightweight)"""
        # Set default parameters if not provided
        if cutoff_layers is None:
            cutoff_layers = [28]
        if compress_ratio is None:
            compress_ratio = 2
        if compress_layer is None:
            compress_layer = [24, 40]
        
        # Prepare pairs for processing
        pairs = [[query, doc] for doc in documents]
        
        with torch.no_grad():
            inputs, query_lengths, prompt_lengths = self._get_lightweight_inputs(pairs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Call model with lightweight parameters
            outputs = self.model(
                **inputs,
                return_dict=True,
                cutoff_layers=cutoff_layers,
                compress_ratio=compress_ratio,
                compress_layer=compress_layer,
                query_lengths=query_lengths,
                prompt_lengths=prompt_lengths
            )
            
            scores = []
            for i in range(len(outputs.logits)):
                logits = self._last_logit_pool(outputs.logits[i], outputs.attention_masks[i])
                scores.extend(logits.cpu().float().tolist())
            
            # Normalize if requested
            if normalize:
                scores = torch.sigmoid(torch.tensor(scores)).tolist()
        
        return scores
    
    def rank(self, query, documents, top_n=None, normalize=False, cutoff_layers=None, compress_ratio=None, compress_layer=None):
        """
        Rank documents based on relevance to the query
        
        Args:
            query (str): The query string
            documents (list): List of document strings
            top_n (int): Number of top documents to return (None for all)
            normalize (bool): Whether to normalize scores to 0-1 range using sigmoid
            cutoff_layers (list): For layerwise models, which layers to use for scoring
            compress_ratio (int): For lightweight models, compression ratio
            compress_layer (list): For lightweight models, which layers to compress
            
        Returns:
            list: List of (document, score) tuples sorted by score
        """
        scores = self.compute_score(query, documents, normalize, cutoff_layers=cutoff_layers, compress_ratio=compress_ratio, compress_layer=compress_layer)
        ranked = list(zip(documents, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            return ranked[:top_n]
        return ranked


# Convenience classes for different BGE model variants
class BGERerankerBase(BGEReranker):
    """BGE Base Reranker - Fastest, smaller model"""
    def __init__(self, device='cpu', use_fp16=False):
        super().__init__(model_size='base', device=device, use_fp16=use_fp16)


class BGERerankerLarge(BGEReranker):
    """BGE Large Reranker - Larger, more accurate model"""
    def __init__(self, device='cpu', use_fp16=False):
        super().__init__(model_size='large', device=device, use_fp16=use_fp16)


class BGERerankerV2M3(BGEReranker):
    """BGE V2 M3 Reranker - Default, balanced model"""
    def __init__(self, device='cpu', use_fp16=False):
        super().__init__(model_size='v2-m3', device=device, use_fp16=use_fp16)


class BGERerankerV2Gemma(BGEReranker):
    """BGE V2 Gemma Reranker - LLM-based reranker"""
    def __init__(self, device='cpu', use_fp16=False, use_bf16=False):
        super().__init__(model_size='v2-gemma', device=device, use_fp16=use_fp16, use_bf16=use_bf16)


class BGERerankerV2MiniCPMLayerwise(BGEReranker):
    """BGE V2 MiniCPM Layerwise Reranker - Advanced layerwise reranker"""
    def __init__(self, device='cpu', use_bf16=True):
        super().__init__(model_size='v2-minicpm-layerwise', device=device, use_bf16=use_bf16)


class BGERerankerV25Gemma2Lightweight(BGEReranker):
    """BGE V2.5 Gemma2 Lightweight Reranker - Efficient lightweight LLM reranker"""
    def __init__(self, device='cpu', use_fp16=True):
        super().__init__(model_size='v2.5-gemma2-lightweight', device=device, use_fp16=use_fp16)
