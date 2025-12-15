

import logging
import os
import pathlib

import numpy as np

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer


class Qwen3EmbeddingModel:
    def __init__(self, model_path="Qwen/Qwen3-Embedding-0.6B", **kwargs):

        self.model = SentenceTransformer(model_path, **kwargs)
        
        # Optional: Enable flash_attention_2 for better performance
        # Uncomment below if you have flash-attn installed:
        # self.model = SentenceTransformer(
        #     model_path,
        #     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        #     tokenizer_kwargs={"padding_side": "left"},
        # )
    
    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs) -> np.ndarray:

        # Use the query prompt for better query representation
        query_embeddings = self.model.encode(
            queries, 
            prompt_name="query",
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            **kwargs
        )
        return np.asarray(query_embeddings)
    
    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
 
        sentences = [(doc.get("title", "") + " " + doc.get("text", "")).strip() for doc in corpus]
        
        # Encode documents (no special prompt needed for documents)
        doc_embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            **kwargs
        )
        return np.asarray(doc_embeddings)