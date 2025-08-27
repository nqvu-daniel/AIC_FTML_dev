"""Text indexing using BM25"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import math

from ..core.base import Index


class BM25Index(Index):
    """BM25 text search index"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, config_dict: Dict[str, Any] = None):
        super().__init__("BM25Index", config_dict)
        self.k1 = k1
        self.b = b
        
        # Index data structures
        self.documents: List[Dict[str, Any]] = []
        self.term_frequencies: List[Counter] = []
        self.document_frequencies: Counter = Counter()
        self.document_lengths: List[int] = []
        self.average_document_length: float = 0.0
        self.num_documents: int = 0
        
    def add(self, documents: List[Dict[str, Any]], metadata: List[Dict[str, Any]] = None):
        """Add documents to the BM25 index"""
        for i, doc in enumerate(documents):
            tokens = doc.get('tokens', [])
            if not tokens:
                continue
                
            # Store document
            self.documents.append(doc)
            
            # Count term frequencies
            tf = Counter(tokens)
            self.term_frequencies.append(tf)
            
            # Update document frequencies
            for term in tf.keys():
                self.document_frequencies[term] += 1
                
            # Store document length
            self.document_lengths.append(len(tokens))
            
        # Update statistics
        self.num_documents = len(self.documents)
        if self.num_documents > 0:
            self.average_document_length = sum(self.document_lengths) / self.num_documents
            
        print(f"Added {len(documents)} documents to BM25 index")
        print(f"Total documents: {self.num_documents}")
        print(f"Vocabulary size: {len(self.document_frequencies)}")
        
    def search(self, query: str, k: int = 10) -> Tuple[List[float], List[int]]:
        """Search using BM25 scoring"""
        # Tokenize query
        query_tokens = query.lower().replace(",", "").split()
        
        if not query_tokens:
            return [], []
            
        # Calculate BM25 scores
        scores = []
        for doc_idx in range(self.num_documents):
            score = self._calculate_bm25_score(query_tokens, doc_idx)
            scores.append(score)
            
        # Sort by score and return top k
        scored_docs = list(enumerate(scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        top_k = scored_docs[:k]
        indices = [idx for idx, score in top_k]
        scores = [score for idx, score in top_k]
        
        return scores, indices
        
    def _calculate_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_tf = self.term_frequencies[doc_idx]
        doc_length = self.document_lengths[doc_idx]
        
        for term in query_tokens:
            if term in doc_tf:
                # Term frequency in document
                tf = doc_tf[term]
                
                # Document frequency (number of documents containing term)
                df = self.document_frequencies[term]
                
                # IDF calculation
                idf = math.log((self.num_documents - df + 0.5) / (df + 0.5))
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
                
                score += idf * (numerator / denominator)
                
        return score
        
    def save(self, index_path: Path):
        """Save BM25 index to disk"""
        index_data = {
            'documents': self.documents,
            'term_frequencies': [dict(tf) for tf in self.term_frequencies],
            'document_frequencies': dict(self.document_frequencies),
            'document_lengths': self.document_lengths,
            'average_document_length': self.average_document_length,
            'num_documents': self.num_documents,
            'k1': self.k1,
            'b': self.b
        }
        
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
            
        print(f"Saved BM25 index: {index_path}")
        
    def load(self, index_path: Path):
        """Load BM25 index from disk"""
        with open(index_path, 'r') as f:
            index_data = json.load(f)
            
        self.documents = index_data['documents']
        self.term_frequencies = [Counter(tf) for tf in index_data['term_frequencies']]
        self.document_frequencies = Counter(index_data['document_frequencies'])
        self.document_lengths = index_data['document_lengths']
        self.average_document_length = index_data['average_document_length']
        self.num_documents = index_data['num_documents']
        self.k1 = index_data.get('k1', 1.2)
        self.b = index_data.get('b', 0.75)
        
        print(f"Loaded BM25 index with {self.num_documents} documents")
        
    def process(self, corpus_data: List[Dict[str, Any]]) -> Any:
        """Process corpus data for pipeline compatibility"""
        self.add(corpus_data)
        return corpus_data