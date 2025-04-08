# Advanced document retrieval functionality
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import config
from typing import List, Dict, Tuple, Any, Optional

class AdvancedRetriever:
    def __init__(self, chunks, embeddings, processor):
        """Initialize the retriever with document chunks and their embeddings"""
        self.chunks = chunks
        self.embeddings = embeddings
        self.processor = processor
        
        # Prepare BM25-like components
        self.tokenized_corpus = processor.prepare_bm25_corpus(chunks)
        self.doc_freqs = self._calculate_doc_frequencies()
        self.total_docs = len(chunks)
        
        # Calculate average document length for BM25
        self.avg_doc_len = sum(len(d) for d in self.tokenized_corpus) / len(self.tokenized_corpus) if self.tokenized_corpus else 0
        
        # BM25 parameters
        self.k1 = 1.5  # Term frequency saturation parameter
        self.b = 0.75  # Length normalization parameter
    
    def _calculate_doc_frequencies(self):
        """Calculate document frequencies for each term"""
        doc_freqs = {}
        
        for doc_tokens in self.tokenized_corpus:
            # Count each term only once per document
            for token in set(doc_tokens):
                if token in doc_freqs:
                    doc_freqs[token] += 1
                else:
                    doc_freqs[token] = 1
        
        return doc_freqs
    
    def hybrid_search(self, query_text: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Perform hybrid search combining semantic and keyword-based retrieval"""
        # Get semantic search results
        semantic_scores = self._semantic_search(query_embedding)
        
        # Get keyword search results
        keyword_scores = self._keyword_search(query_text)
        
        # Combine scores
        combined_scores = []
        
        for i in range(len(self.chunks)):
            # Normalize and combine scores with weighting
            sem_score = semantic_scores[i]
            key_score = keyword_scores[i] if i < len(keyword_scores) else 0
            
            # Combined score with configurable weighting
            score = (config.SEMANTIC_WEIGHT * sem_score + 
                    (1 - config.SEMANTIC_WEIGHT) * key_score)
            
            combined_scores.append((i, score))
        
        # Sort by score and get top results
        top_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k*2]  # Get more results for MMR
        
        # Apply Maximum Marginal Relevance for diversity if enabled
        if config.USE_MMR:
            selected_indices = self._maximum_marginal_relevance(
                query_embedding=query_embedding,
                candidate_indices=[idx for idx, _ in top_results],
                lambda_param=config.MMR_LAMBDA,
                k=top_k
            )
            results = [self.chunks[idx] for idx in selected_indices]
        else:
            # Just take the top k results
            results = [self.chunks[idx] for idx, _ in top_results[:top_k]]
        
        # Add scores to results for debugging/analysis
        for i, result in enumerate(results):
            result["score"] = top_results[i][1] if i < len(top_results) else 0.0
        
        return results
    
    def _semantic_search(self, query_embedding: np.ndarray) -> np.ndarray:
        """Perform semantic search using cosine similarity"""
        # Calculate cosine similarity between query and all document embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return similarities
    
    def _keyword_search(self, query_text: str) -> List[float]:
        """Perform keyword-based search using BM25 algorithm"""
        # Tokenize query
        query_tokens = word_tokenize(query_text.lower())
        
        # Calculate scores for each document
        scores = [0] * len(self.tokenized_corpus)
        
        for token in query_tokens:
            if token in self.doc_freqs:
                # Get document frequency
                df = self.doc_freqs[token]
                idf = np.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                # Score each document
                for i, doc_tokens in enumerate(self.tokenized_corpus):
                    # Calculate term frequency
                    tf = doc_tokens.count(token)
                    doc_len = len(doc_tokens)
                    
                    # BM25 score
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[i] += idf * (numerator / denominator) if denominator != 0 else 0
        
        return scores
    
    def _maximum_marginal_relevance(self, 
                                    query_embedding: np.ndarray, 
                                    candidate_indices: List[int], 
                                    lambda_param: float = 0.5, 
                                    k: int = 5) -> List[int]:
        """Apply Maximum Marginal Relevance to rerank results for diversity
        
        Args:
            query_embedding: The query embedding vector
            candidate_indices: Indices of candidate documents
            lambda_param: Balance between relevance and diversity (0-1)
                          Higher values favor relevance over diversity
            k: Number of results to return
            
        Returns:
            List of selected document indices
        """
        if not candidate_indices:
            return []
        
        # Get embeddings for candidates
        candidate_embeddings = self.embeddings[candidate_indices]
        
        # Calculate relevance scores (cosine similarity to query)
        relevance_scores = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # Initialize selected indices and remaining candidates
        selected_indices = []
        remaining_indices = list(range(len(candidate_indices)))
        
        # Select the most relevant document first
        best_idx = np.argmax(relevance_scores)
        selected_indices.append(candidate_indices[best_idx])
        remaining_indices.remove(best_idx)
        
        # Select remaining documents using MMR
        while len(selected_indices) < k and remaining_indices:
            # Calculate similarity to already selected documents
            selected_embeddings = self.embeddings[selected_indices]
            similarities = cosine_similarity(candidate_embeddings[remaining_indices], selected_embeddings)
            
            # Calculate MMR scores
            mmr_scores = []
            for i, idx in enumerate(remaining_indices):
                # Relevance term
                relevance = relevance_scores[idx]
                # Diversity term (max similarity to any selected document)
                max_similarity = np.max(similarities[i]) if len(selected_indices) > 0 else 0
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            next_best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(candidate_indices[next_best_idx])
            remaining_indices.remove(next_best_idx)
        
        return selected_indices
    
    def contextual_search(self, query_text: str, query_embedding: np.ndarray, 
                         conversation_history: List[str] = None, top_k: int = 5) -> List[Dict]:
        """Perform search with conversation context awareness
        
        Args:
            query_text: The current query text
            query_embedding: The current query embedding
            conversation_history: List of previous queries in the conversation
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks
        """
        # If no conversation history, just do regular search
        if not conversation_history or len(conversation_history) == 0:
            return self.hybrid_search(query_text, query_embedding, top_k)
        
        # Get embeddings for conversation history
        history_embeddings = self.processor.embedding_model.encode(conversation_history)
        
        # Normalize embeddings
        history_embeddings = history_embeddings / np.linalg.norm(history_embeddings, axis=1, keepdims=True)
        
        # Calculate context-aware query embedding (weighted average)
        # More weight to current query, less to history
        weights = np.array([0.7] + [0.3 / len(history_embeddings)] * len(history_embeddings))
        combined_embedding = np.average(
            np.vstack([query_embedding.reshape(1, -1), history_embeddings]), 
            axis=0, 
            weights=weights
        )
        
        # Normalize the combined embedding
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
        # Use the combined embedding for search
        return self.hybrid_search(query_text, combined_embedding, top_k)