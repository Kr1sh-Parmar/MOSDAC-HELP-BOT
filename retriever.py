# Document retrieval functionality
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import config

class SimpleRetriever:
    def __init__(self, chunks, embeddings, processor):
        """Initialize the retriever with document chunks and their embeddings"""
        self.chunks = chunks
        self.embeddings = embeddings
        self.processor = processor
        
        # Prepare BM25-like components
        self.tokenized_corpus = processor.prepare_bm25_corpus(chunks)
        self.doc_freqs = self._calculate_doc_frequencies()
        self.total_docs = len(chunks)
    
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
    
    def hybrid_search(self, query_text, query_embedding, top_k=5):
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
        top_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Return the chunks with highest scores
        return [self.chunks[idx] for idx, score in top_results]
    
    def _semantic_search(self, query_embedding):
        """Perform semantic search using cosine similarity"""
        # Calculate cosine similarity between query and all document embeddings
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return similarities
    
    def _keyword_search(self, query_text):
        """Perform keyword-based search similar to BM25"""
        # Tokenize query
        query_tokens = word_tokenize(query_text.lower())
        
        # Calculate scores for each document
        scores = [0] * len(self.tokenized_corpus)
        
        for token in query_tokens:
            if token in self.doc_freqs:
                # Get document frequency
                df = self.doc_freqs[token]
                idf = np.log((self.total_docs + 1) / (df + 0.5))
                
                # Score each document
                for i, doc_tokens in enumerate(self.tokenized_corpus):
                    # Calculate term frequency
                    tf = doc_tokens.count(token) / len(doc_tokens) if doc_tokens else 0
                    
                    # BM25-like score (simplified)
                    k1 = 1.2
                    b = 0.75
                    doc_len = len(doc_tokens)
                    avg_doc_len = sum(len(d) for d in self.tokenized_corpus) / len(self.tokenized_corpus)
                    
                    tf_score = ((k1 + 1) * tf) / (k1 * (1 - b + b * (doc_len / avg_doc_len)) + tf)
                    scores[i] += idf * tf_score
        
        return scores
