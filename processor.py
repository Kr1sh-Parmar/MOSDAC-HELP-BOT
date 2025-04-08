# Data processing utilities
import json
import os
import re
import nltk
import hashlib
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import config

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DataProcessor:
    def __init__(self):
        """Initialize the data processor"""
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"Loaded embedding model: {config.EMBEDDING_MODEL}")
        
        # Load NLTK stopwords for text processing
        from nltk.corpus import stopwords
        self.stopwords = set(stopwords.words('english'))
        
        # Initialize TF-IDF vectorizer for document clustering
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def load_json_data(self, file_path):
        """Load JSON data from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Ensure all items in the list are dictionaries
            data = [json.loads(doc) if isinstance(doc, str) else doc for doc in data]
        
        print(f"Loaded {len(data)} documents from {file_path}")
        return data
    
    def save_json_data(self, data, file_path):
        """Save data to JSON file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved data to {file_path}")
    
    def hierarchical_chunking(self, documents):
        """Process documents with hierarchical chunking and improved metadata"""
        chunks = []
        
        for doc_id, doc in enumerate(tqdm(documents, desc="Processing documents")):
            # Extract document content and metadata
            content = doc.get("content", "")
            title = doc.get("title", "")
            url = doc.get("url", "")
            date = doc.get("date", "")
            author = doc.get("author", "")
            category = doc.get("category", "")
            
            if not content:
                continue
            
            # Clean HTML if present
            content = self._clean_html(content)
            
            # Extract key phrases for better metadata
            key_phrases = self._extract_key_phrases(content)
            
            # Generate a document hash for tracking
            doc_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # First split into larger parent chunks (semantic sections)
            parent_chunks = self._split_text_semantic(content, config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)
            
            for parent_id, parent_chunk in enumerate(parent_chunks):
                # Generate a summary for the parent chunk
                parent_summary = self._generate_chunk_summary(parent_chunk)
                
                # Split parent chunk into smaller chunks
                child_chunks = self._split_text(parent_chunk, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                
                for child_id, child_chunk in enumerate(child_chunks):
                    # Calculate chunk quality score
                    quality_score = self._calculate_chunk_quality(child_chunk)
                    
                    # Create chunk ID for reference
                    chunk_id = f"chunk_{doc_hash}_{parent_id}_{child_id}"
                    
                    # Add chunk with enhanced metadata
                    chunks.append({
                        "text": child_chunk,
                        "metadata": {
                            "doc_id": doc_id,
                            "title": title,
                            "url": url,
                            "date": date,
                            "author": author,
                            "category": category,
                            "parent_id": f"parent_{doc_id}_{parent_id}",
                            "child_id": chunk_id,
                            "parent_summary": parent_summary,
                            "key_phrases": key_phrases[:5],  # Top 5 key phrases
                            "quality_score": quality_score,
                            "word_count": len(child_chunk.split())
                        }
                    })
        
        return chunks
    
    def _clean_html(self, html_content):
        """Clean HTML content - enhanced version"""
        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', ' ', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<style[^>]*>.*?</style>', ' ', html_content, flags=re.DOTALL)
        
        # Remove HTML comments
        html_content = re.sub(r'<!--.*?-->', ' ', html_content, flags=re.DOTALL)
        
        # Replace common HTML entities
        html_content = html_content.replace('&nbsp;', ' ')
        html_content = html_content.replace('&amp;', '&')
        html_content = html_content.replace('&lt;', '<')
        html_content = html_content.replace('&gt;', '>')
        
        # Remove HTML tags but preserve headings with special markers
        html_content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'## HEADING1: \1 ##', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## HEADING2: \1 ##', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'## HEADING3: \1 ##', html_content, flags=re.DOTALL)
        
        # Preserve list items
        html_content = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1', html_content, flags=re.DOTALL)
        
        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Fix whitespace issues
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common artifacts
        text = re.sub(r'\s+\.', '.', text)  # Remove space before period
        text = re.sub(r'\s+,', ',', text)    # Remove space before comma
        
        return text
    
    def _split_text(self, text, chunk_size, chunk_overlap):
        """Split text into chunks by sentences"""
        if not text:
            return []
            
        # Get sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds the chunk size and we already have content,
            # then store the current chunk and start a new one
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep some sentences for overlap
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_text_semantic(self, text, chunk_size, chunk_overlap):
        """Split text into semantic chunks based on headings and natural breaks"""
        if not text:
            return []
        
        # First look for semantic section breaks (headings)
        heading_pattern = r'## HEADING\d: (.*?) ##'
        headings = re.findall(heading_pattern, text)
        
        # If we found headings, use them to split the text
        if headings:
            # Split by headings
            sections = re.split(heading_pattern, text)
            
            # Recombine heading with its content
            semantic_sections = []
            for i in range(1, len(sections), 2):
                if i < len(sections) - 1:
                    section_title = sections[i]
                    section_content = sections[i+1]
                    semantic_sections.append(f"## {section_title} ##\n{section_content}")
            
            # Further split large sections
            chunks = []
            for section in semantic_sections:
                if len(section) > chunk_size:
                    # Split large sections into smaller chunks
                    section_chunks = self._split_text(section, chunk_size, chunk_overlap)
                    chunks.extend(section_chunks)
                else:
                    chunks.append(section)
            
            return chunks
        
        # If no headings, try to split by natural paragraph breaks
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            # Combine paragraphs into chunks
            chunks = []
            current_chunk = []
            current_size = 0
            
            for paragraph in paragraphs:
                paragraph_size = len(paragraph)
                
                if current_size + paragraph_size > chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    
                    # Calculate overlap
                    overlap_size = 0
                    overlap_paragraphs = []
                    for p in reversed(current_chunk):
                        if overlap_size + len(p) <= chunk_overlap:
                            overlap_paragraphs.insert(0, p)
                            overlap_size += len(p)
                        else:
                            break
                    
                    current_chunk = overlap_paragraphs
                    current_size = sum(len(p) for p in current_chunk)
                
                current_chunk.append(paragraph)
                current_size += paragraph_size
            
            # Add the last chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            
            return chunks
        
        # Fall back to sentence-based splitting
        return self._split_text(text, chunk_size, chunk_overlap)
    
    def _extract_key_phrases(self, text):
        """Extract key phrases from text using TF-IDF"""
        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stopwords]
        
        # Count term frequencies
        term_freq = {}
        for token in tokens:
            if token in term_freq:
                term_freq[token] += 1
            else:
                term_freq[token] = 1
        
        # Get bigrams (pairs of adjacent words)
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            bigrams.append(bigram)
        
        # Count bigram frequencies
        bigram_freq = {}
        for bigram in bigrams:
            if bigram in bigram_freq:
                bigram_freq[bigram] += 1
            else:
                bigram_freq[bigram] = 1
        
        # Combine unigrams and bigrams, sort by frequency
        all_phrases = {**term_freq, **bigram_freq}
        sorted_phrases = sorted(all_phrases.items(), key=lambda x: x[1], reverse=True)
        
        # Return top phrases
        return [phrase for phrase, _ in sorted_phrases[:20]]
    
    def _calculate_chunk_quality(self, chunk):
        """Calculate a quality score for a chunk based on various heuristics"""
        # Initialize score
        score = 1.0
        
        # Penalize very short chunks
        words = chunk.split()
        word_count = len(words)
        if word_count < 20:
            score *= 0.5
        
        # Reward chunks with headings (likely more informative)
        if re.search(r'## HEADING\d:', chunk):
            score *= 1.5
        
        # Reward chunks with bullet points (structured information)
        bullet_count = chunk.count('•')
        if bullet_count > 0:
            score *= (1.0 + min(bullet_count / 10, 0.5))
        
        # Penalize chunks with very little information density
        unique_words = set(word.lower() for word in words if word.isalnum())
        if len(unique_words) > 0:
            info_density = len(unique_words) / word_count
            if info_density < 0.4:  # Low unique word ratio
                score *= 0.7
        
        return score
    
    def _generate_chunk_summary(self, chunk):
        """Generate a brief summary of a chunk"""
        # Extract the first sentence as a summary
        sentences = sent_tokenize(chunk)
        if not sentences:
            return ""
        
        first_sentence = sentences[0]
        
        # If there's a heading, use it as part of the summary
        heading_match = re.search(r'## HEADING\d: (.*?) ##', chunk)
        if heading_match:
            heading = heading_match.group(1)
            return f"{heading}: {first_sentence}"
        
        # Truncate long summaries
        if len(first_sentence) > 100:
            return first_sentence[:97] + "..."
        
        return first_sentence
    
    def generate_embeddings(self, chunks):
        """Generate embeddings for all text chunks"""
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batches to save memory
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized_embeddings = embeddings_array / norms
        
        return normalized_embeddings
    
    def prepare_bm25_corpus(self, chunks):
        """Prepare corpus for BM25 search"""
        tokenized_corpus = []
        
        for chunk in tqdm(chunks, desc="Preparing BM25 corpus"):
            # Tokenize and lowercase
            tokens = word_tokenize(chunk["text"].lower())
            tokenized_corpus.append(tokens)
        
        return tokenized_corpus
