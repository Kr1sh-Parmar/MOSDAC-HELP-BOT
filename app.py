# Main application with RAG pipeline
import os
import json
import numpy as np
import requests
import flask
import time
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import config
from processor import DataProcessor
from retriever import SimpleRetriever
from advanced_retriever import AdvancedRetriever
from cache import QueryCache
from evaluator import RAGEvaluator, QueryAnalyzer
import nltk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global variables for the RAG system
processor = None
retriever = None
chunks = None
embeddings = None
embedding_model = None
query_cache = None
evaluator = None
query_analyzer = None

# Store conversation history for context-aware retrieval
conversation_history = {}

def initialize_system():
    """Initialize the RAG system by loading models and processing data."""
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        print("Downloading NLTK punkt tokenizer data...")
        nltk.download('punkt_tab')
        nltk.download('punkt')
    
    global retriever, processor, embedding_model, query_cache, evaluator, query_analyzer
    
    # Initialize embedding model
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"Loaded embedding model: {config.EMBEDDING_MODEL}")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Load or process data
    if (os.path.exists(config.CHUNKS_PATH) and 
        os.path.exists(config.VECTORS_PATH)):
        
        print(f"Loading preprocessed data...")
        with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        embeddings = np.load(config.VECTORS_PATH)
        print(f"Loaded {len(chunks)} chunks and their embeddings")
    
    else:
        if not os.path.exists(config.RAW_DATA_PATH):
            raise FileNotFoundError(f"Raw data file not found at {config.RAW_DATA_PATH}")
        
        print(f"Processing raw data...")
        raw_data = processor.load_json_data(config.RAW_DATA_PATH)
        chunks = processor.hierarchical_chunking(raw_data)
        embeddings = processor.generate_embeddings(chunks)
        
        processor.save_json_data(chunks, config.CHUNKS_PATH)
        np.save(config.VECTORS_PATH, embeddings)
    
    # Initialize retriever (advanced or simple based on config)
    if config.USE_ADVANCED_RETRIEVER:
        retriever = AdvancedRetriever(chunks, embeddings, processor)
        print("Initialized advanced retriever with MMR and improved BM25")
    else:
        retriever = SimpleRetriever(chunks, embeddings, processor)
        print("Initialized simple retriever")
    
    # Initialize query cache if enabled
    if config.ENABLE_QUERY_CACHE:
        query_cache = QueryCache(
            cache_dir=config.CACHE_DIR,
            max_cache_size=config.MAX_CACHE_SIZE,
            cache_ttl=config.CACHE_TTL
        )
        print(f"Initialized query cache with {query_cache.cache_stats['size']} entries")
    
    # Initialize evaluation components
    if config.LOG_QUERIES:
        evaluator = RAGEvaluator(log_dir=config.EVAL_DIR)
        query_analyzer = QueryAnalyzer(log_dir=config.EVAL_DIR)
        print("Initialized evaluation system")
    
    print("RAG system initialized successfully!")

def query_gemini(prompt):
    """Query the Google Gemini API"""
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(
            f"{config.GEMINI_API_URL}?key={config.GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error ({response.status_code}): {response.text}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def process_query(query, session_id=None):
    """Process a query through the RAG pipeline"""
    start_time = time.time()
    
    # Check cache first if enabled
    if config.ENABLE_QUERY_CACHE and query_cache:
        cached_response = query_cache.get(query)
        if cached_response:
            print(f"Cache hit for query: {query}")
            # Log the cached query if evaluation is enabled
            if config.LOG_QUERIES and evaluator:
                latency = time.time() - start_time
                evaluator.log_query(
                    query=query,
                    retrieved_chunks=cached_response.get("chunks", []),
                    response=cached_response.get("answer", ""),
                    latency=latency,
                    token_usage={"source": "cache"}
                )
            return cached_response
    
    # Get query embedding
    query_embedding = embedding_model.encode(query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Get conversation history for this session if available
    history = None
    if session_id and session_id in conversation_history:
        history = conversation_history[session_id]["queries"][-5:] if conversation_history[session_id]["queries"] else None
    
    # Retrieve relevant chunks
    if config.USE_ADVANCED_RETRIEVER and isinstance(retriever, AdvancedRetriever) and history:
        # Use contextual search with conversation history
        retrieved_chunks = retriever.contextual_search(
            query_text=query,
            query_embedding=query_embedding,
            conversation_history=history,
            top_k=config.TOP_K
        )
    else:
        # Use standard search
        retrieved_chunks = retriever.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            top_k=config.TOP_K
        )
    
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        if config.INCLUDE_METADATA_IN_CONTEXT:
            context_part = f"[Document {i+1}] {chunk['metadata']['title']}\n"
            if chunk['metadata'].get('url'):
                context_part += f"Source: {chunk['metadata']['url']}\n"
            if 'score' in chunk:
                context_part += f"Relevance: {chunk['score']:.4f}\n"
            context_part += f"Content: {chunk['text']}\n"
        else:
            context_part = f"[Document {i+1}]\n{chunk['text']}"
        context_parts.append(context_part)
    
    context = "\n\n".join(context_parts)
    
    # Generate prompt based on configuration
    if config.USE_STRUCTURED_PROMPT:
        prompt = f"""You are a helpful assistant for ISRO MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre). 
Answer the question based ONLY on the provided context.

If the answer isn't in the context, respond "I don't have enough information to answer this question."
If you're unsure, explain what information is missing.

Context:
{context}

Question: {query}

Provide a clear, concise answer with the following structure:
1. Direct answer to the question
2. Brief explanation with supporting details from the context
3. Mention the source document numbers that contain the information"""
    else:
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.
If the answer isn't in the context, respond "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer in clear, concise English:"""
    
    # Query the LLM
    response = query_gemini(prompt)
    
    # Update conversation history
    if session_id:
        if session_id not in conversation_history:
            conversation_history[session_id] = {"queries": []}
        conversation_history[session_id]["queries"].append(query)
        # Limit history size
        if len(conversation_history[session_id]["queries"]) > 10:
            conversation_history[session_id]["queries"].pop(0)
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Prepare result
    result = {
        "query": query,
        "answer": response,
        "sources": [chunk["metadata"] for chunk in retrieved_chunks],
        "latency": f"{latency:.2f} seconds"
    }
    
    # Cache the result if caching is enabled
    if config.ENABLE_QUERY_CACHE and query_cache:
        cache_entry = {
            "answer": response,
            "sources": [chunk["metadata"] for chunk in retrieved_chunks],
            "chunks": retrieved_chunks
        }
        query_cache.set(query, cache_entry)
    
    # Log the query if evaluation is enabled
    if config.LOG_QUERIES and evaluator:
        evaluator.log_query(
            query=query,
            retrieved_chunks=retrieved_chunks,
            response=response,
            latency=latency
        )
    
    return result

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query', '')
    session_id = data.get('session_id', None)
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    result = process_query(query, session_id)
    
    return jsonify(result)

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    stats = {
        "system": {
            "chunks_count": len(retriever.chunks) if retriever else 0,
            "embedding_model": config.EMBEDDING_MODEL,
            "advanced_retriever": config.USE_ADVANCED_RETRIEVER
        }
    }
    
    # Add cache stats if available
    if config.ENABLE_QUERY_CACHE and query_cache:
        stats["cache"] = query_cache.get_stats()
    
    # Add evaluation stats if available
    if config.LOG_QUERIES and evaluator and query_analyzer:
        stats["evaluation"] = evaluator.calculate_metrics()
        stats["query_patterns"] = query_analyzer.analyze_query_patterns()
    
    return jsonify(stats)

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the query cache"""
    if config.ENABLE_QUERY_CACHE and query_cache:
        query_cache.clear()
        return jsonify({"status": "success", "message": "Cache cleared successfully"})
    else:
        return jsonify({"status": "error", "message": "Caching is not enabled"}), 400

@app.route('/chat')
def chat_interface():
    return flask.render_template('chat.html')

@app.route('/system-stats')
def system_stats():
    """Display system statistics page"""
    return flask.render_template('stats.html')

if __name__ == '__main__':
    initialize_system()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
