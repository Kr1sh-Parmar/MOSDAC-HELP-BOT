# Configuration settings
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Data paths
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "data.json")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")
VECTORS_PATH = os.path.join(PROCESSED_DIR, "vectors.npy")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")

os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200

# Retrieval parameters
TOP_K = 5
SEMANTIC_WEIGHT = 0.7
RERANKING_ENABLED = True

# Advanced retrieval parameters
USE_ADVANCED_RETRIEVER = True  # Set to True to use the advanced retriever
USE_MMR = True  # Use Maximum Marginal Relevance for diversity
MMR_LAMBDA = 0.7  # Balance between relevance (1.0) and diversity (0.0)
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# Caching parameters
ENABLE_QUERY_CACHE = True  # Enable query caching
MAX_CACHE_SIZE = 100  # Maximum number of queries to cache
CACHE_TTL = 86400  # Cache time-to-live in seconds (24 hours)

# Prompt engineering
USE_STRUCTURED_PROMPT = True  # Use structured prompts with specific sections
INCLUDE_METADATA_IN_CONTEXT = True  # Include document metadata in context
CONTEXT_STRATEGY = "ranked"  # Options: ranked, mmr, clustered

# Evaluation settings
LOG_QUERIES = True  # Log queries for evaluation
EVAL_METRICS = ["latency", "token_usage"]  # Metrics to track

# Server settings
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# Embedding model
EMBEDDING_MODEL = "all-mpnet-base-v2"
