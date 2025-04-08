# Query cache implementation
import json
import os
import time
from datetime import datetime
import hashlib

class QueryCache:
    def __init__(self, cache_dir="data/cache", max_cache_size=100, cache_ttl=86400):
        """Initialize the query cache
        
        Args:
            cache_dir: Directory to store cache files
            max_cache_size: Maximum number of queries to cache
            cache_ttl: Time-to-live for cache entries in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl
        self.cache_index_path = os.path.join(cache_dir, "cache_index.json")
        self.cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "size": 0
        }
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache if available
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load the cache index from disk"""
        if os.path.exists(self.cache_index_path):
            try:
                with open(self.cache_index_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.cache = cache_data.get("entries", {})
                    self.cache_stats = cache_data.get("stats", self.cache_stats)
                print(f"Loaded {len(self.cache)} entries from cache")
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
                self.cache = {}
                self.cache_stats = {"hits": 0, "misses": 0, "size": 0}
    
    def _save_cache_index(self):
        """Save the cache index to disk"""
        try:
            cache_data = {
                "entries": self.cache,
                "stats": self.cache_stats,
                "last_updated": datetime.now().isoformat()
            }
            with open(self.cache_index_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")
    
    def _generate_key(self, query):
        """Generate a unique key for the query"""
        # Normalize the query (lowercase, remove extra whitespace)
        normalized_query = " ".join(query.lower().split())
        # Generate a hash of the normalized query
        return hashlib.md5(normalized_query.encode('utf-8')).hexdigest()
    
    def get(self, query):
        """Get a cached response for a query
        
        Args:
            query: The user query string
            
        Returns:
            The cached response or None if not found or expired
        """
        key = self._generate_key(query)
        
        if key in self.cache:
            cache_entry = self.cache[key]
            # Check if the entry is expired
            if time.time() - cache_entry["timestamp"] > self.cache_ttl:
                # Entry expired
                self.cache_stats["misses"] += 1
                return None
            
            # Cache hit
            self.cache_stats["hits"] += 1
            # Update access time
            self.cache[key]["last_accessed"] = time.time()
            return cache_entry["response"]
        
        # Cache miss
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, query, response):
        """Store a query response in the cache
        
        Args:
            query: The user query string
            response: The response to cache
        """
        key = self._generate_key(query)
        
        # Check if we need to evict entries (LRU policy)
        if len(self.cache) >= self.max_cache_size and key not in self.cache:
            # Find the least recently accessed entry
            lru_key = min(self.cache.items(), 
                          key=lambda x: x[1]["last_accessed"])[0]
            del self.cache[lru_key]
        
        # Store the new entry
        self.cache[key] = {
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "last_accessed": time.time()
        }
        
        self.cache_stats["size"] = len(self.cache)
        
        # Periodically save the cache index
        if self.cache_stats["size"] % 10 == 0:
            self._save_cache_index()
    
    def clear(self):
        """Clear the entire cache"""
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "size": 0}
        self._save_cache_index()
        
        # Also remove any cache files
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json") and filename != "cache_index.json":
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except Exception as e:
                    print(f"Error removing cache file {filename}: {str(e)}")
    
    def get_stats(self):
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": self.cache_stats["size"],
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": f"{hit_rate:.2f}%",
            "max_size": self.max_cache_size
        }