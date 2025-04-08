# RAG system evaluation utilities
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

class RAGEvaluator:
    def __init__(self, log_dir="data/evaluation"):
        """Initialize the RAG evaluator
        
        Args:
            log_dir: Directory to store evaluation logs
        """
        self.log_dir = log_dir
        self.current_session = {
            "queries": [],
            "start_time": time.time(),
            "metrics": {
                "latency": [],
                "retrieval_precision": [],
                "token_usage": []
            }
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def log_query(self, query: str, retrieved_chunks: List[Dict], 
                 response: str, latency: float, token_usage: Dict = None):
        """Log a query and its results for evaluation
        
        Args:
            query: The user query
            retrieved_chunks: The chunks retrieved by the system
            response: The generated response
            latency: Time taken to process the query (seconds)
            token_usage: Token usage information (if available)
        """
        query_log = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "retrieved_chunks": [
                {
                    "id": chunk.get("metadata", {}).get("child_id", "unknown"),
                    "title": chunk.get("metadata", {}).get("title", "unknown"),
                    "score": chunk.get("score", 0.0)
                } for chunk in retrieved_chunks
            ],
            "response": response,
            "latency": latency
        }
        
        if token_usage:
            query_log["token_usage"] = token_usage
        
        self.current_session["queries"].append(query_log)
        self.current_session["metrics"]["latency"].append(latency)
        
        # Save session periodically
        if len(self.current_session["queries"]) % 10 == 0:
            self.save_session()
    
    def save_session(self):
        """Save the current evaluation session"""
        # Calculate session metrics
        self.current_session["end_time"] = time.time()
        self.current_session["duration"] = self.current_session["end_time"] - self.current_session["start_time"]
        
        # Calculate average metrics
        metrics = self.current_session["metrics"]
        if metrics["latency"]:
            metrics["avg_latency"] = sum(metrics["latency"]) / len(metrics["latency"])
            metrics["min_latency"] = min(metrics["latency"])
            metrics["max_latency"] = max(metrics["latency"])
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def calculate_metrics(self, sessions_to_analyze: int = 5) -> Dict[str, Any]:
        """Calculate overall metrics from recent sessions
        
        Args:
            sessions_to_analyze: Number of recent sessions to analyze
            
        Returns:
            Dictionary of calculated metrics
        """
        # Get list of session files
        session_files = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith("session_") and filename.endswith(".json"):
                filepath = os.path.join(self.log_dir, filename)
                session_files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (newest first) and take the specified number
        session_files.sort(key=lambda x: x[1], reverse=True)
        session_files = session_files[:sessions_to_analyze]
        
        all_latencies = []
        query_count = 0
        
        # Analyze each session
        for filepath, _ in session_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                    
                    all_latencies.extend(session["metrics"]["latency"])
                    query_count += len(session["queries"])
            except Exception as e:
                print(f"Error analyzing session {filepath}: {str(e)}")
        
        # Calculate metrics
        metrics = {}
        
        if all_latencies:
            metrics["avg_latency"] = sum(all_latencies) / len(all_latencies)
            metrics["min_latency"] = min(all_latencies)
            metrics["max_latency"] = max(all_latencies)
            metrics["p95_latency"] = np.percentile(all_latencies, 95)
        
        metrics["total_queries"] = query_count
        metrics["analyzed_sessions"] = len(session_files)
        
        return metrics
    
    def reset_session(self):
        """Reset the current evaluation session"""
        # Save the current session first
        if self.current_session["queries"]:
            self.save_session()
        
        # Start a new session
        self.current_session = {
            "queries": [],
            "start_time": time.time(),
            "metrics": {
                "latency": [],
                "retrieval_precision": [],
                "token_usage": []
            }
        }

class QueryAnalyzer:
    def __init__(self, log_dir="data/evaluation"):
        """Initialize the query analyzer
        
        Args:
            log_dir: Directory containing evaluation logs
        """
        self.log_dir = log_dir
        self.frequent_queries = {}
        self.topic_clusters = {}
    
    def analyze_query_patterns(self, min_frequency: int = 2) -> Dict[str, Any]:
        """Analyze query patterns from evaluation logs
        
        Args:
            min_frequency: Minimum frequency to consider a query pattern
            
        Returns:
            Dictionary of query patterns and statistics
        """
        all_queries = []
        
        # Load all session files
        for filename in os.listdir(self.log_dir):
            if filename.startswith("session_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(self.log_dir, filename), 'r', encoding='utf-8') as f:
                        session = json.load(f)
                        for query_log in session["queries"]:
                            all_queries.append(query_log["query"])
                except Exception as e:
                    print(f"Error loading session {filename}: {str(e)}")
        
        # Count query frequencies (case-insensitive)
        query_counts = {}
        for query in all_queries:
            normalized_query = query.lower().strip()
            if normalized_query in query_counts:
                query_counts[normalized_query] += 1
            else:
                query_counts[normalized_query] = 1
        
        # Filter by minimum frequency
        frequent_queries = {q: count for q, count in query_counts.items() 
                          if count >= min_frequency}
        
        # Sort by frequency (descending)
        sorted_queries = sorted(frequent_queries.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_queries": len(all_queries),
            "unique_queries": len(query_counts),
            "frequent_queries": dict(sorted_queries),
            "frequency_threshold": min_frequency
        }