import json
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_semantic_cached_response(query: str, redis_client, similarity_threshold: float = 0.96):
    query_vector = embedding_model.encode(query).tolist()
    
    # Retrieve active cached vector queries from Redis
    cached_keys = redis_client.keys("cache:vector:*")
    
    for key in cached_keys:
        cached_data = json.loads(redis_client.get(key))
        cached_vector = cached_data["vector"]
        
        # Calculate Cosine Similarity
        cosine_sim = np.dot(query_vector, cached_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(cached_vector)
        )
        
        # Cache Hit if semantic similarity is 96%+
        if cosine_sim >= similarity_threshold:
            print(f"🎯 Semantic Cache Hit! Similarity: {cosine_sim:.4f}")
            return cached_data["response"]
            
    return None

def set_semantic_cache(query: str, response_text: str, redis_client, ttl_seconds: int = 86400):
    query_vector = embedding_model.encode(query).tolist()
    cache_key = f"cache:vector:{hash(query)}"
    
    payload = {
        "query": query,
        "vector": query_vector,
        "response": response_text
    }
    
    redis_client.setex(cache_key, ttl_seconds, json.dumps(payload))