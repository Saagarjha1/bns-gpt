from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def hybrid_search_and_rerank(query: str, top_k_initial: int = 15, top_n_final: int = 3):
    # 1. Broad Retrieval (Top 15 Candidates via Dense + Sparse Hybrid Search)
    initial_results = index.query(
        vector=dense_embedding,
        sparse_vector=sparse_embedding,
        top_k=top_k_initial,
        include_metadata=True
    )
    
    # Format candidates for Pinecone Inference Reranker
    candidate_docs = [
        {"id": match.id, "text": match.metadata["text"]} 
        for match in initial_results.matches
    ]
    
    # 2. Cross-Encoder Precision Reranking via Pinecone Inference API
    rerank_response = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=candidate_docs,
        top_n=top_n_final,
        return_documents=True
    )
    
    # 3. Extract Top-N precision re-ranked context
    reranked_docs = [item.document.text for item in rerank_response.data]
    return reranked_docs